import json
import functools
from typing import Any, Dict, Iterable, List, Type, cast

from pydantic import BaseModel
import google.generativeai as genai
from opentelemetry import trace

from mcp_agent.executor.workflow_task import workflow_task
from mcp_agent.tracing.telemetry import get_tracer, telemetry
from mcp_agent.tracing.token_tracking_decorator import track_tokens
from mcp_agent.tracing.semconv import (
    GEN_AI_AGENT_NAME,
    GEN_AI_REQUEST_MODEL,
    GEN_AI_RESPONSE_FINISH_REASONS,
    GEN_AI_TOOL_CALL_ID,
    GEN_AI_TOOL_NAME,
    GEN_AI_USAGE_INPUT_TOKENS,
    GEN_AI_USAGE_OUTPUT_TOKENS,
)
from mcp_agent.tracing.telemetry import is_otel_serializable
from mcp_agent.utils.common import ensure_serializable
from mcp_agent.utils.pydantic_type_serializer import serialize_model, deserialize_model
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageTypes,
    ModelT,
    MCPMessageParam,
    MCPMessageResult,
    ProviderToMCPConverter,
    RequestParams,
    CallToolResult,
)
from mcp_agent.logging.logger import get_logger
from mcp.types import (
    CallToolRequestParams,
    CallToolRequest,
    ImageContent,
    TextContent,
    EmbeddedResource,
    TextResourceContents,
)


class GeminiAugmentedLLM(AugmentedLLM[dict, dict]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, type_converter=MCPGeminiTypeConverter, **kwargs)
        self.provider = "Gemini"
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)

        default_model = "gemini-1.5-flash"
        if hasattr(self.context.config, "gemini") and self.context.config.gemini:
            if hasattr(self.context.config.gemini, "default_model"):
                default_model = self.context.config.gemini.default_model
            if hasattr(self.context.config.gemini, "api_key"):
                genai.configure(api_key=self.context.config.gemini.api_key)

        self.default_request_params = self.default_request_params or RequestParams(
            model=default_model,
            maxTokens=8192,
            systemPrompt=self.instruction,
            max_iterations=10,
            use_history=True,
        )

    @classmethod
    def convert_message_to_message_param(cls, message: dict, **kwargs) -> dict:
        return message

    @track_tokens()
    async def generate(self, message, request_params: RequestParams | None = None):
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(f"{self.__class__.__name__}.{self.name}.generate") as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)

            params = self.get_request_params(request_params)

            history = []
            if params.use_history:
                mcp_history = self.history.get()
                if mcp_history:
                    history = self.type_converter.from_mcp_history(mcp_history)

            user_messages = self.type_converter.from_mcp_message_param(message)

            list_tools_result = await self.agent.list_tools()
            available_tools = [
                {
                    "function_declarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        }
                    ]
                }
                for tool in list_tools_result.tools
            ]

            model = await self.select_model(params)
            if model:
                span.set_attribute(GEN_AI_REQUEST_MODEL, model)

            gemini_model = genai.GenerativeModel(
                model_name=model,
                tools=available_tools,
                system_instruction=self.instruction or params.systemPrompt,
            )

            chat = gemini_model.start_chat(history=history)

            response = await chat.send_message_async(user_messages)

            while response.candidates[0].finish_reason.name == "TOOL_CODE":
                tool_calls = [part.function_call for part in response.candidates[0].content.parts if hasattr(part, 'function_call')]

                tool_results = []
                for tool_call in tool_calls:
                    tool_result = await self.execute_tool_call(tool_call)
                    tool_results.append(tool_result)

                response = await chat.send_message_async(tool_results)

            if params.use_history:
                self.history.set(chat.history)

            return [self.type_converter.to_mcp_message_result(response.candidates[0].content)]

    async def generate_str(self, message, request_params: RequestParams | None = None):
        responses = await self.generate(message=message, request_params=request_params)
        return responses[0].content.text

    async def execute_tool_call(self, tool_call: dict):
        tool_name = tool_call.name
        tool_args = dict(tool_call.args)

        tool_call_request = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name=tool_name, arguments=tool_args),
        )

        result = await self.call_tool(request=tool_call_request, tool_call_id=None)

        return {
            "function_response": {
                "name": tool_name,
                "response": {"content": result.content[0].text if result.content else ""},
            }
        }

class MCPGeminiTypeConverter(ProviderToMCPConverter[dict, dict]):
    logger = get_logger(__name__)

    @classmethod
    def from_mcp_history(cls, history: List[MCPMessageParam]) -> List[dict]:
        return [cls.from_mcp_message_param_single(h) for h in history]

    @classmethod
    def from_mcp_message_param(cls, message: MessageTypes) -> List[dict]:
        if isinstance(message, str):
            return [{"role": "user", "parts": [{"text": message}]}]
        if isinstance(message, dict):
            return [cls.from_mcp_message_param_single(message)]
        if isinstance(message, list):
            result = []
            for item in message:
                if isinstance(item, str):
                    result.append({"role": "user", "parts": [{"text": item}]})
                elif isinstance(item, dict):
                    result.append(cls.from_mcp_message_param_single(item))
            return result
        raise TypeError(f"Unsupported message type: {type(message)}")

    @classmethod
    def from_mcp_message_param_single(cls, param: dict) -> dict:
        role = "user" if param.get("role") == "user" else "model"
        content = param.get("content")

        parts = []
        if isinstance(content, TextContent):
            parts.append({"text": content.text})
        elif isinstance(content, ImageContent):
            parts.append({"inline_data": {"mime_type": content.mimeType, "data": content.data}})
        elif isinstance(content, EmbeddedResource):
            if isinstance(content.resource, TextResourceContents):
                parts.append({"text": content.resource.text})
            else: # BlobResourceContents
                parts.append({"inline_data": {"mime_type": content.resource.mimeType, "data": content.resource.blob}})
        elif isinstance(content, str):
            parts.append({"text": content})
        else:
            parts.append({"text": str(content)})

        return {"role": role, "parts": parts}

    @classmethod
    def to_mcp_message_result(cls, result: dict) -> MCPMessageResult:
        content_parts = []
        for part in result.get('parts', []):
            if hasattr(part, 'text'):
                content_parts.append(TextContent(type="text", text=part.text))
            elif hasattr(part, 'inline_data'):
                cls.logger.warning("Received an image from Gemini, but it will be converted to a string representation.")
                content_parts.append(ImageContent(type="image", mimeType=part.inline_data.mime_type, data=part.inline_data.data))
            else:
                cls.logger.warning(f"Received an unknown part from Gemini, it will be converted to a string representation: {part}")
                content_parts.append(TextContent(type="text", text=str(part)))

        if len(content_parts) == 1:
            content = content_parts[0]
        else:
            text_content = ""
            for part in content_parts:
                if isinstance(part, TextContent):
                    text_content += part.text
                else:
                    text_content += str(part)
            content = TextContent(type="text", text=text_content)

        return MCPMessageResult(
            role="assistant",
            content=content,
            model="",
            stopReason=None,
        )
