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
)


class GeminiSettings(BaseModel):
    api_key: str = ""
    default_model: str = "gemini-1.5-flash"


class RequestCompletionRequest(BaseModel):
    config: GeminiSettings
    payload: dict


class RequestStructuredCompletionRequest(BaseModel):
    config: GeminiSettings
    response_model: Any | None = None
    serialized_response_model: str | None = None
    response_str: str
    model: str


class GeminiAugmentedLLM(AugmentedLLM[dict, dict]):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, type_converter=MCPGeminiTypeConverter, **kwargs)
        self.provider = "Gemini"
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)

        gemini_config_dict = self.context.config.get("gemini", {})
        if self.context.secrets and "gemini" in self.context.secrets:
            gemini_config_dict.update(self.context.secrets["gemini"])
        self.gemini_settings = GeminiSettings(**gemini_config_dict)

        default_model = self.gemini_settings.default_model

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

            messages: List[dict] = []
            params = self.get_request_params(request_params)

            if params.use_history:
                messages.extend(self.history.get())

            messages.extend(self.type_converter.from_mcp_message_param(message))

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

            responses: List[dict] = []
            model = await self.select_model(params)
            if model:
                span.set_attribute(GEN_AI_REQUEST_MODEL, model)

            total_input_tokens = 0
            total_output_tokens = 0
            finish_reasons = []

            for i in range(params.max_iterations):
                genai.configure(api_key=self.gemini_settings.api_key)
                gemini_model = genai.GenerativeModel(
                    model_name=model,
                    tools=available_tools,
                    system_instruction=self.instruction or params.systemPrompt,
                )

                chat = gemini_model.start_chat(history=messages)

                # The message is the last one in the history
                last_message = messages[-1]

                response = await chat.send_message_async(last_message['parts'])

                responses.append(response.candidates[0].content)
                messages.append(response.candidates[0].content)

                if response.candidates[0].finish_reason.name == "TOOL_CODE":
                    for tool_call in response.candidates[0].content.parts:
                        if tool_call.function_call:
                            result = await self.execute_tool_call(tool_call.function_call)
                            messages.append(result)
                else:
                    break

            if params.use_history:
                self.history.set(messages)

            return responses

    async def generate_str(self, message, request_params: RequestParams | None = None):
        responses = await self.generate(message=message, request_params=request_params)
        final_text: List[str] = []
        for response in responses:
            for part in response.parts:
                if part.text:
                    final_text.append(part.text)
        return "\n".join(final_text)

    async def generate_structured(self, message, response_model: Type[ModelT], request_params: RequestParams | None = None) -> ModelT:
        response = await self.generate_str(message=message, request_params=request_params)
        # Simplified for now, will need to implement proper structured generation
        return response_model.model_validate_json(response)

    async def execute_tool_call(self, tool_call: dict):
        tool_name = tool_call.name
        tool_args = dict(tool_call.args)
        tool_call_id = "N/A"  # Gemini doesn't provide a tool_call_id

        tool_call_request = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name=tool_name, arguments=tool_args),
        )

        result = await self.call_tool(request=tool_call_request, tool_call_id=tool_call_id)

        return {
            "role": "tool",
            "parts": [{"function_response": {"name": tool_name, "response": {"content": result.content[0].text}}}],
        }

class MCPGeminiTypeConverter(ProviderToMCPConverter[dict, dict]):
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

        if hasattr(content, 'text'):
            text = content.text
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)

        return {"role": role, "parts": [{"text": text}]}

    @classmethod
    def to_mcp_message_result(cls, result: dict) -> MCPMessageResult:
        text_content = ""
        for part in result.get('parts', []):
            if 'text' in part:
                text_content += part['text']

        return MCPMessageResult(
            role="assistant",
            content=TextContent(type="text", text=text_content),
            model="",
            stopReason=None,
        )
