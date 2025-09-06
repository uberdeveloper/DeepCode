"""
Clean Gemini LLM Provider Implementation

This module provides a standalone Gemini integration that avoids schema conflicts
with the existing configuration system.
"""

import json
import re
import os
from typing import List, Type
from dataclasses import dataclass

import google.generativeai as genai

from mcp_agent.tracing.telemetry import get_tracer
from mcp_agent.tracing.semconv import GEN_AI_AGENT_NAME, GEN_AI_REQUEST_MODEL
from mcp_agent.tracing.token_tracking_decorator import track_tokens
from mcp_agent.workflows.llm.augmented_llm import (
    AugmentedLLM,
    MessageTypes,
    ModelT,
    MCPMessageResult,
    ProviderToMCPConverter,
    RequestParams,
)
from mcp_agent.logging.logger import get_logger
from mcp.types import TextContent, ImageContent, EmbeddedResource, TextResourceContents


@dataclass
class GeminiConfig:
    """Configuration for Gemini provider"""

    api_key: str
    default_model: str = "gemini-2.5-flash"
    max_tokens: int = 8192
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 40


class GeminiTypeConverter(ProviderToMCPConverter[dict, dict]):
    """Type converter for Gemini messages"""

    @classmethod
    def from_mcp_message_param(cls, message: MessageTypes) -> List[dict]:
        """Convert MCP message param to Gemini format"""
        if isinstance(message, str):
            return [{"role": "user", "parts": [message]}]
        elif isinstance(message, dict):
            return [cls._convert_single_message(message)]
        elif isinstance(message, list):
            result = []
            for item in message:
                if isinstance(item, str):
                    result.append({"role": "user", "parts": [item]})
                elif isinstance(item, dict):
                    result.append(cls._convert_single_message(item))
            return result
        else:
            raise TypeError(f"Unsupported message type: {type(message)}")

    @classmethod
    def _convert_single_message(cls, param: dict) -> dict:
        """Convert a single MCP message param"""
        role = "user" if param.get("role") == "user" else "model"
        content = param.get("content")

        parts = []
        if isinstance(content, TextContent):
            parts.append(content.text)
        elif isinstance(content, ImageContent):
            parts.append(
                {"inline_data": {"mime_type": content.mimeType, "data": content.data}}
            )
        elif isinstance(content, EmbeddedResource):
            if isinstance(content.resource, TextResourceContents):
                parts.append(content.resource.text)
            else:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": content.resource.mimeType,
                            "data": content.resource.blob,
                        }
                    }
                )
        elif isinstance(content, str):
            parts.append(content)
        else:
            parts.append(str(content))

        return {"role": role, "parts": parts}

    @classmethod
    def to_mcp_message_result(cls, result: dict) -> MCPMessageResult:
        """Convert Gemini response to MCP message result"""
        content_parts = []
        for part in result.get("parts", []):
            if hasattr(part, "text"):
                content_parts.append(TextContent(type="text", text=part.text))
            elif hasattr(part, "inline_data"):
                content_parts.append(
                    ImageContent(
                        type="image",
                        mimeType=part.inline_data.mime_type,
                        data=part.inline_data.data,
                    )
                )
            else:
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

    @classmethod
    def from_mcp_history(cls, mcp_history):
        """Convert MCP history to Gemini format"""
        history = []
        for message in mcp_history:
            if isinstance(message, dict):
                if message.get("role") in ["user", "assistant"]:
                    converted = cls._convert_single_message(message)
                    history.append(converted)
        return history


class GeminiProvider(AugmentedLLM[GeminiConfig, dict]):
    """
    Clean Gemini LLM provider implementation that avoids schema conflicts.
    """

    def __init__(self, *args, config=None, **kwargs):
        super().__init__(*args, type_converter=GeminiTypeConverter, **kwargs)
        self.provider = "Gemini"
        self.logger = get_logger(f"{__name__}.{self.name}" if self.name else __name__)

        # Handle configuration safely - bypass schema validation
        if config is None:
            # Create config from kwargs if provided
            api_key = kwargs.pop("api_key", None)
            if api_key:
                self.config = GeminiConfig(api_key=api_key, **kwargs)
            else:
                # Try to get from environment or use default
                import os

                api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                    "GOOGLE_API_KEY"
                )
                self.config = GeminiConfig(api_key=api_key or "default")
        else:
            self.config = config

        # Initialize Gemini
        genai.configure(api_key=self.config.api_key)

        self.default_request_params = RequestParams(
            model=self.config.default_model,
            maxTokens=self.config.max_tokens,
            systemPrompt=self.instruction,
            max_iterations=10,
            use_history=True,
        )

    @classmethod
    def convert_message_to_message_param(cls, message: dict, **kwargs) -> dict:
        """Convert message to MCP message param"""
        return message

    @track_tokens()
    async def generate(self, message, request_params: RequestParams | None = None):
        """Generate response from Gemini"""
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(
            f"{self.__class__.__name__}.{self.name}.generate"
        ) as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)

            params = self.get_request_params(request_params)
            history = []

            if params.use_history:
                mcp_history = self.history.get()
                if mcp_history:
                    history = self.type_converter.from_mcp_history(mcp_history)

            # Simplified message handling - directly use the message as string
            if isinstance(message, str):
                message_content = message
            elif isinstance(message, dict):
                # Extract content from dict format
                content = message.get("content", "")
                if hasattr(content, "text"):
                    message_content = content.text
                elif isinstance(content, str):
                    message_content = content
                else:
                    message_content = str(content)
            else:
                message_content = str(message)

            # Get available tools
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
            response = await chat.send_message_async(message_content)

            # Handle tool calls
            while response.candidates[0].finish_reason.name == "TOOL_CODE":
                tool_calls = [
                    part.function_call
                    for part in response.candidates[0].content.parts
                    if hasattr(part, "function_call")
                ]

                tool_results = []
                for tool_call in tool_calls:
                    tool_result = await self._execute_tool_call(tool_call)
                    tool_results.append(tool_result)

                response = await chat.send_message_async(tool_results)

            if params.use_history:
                self.history.set(chat.history)

            # Extract text content from response
            response_text = ""
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text"):
                    response_text += part.text

            # Create a simple text content result
            content = TextContent(type="text", text=response_text)
            return [
                MCPMessageResult(
                    role="assistant",
                    content=content,
                    model="",
                    stopReason=None,
                )
            ]

    async def generate_str(self, message, request_params: RequestParams | None = None):
        """Generate string response using direct API call to avoid format issues"""
        # Handle file paths - extract content if it's a file
        if isinstance(message, str) and os.path.exists(message):
            if message.endswith(".pdf"):
                # For PDF files, we need to extract text content first
                try:
                    extracted_text = await self._extract_pdf_text(message)
                    message = f"Please analyze the following academic paper and extract the key algorithms, methods, and implementation details. Focus on any mathematical models, machine learning approaches, or computational methods described:\n\n{extracted_text}"
                except Exception as e:
                    self.logger.error(f"Failed to extract PDF text: {e}")
                    message = f"Please analyze the following PDF file: {message}"
            elif message.endswith((".txt", ".md", ".py", ".js", ".html")):
                # For text files, read the content
                try:
                    with open(message, "r", encoding="utf-8") as f:
                        content = f.read()
                    message = (
                        f"I have analyzed the following file content:\n\n{content}"
                    )
                except Exception as e:
                    self.logger.error(f"Failed to read file: {e}")
                    message = f"Please analyze the following file: {message}"

        # Use direct Gemini API call to avoid format conversion issues
        import google.generativeai as genai

        # Create model instance
        model = genai.GenerativeModel(
            model_name=self.config.default_model,
            system_instruction=self.instruction
            or (request_params.systemPrompt if request_params else None),
        )

        # Send message directly using the simple API
        response = model.generate_content(message)

        # Check if response has valid content
        if not response.candidates or not response.candidates[0].content.parts:
            # Handle empty or filtered responses
            finish_reason = (
                response.candidates[0].finish_reason
                if response.candidates
                else "unknown"
            )
            raise ValueError(
                f"Gemini API returned empty response. Finish reason: {finish_reason}"
            )

        return response.text

    async def generate_structured(
        self,
        message,
        response_model: Type[ModelT],
        request_params: RequestParams | None = None,
    ) -> ModelT:
        """Generate structured response as Pydantic model"""
        tracer = get_tracer(self.context)
        with tracer.start_as_current_span(
            f"{self.__class__.__name__}.{self.name}.generate_structured"
        ) as span:
            span.set_attribute(GEN_AI_AGENT_NAME, self.agent.name)

            # First generate a string response
            response_str = await self.generate_str(
                message=message,
                request_params=request_params,
            )

            # Parse the JSON response and validate against the Pydantic model
            try:
                # Try to parse as JSON first (Gemini might return JSON in markdown)
                json_match = re.search(r"\{.*\}", response_str, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_data = json.loads(json_str)
                    return response_model(**parsed_data)
                else:
                    # If no JSON found, wrap in a basic response structure
                    return response_model(**{"response": response_str})

            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON response from Gemini: {e}")
                self.logger.error(f"Response content: {response_str}")
                # Fallback: return a basic response model
                fallback_data = {
                    "response": response_str,
                    "error": "JSON parsing failed",
                }
                return response_model(**fallback_data)
            except Exception as e:
                self.logger.error(f"Failed to create response model: {e}")
                fallback_data = {"response": response_str, "error": str(e)}
                return response_model(**fallback_data)

    async def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            # Check if file exists
            if not os.path.exists(pdf_path):
                return f"PDF file not found: {pdf_path}"

            # Try to use PyPDF2 or similar library
            try:
                import PyPDF2

                text_content = ""
                with open(pdf_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        text_content += page.extract_text() + "\n"
                return text_content
            except ImportError:
                # Fallback to using the file-downloader tool
                from mcp.types import CallToolRequest, CallToolRequestParams

                tool_request = CallToolRequest(
                    method="tools/call",
                    params=CallToolRequestParams(
                        name="file-downloader",
                        arguments={"file_path": pdf_path, "extract_text": True},
                    ),
                )

                result = await self.call_tool(request=tool_request, tool_call_id=None)
                return result.content[0].text if result.content else ""

        except Exception as e:
            self.logger.error(f"Failed to extract PDF text from {pdf_path}: {e}")
            return f"Unable to extract text from PDF file: {pdf_path}"

    async def _execute_tool_call(self, tool_call: dict):
        """Execute a tool call"""
        tool_name = tool_call.name
        tool_args = dict(tool_call.args)

        from mcp.types import CallToolRequest, CallToolRequestParams

        tool_call_request = CallToolRequest(
            method="tools/call",
            params=CallToolRequestParams(name=tool_name, arguments=tool_args),
        )

        result = await self.call_tool(request=tool_call_request, tool_call_id=None)

        return {
            "function_response": {
                "name": tool_name,
                "response": {
                    "content": result.content[0].text if result.content else ""
                },
            }
        }

    async def select_model(self, request_params: RequestParams | None = None) -> str:
        """Select model based on request parameters"""
        if request_params and request_params.model:
            return request_params.model
        return self.context.config.default_model


def create_gemini_provider(config: GeminiConfig, **kwargs) -> GeminiProvider:
    """Factory function to create Gemini provider instance"""
    return GeminiProvider(config=config, **kwargs)
