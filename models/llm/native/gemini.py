import json
from base64 import b64encode
from contextlib import suppress
from typing import Any, Callable, Generator, Optional, Union
from urllib.parse import urlparse

import requests

from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageRole,
    PromptMessageTool,
    TextPromptMessageContent,
    ToolPromptMessage,
)
from dify_plugin.errors.model import InvokeError

from models.llm.invocation_logging import http_response_summary
from models.llm.parameter_conversion import build_web_search_tool


DEFAULT_THOUGHT_SIGNATURE = b64encode(b"skip_thought_signature_validator").decode("ascii")


class GeminiNativeDocumentAdapter:
    def __init__(
        self,
        endpoint_url: Callable[[dict, str], str],
        normalize_model_parameters: Callable[[str, dict], dict],
        calc_response_usage: Callable[[str, dict, int, int], object],
        build_dify_usage: Optional[Callable[[str, dict, dict], object]] = None,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._normalize_model_parameters = normalize_model_parameters
        self._build_dify_usage = build_dify_usage or (
            lambda model, credentials, raw_usage: calc_response_usage(
                model,
                credentials,
                raw_usage.get("input_tokens", 0),
                raw_usage.get("output_tokens", 0),
            )
        )

    def invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]],
        stop: Optional[list[str]],
        stream: bool,
        user: Optional[str],
        invocation_log=None,
    ) -> Union[LLMResult, Generator]:
        method = "streamGenerateContent" if stream else "generateContent"
        request_url = self._endpoint_url(credentials, f"models/{model}:{method}")
        request_body = self.build_body(model, prompt_messages, model_parameters, tools, stop)
        if invocation_log is not None:
            invocation_log.set_replay_request(endpoint=request_url, body=request_body)
        params = {"key": credentials["api_key"]}
        if stream:
            params["alt"] = "sse"

        try:
            response = requests.post(
                request_url,
                headers={"Content-Type": "application/json"},
                params=params,
                json=request_body,
                stream=stream,
                timeout=(10, 120) if stream else (10, 300),
            )
        except Exception as error:
            raise InvokeError(f"Gemini 原生请求失败：{error}") from error

        if invocation_log is not None:
            invocation_log.set_response(
                http=http_response_summary(response),
                provider_request_id=self._provider_request_id(response.headers) or None,
            )

        if response.status_code >= 400:
            raise InvokeError(f"Gemini 原生请求失败，状态码：{response.status_code}，响应：{response.text}")

        if stream:
            return self._handle_stream(model, credentials, response, invocation_log)
        return self._handle_response(model, credentials, response, invocation_log)

    @staticmethod
    def _provider_request_id(headers: object) -> str:
        get = getattr(headers, "get", None)
        if not callable(get):
            return ""
        return (
            get("x-request-id")
            or get("x-goog-request-id")
            or get("x-google-request-id")
            or ""
        )

    def build_body(
        self,
        model: str,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]],
        stop: Optional[list[str]],
    ) -> dict:
        normalized_parameters = self._normalize_model_parameters(model, model_parameters)
        body: dict[str, Any] = {"contents": self._convert_messages(prompt_messages)}

        system_instruction = self._system_instruction(prompt_messages)
        if system_instruction:
            body["systemInstruction"] = system_instruction

        generation_config = self._generation_config(normalized_parameters, stop)
        if generation_config:
            body["generationConfig"] = generation_config

        gemini_tools = self._convert_tools(tools or [])
        web_search_tool = build_web_search_tool(model, normalized_parameters)
        if web_search_tool == {"google_search": {}}:
            web_search_tool = {"googleSearch": {}}
        if web_search_tool:
            gemini_tools.append(web_search_tool)
            if tools:
                body["toolConfig"] = {"includeServerSideToolInvocations": True}
        if gemini_tools:
            body["tools"] = gemini_tools

        return body

    def _convert_messages(self, prompt_messages: list[PromptMessage]) -> list[dict]:
        contents: list[dict] = []
        for prompt_message in prompt_messages:
            if prompt_message.role == PromptMessageRole.SYSTEM:
                continue
            role = "model" if prompt_message.role == PromptMessageRole.ASSISTANT else "user"
            parts = self._convert_message_parts(prompt_message)
            if isinstance(prompt_message, AssistantPromptMessage):
                parts = [self._with_thought_signature(part) for part in parts]
                parts.extend(self._function_call_parts(prompt_message))
            elif isinstance(prompt_message, ToolPromptMessage):
                parts = self._function_response_parts(prompt_message)
            if parts:
                if contents and contents[-1]["role"] == role:
                    contents[-1]["parts"].extend(parts)
                else:
                    contents.append({"role": role, "parts": parts})
        return contents

    @staticmethod
    def _with_thought_signature(part: dict) -> dict:
        return {**part, "thoughtSignature": DEFAULT_THOUGHT_SIGNATURE}

    @staticmethod
    def _system_instruction(prompt_messages: list[PromptMessage]) -> Optional[dict]:
        parts: list[dict] = []
        for prompt_message in prompt_messages:
            if prompt_message.role != PromptMessageRole.SYSTEM:
                continue
            if isinstance(prompt_message.content, str) and prompt_message.content:
                parts.append({"text": prompt_message.content})
        if not parts:
            return None
        return {"parts": parts}

    def _convert_message_parts(self, prompt_message: PromptMessage) -> list[dict]:
        content = prompt_message.content
        if isinstance(content, str):
            return [{"text": content}] if content else []
        if not isinstance(content, list):
            return []

        parts: list[dict] = []
        for item in content:
            if item.type == PromptMessageContentType.TEXT:
                text_content: TextPromptMessageContent = item
                if text_content.data:
                    parts.append({"text": text_content.data})
            elif item.type in {
                PromptMessageContentType.IMAGE,
                PromptMessageContentType.DOCUMENT,
                PromptMessageContentType.AUDIO,
                PromptMessageContentType.VIDEO,
            }:
                parts.append(self._inline_data_part(item))
        return parts

    @staticmethod
    def _function_call_parts(message: AssistantPromptMessage) -> list[dict]:
        parts: list[dict] = []
        for tool_call in message.tool_calls:
            try:
                args = json.loads(tool_call.function.arguments)
            except (TypeError, ValueError):
                raise InvokeError(f"Gemini 原生 functionCall 参数不是有效 JSON：{tool_call.function.name}") from None
            if not isinstance(args, dict):
                raise InvokeError(f"Gemini 原生 functionCall 参数必须是 JSON 对象：{tool_call.function.name}")
            parts.append(
                {
                    "functionCall": {
                        "name": tool_call.function.name,
                        "args": args,
                        "id": tool_call.id,
                    },
                    "thoughtSignature": DEFAULT_THOUGHT_SIGNATURE,
                }
            )
        return parts

    @staticmethod
    def _function_response_parts(message: ToolPromptMessage) -> list[dict]:
        if not message.name:
            return []

        return [
            {
                "functionResponse": {
                    "name": message.name,
                    "response": {"response": message.content},
                    "id": message.tool_call_id,
                }
            }
        ]

    @staticmethod
    def _inline_data_part(content: Any) -> dict:
        base64_data = getattr(content, "base64_data", "")
        if base64_data:
            return {"inlineData": {"mimeType": content.mime_type, "data": base64_data}}

        file_url = GeminiNativeDocumentAdapter._public_file_url(content)
        if file_url:
            return {"fileData": {"mimeType": content.mime_type, "fileUri": file_url}}

        raise InvokeError("Gemini 原生文件路径需要 Dify 提供 base64_data 或公开 http/https URL。")

    @staticmethod
    def _public_file_url(content: Any) -> str:
        for value in (getattr(content, "url", ""), getattr(content, "data", "")):
            if not isinstance(value, str):
                continue
            parsed = urlparse(value.strip())
            if parsed.scheme in {"http", "https"} and parsed.hostname not in {
                None,
                "localhost",
                "127.0.0.1",
                "0.0.0.0",
                "web",
                "nginx",
                "api",
            }:
                return value
        return ""

    @staticmethod
    def _generation_config(model_parameters: dict, stop: Optional[list[str]]) -> dict:
        config: dict[str, Any] = {}
        parameter_map = {
            "temperature": "temperature",
            "top_p": "topP",
            "max_tokens": "maxOutputTokens",
            "max_completion_tokens": "maxOutputTokens",
        }
        for source_name, target_name in parameter_map.items():
            if source_name in model_parameters and model_parameters[source_name] is not None:
                config[target_name] = model_parameters[source_name]

        if stop:
            config["stopSequences"] = stop

        if model_parameters.get("response_format") in {"json_object", "json_schema"}:
            config["responseMimeType"] = "application/json"
        if model_parameters.get("response_format") == "json_schema" and model_parameters.get("json_schema"):
            json_schema = model_parameters["json_schema"]
            if isinstance(json_schema, str):
                with suppress(ValueError):
                    json_schema = json.loads(json_schema)
            config["responseSchema"] = json_schema.get("schema", json_schema) if isinstance(json_schema, dict) else json_schema

        thinking_config = model_parameters.get("thinking_config")
        if isinstance(thinking_config, dict) and thinking_config:
            config["thinkingConfig"] = {
                snake_to_lower_camel(key): value
                for key, value in thinking_config.items()
                if value is not None
            }

        return config

    @staticmethod
    def _convert_tools(tools: list[PromptMessageTool]) -> list[dict]:
        declarations: list[dict] = []
        for tool in tools:
            tool_dict = tool.model_dump(mode="json") if hasattr(tool, "model_dump") else dict(tool)
            function = tool_dict.get("function", tool_dict)
            name = function.get("name")
            if not name:
                continue
            declarations.append(
                {
                    "name": name,
                    "description": function.get("description", ""),
                    "parameters": GeminiNativeDocumentAdapter._normalize_function_schema(
                        function.get("parameters", {})
                    ),
                }
            )
        return [{"functionDeclarations": declarations}] if declarations else []

    @staticmethod
    def _normalize_function_schema(schema: Any, property_name: Optional[str] = None) -> Any:
        """Normalize Dify JSON Schema features that Gemini function declarations reject."""
        if schema is True:
            return {"type": "object"} if property_name == "params" else {}
        if schema is False:
            return {}
        if not isinstance(schema, dict):
            return schema

        normalized: dict[str, Any] = {}
        for key, value in schema.items():
            if key == "additionalProperties":
                continue
            if key == "description" and value is None:
                continue
            if key == "type" and isinstance(value, list):
                non_null_types = [item for item in value if item != "null"]
                if len(non_null_types) == 1:
                    normalized[key] = non_null_types[0]
                else:
                    normalized[key] = value
                continue
            if key == "properties" and isinstance(value, dict):
                normalized[key] = {
                    name: GeminiNativeDocumentAdapter._normalize_function_schema(item, name)
                    for name, item in value.items()
                    if item is not False
                }
                continue
            if key == "items" and value is False:
                normalized["maxItems"] = 0
                continue
            normalized[key] = GeminiNativeDocumentAdapter._normalize_function_schema(value)
        if isinstance(normalized.get("properties"), dict) and isinstance(normalized.get("required"), list):
            normalized["required"] = [
                name for name in normalized["required"] if name in normalized["properties"]
            ]
        return normalized

    def _handle_response(self, model: str, credentials: dict, response: requests.Response, invocation_log=None) -> LLMResult:
        payload = response.json()
        content = self._extract_text(payload)
        tool_calls = self._extract_tool_calls(payload)
        usage_payload = payload.get("usageMetadata") or {}
        raw_usage = self._raw_usage(usage_payload)
        if invocation_log is not None:
            invocation_log.set_response(upstream_usage=raw_usage)
        usage = self._build_dify_usage(model, credentials, raw_usage)
        return LLMResult(
            model=model,
            message=AssistantPromptMessage(content=content, tool_calls=tool_calls),
            usage=usage,
        )

    def _handle_stream(self, model: str, credentials: dict, response: requests.Response, invocation_log=None) -> Generator:
        chunk_index = 0
        usage_payload: Optional[dict] = None
        finish_reason: Optional[str] = None
        in_thought = False
        pending_tool_calls: list[AssistantPromptMessage.ToolCall] = []

        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line.startswith(":") or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            try:
                event = json.loads(data)
            except ValueError:
                continue

            usage_payload = event.get("usageMetadata") or usage_payload
            finish_reason = self._extract_finish_reason(event) or finish_reason
            delta, in_thought = self._extract_stream_text(event, in_thought)
            pending_tool_calls.extend(self._extract_tool_calls(event))
            if not delta:
                continue
            chunk_index += 1
            yield LLMResultChunk(
                model=model,
                delta=LLMResultChunkDelta(
                    index=chunk_index,
                    message=AssistantPromptMessage(content=delta),
                ),
            )

        raw_usage = self._raw_usage(usage_payload or {})
        if invocation_log is not None:
            invocation_log.set_response(upstream_usage=raw_usage)
        usage = self._build_dify_usage(model, credentials, raw_usage)
        if in_thought:
            chunk_index += 1
            yield LLMResultChunk(
                model=model,
                delta=LLMResultChunkDelta(
                    index=chunk_index,
                    message=AssistantPromptMessage(content="</think>\n"),
                ),
            )

        yield LLMResultChunk(
            model=model,
            delta=LLMResultChunkDelta(
                index=chunk_index + 1,
                message=AssistantPromptMessage(content="", tool_calls=pending_tool_calls),
                finish_reason="tool_calls" if pending_tool_calls else finish_reason or "STOP",
                usage=usage,
            ),
        )

    @staticmethod
    def _raw_usage(usage_payload: dict) -> dict:
        return {
            "input_tokens": usage_payload.get("promptTokenCount", 0),
            "output_tokens": usage_payload.get("candidatesTokenCount", 0),
            "total_tokens": usage_payload.get("totalTokenCount"),
            "prompt_tokens_details": {
                "cached_tokens": usage_payload.get("cachedContentTokenCount"),
            },
            "completion_tokens_details": {
                "reasoning_tokens": usage_payload.get("thoughtsTokenCount"),
            },
        }

    @staticmethod
    def _extract_text(payload: dict) -> str:
        text, in_thought = GeminiNativeDocumentAdapter._extract_stream_text(payload, in_thought=False)
        return text + ("</think>\n" if in_thought else "")

    @staticmethod
    def _extract_stream_text(payload: dict, in_thought: bool) -> tuple[str, bool]:
        """将 Gemini ``thought`` part 转为 Dify 可识别的 ``<think>`` 流。"""
        pieces: list[str] = []
        for candidate in payload.get("candidates") or []:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                text = part.get("text")
                if not text:
                    continue
                is_thought = part.get("thought") is True
                if is_thought and not in_thought:
                    pieces.append("<think>\n")
                    in_thought = True
                elif not is_thought and in_thought:
                    pieces.append("</think>\n")
                    in_thought = False
                pieces.append(str(text))
        return "".join(pieces), in_thought

    @staticmethod
    def _extract_tool_calls(payload: dict) -> list[AssistantPromptMessage.ToolCall]:
        """把 Gemini 原生 ``functionCall`` part 转换为 Dify 工具调用。"""
        tool_calls: list[AssistantPromptMessage.ToolCall] = []
        for candidate_index, candidate in enumerate(payload.get("candidates") or []):
            content = candidate.get("content") or {}
            for part_index, part in enumerate(content.get("parts") or []):
                function_call = part.get("functionCall")
                if not isinstance(function_call, dict) or not function_call.get("name"):
                    continue
                tool_calls.append(
                    AssistantPromptMessage.ToolCall(
                        id=function_call.get("id") or f"gemini-call-{candidate_index}-{part_index}",
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name=function_call["name"],
                            arguments=json.dumps(function_call.get("args") or {}, ensure_ascii=False),
                        ),
                    )
                )
        return tool_calls

    @staticmethod
    def _extract_finish_reason(payload: dict) -> Optional[str]:
        for candidate in payload.get("candidates") or []:
            finish_reason = candidate.get("finishReason")
            if finish_reason:
                return finish_reason
        return None


def snake_to_lower_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])
