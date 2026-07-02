import json
from contextlib import suppress
from typing import Any, Callable, Generator, Optional, Union

import requests

from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageRole,
    PromptMessageTool,
    TextPromptMessageContent,
)
from dify_plugin.errors.model import InvokeError


class GeminiNativeDocumentAdapter:
    def __init__(
        self,
        endpoint_url: Callable[[dict, str], str],
        normalize_model_parameters: Callable[[str, dict], dict],
        calc_response_usage: Callable[[str, dict, int, int], object],
    ) -> None:
        self._endpoint_url = endpoint_url
        self._normalize_model_parameters = normalize_model_parameters
        self._calc_response_usage = calc_response_usage

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
    ) -> Union[LLMResult, Generator]:
        method = "streamGenerateContent" if stream else "generateContent"
        request_url = self._endpoint_url(credentials, f"models/{model}:{method}")
        request_body = self.build_body(model, prompt_messages, model_parameters, tools, stop)
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
                timeout=(10, 300),
            )
        except Exception as error:
            raise InvokeError(f"Gemini 原生请求失败：{error}") from error

        if response.status_code >= 400:
            raise InvokeError(f"Gemini 原生请求失败，状态码：{response.status_code}，响应：{response.text}")

        if stream:
            return self._handle_stream(model, credentials, response)
        return self._handle_response(model, credentials, response)

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
            if parts:
                contents.append({"role": role, "parts": parts})
        return contents

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
            return [{"text": content}]
        if not isinstance(content, list):
            return []

        parts: list[dict] = []
        for item in content:
            if item.type == PromptMessageContentType.TEXT:
                text_content: TextPromptMessageContent = item
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
    def _inline_data_part(content: Any) -> dict:
        base64_data = getattr(content, "base64_data", "")
        if not base64_data:
            raise InvokeError("Gemini 原生文件路径需要 Dify 提供 base64_data，URL 文件输入暂未实现。")
        return {"inlineData": {"mimeType": content.mime_type, "data": base64_data}}

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
                    "parameters": function.get("parameters", {}),
                }
            )
        return [{"functionDeclarations": declarations}] if declarations else []

    def _handle_response(self, model: str, credentials: dict, response: requests.Response) -> LLMResult:
        payload = response.json()
        content = self._extract_text(payload)
        usage_payload = payload.get("usageMetadata") or {}
        usage = self._calc_response_usage(
            model,
            credentials,
            usage_payload.get("promptTokenCount", 0),
            usage_payload.get("candidatesTokenCount", 0),
        )
        return LLMResult(model=model, message=AssistantPromptMessage(content=content), usage=usage)

    def _handle_stream(self, model: str, credentials: dict, response: requests.Response) -> Generator:
        chunk_index = 0
        usage_payload: Optional[dict] = None
        finish_reason: Optional[str] = None

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
            delta = self._extract_text(event)
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

        usage = self._calc_response_usage(
            model,
            credentials,
            (usage_payload or {}).get("promptTokenCount", 0),
            (usage_payload or {}).get("candidatesTokenCount", 0),
        )
        yield LLMResultChunk(
            model=model,
            delta=LLMResultChunkDelta(
                index=chunk_index + 1,
                message=AssistantPromptMessage(content=""),
                finish_reason=finish_reason or "STOP",
                usage=usage,
            ),
        )

    @staticmethod
    def _extract_text(payload: dict) -> str:
        pieces: list[str] = []
        for candidate in payload.get("candidates") or []:
            content = candidate.get("content") or {}
            for part in content.get("parts") or []:
                text = part.get("text")
                if text:
                    pieces.append(str(text))
        return "".join(pieces)

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
