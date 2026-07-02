import json
from contextlib import suppress
from typing import Any, Callable, Generator, Optional, Union

import requests

from dify_plugin.entities.model.llm import LLMResult, LLMResultChunk, LLMResultChunkDelta
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    DocumentPromptMessageContent,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageRole,
    PromptMessageTool,
    TextPromptMessageContent,
)
from dify_plugin.errors.model import InvokeError

from models.llm.native.base import file_bytes


class OpenAIResponsesDocumentAdapter:
    def __init__(
        self,
        endpoint_url: Callable[[dict, str], str],
        request_headers: Callable[[dict], dict],
        normalize_model_parameters: Callable[[str, dict], dict],
        calc_response_usage: Callable[[str, dict, int, int], object],
        create_final_chunk: Callable[..., LLMResultChunk],
    ) -> None:
        self._endpoint_url = endpoint_url
        self._request_headers = request_headers
        self._normalize_model_parameters = normalize_model_parameters
        self._calc_response_usage = calc_response_usage
        self._create_final_chunk = create_final_chunk

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
        request_url = self._endpoint_url(credentials, "responses")
        request_body = self.build_body(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters,
            tools=tools,
            stop=stop,
            stream=stream,
            user=user,
        )

        try:
            response = requests.post(
                request_url,
                headers=self._request_headers(credentials),
                json=request_body,
                stream=stream,
                timeout=(10, 300),
            )
        except Exception as error:
            raise InvokeError(f"OpenAI Responses 请求失败：{error}") from error

        if response.status_code >= 400:
            raise InvokeError(f"OpenAI Responses 请求失败，状态码：{response.status_code}，响应：{response.text}")

        if stream:
            return self._handle_stream(model, credentials, response, prompt_messages)
        return self._handle_response(model, credentials, response)

    def build_body(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]],
        stop: Optional[list[str]],
        stream: bool,
        user: Optional[str],
    ) -> dict:
        normalized_parameters = self._normalize_model_parameters(model, model_parameters)
        body: dict[str, Any] = {
            "model": model,
            "input": self._convert_messages(credentials, prompt_messages),
            "stream": stream,
        }

        if user:
            body["user"] = user
        if stop:
            body["stop"] = stop

        parameter_map = {
            "max_tokens": "max_output_tokens",
            "max_completion_tokens": "max_output_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
        }
        for source_name, target_name in parameter_map.items():
            if source_name in normalized_parameters and normalized_parameters[source_name] is not None:
                body[target_name] = normalized_parameters[source_name]

        reasoning_effort = normalized_parameters.get("reasoning_effort")
        if reasoning_effort and reasoning_effort != "none":
            body["reasoning"] = {"effort": reasoning_effort}

        text_format = self._text_format(normalized_parameters)
        if text_format:
            body["text"] = {"format": text_format}

        if tools:
            body["tools"] = self._convert_tools(tools)

        return body

    def _convert_messages(self, credentials: dict, prompt_messages: list[PromptMessage]) -> list[dict]:
        responses_input: list[dict] = []
        for prompt_message in prompt_messages:
            role = self._role(prompt_message.role)
            content = prompt_message.content
            if isinstance(content, str):
                content_parts = [{"type": "input_text", "text": content}]
            elif isinstance(content, list):
                content_parts = []
                for part in content:
                    content_parts.extend(self._convert_content_part(part, credentials))
            else:
                content_parts = []

            if content_parts:
                responses_input.append({"role": role, "content": content_parts})
        return responses_input

    @staticmethod
    def _role(role: PromptMessageRole) -> str:
        if role == PromptMessageRole.ASSISTANT:
            return "assistant"
        if role == PromptMessageRole.SYSTEM:
            return "system"
        return "user"

    def _convert_content_part(self, content: Any, credentials: dict) -> list[dict]:
        if content.type == PromptMessageContentType.TEXT:
            text_content: TextPromptMessageContent = content
            return [{"type": "input_text", "text": text_content.data}]
        if content.type == PromptMessageContentType.IMAGE:
            image_content: ImagePromptMessageContent = content
            return [{"type": "input_image", "image_url": image_content.data}]
        if content.type == PromptMessageContentType.DOCUMENT:
            document_content: DocumentPromptMessageContent = content
            uploaded_file_id = self._upload_file(credentials, document_content)
            if uploaded_file_id:
                return [{"type": "input_file", "file_id": uploaded_file_id}]
            return [
                {
                    "type": "input_file",
                    "filename": document_content.filename or "document",
                    "file_data": document_content.data,
                }
            ]
        if content.type in {PromptMessageContentType.AUDIO, PromptMessageContentType.VIDEO}:
            raise InvokeError("GPT 原生文档路径暂不支持音频/视频输入，请使用支持该模态的模型原生路径。")
        return []

    def _upload_file(self, credentials: dict, document_content: DocumentPromptMessageContent) -> Optional[str]:
        try:
            headers = dict(self._request_headers(credentials))
            headers.pop("Content-Type", None)
            response = requests.post(
                self._endpoint_url(credentials, "files"),
                headers=headers,
                data={"purpose": "user_data"},
                files={
                    "file": (
                        document_content.filename or "document",
                        file_bytes(document_content),
                        document_content.mime_type,
                    )
                },
                timeout=(10, 300),
            )
        except Exception:
            return None

        if response.status_code >= 400:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        file_id = payload.get("id")
        return file_id if isinstance(file_id, str) and file_id else None

    @staticmethod
    def _text_format(model_parameters: dict) -> Optional[dict]:
        response_format = model_parameters.get("response_format")
        if response_format == "json_object":
            return {"type": "json_object"}
        if response_format != "json_schema":
            return None

        json_schema = model_parameters.get("json_schema")
        if not json_schema:
            return None
        if isinstance(json_schema, str):
            with suppress(ValueError):
                json_schema = json.loads(json_schema)

        if isinstance(json_schema, dict) and "schema" in json_schema:
            schema = json_schema.get("schema")
            name = json_schema.get("name") or "structured_output"
            strict = json_schema.get("strict", True)
        else:
            schema = json_schema
            name = "structured_output"
            strict = True

        return {
            "type": "json_schema",
            "name": name,
            "schema": schema,
            "strict": strict,
        }

    @staticmethod
    def _convert_tools(tools: list[PromptMessageTool]) -> list[dict]:
        responses_tools: list[dict] = []
        for tool in tools:
            tool_dict = tool.model_dump(mode="json") if hasattr(tool, "model_dump") else dict(tool)
            function = tool_dict.get("function", tool_dict)
            name = function.get("name")
            if not name:
                continue
            responses_tools.append(
                {
                    "type": "function",
                    "name": name,
                    "description": function.get("description", ""),
                    "parameters": function.get("parameters", {}),
                }
            )
        return responses_tools

    def _handle_response(self, model: str, credentials: dict, response: requests.Response) -> LLMResult:
        payload = response.json()
        content = self._extract_output_text(payload)
        usage_payload = payload.get("usage") or {}
        usage = self._calc_response_usage(
            model,
            credentials,
            usage_payload.get("input_tokens", 0),
            usage_payload.get("output_tokens", 0),
        )
        return LLMResult(
            model=payload.get("model", model),
            message=AssistantPromptMessage(content=content),
            usage=usage,
        )

    def _handle_stream(
        self,
        model: str,
        credentials: dict,
        response: requests.Response,
        prompt_messages: list[PromptMessage],
    ) -> Generator:
        chunk_index = 0
        full_content = ""
        usage_payload: Optional[dict] = None
        finish_reason: Optional[str] = None

        for raw_line in response.iter_lines(decode_unicode=False):
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace").strip()
            if line.startswith(":") or not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                continue

            try:
                event = json.loads(data)
            except ValueError:
                continue

            event_type = event.get("type")
            if event_type in {"response.output_text.delta", "response.refusal.delta"}:
                delta = event.get("delta") or ""
                if not delta:
                    continue
                chunk_index += 1
                full_content += delta
                yield LLMResultChunk(
                    model=model,
                    delta=LLMResultChunkDelta(
                        index=chunk_index,
                        message=AssistantPromptMessage(content=delta),
                    ),
                )
            elif event_type in {"response.completed", "response.incomplete"}:
                response_payload = event.get("response") or {}
                usage_payload = response_payload.get("usage") or usage_payload
                finish_reason = response_payload.get("status") or event_type.removeprefix("response.")
            elif event_type == "response.failed":
                response_payload = event.get("response") or {}
                error = response_payload.get("error") or event.get("error") or {}
                raise InvokeError(f"OpenAI Responses 流式请求失败：{error}")

        yield self._create_final_chunk(
            index=chunk_index + 1,
            message=AssistantPromptMessage(content=""),
            finish_reason=finish_reason or "stop",
            usage=self._usage_to_chat_usage(usage_payload),
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            full_content=full_content,
        )

    @staticmethod
    def _extract_output_text(payload: dict) -> str:
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]

        pieces: list[str] = []
        for item in payload.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if isinstance(content, dict) and content.get("type") in {"output_text", "refusal"}:
                    text = content.get("text") or content.get("refusal") or ""
                    if text:
                        pieces.append(str(text))
        return "".join(pieces)

    @staticmethod
    def _usage_to_chat_usage(usage: Optional[dict]) -> dict:
        usage = usage or {}
        return {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        }
