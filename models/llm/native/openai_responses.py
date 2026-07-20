import json
import os
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
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)
from dify_plugin.errors.model import InvokeError

from models.llm.invocation_logging import http_response_summary, responses_payload_summary
from models.llm.native.base import file_bytes
from models.llm.parameter_conversion import build_web_search_tool


def _debug(message: str, *args: object) -> None:
    if os.getenv("FLYFUS_RESPONSES_DEBUG") != "1":
        return
    if args:
        message = message % args
    with suppress(Exception):
        with open("/tmp/flyfus_responses_debug.log", "a", encoding="utf-8") as debug_file:
            debug_file.write(message + "\n")


class OpenAIResponsesAdapter:
    """OpenAI 系列 Responses API 路径，包含文本、图片、文件、结构化输出和工具调用。"""

    def __init__(
        self,
        endpoint_url: Callable[[dict, str], str],
        request_headers: Callable[[dict], dict],
        normalize_model_parameters: Callable[[str, dict], dict],
        calc_response_usage: Callable[[str, dict, int, int], object],
        create_final_chunk: Callable[..., LLMResultChunk],
        build_dify_usage: Optional[Callable[[str, dict, dict], object]] = None,
    ) -> None:
        self._endpoint_url = endpoint_url
        self._request_headers = request_headers
        self._normalize_model_parameters = normalize_model_parameters
        self._build_dify_usage = build_dify_usage or (
            lambda model, credentials, raw_usage: calc_response_usage(
                model,
                credentials,
                raw_usage.get("input_tokens", 0),
                raw_usage.get("output_tokens", 0),
            )
        )
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
        invocation_log=None,
    ) -> Union[LLMResult, Generator]:
        request_body = self._build_body(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters,
            tools=tools,
            stop=stop,
            stream=stream,
            user=user,
        )
        self._log_request_summary(model, stream, prompt_messages, model_parameters, tools, request_body)

        try:
            response = requests.post(
                self._endpoint_url(credentials, "responses"),
                headers=self._request_headers(credentials),
                json=request_body,
                stream=stream,
                timeout=(10, 120) if stream else (10, 300),
            )
        except Exception as error:
            _debug(
                "[flyfus responses] request_failed model=%s stream=%s error_type=%s error=%s",
                model,
                stream,
                type(error).__name__,
                error,
            )
            raise InvokeError(f"OpenAI Responses 请求失败：{error}") from error

        request_id = response.headers.get("x-request-id") or response.headers.get("openai-request-id") or ""
        if invocation_log is not None:
            invocation_log.set_response(
                http=http_response_summary(response),
                provider_request_id=request_id or None,
            )
        _debug(
            "[flyfus responses] response_headers model=%s stream=%s status=%s request_id=%s content_type=%s",
            model,
            stream,
            response.status_code,
            request_id,
            response.headers.get("content-type", ""),
        )
        if response.status_code >= 400:
            if invocation_log is not None:
                invocation_log.set_response(error_body=response.text[:2000])
            _debug(
                "[flyfus responses] response_error model=%s status=%s body_head=%s",
                model,
                response.status_code,
                response.text[:1000],
            )
            raise InvokeError(f"OpenAI Responses 请求失败，状态码：{response.status_code}，响应：{response.text}")

        if stream:
            return self._handle_stream(model, credentials, response, prompt_messages, invocation_log=invocation_log)
        return self._handle_response(model, credentials, response, prompt_messages, invocation_log=invocation_log)

    def _build_body(
        self,
        *,
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
            "input": self._convert_messages(model, credentials, prompt_messages),
            "stream": stream,
        }

        if stop:
            body["stop"] = stop
        if user:
            body["user"] = user

        for source_name, target_name in {
            "max_tokens": "max_output_tokens",
            "max_completion_tokens": "max_output_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "presence_penalty": "presence_penalty",
            "frequency_penalty": "frequency_penalty",
            "service_tier": "service_tier",
            "verbosity": "verbosity",
        }.items():
            if source_name in normalized_parameters and normalized_parameters[source_name] is not None:
                body[target_name] = normalized_parameters[source_name]

        reasoning_effort = normalized_parameters.get("reasoning_effort")
        if reasoning_effort and reasoning_effort != "none":
            body["reasoning"] = {"effort": reasoning_effort}

        text_format = self._text_format(normalized_parameters)
        if text_format:
            body["text"] = {"format": text_format}

        converted_tools = self._convert_tools(tools) or []
        web_search_tool = build_web_search_tool(model, normalized_parameters)
        if web_search_tool:
            converted_tools.append(web_search_tool)
        if converted_tools:
            body["tools"] = converted_tools
            body.setdefault("tool_choice", "auto")

        return body

    def _convert_messages(self, model: str, credentials: dict, prompt_messages: list[PromptMessage]) -> list[dict]:
        input_items: list[dict] = []
        for message in prompt_messages:
            if isinstance(message, SystemPromptMessage):
                input_items.append(
                    {
                        "type": "message",
                        "role": "system",
                        "content": self._text_from_content(message.content),
                    }
                )
            elif isinstance(message, UserPromptMessage):
                input_items.append(
                    {
                        "type": "message",
                        "role": "user",
                        "content": self._user_content_parts(model, credentials, message.content),
                    }
                )
            elif isinstance(message, AssistantPromptMessage):
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        input_items.append(
                            {
                                "type": "function_call",
                                "call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            }
                        )
                else:
                    input_items.append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": self._text_from_content(message.content),
                        }
                    )
            elif isinstance(message, ToolPromptMessage):
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content if isinstance(message.content, str) else "",
                    }
                )
        return input_items

    @staticmethod
    def _text_from_content(content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "\n".join(
                part.data
                for part in content
                if getattr(part, "type", None) == PromptMessageContentType.TEXT
            )
        return ""

    def _user_content_parts(self, model: str, credentials: dict, content: object) -> str | list[dict]:
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""

        content_parts: list[dict] = []
        for part in content:
            if part.type == PromptMessageContentType.TEXT:
                text_part: TextPromptMessageContent = part
                content_parts.append({"type": "input_text", "text": text_part.data})
            elif part.type == PromptMessageContentType.IMAGE and self._supports_attachments(model):
                image_part: ImagePromptMessageContent = part
                item = {"type": "input_image", "image_url": image_part.data}
                if image_part.detail:
                    item["detail"] = image_part.detail.value
                content_parts.append(item)
            elif part.type == PromptMessageContentType.DOCUMENT and self._supports_attachments(model):
                document_part: DocumentPromptMessageContent = part
                document_url = getattr(document_part, "url", "") or ""
                if document_part.format == "url" and document_url:
                    content_parts.append(
                        {
                            "type": "input_file",
                            "file_url": document_url,
                        }
                    )
                    continue

                uploaded_file_id = self._upload_file(credentials, document_part)
                if uploaded_file_id:
                    content_parts.append({"type": "input_file", "file_id": uploaded_file_id})
                else:
                    content_parts.append(
                        {
                            "type": "input_file",
                            "filename": document_part.filename or "document",
                            "file_data": document_part.data,
                        }
                    )
        return content_parts

    @staticmethod
    def _supports_attachments(model: str) -> bool:
        return not model.lower().startswith("grok-")

    def _upload_file(self, credentials: dict, document_content: DocumentPromptMessageContent) -> Optional[str]:
        filename = document_content.filename or "document"
        mime_type = document_content.mime_type or ""
        try:
            data = file_bytes(document_content)
        except Exception as error:
            _debug(
                "[flyfus responses] file_prepare_failed filename=%s mime=%s error_type=%s error=%s",
                filename,
                mime_type,
                type(error).__name__,
                error,
            )
            return None

        _debug(
            "[flyfus responses] file_upload_start filename=%s mime=%s bytes=%s",
            filename,
            mime_type,
            len(data),
        )
        try:
            headers = dict(self._request_headers(credentials))
            headers.pop("Content-Type", None)
            response = requests.post(
                self._endpoint_url(credentials, "files"),
                headers=headers,
                data={"purpose": "user_data"},
                files={
                    "file": (
                        filename,
                        data,
                        mime_type,
                    )
                },
                timeout=(10, 300),
            )
        except Exception as error:
            _debug(
                "[flyfus responses] file_upload_failed filename=%s mime=%s bytes=%s error_type=%s error=%s",
                filename,
                mime_type,
                len(data),
                type(error).__name__,
                error,
            )
            return None

        if response.status_code >= 400:
            _debug(
                "[flyfus responses] file_upload_http_error filename=%s status=%s body_head=%s",
                filename,
                response.status_code,
                response.text[:500],
            )
            return None
        with suppress(ValueError):
            file_id = response.json().get("id")
            if isinstance(file_id, str) and file_id:
                _debug(
                    "[flyfus responses] file_upload_success filename=%s bytes=%s file_id=%s",
                    filename,
                    len(data),
                    file_id,
                )
                return file_id
        _debug("[flyfus responses] file_upload_no_id filename=%s status=%s", filename, response.status_code)
        return None

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
    def _convert_tools(tools: Optional[list[PromptMessageTool]]) -> Optional[list[dict]]:
        if not tools:
            return None

        converted_tools: list[dict] = []
        for tool in tools:
            if hasattr(tool, "name"):
                name = tool.name
                description = getattr(tool, "description", "")
                parameters = getattr(tool, "parameters", {})
            else:
                tool_dict = tool.model_dump(mode="json") if hasattr(tool, "model_dump") else dict(tool)
                function = tool_dict.get("function", tool_dict)
                name = function.get("name")
                description = function.get("description", "")
                parameters = function.get("parameters", {})
            if not name:
                continue
            converted_tools.append(
                {
                    "type": "function",
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            )
        return converted_tools or None

    def _handle_response(
        self,
        model: str,
        credentials: dict,
        response: requests.Response,
        prompt_messages: list[PromptMessage],
        invocation_log=None,
    ) -> LLMResult:
        payload = response.json()
        usage_payload = payload.get("usage") or {}
        if invocation_log is not None:
            payload_summary = responses_payload_summary(payload)
            invocation_log.set_response(
                **payload_summary,
                upstream_usage=usage_payload,
                provider_request_id=invocation_log.response.get("provider_request_id") or payload.get("id"),
            )
        return LLMResult(
            model=payload.get("model", model),
            prompt_messages=prompt_messages,
            message=AssistantPromptMessage(
                content=self._extract_output_text(payload),
                tool_calls=self._extract_tool_calls(payload),
            ),
            usage=self._build_dify_usage(model, credentials, usage_payload),
        )

    def _handle_stream(
        self,
        model: str,
        credentials: dict,
        response: requests.Response,
        prompt_messages: list[PromptMessage],
        invocation_log=None,
    ) -> Generator:
        index = 0
        full_text = ""
        final_model = model
        finish_reason = "stop"
        usage_payload: Optional[dict] = None
        pending_tool_calls: dict[object, dict] = {}
        event_counts: dict[str, int] = {}
        total_events = 0
        last_event_type = ""
        saw_completed = False
        yielded_chunks = 0
        response_id = ""

        _debug("[flyfus responses] stream_start model=%s prompt_summary=%s", model, self._prompt_summary(prompt_messages))

        try:
            line_iterator = response.iter_lines(decode_unicode=False)
            for raw_line in line_iterator:
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8", errors="replace").strip()
                if line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    _debug(
                        "[flyfus responses] stream_done_marker model=%s total_events=%s last_event=%s",
                        model,
                        total_events,
                        last_event_type,
                    )
                    continue

                try:
                    event = json.loads(data)
                except ValueError:
                    _debug("[flyfus responses] stream_bad_json model=%s line_head=%s", model, data[:500])
                    continue

                event_type = event.get("type") or "unknown"
                total_events += 1
                last_event_type = event_type
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
                self._log_stream_event(model, total_events, event_type, event)

                if event_type in {"response.output_text.delta", "response.refusal.delta"}:
                    delta_text = event.get("delta") or event.get("text") or ""
                    if not delta_text:
                        continue
                    full_text += delta_text
                    yield LLMResultChunk(
                        model=final_model,
                        prompt_messages=prompt_messages,
                        delta=LLMResultChunkDelta(
                            index=index,
                            message=AssistantPromptMessage(content=delta_text),
                        ),
                    )
                    yielded_chunks += 1
                    index += 1
                elif event_type == "response.output_item.added":
                    call_key = self._tool_call_event_key(event)
                    item = event.get("item") or {}
                    if call_key is not None and item.get("type") == "function_call":
                        pending_tool_calls[call_key] = {
                            "call_id": item.get("call_id") or item.get("id") or "",
                            "name": item.get("name") or "",
                            "arguments": item.get("arguments") or "",
                        }
                elif event_type == "response.function_call_arguments.delta":
                    call_key = self._tool_call_event_key(event)
                    if call_key in pending_tool_calls:
                        pending_tool_calls[call_key]["arguments"] += event.get("delta") or ""
                elif event_type == "response.function_call_arguments.done":
                    call_key = self._tool_call_event_key(event)
                    if call_key in pending_tool_calls:
                        pending_tool_calls[call_key]["arguments"] = event.get("arguments") or ""
                        if event.get("name"):
                            pending_tool_calls[call_key]["name"] = event["name"]
                elif event_type == "response.output_item.done":
                    call_key = self._tool_call_event_key(event)
                    item = event.get("item") or {}
                    if call_key is not None and item.get("type") == "function_call":
                        pending_tool_calls.setdefault(call_key, {})
                        pending_tool_calls[call_key].update(
                            {
                                "call_id": item.get("call_id") or item.get("id") or "",
                                "name": item.get("name") or pending_tool_calls[call_key].get("name", ""),
                                "arguments": item.get("arguments") or pending_tool_calls[call_key].get("arguments", ""),
                            }
                        )
                elif event_type in {"response.completed", "response.incomplete"}:
                    saw_completed = event_type == "response.completed"
                    response_payload = event.get("response") or {}
                    response_id = response_payload.get("id") or response_id
                    final_model = response_payload.get("model") or final_model
                    usage_payload = response_payload.get("usage") or usage_payload
                    if invocation_log is not None:
                        payload_summary = responses_payload_summary(response_payload)
                        invocation_log.set_response(
                            **payload_summary,
                            stream_event_counts=event_counts,
                            stream_event_count=total_events,
                            stream_last_event_type=last_event_type,
                        )
                    if not pending_tool_calls:
                        pending_tool_calls = self._extract_pending_tool_calls(response_payload)
                    finish_reason = "tool_calls" if pending_tool_calls else "stop"
                    if event_type == "response.incomplete":
                        incomplete_details = response_payload.get("incomplete_details") or {}
                        _debug(
                            "[flyfus responses] stream_incomplete model=%s details=%s usage=%s",
                            model,
                            incomplete_details,
                            usage_payload,
                        )
                    if not full_text and not pending_tool_calls:
                        completed_text = self._extract_output_text(response_payload)
                        if completed_text:
                            full_text = completed_text
                            yield LLMResultChunk(
                                model=final_model,
                                prompt_messages=prompt_messages,
                                delta=LLMResultChunkDelta(
                                    index=index,
                                    message=AssistantPromptMessage(content=completed_text),
                                ),
                            )
                            yielded_chunks += 1
                            index += 1
                elif event_type == "response.failed":
                    response_payload = event.get("response") or {}
                    error = response_payload.get("error") or event.get("error") or {}
                    if invocation_log is not None:
                        invocation_log.set_response(
                            response_id=response_payload.get("id") or response_id,
                            error=error,
                            stream_event_counts=event_counts,
                            stream_event_count=total_events,
                        )
                    _debug("[flyfus responses] stream_failed model=%s error=%s", model, error)
                    raise InvokeError(f"OpenAI Responses 流式请求失败：{error}")
        except requests.exceptions.ChunkedEncodingError as error:
            _debug(
                "[flyfus responses] stream_chunked_encoding_error model=%s error=%s total_events=%s last_event=%s "
                "event_counts=%s text_chars=%s yielded_chunks=%s saw_completed=%s usage_seen=%s tool_calls=%s",
                model,
                error,
                total_events,
                last_event_type,
                event_counts,
                len(full_text),
                yielded_chunks,
                saw_completed,
                usage_payload is not None,
                len(pending_tool_calls),
            )
            raise InvokeError(f"OpenAI Responses 流式连接提前结束：{error}") from error
        except requests.exceptions.RequestException as error:
            _debug(
                "[flyfus responses] stream_request_error model=%s error_type=%s error=%s total_events=%s last_event=%s "
                "event_counts=%s text_chars=%s yielded_chunks=%s saw_completed=%s",
                model,
                type(error).__name__,
                error,
                total_events,
                last_event_type,
                event_counts,
                len(full_text),
                yielded_chunks,
                saw_completed,
            )
            raise InvokeError(f"OpenAI Responses 流式请求异常：{error}") from error

        _debug(
            "[flyfus responses] stream_end model=%s total_events=%s last_event=%s event_counts=%s text_chars=%s "
            "yielded_chunks=%s saw_completed=%s usage_seen=%s tool_calls=%s",
            final_model,
            total_events,
            last_event_type,
            event_counts,
            len(full_text),
            yielded_chunks,
            saw_completed,
            usage_payload is not None,
            len(pending_tool_calls),
        )

        tool_calls = self._pending_tool_calls_to_messages(pending_tool_calls)
        if invocation_log is not None:
            invocation_log.set_response(
                response_id=response_id,
                model=final_model,
                output_text=full_text,
                usage=usage_payload,
                upstream_usage=usage_payload,
                provider_request_id=invocation_log.response.get("provider_request_id") or response_id,
                finish_reason=finish_reason,
                saw_completed=saw_completed,
                stream_event_counts=event_counts,
                stream_event_count=total_events,
                stream_last_event_type=last_event_type,
                yielded_chunks=yielded_chunks,
                tool_calls_count=len(tool_calls),
            )
        if tool_calls:
            yield LLMResultChunk(
                model=final_model,
                prompt_messages=prompt_messages,
                delta=LLMResultChunkDelta(
                    index=index,
                    message=AssistantPromptMessage(content="", tool_calls=tool_calls),
                    finish_reason="tool_calls",
                ),
            )
            index += 1

        final_chunk = self._create_final_chunk(
            index=index,
            message=AssistantPromptMessage(content=""),
            finish_reason=finish_reason,
            usage=self._usage_to_chat_usage(usage_payload),
            model=final_model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            full_content=full_text,
        )
        if final_chunk.delta.usage is not None:
            final_chunk.delta.usage = self._build_dify_usage(model, credentials, usage_payload or {})
        yield final_chunk

    def _log_request_summary(
        self,
        model: str,
        stream: bool,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]],
        request_body: dict,
    ) -> None:
        input_items = request_body.get("input") or []
        _debug(
            "[flyfus responses] request_summary model=%s stream=%s prompt_summary=%s input_summary=%s "
            "tools=%s text_format=%s body_keys=%s parameter_keys=%s",
            model,
            stream,
            self._prompt_summary(prompt_messages),
            self._input_summary(input_items),
            len(tools or []),
            bool((request_body.get("text") or {}).get("format")),
            sorted(request_body.keys()),
            sorted(model_parameters.keys()),
        )

    @staticmethod
    def _prompt_summary(prompt_messages: list[PromptMessage]) -> dict:
        summary: dict[str, Any] = {
            "messages": len(prompt_messages),
            "roles": {},
            "content": {"text": 0, "image": 0, "document": 0, "audio": 0, "video": 0, "other": 0},
            "tool_calls": 0,
            "text_chars": 0,
            "documents": [],
        }
        for message in prompt_messages:
            role = getattr(message, "role", None)
            role_name = getattr(role, "value", str(role))
            summary["roles"][role_name] = summary["roles"].get(role_name, 0) + 1
            content = getattr(message, "content", None)
            tool_calls = getattr(message, "tool_calls", None) or []
            summary["tool_calls"] += len(tool_calls)
            if isinstance(content, str):
                summary["content"]["text"] += 1
                summary["text_chars"] += len(content)
                continue
            if not isinstance(content, list):
                continue
            for part in content:
                part_type = getattr(part, "type", None)
                if part_type == PromptMessageContentType.TEXT:
                    summary["content"]["text"] += 1
                    summary["text_chars"] += len(getattr(part, "data", "") or "")
                elif part_type == PromptMessageContentType.IMAGE:
                    summary["content"]["image"] += 1
                elif part_type == PromptMessageContentType.DOCUMENT:
                    summary["content"]["document"] += 1
                    summary["documents"].append(
                        {
                            "filename": getattr(part, "filename", "") or "document",
                            "mime": getattr(part, "mime_type", "") or "",
                            "has_base64": bool(getattr(part, "base64_data", "")),
                            "has_url_or_data": bool(getattr(part, "data", "")),
                        }
                    )
                elif getattr(part_type, "value", part_type) == "audio":
                    summary["content"]["audio"] += 1
                elif getattr(part_type, "value", part_type) == "video":
                    summary["content"]["video"] += 1
                else:
                    summary["content"]["other"] += 1
        return summary

    @staticmethod
    def _input_summary(input_items: list[dict]) -> dict:
        summary: dict[str, Any] = {
            "items": len(input_items),
            "item_types": {},
            "roles": {},
            "content_parts": {},
            "text_chars": 0,
        }
        for item in input_items:
            item_type = item.get("type", "unknown")
            summary["item_types"][item_type] = summary["item_types"].get(item_type, 0) + 1
            role = item.get("role")
            if role:
                summary["roles"][role] = summary["roles"].get(role, 0) + 1
            content = item.get("content")
            if isinstance(content, str):
                summary["text_chars"] += len(content)
                summary["content_parts"]["text_string"] = summary["content_parts"].get("text_string", 0) + 1
            elif isinstance(content, list):
                for part in content:
                    part_type = part.get("type", "unknown") if isinstance(part, dict) else "unknown"
                    summary["content_parts"][part_type] = summary["content_parts"].get(part_type, 0) + 1
                    if isinstance(part, dict):
                        summary["text_chars"] += len(part.get("text") or "")
        return summary

    @staticmethod
    def _log_stream_event(model: str, event_number: int, event_type: str, event: dict) -> None:
        if event_type in {"response.output_text.delta", "response.refusal.delta"}:
            if event_number == 1 or event_number % 25 == 0:
                delta_text = event.get("delta") or event.get("text") or ""
                _debug(
                    "[flyfus responses] stream_event model=%s n=%s type=%s delta_chars=%s",
                    model,
                    event_number,
                    event_type,
                    len(delta_text),
                )
            return

        details: dict[str, Any] = {}
        if "sequence_number" in event:
            details["sequence_number"] = event.get("sequence_number")
        if "output_index" in event:
            details["output_index"] = event.get("output_index")
        item = event.get("item")
        if isinstance(item, dict):
            details["item_type"] = item.get("type")
            details["item_id"] = item.get("id")
            details["call_id"] = item.get("call_id")
            details["name"] = item.get("name")
        response_payload = event.get("response")
        if isinstance(response_payload, dict):
            details["response_status"] = response_payload.get("status")
            details["response_model"] = response_payload.get("model")
            details["usage"] = response_payload.get("usage")
            details["incomplete_details"] = response_payload.get("incomplete_details")
            if response_payload.get("error"):
                details["error"] = response_payload.get("error")
        if event.get("error"):
            details["error"] = event.get("error")
        _debug("[flyfus responses] stream_event model=%s n=%s type=%s details=%s", model, event_number, event_type, details)

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

    @classmethod
    def _extract_tool_calls(cls, payload: dict) -> list[AssistantPromptMessage.ToolCall]:
        return cls._pending_tool_calls_to_messages(cls._extract_pending_tool_calls(payload))

    @staticmethod
    def _extract_pending_tool_calls(payload: dict) -> dict[object, dict]:
        pending: dict[int, dict] = {}
        for index, item in enumerate(payload.get("output") or []):
            if isinstance(item, dict) and item.get("type") == "function_call":
                pending[index] = {
                    "call_id": item.get("call_id") or item.get("id") or "",
                    "name": item.get("name") or "",
                    "arguments": item.get("arguments") or "",
                }
        return pending

    @staticmethod
    def _pending_tool_calls_to_messages(
        pending_tool_calls: dict[object, dict],
    ) -> list[AssistantPromptMessage.ToolCall]:
        tool_calls: list[AssistantPromptMessage.ToolCall] = []
        for output_index in sorted(pending_tool_calls, key=str):
            item = pending_tool_calls[output_index]
            name = item.get("name") or ""
            if not name:
                continue
            tool_calls.append(
                AssistantPromptMessage.ToolCall(
                    id=item.get("call_id") or "",
                    type="function",
                    function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                        name=name,
                        arguments=item.get("arguments") or "{}",
                    ),
                )
            )
        return tool_calls

    @staticmethod
    def _tool_call_event_key(event: dict) -> object:
        if event.get("output_index") is not None:
            return event["output_index"]
        if event.get("item_id") is not None:
            return event["item_id"]
        if event.get("call_id") is not None:
            return event["call_id"]
        return None

    @staticmethod
    def _usage_to_chat_usage(usage: Optional[dict]) -> dict:
        usage = usage or {}
        return {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        }
