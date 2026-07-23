import hashlib
import time
import traceback
import uuid
import json
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from typing import Any, Iterator, Optional

from models.llm.sls_logging import write_invocation_log


_MAX_STRING_LENGTH = 30000
_MAX_EVENT_STRING_LENGTH = 8000
_MAX_LIST_ITEMS = 80
_MAX_DICT_ITEMS = 120
_MAX_PREVIEW_LENGTH = 1200


class InvocationLog:
    def __init__(self, *, model: str, credentials: dict, stream: bool, user: Optional[str]) -> None:
        self.invocation_id = str(uuid.uuid4())
        self.started_at = time.time()
        self.model = model
        self.stream = stream
        self.user = user
        self.credentials = credentials
        self.request: dict[str, Any] = {}
        self.response: dict[str, Any] = {}
        self.events: list[dict] = []
        self.result: dict[str, Any] = {}
        self.flushed = False

    @classmethod
    def from_credentials(
        cls,
        *,
        model: str,
        credentials: dict,
        stream: bool,
        user: Optional[str],
    ) -> "InvocationLog":
        return cls(model=model, credentials=credentials, stream=stream, user=user)

    def event(self, name: str, **fields: Any) -> None:
        self.events.append(
            {
                "time": _iso_now(),
                "elapsed_ms": int((time.time() - self.started_at) * 1000),
                "name": name,
                "invocation_id": self.invocation_id,
                **_sanitize(fields, max_string_length=_MAX_EVENT_STRING_LENGTH),
            }
        )

    @contextmanager
    def step(self, name: str, **fields: Any) -> Iterator[None]:
        step_started_at = time.time()
        try:
            yield
        except Exception as error:
            self.event(
                name,
                status="error",
                duration_ms=int((time.time() - step_started_at) * 1000),
                error_type=type(error).__name__,
                error=str(error),
                **fields,
            )
            raise
        else:
            self.event(
                name,
                status="success",
                duration_ms=int((time.time() - step_started_at) * 1000),
                **fields,
            )

    def set_request(self, **fields: Any) -> None:
        self.request.update(_sanitize(fields))

    def set_replay_request(self, *, endpoint: str, body: dict[str, Any]) -> None:
        """Store the exact JSON body sent upstream, excluding credentials.

        This intentionally bypasses the diagnostic sanitizer: truncating a prompt,
        tool argument, or tool output would make the record impossible to replay.
        Authentication is never part of a replay record and must be supplied by the
        caller when it is sent again.
        """
        self.request["replay_request"] = {
            "endpoint": endpoint,
            "body": body,
        }

    def set_response(self, **fields: Any) -> None:
        output_text = fields.get("output_text")
        if isinstance(output_text, str) and output_text:
            self.response["output_text_md5"] = hashlib.md5(output_text.encode("utf-8")).hexdigest()
        self.response.update(_sanitize(fields))

    def success(self, **fields: Any) -> None:
        self.result = {"status": "success", **_sanitize(fields)}

    def failure(self, error: BaseException) -> None:
        self.result = {
            "status": "error",
            "error_type": type(error).__name__,
            "error": _truncate(str(error)),
            "traceback": _truncate("".join(traceback.format_exception(type(error), error, error.__traceback__))),
        }

    def flush(self) -> None:
        if self.flushed:
            return
        self.flushed = True

        response_id = _first_present(self.response.get("response_id"), self.response.get("id"))
        upstream_request_id = _nested_get(self.response, "http", "headers", "x-request-id")
        upstream_client_request_id = _nested_get(self.response, "http", "headers", "x-client-request-id")
        upstream_cf_ray = _nested_get(self.response, "http", "headers", "cf-ray")
        upstream_request = self.request.get("upstream_request") or {}
        output_text = self.response.get("output_text")
        output_text_md5 = self.response.get("output_text_md5") or ""
        event = {
            "time": _iso_now(),
            "source": "flyfus_llm_provider",
            "schema_version": 4,
            "event_type": "llm_invocation",
            "invocation_id": self.invocation_id,
            "client_request_id": self.invocation_id,
            "response_id": response_id,
            "upstream_request_id": upstream_request_id,
            "upstream_client_request_id": upstream_client_request_id,
            "upstream_cf_ray": upstream_cf_ray,
            "model": self.model,
            "stream": self.stream,
            "user": self.user,
            "duration_ms": int((time.time() - self.started_at) * 1000),
            "status": (self.result or {}).get("status", "unknown"),
            "ids": {
                "invocation_id": self.invocation_id,
                "client_request_id": self.invocation_id,
                "response_id": response_id,
                "upstream_request_id": upstream_request_id,
                "upstream_client_request_id": upstream_client_request_id,
                "upstream_cf_ray": upstream_cf_ray,
            },
            "input": {
                "kind": _input_kind(self.request.get("prompt_metrics_final") or self.request.get("prompt_metrics_initial")),
                "model": self.model,
                "model_family": self.request.get("model_family"),
                "adapter": self.request.get("adapter"),
                "stream": self.stream,
                "user": self.user,
                "stop": self.request.get("stop"),
                "model_parameters": self.request.get("model_parameters_final")
                or self.request.get("model_parameters"),
                "metrics": self.request.get("prompt_metrics_final")
                or self.request.get("prompt_metrics_initial"),
                "messages": self.request.get("prompt_messages_final")
                or self.request.get("prompt_messages_initial"),
                "tools": self.request.get("tools"),
            },
            "upstream": {
                "endpoint": upstream_request.get("endpoint"),
                "headers": upstream_request.get("headers"),
                "body_summary": upstream_request.get("body_summary"),
                "replay": self.request.get("replay_request"),
                "http": self.response.get("http"),
                "stream_event_count": self.response.get("stream_event_count"),
                "stream_event_counts": self.response.get("stream_event_counts"),
                "finish_reason": self.response.get("finish_reason"),
            },
            "output": {
                "text": output_text,
                "text_md5": output_text_md5,
                "usage": self.response.get("usage"),
                "chunk_count": self.response.get("chunk_count") or self.response.get("yielded_chunks"),
                "tool_calls_count": self.response.get("tool_calls_count"),
                "message": self.response.get("message"),
                "error": self.response.get("error") or self.response.get("error_body"),
            },
            "timeline": self.events,
        }
        error_result = _compact_error_result(self.result)
        if error_result:
            event["error"] = error_result
        write_invocation_log(self.credentials, event)


def wrap_stream_with_invocation_log(stream_result, invocation_log: InvocationLog, usage_reporter=None, error_chunk_factory=None):
    chunk_count = 0
    output_parts: list[str] = []
    usage = None
    try:
        for chunk in stream_result:
            chunk_count += 1
            raw_usage = getattr(getattr(chunk, "delta", None), "usage", None)
            if raw_usage is not None:
                usage = raw_usage
            chunk_summary = llm_chunk_summary(chunk)
            if usage is None and chunk_summary.get("usage") is not None:
                usage = chunk_summary["usage"]
            chunk_text = chunk_summary.get("text")
            if isinstance(chunk_text, str) and chunk_text:
                output_parts.append(chunk_text)
            yield chunk
    except Exception as error:
        invocation_log.failure(error)
        invocation_log.set_response(
            output_text="".join(output_parts),
        )
        invocation_log.event("stream_error", chunk_count=chunk_count, output_text="".join(output_parts))
        if error_chunk_factory is not None:
            yield error_chunk_factory(failure_output_text(invocation_log, error), chunk_count)
            return
        raise
    else:
        output_text = "".join(output_parts)
        invocation_log.set_response(
            output_text=output_text,
            chunk_count=chunk_count,
        )
        invocation_log.success(chunk_count=chunk_count, output_text=output_text)
        if usage_reporter is not None:
            with suppress(Exception):
                usage_reporter(usage or invocation_log.response.get("usage"))
    finally:
        invocation_log.flush()


def failure_output_text(invocation_log: InvocationLog, error: BaseException) -> str:
    """Build a user-visible failure report without exposing provider credentials."""
    metrics = invocation_log.request.get("prompt_metrics_final") or invocation_log.request.get("prompt_metrics_initial") or {}
    response = invocation_log.response
    upstream_headers = _nested_get(response, "http", "headers") or {}
    raw_error = response.get("error") or response.get("error_body")
    if raw_error is None:
        raw_error = repr(error.__cause__ or error)
    if isinstance(raw_error, (dict, list)):
        raw_error = json.dumps(raw_error, ensure_ascii=False, default=str)

    lines = [
        "[模型调用失败]",
        "provider: qaqab/flyfus_llm_provider/flyfus_llm_provider",
        f"model: {invocation_log.model}",
        f"stream: {str(invocation_log.stream).lower()}",
        f"user: {invocation_log.user or '<empty>'}",
        f"input_message_count: {metrics.get('message_count', 0)} messages",
        f"input_content_characters: {metrics.get('total_content_chars', 0)} Unicode characters (not tokens)",
        f"input_roles: {json.dumps(metrics.get('role_counts') or {}, ensure_ascii=False)}",
        f"invocation_id: {invocation_log.invocation_id}",
        f"upstream_request_id: {upstream_headers.get('x-request-id') or upstream_headers.get('openai-request-id') or '<none>'}",
        f"upstream_client_request_id: {upstream_headers.get('x-client-request-id') or '<none>'}",
        f"upstream_cf_ray: {upstream_headers.get('cf-ray') or '<none>'}",
        f"stream_event_count: {response.get('stream_event_count', 0)}",
        f"stream_last_event: {response.get('stream_last_event_type') or '<none>'}",
        f"partial_output_characters: {len(str(response.get('output_text') or ''))}",
        f"error_type: {type(error).__name__}",
        f"error: {error}",
        f"raw_error: {raw_error}",
    ]
    return "\n".join(lines)


def llm_result_summary(result: Any) -> dict:
    message = getattr(result, "message", None)
    content = getattr(message, "content", None)
    return _sanitize(
        {
            "type": type(result).__name__,
            "id": getattr(result, "id", None),
            "model": getattr(result, "model", None),
            "output_text": content if isinstance(content, str) else str(content or ""),
            "message": {
                "type": type(message).__name__ if message is not None else None,
                "content": content,
                "tool_calls": getattr(message, "tool_calls", None),
            },
            "usage": getattr(result, "usage", None),
        }
    )


def http_response_summary(response: Any) -> dict:
    headers = getattr(response, "headers", {}) or {}
    return _sanitize(
        {
            "http_status": getattr(response, "status_code", None),
            "headers": {
                "x-request-id": _header_value(headers, "x-request-id"),
                "x-client-request-id": _header_value(headers, "x-client-request-id"),
                "openai-request-id": _header_value(headers, "openai-request-id"),
                "cf-ray": _header_value(headers, "cf-ray"),
                "content-type": _header_value(headers, "content-type"),
            },
        }
    )


def responses_payload_summary(payload: dict) -> dict:
    return _sanitize(
        {
            "response_id": payload.get("id"),
            "object": payload.get("object"),
            "model": payload.get("model"),
            "status": payload.get("status"),
            "created_at": payload.get("created_at"),
            "completed_at": payload.get("completed_at"),
            "error": payload.get("error"),
            "incomplete_details": payload.get("incomplete_details"),
            "usage": payload.get("usage"),
            "output_text": _extract_responses_output_text(payload),
            "output_count": len(payload.get("output") or []),
        }
    )


def llm_chunk_summary(chunk: Any) -> dict:
    if isinstance(chunk, str):
        return {"type": "str", "text": _sanitize_string(chunk)}

    delta = getattr(chunk, "delta", None)
    message = getattr(delta, "message", None)
    text = _message_text(message)
    return _sanitize(
        {
            "type": type(chunk).__name__,
            "model": getattr(chunk, "model", None),
            "text": text,
            "finish_reason": getattr(delta, "finish_reason", None),
            "usage": getattr(delta, "usage", None),
            "tool_calls": getattr(message, "tool_calls", None),
        }
    )


def prompt_messages_summary(prompt_messages: list) -> list[dict]:
    summary = []
    for message in prompt_messages:
        role = getattr(getattr(message, "role", None), "value", getattr(message, "role", None))
        content = getattr(message, "content", None)
        tool_output = _tool_output_summary(message)
        content_summary = _content_summary(content, role=role)
        summary.append(
            _compact_dict({
                "role": role,
                "type": type(message).__name__,
                "content": content_summary,
                "content_length": _content_length(content),
                "name": getattr(message, "name", None),
                "tool_call_id": getattr(message, "tool_call_id", None),
                "tool_output": tool_output,
            })
        )
    return _sanitize(summary)


def prompt_messages_metrics(prompt_messages: list) -> dict:
    role_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    total_chars = 0
    latest_user_message = ""
    latest_assistant_message = ""
    tool_names: list[str] = []

    for message in prompt_messages:
        role = str(getattr(getattr(message, "role", None), "value", getattr(message, "role", None)))
        message_type = type(message).__name__
        role_counts[role] = role_counts.get(role, 0) + 1
        type_counts[message_type] = type_counts.get(message_type, 0) + 1
        content = getattr(message, "content", None)
        text = _content_text(content)
        total_chars += len(text)
        if role == "user":
            latest_user_message = text
        elif role == "assistant" and text:
            latest_assistant_message = text
        elif role == "tool":
            tool_name = getattr(message, "name", None)
            if tool_name:
                tool_names.append(tool_name)

    return _sanitize(
        {
            "message_count": len(prompt_messages),
            "role_counts": role_counts,
            "type_counts": type_counts,
            "total_content_chars": total_chars,
            "latest_user_message": latest_user_message,
            "latest_user_message_md5": hashlib.md5(latest_user_message.encode("utf-8")).hexdigest()
            if latest_user_message
            else "",
            "latest_assistant_message": latest_assistant_message,
            "tool_names": tool_names,
        }
    )


def upstream_openai_compatible_request_summary(
    *,
    model: str,
    credentials: dict,
    prompt_messages: list,
    model_parameters: dict,
    tools: Optional[list],
    stop: Optional[list[str]],
    stream: bool,
    user: Optional[str],
    convert_message,
    headers: Optional[dict] = None,
) -> dict:
    data = {
        "model": credentials.get("endpoint_model_name", model),
        "stream": stream,
        **dict(model_parameters),
        "messages": [convert_message(message, credentials) for message in prompt_messages],
    }
    if tools:
        data["tool_choice"] = "auto"
        data["tools"] = tools_summary(tools)
    if stop:
        data["stop"] = stop
    if user:
        data["user"] = user
    return _sanitize(
        {
            "endpoint": _join_endpoint(credentials.get("endpoint_url", ""), "chat/completions"),
            "headers": headers or {},
            "body_summary": _request_body_summary(data),
            "message_count": len(data["messages"]),
            "tool_count": len(tools or []),
        }
    )


def tools_summary(tools: Optional[list]) -> list:
    summaries = []
    for tool in tools or []:
        summaries.append(
            _compact_dict({
                "type": type(tool).__name__,
                "name": getattr(tool, "name", None),
                "description": getattr(tool, "description", None),
                "parameters": _parameters_summary(getattr(tool, "parameters", None)),
            })
        )
    return _sanitize(summaries)


def _header_value(headers: Any, name: str) -> Optional[str]:
    with_header_get = getattr(headers, "get", None)
    if callable(with_header_get):
        return with_header_get(name) or with_header_get(name.title()) or with_header_get(name.upper())
    if isinstance(headers, dict):
        lowered = {str(key).lower(): value for key, value in headers.items()}
        value = lowered.get(name.lower())
        return str(value) if value is not None else None
    return None


def _extract_responses_output_text(payload: dict) -> str:
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]

    parts: list[str] = []
    for item in payload.get("output") or []:
        if not isinstance(item, dict):
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") in {"output_text", "refusal"}:
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


def _tool_output_summary(message: Any) -> Optional[dict]:
    role = getattr(getattr(message, "role", None), "value", getattr(message, "role", None))
    if role != "tool":
        return None
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return None

    parsed = _try_parse_json(content)
    if not isinstance(parsed, dict):
        return {"raw_text": content}

    result: dict[str, Any] = {"keys": sorted(str(key) for key in parsed.keys())}
    for key in ("query_result", "prepare_result", "confirm_result", "write_result"):
        if key not in parsed:
            continue
        nested = parsed[key]
        if isinstance(nested, str):
            nested_payload = _try_parse_json(nested)
            if isinstance(nested_payload, dict):
                result[key] = _report_tool_payload_summary(nested_payload)
            else:
                result[key] = nested
        elif isinstance(nested, dict):
            result[key] = _report_tool_payload_summary(nested)
    return result


def _report_tool_payload_summary(payload: dict) -> dict:
    raw = payload.get("raw")
    return _sanitize(
        {
            "status": payload.get("status"),
            "message": payload.get("message"),
            "answer": payload.get("answer"),
            "report_key": payload.get("report_key"),
            "report_name": payload.get("report_name"),
            "selected_paths": payload.get("selected_paths"),
            "pending_patch": payload.get("pending_patch"),
            "local_rejected": payload.get("local_rejected"),
            "raw_status": raw.get("status") if isinstance(raw, dict) else None,
            "raw_ok": raw.get("ok") if isinstance(raw, dict) else None,
            "raw_response_code": (
                raw.get("response", {}).get("code")
                if isinstance(raw, dict) and isinstance(raw.get("response"), dict)
                else None
            ),
        }
    )


def _message_text(message: Any) -> str:
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return str(content)


def _content_summary(content: Any, *, role: Optional[str] = None) -> Any:
    if isinstance(content, str):
        if role == "tool":
            return _truncate(content, max_string_length=_MAX_PREVIEW_LENGTH)
        return content
    if isinstance(content, list):
        return [_sanitize(_prompt_part_summary(part)) for part in content[:_MAX_LIST_ITEMS]]
    return _sanitize(content)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            data = getattr(item, "data", None)
            if isinstance(data, str):
                parts.append(data)
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _content_length(content: Any) -> int:
    return len(_content_text(content))


def _prompt_part_summary(part: Any) -> dict:
    part_type = getattr(getattr(part, "type", None), "value", getattr(part, "type", None))
    data = getattr(part, "data", None)
    return {
        "type": part_type or type(part).__name__,
        "data": data,
        "filename": getattr(part, "filename", None),
        "mime_type": getattr(part, "mime_type", None),
    }


def _sanitize(value: Any, *, max_string_length: int = _MAX_STRING_LENGTH) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _sanitize_string(value, max_string_length=max_string_length)
    if isinstance(value, dict):
        sanitized = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_DICT_ITEMS:
                sanitized["_truncated_items"] = len(value) - _MAX_DICT_ITEMS
                break
            key_text = str(key)
            sanitized[key_text] = _sanitize(item, max_string_length=max_string_length)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        sanitized_items = [_sanitize(item, max_string_length=max_string_length) for item in items[:_MAX_LIST_ITEMS]]
        if len(items) > _MAX_LIST_ITEMS:
            sanitized_items.append({"_truncated_items": len(items) - _MAX_LIST_ITEMS})
        return sanitized_items
    return _truncate(str(value), max_string_length=max_string_length)


def _sanitize_string(value: str, *, max_string_length: int = _MAX_STRING_LENGTH) -> str:
    return _truncate(value, max_string_length=max_string_length)


def _truncate(value: str, *, max_string_length: int = _MAX_STRING_LENGTH) -> str:
    if len(value) <= max_string_length:
        return value
    return value[:max_string_length] + f"...[truncated {len(value) - max_string_length} chars]"


def _try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _request_body_summary(body: dict) -> dict:
    return {
        "model": body.get("model"),
        "stream": body.get("stream"),
        "temperature": body.get("temperature"),
        "max_tokens": body.get("max_tokens"),
        "input_count": len(body.get("input") or body.get("messages") or []),
        "tool_count": len(body.get("tools") or []),
        "tool_choice": body.get("tool_choice"),
        "has_response_format": bool(body.get("response_format") or body.get("text")),
    }


def _compact_error_result(result: dict) -> Optional[dict]:
    if not result or result.get("status") != "error":
        return None
    return {
        key: value
        for key, value in result.items()
        if key != "status"
    }


def _input_kind(metrics: Optional[dict]) -> str:
    if not isinstance(metrics, dict):
        return "unknown"
    role_counts = metrics.get("role_counts") or {}
    if role_counts.get("assistant") or role_counts.get("tool"):
        return "conversation"
    return "single_call"


def _parameters_summary(parameters: Any) -> Any:
    if not isinstance(parameters, dict):
        return parameters
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return parameters
    return {
        "type": parameters.get("type"),
        "required": parameters.get("required"),
        "properties": {
            name: {
                "type": schema.get("type") if isinstance(schema, dict) else None,
                "description": schema.get("description") if isinstance(schema, dict) else None,
            }
            for name, schema in properties.items()
        },
    }


def _compact_dict(value: dict) -> dict:
    return {
        key: item
        for key, item in value.items()
        if item not in (None, "", [], {})
    }


def _nested_get(value: dict, *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def _join_endpoint(endpoint_url: str, path: str) -> str:
    if not endpoint_url:
        return path
    return endpoint_url.rstrip("/") + "/" + path.lstrip("/")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()
