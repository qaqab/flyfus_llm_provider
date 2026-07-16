import json
import re
from contextlib import suppress
from typing import Any, Optional

import requests
from dify_plugin.entities.model.message import TextPromptMessageContent


_TOKEN_USAGE_URL = "https://geo.dev.vocscope.com/api/geo/v2/dify_llm/token-usage"
_TOKEN_USAGE_API_KEY = "test_dify_1780389317_af8ade862225"
_USAGE_CONTEXT_PATTERN = re.compile(r"<FP_USAGE_CONTEXT>(.*?)</FP_USAGE_CONTEXT>", re.DOTALL)
_USAGE_OWNER_TYPES = {
    "dify_run_listing",
    "dify_run_keyword",
    "dify_run_category",
    "dify_run_qa",
    "chat_session",
    "report_chat_session",
}


def normalize_usage(request_id: str, model: str, raw_usage: Optional[dict]) -> dict:
    """Convert provider-specific token usage into the backend contract."""
    usage = _as_dict(raw_usage)
    prompt_details = _as_dict(usage.get("prompt_tokens_details"))
    input_details = _as_dict(usage.get("input_tokens_details"))
    completion_details = _as_dict(usage.get("completion_tokens_details"))
    output_details = _as_dict(usage.get("output_tokens_details"))

    input_tokens = _first_value(usage, "input_tokens", "prompt_tokens")
    output_tokens = _first_value(usage, "output_tokens", "completion_tokens")
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    return {
        "request_id": request_id,
        "model": model,
        "input_tokens": input_tokens,
        "cached_tokens": _first_not_none(
            input_details.get("cached_tokens"),
            prompt_details.get("cached_tokens"),
            usage.get("prompt_cache_hit_tokens"),
            usage.get("cache_read_input_tokens"),
        ),
        "cache_write_tokens": _first_not_none(
            input_details.get("cache_write_tokens"),
            usage.get("cache_creation_input_tokens"),
        ),
        "output_tokens": output_tokens,
        "reasoning_tokens": _first_not_none(
            output_details.get("reasoning_tokens"),
            completion_details.get("reasoning_tokens"),
        ),
        "total_tokens": total_tokens,
    }


def post_token_usage(
    payload: dict,
    *,
    timeout: tuple[int, int] = (3, 10),
) -> requests.Response:
    """Post normalized token usage to the dedicated backend endpoint."""
    # Temporary: do not report token usage until the accounting backend is ready.
    # Remove this return to re-enable token usage uploads.
    return None

    response = requests.post(
        _TOKEN_USAGE_URL,
        headers={
            "Authorization": f"Bearer {_TOKEN_USAGE_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response


def extract_usage_context(prompt_messages: list) -> Optional[dict[str, str]]:
    """Remove the private usage tag from prompts and return its last valid payload."""
    usage_context = None
    for message in prompt_messages:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            cleaned, contexts = _strip_usage_context(content)
            message.content = cleaned
            if contexts:
                usage_context = contexts[-1]
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, TextPromptMessageContent):
                continue
            cleaned, contexts = _strip_usage_context(part.data)
            part.data = cleaned
            if contexts:
                usage_context = contexts[-1]
    return usage_context


def report_token_usage(
    request_id: str,
    model: str,
    raw_usage: Optional[dict],
    usage_context: Optional[dict[str, str]],
) -> bool:
    """Best-effort usage reporting; accounting failures must not fail the LLM call."""
    if not usage_context:
        return False
    payload = normalize_usage(request_id, model, raw_usage)
    payload.update(usage_context)
    try:
        post_token_usage(payload)
    except requests.RequestException:
        return False
    return True


def _strip_usage_context(text: str) -> tuple[str, list[dict[str, str]]]:
    contexts: list[dict[str, str]] = []
    for match in _USAGE_CONTEXT_PATTERN.finditer(text):
        with suppress(ValueError):
            payload = json.loads(match.group(1).strip())
            if _valid_usage_context(payload):
                contexts.append(
                    {
                        "usage_owner_type": payload["usage_owner_type"],
                        "usage_owner_id": payload["usage_owner_id"],
                    }
                )
    return _USAGE_CONTEXT_PATTERN.sub("", text), contexts


def _valid_usage_context(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    return (
        payload.get("usage_owner_type") in _USAGE_OWNER_TYPES
        and isinstance(payload.get("usage_owner_id"), str)
        and bool(payload["usage_owner_id"])
    )


def _first_value(values: dict[str, Any], *keys: str) -> Any:
    return _first_not_none(*(values.get(key) for key in keys))


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else {}
    attributes = getattr(value, "__dict__", None)
    return attributes if isinstance(attributes, dict) else {}
