import re
from typing import Any, Optional

import requests


_USAGE_USER_PATTERN = re.compile(r"^[^:\s]+:(?:web_chat|report_chat):[^:\s]+$")


def normalize_upstream_usage(raw_usage: Optional[dict]) -> dict:
    """Convert provider-specific usage into the shared internal token schema."""
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


def to_geo_payload(request_id: str, model: str, raw_usage: Optional[dict]) -> dict:
    """Build the Geo token-usage request body from upstream usage."""
    return {
        "request_id": request_id,
        "model": model,
        **normalize_upstream_usage(raw_usage),
    }


def normalize_usage(request_id: str, model: str, raw_usage: Optional[dict]) -> dict:
    """Backward-compatible alias for callers that need the Geo request body."""
    return to_geo_payload(request_id, model, raw_usage)


def format_usage_currency(raw_usage: Optional[dict], *, log_id: Optional[str] = None) -> str:
    """Render the complete normalized token breakdown into Dify's string-only currency field."""
    usage = normalize_upstream_usage(raw_usage)

    def value(key: str) -> str:
        item = usage[key]
        return str(item) if item is not None else "未提供"

    summary = "；".join(
        (
            f"输入 Token: {value('input_tokens')}",
            f"缓存命中 Token: {value('cached_tokens')}",
            f"缓存写入 Token: {value('cache_write_tokens')}",
            f"输出 Token: {value('output_tokens')}",
            f"推理 Token: {value('reasoning_tokens')}",
            f"总 Token: {value('total_tokens')}",
        )
    )
    return f"{summary} | log_id={log_id}" if log_id else summary


def post_token_usage(
    payload: dict,
    credentials: dict,
    *,
    timeout: tuple[int, int] = (3, 10),
) -> requests.Response:
    """Post normalized token usage to the dedicated backend endpoint."""
    geo_url = str(credentials.get("geo_url") or "").strip().rstrip("/")
    response = requests.post(
        f"{geo_url}/dify_llm/token-usage",
        headers={
            "Authorization": f"Bearer {str(credentials.get('geo_key') or '').strip()}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response


def report_token_usage(
    request_id: str,
    model: str,
    raw_usage: Optional[dict],
    user: Optional[str],
    credentials: dict,
) -> bool:
    """Best-effort usage reporting; accounting failures must not fail the LLM call."""
    normalized_user = user.strip() if isinstance(user, str) else ""
    if not _USAGE_USER_PATTERN.fullmatch(normalized_user):
        return False
    payload = to_geo_payload(request_id, model, raw_usage)
    payload["user"] = normalized_user
    try:
        post_token_usage(payload, credentials)
    except requests.RequestException:
        return False
    return True


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
