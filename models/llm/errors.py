from __future__ import annotations

import json
import requests

from dify_plugin.errors.model import InvokeError


class FlyfusInvokeError(InvokeError):
    flyfus_error_type = "model_error"
    flyfus_user_message = "模型服务暂时不可用，请稍后重试。"
    flyfus_retryable = False
    flyfus_internal_retryable = False


class UpstreamTimeoutError(FlyfusInvokeError):
    flyfus_error_type = "upstream_timeout"
    flyfus_user_message = "模型服务响应超时，请稍后重试。"
    flyfus_retryable = True
    flyfus_internal_retryable = True


class UpstreamConnectionError(FlyfusInvokeError):
    flyfus_error_type = "upstream_connection_error"
    flyfus_user_message = "模型服务连接失败，请稍后重试。"
    flyfus_retryable = True
    flyfus_internal_retryable = True


class UpstreamRateLimitedError(FlyfusInvokeError):
    flyfus_error_type = "upstream_rate_limited"
    flyfus_user_message = "模型服务当前请求过多，请稍后重试。"
    flyfus_retryable = True
    flyfus_internal_retryable = True


class UpstreamServerError(FlyfusInvokeError):
    flyfus_error_type = "upstream_server_error"
    flyfus_user_message = "模型服务暂时不可用，请稍后重试。"
    flyfus_retryable = True
    flyfus_internal_retryable = True


class UpstreamAuthError(FlyfusInvokeError):
    flyfus_error_type = "upstream_auth_error"
    flyfus_user_message = "模型服务配置异常，请联系管理员处理。"


class UpstreamBadRequestError(FlyfusInvokeError):
    flyfus_error_type = "upstream_bad_request"
    flyfus_user_message = "本次请求暂时无法处理，请调整内容后重试；如仍失败，请联系管理员。"


class UpstreamResponseIncompleteError(FlyfusInvokeError):
    flyfus_error_type = "upstream_response_incomplete"
    flyfus_user_message = "模型未能生成完整回答，请调整请求后重试。"


class UpstreamResponseFailedError(FlyfusInvokeError):
    flyfus_error_type = "upstream_response_failed"
    flyfus_user_message = "模型生成失败，请稍后重试。"
    flyfus_retryable = True


class GeminiStreamIncompleteError(FlyfusInvokeError):
    flyfus_error_type = "gemini_stream_incomplete"
    flyfus_user_message = "模型响应中断，请稍后重试。"
    flyfus_retryable = True


class GeminiSafetyBlockedError(FlyfusInvokeError):
    flyfus_error_type = "gemini_safety_blocked"
    flyfus_user_message = "该请求未能通过模型安全检查，请调整内容后重试。"


def request_error(provider: str, error: requests.RequestException) -> FlyfusInvokeError:
    message = f"{provider} 请求失败：{error}"
    if isinstance(error, (requests.ConnectTimeout, requests.ReadTimeout, requests.Timeout)):
        return UpstreamTimeoutError(message)
    return UpstreamConnectionError(message)


def http_error(provider: str, status_code: int, response_text: str) -> FlyfusInvokeError:
    message = f"{provider} 请求失败，状态码：{status_code}，响应：{response_text[:2000]}"
    if status_code in {401, 403}:
        return UpstreamAuthError(message)
    if status_code == 408:
        return UpstreamTimeoutError(message)
    if status_code == 429:
        return UpstreamRateLimitedError(message)
    if status_code >= 500:
        return UpstreamServerError(message)
    return UpstreamBadRequestError(message)


def response_failed_error(provider: str, error: object) -> FlyfusInvokeError:
    raw_error = json.dumps(error, ensure_ascii=False, default=str) if isinstance(error, (dict, list)) else str(error)
    message = f"{provider} 返回失败响应：{raw_error}"
    normalized = raw_error.lower()
    if any(marker in normalized for marker in ("rate_limit", "rate limit", "too many requests")):
        return UpstreamRateLimitedError(message)
    if any(marker in normalized for marker in ("timeout", "timed out")):
        return UpstreamTimeoutError(message)
    if any(marker in normalized for marker in ("unauthorized", "forbidden", "invalid_api_key", "authentication")):
        return UpstreamAuthError(message)
    if any(marker in normalized for marker in ("server_error", "internal error", "service unavailable", "overloaded")):
        return UpstreamServerError(message)
    if any(marker in normalized for marker in ("invalid_request", "bad request", "unsupported")):
        return UpstreamBadRequestError(message)
    return UpstreamResponseFailedError(message)


def retry_delay_seconds(status_code: int, headers: object) -> float:
    if status_code != 429:
        return 0
    get = getattr(headers, "get", None)
    raw_value = get("Retry-After") if callable(get) else None
    try:
        return min(10.0, max(0.0, float(raw_value)))
    except (TypeError, ValueError):
        return 1.0
