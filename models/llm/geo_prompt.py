"""Geo Prompt 引用的渲染与局部替换。"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import time
from typing import Optional

import requests

from dify_plugin.entities.model.message import PromptMessage, PromptMessageRole
from dify_plugin.errors.model import InvokeError

from models.llm.invocation_logging import InvocationLog


_GEO_PROMPT_REFERENCE_PATTERN = re.compile(
    r"\{\{dify_admin:(?P<name>[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+)}}"
)
_GEO_PROMPT_TOKEN_PATTERN = re.compile(r"\{\{dify_admin:[^}]*}}")
_GEO_PROMPT_RENDER_ATTEMPTS = 3
_GEO_PROMPT_RENDER_RETRY_DELAY_SECONDS = 10
_GEO_PROMPT_RENDER_MAX_WORKERS = 4


def render_geo_prompt_text(
    text: str,
    credentials: dict,
    invocation_log: Optional[InvocationLog] = None,
) -> str:
    """渲染一段包含 Geo Prompt 引用的文本。"""
    normalized_text = _normalize_geo_prompt_references(text)
    if normalized_text == text and not _GEO_PROMPT_TOKEN_PATTERN.search(text):
        return text

    geo_base_url = str(credentials.get("geo_prompt_render_url") or "").strip().rstrip("/")
    if invocation_log is not None:
        invocation_log.event(
            "geo_prompt_render_request",
            endpoint=f"{geo_base_url}/dify_admin/render",
            reference=normalized_text,
            reference_count=len(_GEO_PROMPT_TOKEN_PATTERN.findall(text)),
        )
    for attempt in range(1, _GEO_PROMPT_RENDER_ATTEMPTS + 1):
        if invocation_log is not None:
            invocation_log.event(
                "geo_prompt_render_attempt_started",
                reference=normalized_text,
                attempt=attempt,
                max_attempts=_GEO_PROMPT_RENDER_ATTEMPTS,
            )
        try:
            response = requests.post(
                f"{geo_base_url}/dify_admin/render",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {str(credentials.get('geo_prompt_api_key') or '').strip()}",
                },
                json={"text": normalized_text},
                timeout=(10, 60),
            )
            break
        except requests.RequestException as error:
            if attempt == _GEO_PROMPT_RENDER_ATTEMPTS:
                if invocation_log is not None:
                    invocation_log.event(
                        "geo_prompt_render_failed",
                        reference=normalized_text,
                        attempt=attempt,
                        max_attempts=_GEO_PROMPT_RENDER_ATTEMPTS,
                        error_type=type(error).__name__,
                        error=str(error),
                    )
                raise InvokeError(
                    f"Geo Prompt 渲染请求失败，已尝试 {attempt} 次：{error}"
                ) from error
            if invocation_log is not None:
                invocation_log.event(
                    "geo_prompt_render_retry",
                    reference=normalized_text,
                    attempt=attempt,
                    retry_delay_seconds=_GEO_PROMPT_RENDER_RETRY_DELAY_SECONDS,
                    error=str(error),
                )
            time.sleep(_GEO_PROMPT_RENDER_RETRY_DELAY_SECONDS)

    if invocation_log is not None:
        invocation_log.event(
            "geo_prompt_render_response",
            reference=normalized_text,
            status_code=response.status_code,
        )

    if response.status_code != 200:
        raise InvokeError(
            f"Geo Prompt 渲染失败，状态码：{response.status_code}，响应：{response.text}"
        )

    try:
        payload = response.json()
    except ValueError as error:
        raise InvokeError("Geo Prompt 渲染接口返回的不是 JSON。") from error

    rendered_text = payload.get("data", {}).get("rendered_text")
    if not isinstance(rendered_text, str):
        raise InvokeError("Geo Prompt 渲染接口返回格式缺少 data.rendered_text。")

    return rendered_text


def render_geo_prompt_references(
    prompt_messages: list[PromptMessage],
    credentials: dict,
    invocation_log: Optional[InvocationLog] = None,
) -> None:
    """并发渲染 system prompt 中的 Geo Prompt 引用。"""
    for prompt_message in prompt_messages:
        if prompt_message.role != PromptMessageRole.SYSTEM:
            continue
        if not isinstance(prompt_message.content, str):
            continue

        references = list(dict.fromkeys(_GEO_PROMPT_TOKEN_PATTERN.findall(prompt_message.content)))
        if not references:
            continue

        if invocation_log is not None:
            invocation_log.event(
                "geo_prompt_render_batch_started",
                reference_count=len(_GEO_PROMPT_TOKEN_PATTERN.findall(prompt_message.content)),
                unique_reference_count=len(references),
                max_workers=min(_GEO_PROMPT_RENDER_MAX_WORKERS, len(references)),
            )

        rendered_references: dict[str, str] = {}
        with ThreadPoolExecutor(
            max_workers=min(_GEO_PROMPT_RENDER_MAX_WORKERS, len(references))
        ) as executor:
            futures = {
                executor.submit(render_geo_prompt_text, reference, credentials, invocation_log): reference
                for reference in references
            }
            for future in as_completed(futures):
                reference = futures[future]
                try:
                    rendered_references[reference] = future.result()
                except InvokeError as error:
                    if invocation_log is not None:
                        invocation_log.event(
                            "geo_prompt_render_fallback",
                            reference=reference,
                            fallback="leave_reference_unchanged",
                            error_type=type(error).__name__,
                            error=str(error),
                        )

        for reference, rendered_text in rendered_references.items():
            prompt_message.content = prompt_message.content.replace(reference, rendered_text)


def _normalize_geo_prompt_references(text: str) -> str:
    geo_prompt_tokens = _GEO_PROMPT_TOKEN_PATTERN.findall(text)
    if not geo_prompt_tokens:
        return text
    for token in geo_prompt_tokens:
        reference = _GEO_PROMPT_REFERENCE_PATTERN.fullmatch(token)
        if reference is None:
            raise InvokeError("Geo Prompt 引用必须使用 {{dify_admin:agent.prompt}} 格式。")
    return text
