"""Resolve AI Mode references embedded in Flyfus context blocks."""

import re

import requests

from dify_plugin.entities.model.message import (
    PromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)

from models.llm.model_route import ModelRouteResult


_CONTEXT_PATTERN = re.compile(r"<FLYFUS_CONTEXT>(?P<content>.*?)</FLYFUS_CONTEXT>", re.DOTALL)
_AI_MODE_PATTERN = re.compile(r"\{\{dify_admin:ai_mode\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+}}")


def apply_ai_mode(
    model: str,
    model_parameters: dict,
    prompt_messages: list[PromptMessage],
    credentials: dict,
) -> ModelRouteResult:
    """Apply the last AI Mode reference found in User or Tool context blocks."""
    reference = _extract_and_remove_references(prompt_messages)
    fallback = ModelRouteResult(model=model, parameters=dict(model_parameters))
    if reference is None:
        return fallback

    base_url = str(credentials.get("geo_prompt_render_url") or "").strip().rstrip("/")
    if not base_url:
        return fallback

    try:
        response = requests.post(
            f"{base_url}/dify_admin/ai_mode/resolve_reference",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {str(credentials.get('geo_prompt_api_key') or '').strip()}",
            },
            json={"reference": reference},
            timeout=(10, 60),
        )
        payload = response.json()
        config = payload["data"]["config"]
        target_model = config["model"]
        target_parameters = config["parameters"]
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return fallback

    if (
        response.status_code != 200
        or payload.get("code") != 200
        or not isinstance(target_model, str)
        or not isinstance(target_parameters, dict)
    ):
        return fallback
    return ModelRouteResult(model=target_model, parameters=target_parameters, applied=True)


def _extract_and_remove_references(prompt_messages: list[PromptMessage]) -> str | None:
    last_reference: str | None = None
    for message in prompt_messages:
        if not isinstance(message, (UserPromptMessage, ToolPromptMessage)):
            continue
        text_parts = (
            [message]
            if isinstance(message.content, str)
            else [part for part in message.content or [] if isinstance(part, TextPromptMessageContent)]
        )
        for part in text_parts:
            text = part.content if isinstance(part, PromptMessage) else part.data
            cleaned, references = _clean_text(text)
            if references:
                cleaned = cleaned.strip()
            if isinstance(part, PromptMessage):
                part.content = cleaned
            else:
                part.data = cleaned
            if references:
                last_reference = references[-1]
    return last_reference


def _clean_text(text: str) -> tuple[str, list[str]]:
    references: list[str] = []

    def clean_context(match: re.Match) -> str:
        content = match.group("content")
        references.extend(_AI_MODE_PATTERN.findall(content))
        cleaned_content = _AI_MODE_PATTERN.sub("", content)
        return f"<FLYFUS_CONTEXT>{cleaned_content}</FLYFUS_CONTEXT>" if cleaned_content.strip() else ""

    return _CONTEXT_PATTERN.sub(clean_context, text), references
