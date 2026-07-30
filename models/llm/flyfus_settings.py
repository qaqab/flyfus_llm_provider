import json
import re
from dataclasses import dataclass

from dify_plugin.entities.model.message import (
    PromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)

from models.llm.context_tags import FlyfusContextTag, FlyfusSettingType


_SETTING_PATTERN = re.compile(
    rf"<{FlyfusContextTag.SETTING}>(?P<content>.*?)</{FlyfusContextTag.SETTING}>",
    re.DOTALL,
)
_AI_MODE_REFERENCE_PATTERN = re.compile(
    r"\{\{dify_admin:ai_mode\.[A-Za-z0-9_-]+\.(?P<mode>[A-Za-z0-9_-]+)}}"
)
_LOG_CONTEXT_FIELDS = (
    "user_id",
    "app_id",
    "workflow_id",
    "workflow_run_id",
    "conversation_id",
)


@dataclass(frozen=True)
class FlyfusSettings:
    ai_mode_reference: str | None = None
    ai_mode_name: str = ""
    user_id: str = ""
    app_id: str = ""
    workflow_id: str = ""
    workflow_run_id: str = ""
    conversation_id: str = ""

    def log_context(self) -> dict[str, str]:
        return {field: getattr(self, field) for field in _LOG_CONTEXT_FIELDS}


def extract_flyfus_settings(prompt_messages: list[PromptMessage]) -> FlyfusSettings:
    """Remove JSON setting blocks and return their model and logging controls."""
    ai_mode_reference: str | None = None
    ai_mode_name = ""
    log_context = {field: "" for field in _LOG_CONTEXT_FIELDS}
    latest_user_message = next(
        (message for message in reversed(prompt_messages) if isinstance(message, UserPromptMessage)),
        None,
    )

    for message in prompt_messages:
        if not isinstance(message, (UserPromptMessage, ToolPromptMessage)):
            continue

        payloads: list[dict] = []
        if isinstance(message.content, str):
            message.content, payloads = _extract_payloads(message.content)
        elif isinstance(message.content, list):
            for part in message.content:
                if not isinstance(part, TextPromptMessageContent):
                    continue
                part.data, part_payloads = _extract_payloads(part.data)
                payloads.extend(part_payloads)

        for payload in payloads:
            setting_type = payload.get("type")
            if setting_type == FlyfusSettingType.AI_MODE:
                reference = payload.get("reference")
                if isinstance(reference, str):
                    reference = reference.strip()
                    if match := _AI_MODE_REFERENCE_PATTERN.fullmatch(reference):
                        ai_mode_reference = reference
                        ai_mode_name = match.group("mode")
            elif setting_type == FlyfusSettingType.LOG_CONTEXT and message is latest_user_message:
                log_context = _log_context(payload)

    return FlyfusSettings(
        ai_mode_reference=ai_mode_reference,
        ai_mode_name=ai_mode_name,
        **log_context,
    )


def _extract_payloads(text: str) -> tuple[str, list[dict]]:
    payloads: list[dict] = []

    def remove_setting(match: re.Match) -> str:
        try:
            payload = json.loads(match.group("content"))
        except (json.JSONDecodeError, TypeError):
            return ""
        if isinstance(payload, dict):
            payloads.append(payload)
        return ""

    cleaned, count = _SETTING_PATTERN.subn(remove_setting, text)
    return (cleaned.strip() if count else cleaned), payloads


def _log_context(payload: dict) -> dict[str, str]:
    result = {}
    for field in _LOG_CONTEXT_FIELDS:
        value = payload.get(field)
        result[field] = value.strip() if isinstance(value, str) else ""
    return result
