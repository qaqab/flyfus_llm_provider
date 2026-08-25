import json
import re

from dify_plugin.entities.model.message import (
    PromptMessage,
    SystemPromptMessage,
    TextPromptMessageContent,
    UserPromptMessage,
)

from models.llm.context_tags import FlyfusContextTag


_MEMORY_PATTERN = re.compile(
    rf"<{FlyfusContextTag.MEMORY}>(?P<content>.*?)</{FlyfusContextTag.MEMORY}>",
    re.DOTALL,
)


def apply_flyfus_memory(prompt_messages: list[PromptMessage]) -> None:
    """Remove transported memory blocks and inject their content by type."""
    system_memory: str | None = None
    current_user_memory: str | None = None
    latest_user_message = next(
        (message for message in reversed(prompt_messages) if isinstance(message, UserPromptMessage)),
        None,
    )

    for message in prompt_messages:
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
            memory_type = payload.get("type")
            content = payload.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            if memory_type == "system":
                system_memory = content
            elif memory_type == "user" and message is latest_user_message:
                current_user_memory = content

    if system_memory:
        system_message = next(
            (message for message in prompt_messages if isinstance(message, SystemPromptMessage)),
            None,
        )
        if system_message is None:
            prompt_messages.insert(0, SystemPromptMessage(content=system_memory))
        else:
            _append_text(system_message, system_memory)

    if current_user_memory and latest_user_message:
        _append_text(latest_user_message, current_user_memory)


def _extract_payloads(text: str) -> tuple[str, list[dict]]:
    payloads: list[dict] = []

    def remove_memory(match: re.Match) -> str:
        try:
            payload = json.loads(match.group("content"))
        except (json.JSONDecodeError, TypeError):
            return ""
        if isinstance(payload, dict):
            payloads.append(payload)
        return ""

    cleaned, count = _MEMORY_PATTERN.subn(remove_memory, text)
    return (cleaned.strip() if count else cleaned), payloads


def _append_text(message: PromptMessage, content: str) -> None:
    if isinstance(message.content, str):
        message.content = _join_text(message.content, content)
        return
    if isinstance(message.content, list):
        text_part = next(
            (
                part
                for part in reversed(message.content)
                if isinstance(part, TextPromptMessageContent)
            ),
            None,
        )
        if text_part is None:
            message.content.append(TextPromptMessageContent(data=content))
        else:
            text_part.data = _join_text(text_part.data, content)
        return
    message.content = content


def _join_text(existing: str, appended: str) -> str:
    if not existing.strip():
        return appended
    return f"{existing.rstrip()}\n\n{appended}"
