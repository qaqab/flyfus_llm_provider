"""从工作流工具输出提取下一步推理强度。"""

import json
from typing import Optional

from dify_plugin.entities.model.message import PromptMessage, ToolPromptMessage


_REASONING_EFFORT_TOOL_NAME = "set_next_step"
_REASONING_EFFORT_VALUES = {"low", "medium", "high", "xhigh"}


def reasoning_effort_from_tool_messages(prompt_messages: list[PromptMessage]) -> Optional[str]:
    """Read the next reasoning effort from the dedicated workflow tool only."""
    reasoning_effort = None
    for prompt_message in prompt_messages:
        if not isinstance(prompt_message, ToolPromptMessage):
            continue
        if prompt_message.name != _REASONING_EFFORT_TOOL_NAME:
            continue
        if not isinstance(prompt_message.content, str):
            continue
        parsed_effort = _extract_reasoning_effort(prompt_message.content)
        if parsed_effort:
            reasoning_effort = parsed_effort
    return reasoning_effort


def _extract_reasoning_effort(content: str) -> Optional[str]:
    payload = _try_parse_json(content.strip())
    for _ in range(4):
        if not isinstance(payload, dict):
            return None

        effort = payload.get("reasoning_effort")
        if isinstance(effort, str) and effort.lower() in _REASONING_EFFORT_VALUES:
            return effort.lower()

        if len(payload) != 1:
            return None
        wrapped_value = next(iter(payload.values()))
        if not isinstance(wrapped_value, str):
            return None
        payload = _try_parse_json(wrapped_value.strip())
    return None


def _try_parse_json(text: str):
    if not text or text[0] not in "{[":
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None
