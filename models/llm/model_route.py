"""Parse model routing directives embedded in Dify system prompts."""

import json
import re
from dataclasses import dataclass
from typing import Any

from dify_plugin.entities.model.message import PromptMessage, SystemPromptMessage

from models.llm.model_catalog import load_model_configs, load_model_extra, load_predefined_chat_models


_ROUTE_PATTERN = re.compile(
    r"<flyfus_model_route>\s*(.*?)\s*</flyfus_model_route>",
    re.DOTALL,
)
_ROUTE_PARAMETER_NAMES = frozenset(
    {
        "enable_thinking",
        "enable_web_search",
        "max_tokens",
        "reasoning_effort",
        "temperature",
        "top_p",
    }
)
_REASONING_EFFORT_VALUES = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})


@dataclass(frozen=True)
class ModelRouteResult:
    model: str
    parameters: dict[str, Any]
    applied: bool = False


def apply_model_route(
    model: str,
    model_parameters: dict[str, Any],
    prompt_messages: list[PromptMessage],
) -> ModelRouteResult:
    """Apply the first valid system-prompt route and remove all route markers."""
    route_payload: dict[str, Any] | None = None
    retained_messages: list[PromptMessage] = []

    for message in prompt_messages:
        if not isinstance(message, SystemPromptMessage) or not isinstance(message.content, str):
            retained_messages.append(message)
            continue

        if route_payload is None:
            route_payload = _first_valid_payload(message.content)
        cleaned_content = _ROUTE_PATTERN.sub("", message.content)
        if cleaned_content.strip():
            message.content = cleaned_content
            retained_messages.append(message)

    prompt_messages[:] = retained_messages
    if route_payload is None:
        return ModelRouteResult(model=model, parameters=dict(model_parameters))

    target_model = route_payload.get("model")
    if not isinstance(target_model, str):
        return ModelRouteResult(model=model, parameters=dict(model_parameters))
    target_model = target_model.strip()
    if target_model not in load_predefined_chat_models():
        return ModelRouteResult(model=model, parameters=dict(model_parameters))

    allowed_parameters = supported_route_parameters(target_model)
    effective_parameters = {
        name: value
        for name, value in model_parameters.items()
        if name in allowed_parameters or name in _declared_parameter_rules(target_model)
    }
    route_parameters = route_payload.get("parameters")
    if isinstance(route_parameters, dict):
        for name, value in route_parameters.items():
            if name not in allowed_parameters:
                continue
            normalized_value = _normalize_route_value(target_model, name, value)
            if normalized_value is not None:
                effective_parameters[name] = normalized_value

    return ModelRouteResult(model=target_model, parameters=effective_parameters, applied=True)


def supported_route_parameters(model: str) -> set[str]:
    declared = set(_declared_parameter_rules(model))
    supported = declared & _ROUTE_PARAMETER_NAMES
    thinking_mode = load_model_extra(model).get("thinking", {}).get("mode")
    if thinking_mode == "gemini":
        supported.add("reasoning_effort")
    return supported


def _first_valid_payload(content: str) -> dict[str, Any] | None:
    for match in _ROUTE_PATTERN.finditer(content):
        try:
            payload = json.loads(match.group(1))
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _declared_parameter_rules(model: str) -> dict[str, dict[str, Any]]:
    rules = load_model_configs().get(model, {}).get("parameter_rules", [])
    if not isinstance(rules, list):
        return {}
    return {
        rule["name"]: rule
        for rule in rules
        if isinstance(rule, dict) and isinstance(rule.get("name"), str)
    }


def _normalize_route_value(model: str, name: str, value: Any) -> Any | None:
    if name in {"enable_thinking", "enable_web_search"}:
        return value if isinstance(value, bool) else None
    if name == "reasoning_effort":
        if not isinstance(value, str):
            return None
        value = value.strip().lower()
        return value if value in _reasoning_effort_values(model) else None
    if name == "max_tokens":
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        return int(_clamp_to_rule(model, name, value))
    if name in {"temperature", "top_p"}:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return None
        return _clamp_to_rule(model, name, value)
    return None


def _reasoning_effort_values(model: str) -> frozenset[str]:
    rule = _declared_parameter_rules(model).get("reasoning_effort", {})
    options = rule.get("options")
    if isinstance(options, list):
        return frozenset(str(option).lower() for option in options)
    return _REASONING_EFFORT_VALUES


def _clamp_to_rule(model: str, name: str, value: int | float) -> int | float:
    rule = _declared_parameter_rules(model).get(name, {})
    minimum = rule.get("min")
    maximum = rule.get("max")
    if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
        value = max(value, minimum)
    if isinstance(maximum, (int, float)) and not isinstance(maximum, bool):
        value = min(value, maximum)
    return value
