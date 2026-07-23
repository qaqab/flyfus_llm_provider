"""发送上游前的 Dify 消息预处理。"""

import re

from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageRole,
    SystemPromptMessage,
)


_THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def drop_analyze_channel(prompt_messages: list[PromptMessage]) -> None:
    """移除历史 assistant 消息里的思考内容。"""
    for prompt_message in prompt_messages:
        if not isinstance(prompt_message, AssistantPromptMessage):
            continue
        if not isinstance(prompt_message.content, str):
            continue
        if "<think>" not in prompt_message.content:
            continue
        prompt_message.content = _THINK_PATTERN.sub("", prompt_message.content)


def apply_json_schema_prompt(model_parameters: dict, prompt_messages: list[PromptMessage]) -> None:
    """把 Dify 的 json_schema 参数补成兼容端更容易遵循的系统提示。"""
    if model_parameters.get("response_format") != "json_schema":
        return

    json_schema = model_parameters.get("json_schema")
    if not json_schema:
        return

    structured_output_prompt = (
        "Your response must be a JSON object that validates against the following JSON schema, and nothing else.\n"
        f"JSON Schema: ```json\n{json_schema}\n```"
    )
    existing_system_prompt = next(
        (p for p in prompt_messages if p.role == PromptMessageRole.SYSTEM),
        None,
    )
    if existing_system_prompt:
        existing_system_prompt.content = structured_output_prompt + "\n\n" + existing_system_prompt.content
    else:
        prompt_messages.insert(0, SystemPromptMessage(content=structured_output_prompt))
