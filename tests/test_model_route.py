from dify_plugin.entities.model.message import SystemPromptMessage, UserPromptMessage

from models.llm.model_route import apply_model_route


def _route_message(payload: str, prefix: str = "Keep this instruction.\n") -> SystemPromptMessage:
    return SystemPromptMessage(
        content=f"{prefix}<flyfus_model_route>\n{payload}\n</flyfus_model_route>"
    )


def test_gpt_route_filters_unsupported_parameters_and_caps_max_tokens() -> None:
    messages = [
        _route_message(
            """{
  "model": "gpt-5.6-sol",
  "parameters": {
    "max_tokens": 999999,
    "temperature": 0.7,
    "top_p": 0.9,
    "reasoning_effort": "HIGH",
    "enable_thinking": true,
    "enable_web_search": true
  }
}"""
        )
    ]

    result = apply_model_route("medium", {}, messages)

    assert result.applied is True
    assert result.model == "gpt-5.6-sol"
    assert result.parameters == {
        "max_tokens": 128000,
        "temperature": 0.7,
        "top_p": 0.9,
        "reasoning_effort": "high",
        "enable_web_search": True,
    }
    assert messages[0].content == "Keep this instruction.\n"


def test_qwen_route_drops_web_search_and_reasoning_effort() -> None:
    messages = [
        _route_message(
            """{
  "model": "qwen3.7-max",
  "parameters": {
    "max_tokens": 65536,
    "reasoning_effort": "high",
    "enable_thinking": true,
    "enable_web_search": true
  }
}"""
        )
    ]

    result = apply_model_route("gpt-5.6-sol", {"response_format": "text"}, messages)

    assert result.parameters == {
        "response_format": "text",
        "max_tokens": 65536,
        "enable_thinking": True,
    }


def test_gemini_route_keeps_canonical_reasoning_effort_for_existing_adapter() -> None:
    messages = [
        _route_message(
            '{"model":"gemini-3.6-flash","parameters":{"reasoning_effort":"xhigh",'
            '"enable_thinking":true,"enable_web_search":true}}'
        )
    ]

    result = apply_model_route("medium", {}, messages)

    assert "enable_thinking" not in result.parameters
    assert result.parameters["reasoning_effort"] == "xhigh"
    assert result.parameters["enable_web_search"] is True


def test_unknown_model_falls_back_but_still_strips_route_marker() -> None:
    messages = [
        _route_message('{"model":"not-installed","parameters":{"max_tokens":1}}'),
        UserPromptMessage(content="Hello"),
    ]

    result = apply_model_route("medium", {"max_tokens": 12}, messages)

    assert result.applied is False
    assert result.model == "medium"
    assert result.parameters == {"max_tokens": 12}
    assert "flyfus_model_route" not in messages[0].content


def test_route_in_user_message_is_ignored() -> None:
    content = '<flyfus_model_route>{"model":"gpt-5.6-sol"}</flyfus_model_route>'
    messages = [UserPromptMessage(content=content)]

    result = apply_model_route("medium", {"temperature": 1}, messages)

    assert result.applied is False
    assert result.model == "medium"
    assert messages[0].content == content


def test_prompt_without_route_is_unchanged() -> None:
    messages = [SystemPromptMessage(content="Keep this system prompt exactly.")]
    parameters = {"temperature": 0.4, "custom_parameter": "keep"}

    result = apply_model_route("medium", parameters, messages)

    assert result.applied is False
    assert result.model == "medium"
    assert result.parameters == parameters
    assert messages[0].content == "Keep this system prompt exactly."


def test_marker_only_system_message_is_removed() -> None:
    messages = [_route_message('{"model":"qwen3.7-max"}', prefix="")]

    result = apply_model_route("medium", {}, messages)

    assert result.applied is True
    assert messages == []
