import pytest

from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)

from models.llm.context_guard import (
    ContextGuardError,
    guard_prompt_messages,
    is_context_window_error,
    validate_message_protocol,
)


def _tool_call(call_id: str):
    return AssistantPromptMessage.ToolCall(
        id=call_id,
        type="function",
        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
            name="lookup",
            arguments="{}",
        ),
    )


def _counter(messages):
    return sum(len(str(message.content or "")) + 10 for message in messages)


def test_guard_keeps_context_below_soft_limit_unchanged() -> None:
    messages = [
        SystemPromptMessage(content="system"),
        UserPromptMessage(content="current"),
    ]

    result = guard_prompt_messages(
        messages,
        token_counter=_counter,
        context_size=1000,
        max_output_tokens=100,
    )

    assert result.trimmed is False
    assert [message.content for message in messages] == ["system", "current"]


def test_guard_removes_oldest_complete_turn_first() -> None:
    messages = [
        SystemPromptMessage(content="system"),
        UserPromptMessage(content="old-user-1" * 20),
        AssistantPromptMessage(content="old-answer-1" * 20),
        UserPromptMessage(content="old-user-2" * 20),
        AssistantPromptMessage(content="old-answer-2" * 20),
        UserPromptMessage(content="current-user"),
    ]

    result = guard_prompt_messages(
        messages,
        token_counter=_counter,
        context_size=5200,
        max_output_tokens=100,
        extra_input_tokens=4000,
    )

    contents = [str(message.content) for message in messages]
    assert result.trimmed is True
    assert not any("old-user-1" in content for content in contents)
    assert not any("old-answer-1" in content for content in contents)
    assert contents[-1] == "current-user"


def test_guard_removes_historical_tool_call_and_results_as_one_block() -> None:
    messages = [
        SystemPromptMessage(content="system"),
        UserPromptMessage(content="historical"),
        AssistantPromptMessage(content="", tool_calls=[_tool_call("call-1")]),
        ToolPromptMessage(
            name="lookup",
            tool_call_id="call-1",
            content="old-tool-result" * 100,
        ),
        AssistantPromptMessage(content="historical-answer"),
        UserPromptMessage(content="current"),
        AssistantPromptMessage(content="", tool_calls=[_tool_call("call-2")]),
        ToolPromptMessage(
            name="lookup",
            tool_call_id="call-2",
            content="latest-result",
        ),
    ]

    result = guard_prompt_messages(
        messages,
        token_counter=_counter,
        context_size=6000,
        max_output_tokens=100,
        extra_input_tokens=3900,
    )

    assert result.trimmed is True
    assert not any(
        isinstance(message, ToolPromptMessage) and message.tool_call_id == "call-1"
        for message in messages
    )
    assert not any(
        isinstance(message, AssistantPromptMessage)
        and any(call.id == "call-1" for call in (message.tool_calls or []))
        for message in messages
    )
    assert any(
        isinstance(message, ToolPromptMessage) and message.tool_call_id == "call-2"
        for message in messages
    )
    validate_message_protocol(messages)


def test_guard_never_removes_current_turn_tool_cycle() -> None:
    system = SystemPromptMessage(content="system")
    current_user = UserPromptMessage(content="current")
    current_call = AssistantPromptMessage(
        content="",
        tool_calls=[_tool_call("call-1")],
    )
    current_result = ToolPromptMessage(
        name="lookup",
        tool_call_id="call-1",
        content="required-current-result" * 100,
    )
    messages = [
        system,
        current_user,
        current_call,
        current_result,
    ]

    result = guard_prompt_messages(
        messages,
        token_counter=_counter,
        context_size=100000,
        max_output_tokens=1000,
        emergency=True,
        required_user_message=current_user,
    )

    assert result.trimmed is False
    assert messages == [
        system,
        current_user,
        current_call,
        current_result,
    ]


def test_guard_fails_when_mandatory_context_exceeds_budget() -> None:
    messages = [
        SystemPromptMessage(content="system" * 1000),
        UserPromptMessage(content="current" * 1000),
    ]

    with pytest.raises(ContextGuardError, match="MANDATORY_CONTEXT_TOO_LARGE"):
        guard_prompt_messages(
            messages,
            token_counter=_counter,
            context_size=5000,
            max_output_tokens=100,
        )


def test_protocol_validator_rejects_orphan_tool_result() -> None:
    with pytest.raises(ContextGuardError, match="orphan tool result"):
        validate_message_protocol(
            [ToolPromptMessage(tool_call_id="call-1", content="result")]
        )


def test_context_window_error_detection() -> None:
    error = RuntimeError(
        "litellm.APIError: Your input exceeds the context window of this model."
    )

    assert is_context_window_error(error) is True
    assert is_context_window_error(RuntimeError("timeout")) is False


def test_emergency_guard_trims_even_when_configured_window_is_large() -> None:
    messages = [
        SystemPromptMessage(content="system"),
        UserPromptMessage(content="old" * 1000),
        AssistantPromptMessage(content="answer" * 1000),
        UserPromptMessage(content="current"),
    ]

    result = guard_prompt_messages(
        messages,
        token_counter=_counter,
        context_size=100000,
        max_output_tokens=1000,
        emergency=True,
    )

    assert result.trimmed is True
    assert result.final_tokens < result.initial_tokens
    assert [message.content for message in messages] == ["system", "current"]


def test_guard_preserves_real_user_before_injected_context_user() -> None:
    current_user = UserPromptMessage(content="real-current-user")
    injected_context = UserPromptMessage(content="injected-image-context")
    messages = [
        SystemPromptMessage(content="system"),
        UserPromptMessage(content="old" * 1000),
        AssistantPromptMessage(content="answer" * 1000),
        current_user,
        injected_context,
    ]

    result = guard_prompt_messages(
        messages,
        token_counter=_counter,
        context_size=100000,
        max_output_tokens=1000,
        emergency=True,
        required_user_message=current_user,
    )

    assert result.trimmed is True
    assert current_user in messages
    assert injected_context in messages
