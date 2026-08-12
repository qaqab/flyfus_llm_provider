import pytest

from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)

from models.llm.context_guard import (
    CONTEXT_RETRY_STAGES,
    ContextGuardError,
    ContextWindowExceededError,
    guard_prompt_messages,
    is_context_window_error,
    trim_prompt_messages_for_context_retry,
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


def test_context_retry_stage_order() -> None:
    assert CONTEXT_RETRY_STAGES == (
        "p3_history",
        "p2_tool_cycles",
        "p1_history",
    )


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

    with pytest.raises(ContextWindowExceededError, match="MANDATORY_CONTEXT_TOO_LARGE") as raised:
        guard_prompt_messages(
            messages,
            token_counter=_counter,
            context_size=5000,
            max_output_tokens=100,
        )

    assert raised.value.flyfus_error_type == "context_window_exceeded"


def test_guard_accepts_muse_spark_configured_context_size() -> None:
    messages = [
        SystemPromptMessage(content="system"),
        UserPromptMessage(content="current"),
    ]

    result = guard_prompt_messages(
        messages,
        token_counter=lambda _: 31503,
        context_size=1048576,
        max_output_tokens=128000,
    )

    assert result.trimmed is False
    assert result.final_tokens == 31503
    assert result.hard_budget == 899605


def test_protocol_validator_rejects_orphan_tool_result() -> None:
    with pytest.raises(ContextGuardError, match="orphan tool result") as raised:
        validate_message_protocol(
            [ToolPromptMessage(tool_call_id="call-1", content="result")]
        )

    assert raised.value.flyfus_error_type == "invalid_message_protocol"


def test_context_window_error_detection() -> None:
    error = RuntimeError(
        "litellm.APIError: Your input exceeds the context window of this model."
    )

    assert is_context_window_error(error) is True
    assert is_context_window_error(RuntimeError("context_window_exceeded")) is True
    assert is_context_window_error(RuntimeError("timeout")) is False


def test_context_retry_removes_p3_history_before_recent_history() -> None:
    current_user = UserPromptMessage(content="current")
    messages = [
        SystemPromptMessage(content="system"),
        UserPromptMessage(content="old-user"),
        AssistantPromptMessage(content="old-answer"),
        UserPromptMessage(content="recent-user-1"),
        AssistantPromptMessage(content="recent-answer-1"),
        UserPromptMessage(content="recent-user-2"),
        AssistantPromptMessage(content="recent-answer-2"),
        current_user,
    ]

    result = trim_prompt_messages_for_context_retry(
        messages,
        stage="p3_history",
        required_user_message=current_user,
    )

    assert result.stage == "p3_history"
    assert result.removed_message_count == 2
    assert result.removed_block_count == 1
    assert [message.content for message in messages] == [
        "system",
        "recent-user-1",
        "recent-answer-1",
        "recent-user-2",
        "recent-answer-2",
        "current",
    ]


def test_context_retry_removes_p1_history_after_p3_stage() -> None:
    current_user = UserPromptMessage(content="current")
    messages = [
        SystemPromptMessage(content="system"),
        UserPromptMessage(content="recent-user-1"),
        AssistantPromptMessage(content="recent-answer-1"),
        UserPromptMessage(content="recent-user-2"),
        AssistantPromptMessage(content="recent-answer-2"),
        current_user,
    ]

    result = trim_prompt_messages_for_context_retry(
        messages,
        stage="p1_history",
        required_user_message=current_user,
    )

    assert result.stage == "p1_history"
    assert result.removed_message_count == 4
    assert result.removed_block_count == 2
    assert [message.content for message in messages] == ["system", "current"]


def test_context_retry_removes_complete_p2_tool_cycles() -> None:
    current_user = UserPromptMessage(content="current")
    parallel_calls = AssistantPromptMessage(
        content="",
        tool_calls=[_tool_call("call-1"), _tool_call("call-2")],
    )
    messages = [
        SystemPromptMessage(content="system"),
        current_user,
        parallel_calls,
        ToolPromptMessage(
            name="batch_call",
            tool_call_id="call-1",
            content="large-result-1",
        ),
        ToolPromptMessage(
            name="batch_call",
            tool_call_id="call-2",
            content="large-result-2",
        ),
        AssistantPromptMessage(content="progress update"),
    ]

    result = trim_prompt_messages_for_context_retry(
        messages,
        stage="p2_tool_cycles",
        required_user_message=current_user,
    )

    assert result.stage == "p2_tool_cycles"
    assert result.removed_message_count == 3
    assert result.removed_block_count == 1
    assert [message.content for message in messages] == [
        "system",
        "current",
        "progress update",
    ]
    validate_message_protocol(messages)


def test_context_retry_p2_is_noop_without_complete_tool_cycle() -> None:
    current_user = UserPromptMessage(content="current")
    messages = [current_user, AssistantPromptMessage(content="ordinary answer")]

    result = trim_prompt_messages_for_context_retry(
        messages,
        stage="p2_tool_cycles",
        required_user_message=current_user,
    )

    assert result.trimmed is False
    assert [message.content for message in messages] == ["current", "ordinary answer"]


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
