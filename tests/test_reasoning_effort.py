import pytest

from models.llm.errors import GeminiStreamIncompleteError
from models.llm.llm import (
    FlyfusLargeLanguageModel,
    _GEMINI_MAX_RETRIES,
    _GeminiPartialOutputError,
    _GeminiRetryExhaustedError,
)
from models.llm.native.gemini import GeminiNativeDocumentAdapter
from models.llm.reasoning_effort import reasoning_effort_from_tool_messages
from dify_plugin.entities.model.message import ToolPromptMessage, UserPromptMessage
from dify_plugin.errors.model import InvokeError


@pytest.mark.parametrize(
    ("reasoning_effort", "thinking_level"),
    [
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
        ("xhigh", "High"),
    ],
)
def test_set_next_step_effort_maps_to_gemini_thinking_level(
    reasoning_effort: str, thinking_level: str
) -> None:
    parameters = {"reasoning_effort": reasoning_effort, "thinking_config": {"thinking_level": "Low"}}

    FlyfusLargeLanguageModel._apply_reasoning_effort(parameters, "gemini", {})

    body = GeminiNativeDocumentAdapter(
        endpoint_url=lambda _credentials, _path: "https://example.test/",
        normalize_model_parameters=lambda _model, value: value,
        calc_response_usage=lambda *_args: None,
    ).build_body(
        model="gemini-3.6-flash",
        prompt_messages=[UserPromptMessage(content="Reply")],
        model_parameters=parameters,
        tools=None,
        stop=None,
    )

    assert "reasoning_effort" not in parameters
    assert body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == thinking_level


def test_set_next_step_accepts_max_effort() -> None:
    messages = [
        ToolPromptMessage(
            name="set_next_step",
            tool_call_id="call-1",
            content='{"reasoning_effort":"max"}',
        )
    ]

    assert reasoning_effort_from_tool_messages(messages) == "max"


def test_gemini_retry_reason_covers_empty_and_malformed_responses() -> None:
    assert (
        FlyfusLargeLanguageModel._gemini_retry_reason(
            InvokeError("Gemini 原生接口返回空响应（finish_reasons=['STOP']）")
        )
        == "empty_response"
    )
    assert (
        FlyfusLargeLanguageModel._gemini_retry_reason(
            InvokeError("finishReason=MALFORMED_FUNCTION_CALL")
        )
        == "malformed_function_call"
    )
    assert FlyfusLargeLanguageModel._gemini_retry_reason(InvokeError("Gemini 请求超时")) is None
    assert (
        FlyfusLargeLanguageModel._gemini_retry_reason(
            GeminiStreamIncompleteError("未收到 finishReason")
        )
        == "stream_incomplete"
    )
    assert FlyfusLargeLanguageModel._gemini_max_retries("stream_incomplete") == 1


def test_gemini_retry_exhausted_errors_keep_distinct_root_cause_types() -> None:
    empty_response = _GeminiRetryExhaustedError(
        "empty_response",
        _GEMINI_MAX_RETRIES,
        InvokeError("Gemini 原生接口返回空响应"),
    )
    malformed_function_call = _GeminiRetryExhaustedError(
        "malformed_function_call",
        _GEMINI_MAX_RETRIES,
        InvokeError("finishReason=MALFORMED_FUNCTION_CALL"),
    )

    assert empty_response.flyfus_error_type == "gemini_empty_response"
    assert empty_response.flyfus_retryable is False
    assert "未返回有效内容" in empty_response.flyfus_user_message
    assert "已重试 2 次" in str(empty_response)
    assert "reason=empty_response" in str(empty_response)

    assert malformed_function_call.flyfus_error_type == "gemini_malformed_function_call"
    assert malformed_function_call.flyfus_retryable is False
    assert "模型执行任务时发生异常" in malformed_function_call.flyfus_user_message
    assert "已重试 2 次" in str(malformed_function_call)
    assert "reason=malformed_function_call" in str(malformed_function_call)


def test_gemini_partial_output_error_is_distinct_and_not_retryable() -> None:
    error = _GeminiPartialOutputError(InvokeError("Gemini 原生接口流式响应提前结束"))

    assert error.flyfus_error_type == "gemini_partial_output_error"
    assert error.flyfus_retryable is False
    assert "本次回答不完整" in error.flyfus_user_message
    assert "已停止自动重试" in str(error)
