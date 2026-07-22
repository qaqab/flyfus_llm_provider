import pytest
from types import SimpleNamespace

from models.llm.invocation_logging import InvocationLog, failure_output_text, prompt_messages_metrics, wrap_stream_with_invocation_log


def test_invocation_log_uses_provider_credentials(monkeypatch) -> None:
    written = []
    monkeypatch.setattr(
        "models.llm.invocation_logging.write_invocation_log",
        lambda credentials, event: written.append((credentials, event)),
    )

    credentials = {"sls_endpoint": "https://sls.example.com"}
    invocation_log = InvocationLog.from_credentials(model="gpt-5.4", credentials=credentials, stream=False, user=None)
    invocation_log.success()
    invocation_log.flush()

    assert written[0][0] is credentials
    assert written[0][1]["invocation_id"] == invocation_log.invocation_id


def test_invocation_log_posts_clean_event(monkeypatch) -> None:
    written = []
    monkeypatch.setattr(
        "models.llm.invocation_logging.write_invocation_log",
        lambda credentials, event: written.append((credentials, event)),
    )

    invocation_log = InvocationLog.from_credentials(
        model="gpt-5.4",
        credentials={
            "api_key": "model-token",
        },
        stream=False,
        user="user-1",
    )
    invocation_log.event(
        "request",
        payload={"api_key": "model-token", "file": "data:image/png;base64,AAAA"},
    )
    invocation_log.success(result_type="LLMResult")
    invocation_log.flush()

    assert len(written) == 1
    event = written[0][1]
    assert event["status"] == "success"
    assert "result" not in event
    assert event["schema_version"] == 4
    assert event["timeline"][0]["invocation_id"] == event["invocation_id"]
    assert event["timeline"][0]["payload"]["api_key"] == "model-token"
    assert event["timeline"][0]["payload"]["file"] == "data:image/png;base64,AAAA"
    assert "events" not in event


def test_invocation_log_keeps_replay_body_without_truncation(monkeypatch) -> None:
    written = []
    monkeypatch.setattr(
        "models.llm.invocation_logging.write_invocation_log",
        lambda credentials, event: written.append((credentials, event)),
    )

    invocation_log = InvocationLog.from_credentials(model="gpt-5.6-sol", credentials={}, stream=True, user=None)
    full_prompt = "x" * 30001
    invocation_log.set_replay_request(
        endpoint="https://litellm.flyfus.com/responses",
        body={"model": "gpt-5.6-sol", "input": [{"role": "user", "content": full_prompt}]},
    )
    invocation_log.success()
    invocation_log.flush()

    replay = written[0][1]["upstream"]["replay"]
    assert replay["endpoint"] == "https://litellm.flyfus.com/responses"
    assert replay["body"]["input"][0]["content"] == full_prompt


def test_invocation_log_does_not_redact_usage_token_counts(monkeypatch) -> None:
    written = []
    monkeypatch.setattr(
        "models.llm.invocation_logging.write_invocation_log",
        lambda credentials, event: written.append((credentials, event)),
    )

    invocation_log = InvocationLog.from_credentials(model="gpt-5.4", credentials={}, stream=False, user=None)
    invocation_log.set_response(
        response_id="resp_123",
        usage={
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "input_tokens_details": {"cached_tokens": 1},
        },
    )
    invocation_log.success()
    invocation_log.flush()

    event = written[0][1]
    assert event["response_id"] == "resp_123"
    assert event["output"]["usage"]["input_tokens"] == 10
    assert event["output"]["usage"]["output_tokens"] == 20
    assert event["output"]["usage"]["total_tokens"] == 30


def test_invocation_log_classifies_system_user_as_single_call(monkeypatch) -> None:
    written = []
    monkeypatch.setattr(
        "models.llm.invocation_logging.write_invocation_log",
        lambda credentials, event: written.append((credentials, event)),
    )

    invocation_log = InvocationLog.from_credentials(model="gpt-5.4", credentials={}, stream=False, user=None)
    invocation_log.set_request(
        prompt_metrics_final={
            "message_count": 2,
            "role_counts": {"system": 1, "user": 1},
        }
    )
    invocation_log.success()
    invocation_log.flush()

    event = written[0][1]
    assert event["input"]["kind"] == "single_call"


def test_prompt_metrics_adds_md5_for_latest_user_message() -> None:
    user_message = SimpleNamespace(role=SimpleNamespace(value="user"), content="优化产品主图")

    metrics = prompt_messages_metrics([user_message])

    assert metrics["latest_user_message"] == "优化产品主图"
    assert metrics["latest_user_message_md5"] == "e20a41c2041abff3d4c1571ca998fe82"


def test_invocation_log_keeps_error_details_without_success_result(monkeypatch) -> None:
    written = []
    monkeypatch.setattr(
        "models.llm.invocation_logging.write_invocation_log",
        lambda credentials, event: written.append((credentials, event)),
    )

    invocation_log = InvocationLog.from_credentials(model="gpt-5.4", credentials={}, stream=False, user=None)
    try:
        raise RuntimeError("boom")
    except RuntimeError as error:
        invocation_log.failure(error)
    invocation_log.flush()

    event = written[0][1]
    assert event["status"] == "error"
    assert event["error"]["error_type"] == "RuntimeError"
    assert event["error"]["error"] == "boom"
    assert "result" not in event


def test_stream_wrapper_flushes_on_success(monkeypatch) -> None:
    flushed = []
    usages = []
    invocation_log = InvocationLog.from_credentials(
        model="gpt-5.4",
        credentials={},
        stream=True,
        user=None,
    )
    monkeypatch.setattr(invocation_log, "flush", lambda: flushed.append(True))

    chunks = list(
        wrap_stream_with_invocation_log(
            iter(["a", "b"]),
            invocation_log,
            lambda usage: usages.append(usage),
        )
    )

    assert chunks == ["a", "b"]
    assert invocation_log.result == {"status": "success", "chunk_count": 2, "output_text": "ab"}
    assert invocation_log.response["output_text"] == "ab"
    assert invocation_log.response["chunk_count"] == 2
    assert flushed == [True]
    assert usages == [None]


def test_stream_wrapper_flushes_on_error(monkeypatch) -> None:
    flushed = []
    invocation_log = InvocationLog.from_credentials(
        model="gpt-5.4",
        credentials={},
        stream=True,
        user=None,
    )
    monkeypatch.setattr(invocation_log, "flush", lambda: flushed.append(True))

    def broken_stream():
        yield "a"
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        list(wrap_stream_with_invocation_log(broken_stream(), invocation_log))

    assert invocation_log.result["status"] == "error"
    assert invocation_log.result["error_type"] == "RuntimeError"
    assert flushed == [True]


def test_stream_wrapper_returns_normal_error_chunk_when_configured(monkeypatch) -> None:
    flushed = []
    invocation_log = InvocationLog.from_credentials(
        model="gpt-5.6-sol",
        credentials={},
        stream=True,
        user=None,
    )
    invocation_log.set_request(prompt_metrics_initial={"message_count": 2, "total_content_chars": 100})
    monkeypatch.setattr(invocation_log, "flush", lambda: flushed.append(True))

    def broken_stream():
        raise RuntimeError("Response ended prematurely")
        yield  # pragma: no cover

    result = list(
        wrap_stream_with_invocation_log(
            broken_stream(),
            invocation_log,
            error_chunk_factory=lambda content, index: {"content": content, "index": index},
        )
    )

    assert result[0]["index"] == 0
    assert "model: gpt-5.6-sol" in result[0]["content"]
    assert "input_content_characters: 100 Unicode characters (not tokens)" in result[0]["content"]
    assert "raw_error: RuntimeError('Response ended prematurely')" in result[0]["content"]
    assert invocation_log.result["status"] == "error"
    assert flushed == [True]


def test_failure_output_includes_upstream_raw_error() -> None:
    invocation_log = InvocationLog.from_credentials(model="gpt-5.6-sol", credentials={}, stream=True, user="user-1")
    invocation_log.set_request(prompt_metrics_initial={"message_count": 3, "total_content_chars": 99})
    invocation_log.set_response(
        error={"code": "context_length_exceeded", "type": "invalid_request_error"},
        http={"headers": {"x-request-id": "request-1"}},
        stream_event_count=4,
    )

    content = failure_output_text(invocation_log, RuntimeError("upstream failed"))

    assert "user: user-1" in content
    assert "upstream_request_id: request-1" in content
    assert 'raw_error: {"code": "context_length_exceeded", "type": "invalid_request_error"}' in content


def test_stream_wrapper_reports_raw_usage_once(monkeypatch) -> None:
    usages = []
    invocation_log = InvocationLog.from_credentials(model="gpt-5.4", credentials={}, stream=True, user=None)
    monkeypatch.setattr(invocation_log, "flush", lambda: None)
    usage = SimpleNamespace(prompt_tokens=9, completion_tokens=4, total_tokens=13)
    chunk = SimpleNamespace(delta=SimpleNamespace(message=None, finish_reason="stop", usage=usage))

    list(wrap_stream_with_invocation_log(iter([chunk]), invocation_log, usages.append))

    assert usages == [usage]
