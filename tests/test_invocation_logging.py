import pytest

from models.llm.invocation_logging import InvocationLog, wrap_stream_with_invocation_log


def test_invocation_log_uses_hardcoded_token(monkeypatch) -> None:
    posted = []
    monkeypatch.delenv("AXIOM_API_TOKEN", raising=False)
    monkeypatch.setattr("models.llm.invocation_logging.requests.post", lambda *args, **kwargs: posted.append((args, kwargs)))

    invocation_log = InvocationLog.from_credentials(model="gpt-5.4", credentials={}, stream=False, user=None)
    invocation_log.success()
    invocation_log.flush()

    assert len(posted) == 1
    assert posted[0][1]["headers"]["Authorization"].startswith("Bearer xaat-")


def test_invocation_log_posts_clean_event(monkeypatch) -> None:
    posted = []

    def fake_post(*args, **kwargs):
        posted.append((args, kwargs))

    monkeypatch.setattr("models.llm.invocation_logging.requests.post", fake_post)

    invocation_log = InvocationLog.from_credentials(
        model="gpt-5.4",
        credentials={
            "axiom_api_token": "axiom-token",
            "api_key": "model-token",
            "axiom_dataset": "plugins_test",
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

    assert len(posted) == 1
    args, kwargs = posted[0]
    assert args[0] == "https://us-east-1.aws.edge.axiom.co/v1/ingest/plugins_test"
    assert kwargs["headers"]["Authorization"] == "Bearer axiom-token"
    event = kwargs["json"][0]
    assert event["status"] == "success"
    assert "result" not in event
    assert event["schema_version"] == 3
    assert event["timeline"][0]["invocation_id"] == event["invocation_id"]
    assert event["timeline"][0]["payload"]["api_key"] == "model-token"
    assert event["timeline"][0]["payload"]["file"] == "data:image/png;base64,AAAA"
    assert "events" not in event


def test_invocation_log_does_not_redact_usage_token_counts(monkeypatch) -> None:
    posted = []
    monkeypatch.setattr("models.llm.invocation_logging.requests.post", lambda *args, **kwargs: posted.append((args, kwargs)))

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

    event = posted[0][1]["json"][0]
    assert event["response_id"] == "resp_123"
    assert event["output"]["usage"]["input_tokens"] == 10
    assert event["output"]["usage"]["output_tokens"] == 20
    assert event["output"]["usage"]["total_tokens"] == 30


def test_invocation_log_classifies_system_user_as_single_call(monkeypatch) -> None:
    posted = []
    monkeypatch.setattr("models.llm.invocation_logging.requests.post", lambda *args, **kwargs: posted.append((args, kwargs)))

    invocation_log = InvocationLog.from_credentials(model="gpt-5.4", credentials={}, stream=False, user=None)
    invocation_log.set_request(
        prompt_metrics_final={
            "message_count": 2,
            "role_counts": {"system": 1, "user": 1},
        }
    )
    invocation_log.success()
    invocation_log.flush()

    event = posted[0][1]["json"][0]
    assert event["input"]["kind"] == "single_call"


def test_invocation_log_keeps_error_details_without_success_result(monkeypatch) -> None:
    posted = []
    monkeypatch.setattr("models.llm.invocation_logging.requests.post", lambda *args, **kwargs: posted.append((args, kwargs)))

    invocation_log = InvocationLog.from_credentials(model="gpt-5.4", credentials={}, stream=False, user=None)
    try:
        raise RuntimeError("boom")
    except RuntimeError as error:
        invocation_log.failure(error)
    invocation_log.flush()

    event = posted[0][1]["json"][0]
    assert event["status"] == "error"
    assert event["error"]["error_type"] == "RuntimeError"
    assert event["error"]["error"] == "boom"
    assert "result" not in event


def test_stream_wrapper_flushes_on_success(monkeypatch) -> None:
    flushed = []
    invocation_log = InvocationLog.from_credentials(
        model="gpt-5.4",
        credentials={},
        stream=True,
        user=None,
    )
    monkeypatch.setattr(invocation_log, "flush", lambda: flushed.append(True))

    chunks = list(wrap_stream_with_invocation_log(iter(["a", "b"]), invocation_log))

    assert chunks == ["a", "b"]
    assert invocation_log.result == {"status": "success", "chunk_count": 2, "output_text": "ab"}
    assert invocation_log.response["output_text"] == "ab"
    assert invocation_log.response["chunk_count"] == 2
    assert flushed == [True]


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
