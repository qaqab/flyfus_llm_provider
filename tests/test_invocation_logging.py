import json

import pytest
from dify_plugin.errors.model import InvokeError

from models.llm import invocation_logging, sls_logging
from models.llm.invocation_logging import InvocationLog, wrap_stream_with_invocation_log
from models.llm.native.openai_responses import _RetryableStreamError


def _error_payload(message: str) -> dict:
    prefix = "<FLYFUS_ERROR>"
    suffix = "</FLYFUS_ERROR>"
    assert message.startswith(prefix)
    assert message.endswith(suffix)
    return json.loads(message[len(prefix) : -len(suffix)])


def test_stream_failure_is_logged_and_raised_as_error_envelope(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        invocation_logging,
        "write_invocation_log",
        lambda credentials, event: captured.update(event=event),
    )
    log = InvocationLog(model="gemini-3.6-flash", credentials={}, stream=True, user="user-1")
    log.set_request(
        prompt_metrics_final={
            "message_count": 3,
            "total_content_chars": 11623,
            "role_counts": {"system": 1, "user": 2},
        }
    )

    def failing_stream():
        raise RuntimeError("upstream read timed out")
        yield

    with pytest.raises(InvokeError) as exc_info:
        list(wrap_stream_with_invocation_log(failing_stream(), log))

    message = str(exc_info.value)
    payload = _error_payload(message)
    error_details = payload.pop("error")
    assert payload == {
        "type": "model_error",
        "user_message": "模型服务暂时不可用，请稍后重试。",
        "retryable": False,
        "partial_output": False,
        "log_id": log.invocation_id,
        "invocation_id": log.invocation_id,
        "request_id": log.invocation_id,
        "client_request_id": log.invocation_id,
        "response_id": None,
        "upstream_request_id": None,
        "cf_ray": None,
    }
    assert "model: gemini-3.6-flash" in error_details
    assert "error_type: RuntimeError" in error_details
    assert "error: upstream read timed out" in error_details
    assert payload["log_id"] == log.invocation_id
    assert captured["event"]["status"] == "error"
    assert captured["event"]["timeline"][-1]["name"] == "stream_error"


def test_recognized_stream_failure_uses_specific_flyfus_error_type(monkeypatch) -> None:
    monkeypatch.setattr(invocation_logging, "write_invocation_log", lambda *_args: None)
    log = InvocationLog(model="gpt-5.6-sol", credentials={}, stream=True, user="user-1")
    log.set_response(
        response_id="resp_123",
        http={
            "headers": {
                "x-request-id": "request-123",
                "x-client-request-id": "client-request-123",
                "cf-ray": "ray-123",
            }
        },
        stream_event_count=44,
        stream_event_counts={"response.created": 1, "unknown": 1},
        stream_last_event_type="unknown",
    )
    log.event("stream_retry", attempt=1)
    log.event("gemini_retry", retry=1)
    log.event("request_retry", retry=1)

    def failing_stream():
        raise _RetryableStreamError("missing response.completed")
        yield

    with pytest.raises(InvokeError) as exc_info:
        list(wrap_stream_with_invocation_log(failing_stream(), log))

    payload = _error_payload(str(exc_info.value))
    error_details = payload.pop("error")
    assert payload == {
        "type": "upstream_stream_incomplete",
        "user_message": "模型响应中断，请稍后重试。",
        "retryable": True,
        "partial_output": False,
        "log_id": log.invocation_id,
        "invocation_id": log.invocation_id,
        "request_id": log.invocation_id,
        "client_request_id": log.invocation_id,
        "response_id": "resp_123",
        "upstream_request_id": "request-123",
        "cf_ray": "ray-123",
    }
    assert "stream_event_count: 44" in error_details
    assert 'stream_event_counts: {"response.created": 1, "unknown": 1}' in error_details
    assert "retry_count: 3" in error_details
    assert "error_type: _RetryableStreamError" in error_details
    assert "error: missing response.completed" in error_details


def test_partial_stream_failure_preserves_chunks_then_raises_error_envelope(monkeypatch) -> None:
    monkeypatch.setattr(invocation_logging, "write_invocation_log", lambda *_args: None)
    log = InvocationLog(model="gpt-5.6-sol", credentials={}, stream=True, user="user-1")

    class Chunk:
        delta = None

    def failing_stream():
        yield Chunk()
        raise RuntimeError("stream disconnected")

    stream = wrap_stream_with_invocation_log(failing_stream(), log)
    assert isinstance(next(stream), Chunk)
    with pytest.raises(InvokeError) as exc_info:
        next(stream)

    payload = _error_payload(str(exc_info.value))
    assert payload["partial_output"] is True
    assert payload["log_id"] == log.invocation_id


def test_invocation_event_exposes_model_route_fields(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        invocation_logging,
        "write_invocation_log",
        lambda credentials, event: captured.update(event=event),
    )
    log = InvocationLog(model="gpt-5.6-sol", credentials={}, stream=True, user="")
    log.set_request(
        configured_model="medium",
        routed_model="gpt-5.6-sol",
        model_route_applied=True,
        user_id="user-1",
        app_id="app-1",
        workflow_id="workflow-1",
        conversation_id="conversation-1",
    )
    log.success()

    log.flush()

    assert captured["event"]["schema_version"] == 7
    assert captured["event"]["log_id"] == log.invocation_id
    assert captured["event"]["request_id"] == log.invocation_id
    assert captured["event"]["x_request_id"] == log.invocation_id
    assert captured["event"]["invocation_id"] == log.invocation_id
    assert captured["event"]["ids"]["log_id"] == log.invocation_id
    assert captured["event"]["ids"]["request_id"] == log.invocation_id
    assert captured["event"]["ids"]["x_request_id"] == log.invocation_id
    assert captured["event"]["model"] == "gpt-5.6-sol"
    assert captured["event"]["configured_model"] == "medium"
    assert captured["event"]["routed_model"] == "gpt-5.6-sol"
    assert captured["event"]["model_route_applied"] is True
    assert captured["event"]["user_id"] == "user-1"
    assert captured["event"]["app_id"] == "app-1"
    assert captured["event"]["workflow_id"] == "workflow-1"
    assert captured["event"]["workflow_run_id"] == ""
    assert captured["event"]["conversation_id"] == "conversation-1"
    assert captured["event"]["input"]["configured_model"] == "medium"


def test_sls_log_indexes_model_route_fields(monkeypatch) -> None:
    captured = {}

    class FakeLogItem:
        def set_time(self, value):
            captured["time"] = value

        def set_contents(self, value):
            captured["contents"] = dict(value)

    class FakeLogClient:
        def __init__(self, *args):
            pass

        def put_logs(self, request):
            captured["request"] = request

    class FakePutLogsRequest:
        def __init__(self, project, logstore, topic, source, logitems):
            captured["project"] = project
            captured["logstore"] = logstore

    monkeypatch.setattr(sls_logging, "LogItem", FakeLogItem)
    monkeypatch.setattr(sls_logging, "LogClient", FakeLogClient)
    monkeypatch.setattr(sls_logging, "PutLogsRequest", FakePutLogsRequest)
    event = {
        "invocation_id": "invocation-1",
        "model": "gpt-5.6-sol",
        "configured_model": "medium",
        "routed_model": "gpt-5.6-sol",
        "model_route_applied": True,
        "user": "dify-user",
        "user_id": "user-1",
        "workflow_id": "workflow-1",
        "workflow_run_id": "run-1",
        "conversation_id": "conversation-1",
    }
    credentials = {
        "sls_endpoint": "endpoint",
        "sls_project": "project",
        "sls_logstore": "custom-logstore",
        "sls_access_key_id": "key-id",
        "sls_access_key_secret": "key-secret",
    }

    sls_logging.write_invocation_log(credentials, event)

    assert captured["contents"]["log_id"] == "invocation-1"
    assert captured["contents"]["request_id"] == "invocation-1"
    assert captured["contents"]["x_request_id"] == "invocation-1"
    assert captured["contents"]["invocation_id"] == "invocation-1"
    assert captured["contents"]["model"] == "gpt-5.6-sol"
    assert captured["contents"]["configured_model"] == "medium"
    assert captured["contents"]["routed_model"] == "gpt-5.6-sol"
    assert captured["contents"]["model_route_applied"] == "true"
    assert captured["contents"]["user"] == "dify-user"
    assert captured["contents"]["user_id"] == "user-1"
    assert captured["contents"]["app_id"] == ""
    assert captured["contents"]["workflow_id"] == "workflow-1"
    assert captured["contents"]["workflow_run_id"] == "run-1"
    assert captured["contents"]["conversation_id"] == "conversation-1"
    assert captured["logstore"] == "custom-logstore"


def test_sls_logstore_defaults_for_existing_credentials(monkeypatch) -> None:
    captured = {}

    class FakeLogItem:
        def set_time(self, value):
            pass

        def set_contents(self, value):
            pass

    class FakeLogClient:
        def __init__(self, *args):
            pass

        def put_logs(self, request):
            pass

    class FakePutLogsRequest:
        def __init__(self, project, logstore, topic, source, logitems):
            captured["logstore"] = logstore

    monkeypatch.setattr(sls_logging, "LogItem", FakeLogItem)
    monkeypatch.setattr(sls_logging, "LogClient", FakeLogClient)
    monkeypatch.setattr(sls_logging, "PutLogsRequest", FakePutLogsRequest)

    sls_logging.write_invocation_log(
        {
            "sls_endpoint": "endpoint",
            "sls_project": "project",
            "sls_logstore": "   ",
            "sls_access_key_id": "key-id",
            "sls_access_key_secret": "key-secret",
        },
        {"invocation_id": "invocation-1"},
    )

    assert captured["logstore"] == "flyfus-dify-llm-log"
