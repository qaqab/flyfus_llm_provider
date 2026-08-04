import pytest

from dify_plugin.errors.model import InvokeError

from models.llm import invocation_logging, sls_logging
from models.llm.invocation_logging import InvocationLog, wrap_stream_with_invocation_log


def test_stream_failure_is_logged_and_raised_for_dify_retry(monkeypatch) -> None:
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

    with pytest.raises(InvokeError) as raised:
        list(wrap_stream_with_invocation_log(failing_stream(), log))

    message = str(raised.value)
    assert "[模型调用失败]" in message
    assert "invocation_id:" in message
    assert "stream_event_count: 0" in message
    assert "upstream read timed out" in message
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert captured["event"]["status"] == "error"
    assert captured["event"]["timeline"][-1]["name"] == "stream_error"


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

    monkeypatch.setattr(sls_logging, "LogItem", FakeLogItem)
    monkeypatch.setattr(sls_logging, "LogClient", FakeLogClient)
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
        "sls_access_key_id": "key-id",
        "sls_access_key_secret": "key-secret",
    }

    sls_logging.write_invocation_log(credentials, event)

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
