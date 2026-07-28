from models.llm import invocation_logging, sls_logging
from models.llm.invocation_logging import InvocationLog


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
    )
    log.success()

    log.flush()

    assert captured["event"]["model"] == "gpt-5.6-sol"
    assert captured["event"]["configured_model"] == "medium"
    assert captured["event"]["routed_model"] == "gpt-5.6-sol"
    assert captured["event"]["model_route_applied"] is True
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
