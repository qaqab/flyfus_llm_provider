from types import SimpleNamespace

import pytest
from dify_plugin.entities.model.message import AssistantPromptMessage, UserPromptMessage
from dify_plugin.errors.model import InvokeError

from models.llm.invocation_logging import InvocationLog
from models.llm.llm import _usage_request_id, FlyfusLargeLanguageModel


def test_request_id_header_is_unique_per_http_request() -> None:
    log_id = "log-123"
    model = object.__new__(FlyfusLargeLanguageModel)

    first = model._request_headers({"_flyfus_log_id": log_id})
    second = model._request_headers({"_flyfus_log_id": log_id})

    assert first["x-request-id"] != second["x-request-id"]
    assert "X-Client-Request-Id" not in first
    assert "X-Flyfus-Invocation-Id" not in first


def test_openai_compatible_stream_records_sent_request_and_response_ids(monkeypatch) -> None:
    posted = {}
    logged = {}

    class Response:
        status_code = 200
        headers = {"x-litellm-call-id": "litellm-stream-1"}
        encoding = None
        text = ""

        def iter_lines(self, **_kwargs):
            return iter(())

    def post(*_args, **kwargs):
        posted.update(kwargs)
        return Response()

    monkeypatch.setattr(
        "models.llm.llm.requests.post",
        post,
    )
    monkeypatch.setattr("models.llm.llm.report_token_usage", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "models.llm.invocation_logging.write_invocation_log",
        lambda _credentials, event: logged.update(event),
    )

    model = object.__new__(FlyfusLargeLanguageModel)
    monkeypatch.setattr(
        model,
        "_create_final_llm_result_chunk",
        lambda *_args, **_kwargs: SimpleNamespace(
            model="qwen3.6-flash",
            delta=SimpleNamespace(
                message=AssistantPromptMessage(content=""),
                usage=None,
                finish_reason="stop",
            ),
        ),
    )
    stream = model._invoke(
        model="qwen3.6-flash",
        credentials={"endpoint_url": "https://example.test/", "api_key": "key"},
        prompt_messages=[
            UserPromptMessage(
                content=(
                    'hello<FLYFUS_SETTING>{"type":"log_context",'
                    '"workflow_run_id":"workflow-run-1",'
                    '"chat_session_id":"chat-session-1"}</FLYFUS_SETTING>'
                )
            )
        ],
        model_parameters={},
        stream=True,
    )
    list(stream)

    request_id = posted["headers"]["x-request-id"]
    assert posted["json"]["metadata"] == {"session_id": "chat-session-1"}
    assert logged["x_request_ids"] == [request_id]
    assert logged["response_id"] == "litellm-stream-1"


def test_geo_usage_request_id_falls_back_to_log_id() -> None:
    log = InvocationLog(model="test-model", credentials={}, stream=False, user=None)

    assert _usage_request_id(log) == log.log_id

    log.set_response(http={"headers": {"x-litellm-call-id": "litellm-call-1"}})

    assert _usage_request_id(log) == "litellm-call-1"


def test_openai_compatible_sends_generated_request_id(monkeypatch) -> None:
    posted = {}
    logged = {}

    class Response:
        status_code = 200
        headers = {"x-litellm-call-id": "litellm-call-1"}
        encoding = None
        text = ""

        def json(self):
            return {
                "id": "body-response-1",
                "model": "qwen3.6-flash",
                "choices": [{"message": {"content": "complete"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    def post(*_args, **kwargs):
        posted.update(kwargs)
        return Response()

    monkeypatch.setattr("models.llm.llm.requests.post", post)
    monkeypatch.setattr(
        model := object.__new__(FlyfusLargeLanguageModel),
        "_handle_generate_response",
        lambda *_args: SimpleNamespace(
            model="qwen3.6-flash",
            message=AssistantPromptMessage(content="complete"),
            usage=None,
        ),
    )
    monkeypatch.setattr("models.llm.llm.report_token_usage", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(
        "models.llm.invocation_logging.write_invocation_log",
        lambda _credentials, event: logged.update(event),
    )

    model._invoke(
        model="qwen3.6-flash",
        credentials={"endpoint_url": "https://example.test/", "api_key": "key"},
        prompt_messages=[UserPromptMessage(content="hello")],
        model_parameters={},
        stream=False,
    )

    request_id = posted["headers"]["x-request-id"]
    assert request_id != logged["log_id"]
    assert logged["x_request_ids"] == [request_id]
    assert logged["response_id"] == "litellm-call-1"


def test_openai_compatible_extra_headers_cannot_override_request_id(monkeypatch) -> None:
    posted = {}

    class Response:
        status_code = 200
        headers = {}
        encoding = None
        text = ""

        def json(self):
            return {
                "id": "body-response-1",
                "model": "qwen3.6-flash",
                "choices": [{"message": {"content": "complete"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    monkeypatch.setattr(
        "models.llm.llm.requests.post",
        lambda *_args, **kwargs: (posted.update(kwargs) or Response()),
    )
    monkeypatch.setattr(
        model := object.__new__(FlyfusLargeLanguageModel),
        "_handle_generate_response",
        lambda *_args: SimpleNamespace(
            model="qwen3.6-flash",
            message=AssistantPromptMessage(content="complete"),
            usage=None,
        ),
    )
    monkeypatch.setattr("models.llm.llm.report_token_usage", lambda *_args, **_kwargs: True)

    model._invoke(
        model="qwen3.6-flash",
        credentials={
            "endpoint_url": "https://example.test/",
            "api_key": "key",
            "extra_headers": {"x-request-id": "fixed-upstream-id"},
        },
        prompt_messages=[UserPromptMessage(content="hello")],
        model_parameters={},
        stream=False,
    )

    assert posted["headers"]["x-request-id"] != "fixed-upstream-id"


def test_openai_compatible_does_not_record_request_id_before_serialization(monkeypatch) -> None:
    logged = {}
    monkeypatch.setattr(
        "models.llm.invocation_logging.write_invocation_log",
        lambda _credentials, event: logged.update(event),
    )

    model = object.__new__(FlyfusLargeLanguageModel)
    with pytest.raises(InvokeError):
        model._invoke(
            model="qwen3.6-flash",
            credentials={"endpoint_url": "https://example.test/", "api_key": "key"},
            prompt_messages=[UserPromptMessage(content="hello")],
            model_parameters={"response_format": "json_schema", "json_schema": "{"},
            stream=False,
        )

    assert logged["x_request_ids"] == []


def test_openai_compatible_error_includes_litellm_call_id(monkeypatch) -> None:
    logged = {}

    class Response:
        status_code = 500
        headers = {"x-litellm-call-id": "litellm-error-1"}
        encoding = None
        text = "upstream failed"

    monkeypatch.setattr(
        "models.llm.llm.requests.post",
        lambda *_args, **_kwargs: Response(),
    )
    monkeypatch.setattr(
        "models.llm.invocation_logging.write_invocation_log",
        lambda _credentials, event: logged.update(event),
    )

    model = object.__new__(FlyfusLargeLanguageModel)
    with pytest.raises(InvokeError, match='"response_id":"litellm-error-1"'):
        model._invoke(
            model="qwen3.6-flash",
            credentials={"endpoint_url": "https://example.test/", "api_key": "key"},
            prompt_messages=[UserPromptMessage(content="hello")],
            model_parameters={},
            stream=False,
        )

    assert logged["response_id"] == "litellm-error-1"
