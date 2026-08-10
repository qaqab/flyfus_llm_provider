from types import SimpleNamespace

import pytest

from models.llm.errors import UpstreamBadRequestError, UpstreamResponseIncompleteError
from models.llm.native.openai_responses import OpenAIResponsesAdapter
from dify_plugin.entities.model.llm import LLMUsage
from dify_plugin.entities.model.message import SystemPromptMessage
from dify_plugin.errors.model import InvokeError


def _adapter() -> OpenAIResponsesAdapter:
    return OpenAIResponsesAdapter(
        endpoint_url=lambda _credentials, _path: "https://example.test/",
        request_headers=lambda _credentials: {},
        normalize_model_parameters=lambda _model, parameters: parameters,
        calc_response_usage=lambda *_args: LLMUsage.empty_usage(),
        create_final_chunk=lambda *_args, **_kwargs: SimpleNamespace(
            delta=SimpleNamespace(usage=None)
        ),
    )


def test_responses_system_message_omits_type_and_request_omits_user() -> None:
    body = _adapter()._build_body(
        model="gpt-5",
        credentials={},
        prompt_messages=[SystemPromptMessage(content="You are helpful.")],
        model_parameters={},
        tools=None,
        stop=None,
        stream=False,
        user="af158386-224e-4dba-b382-5a3aa1209fde",
    )

    assert body["input"] == [{"role": "system", "content": "You are helpful."}]
    assert "user" not in body


def test_responses_stream_requires_completed_terminal_event() -> None:
    class Response:
        def iter_lines(self, decode_unicode=False):
            assert decode_unicode is False
            yield b'data: {"type":"response.created"}'
            yield b'data: {"type":"response.output_text.delta","delta":"partial"}'
            yield b'data: {"error":{"message":"upstream closed"}}'

    with pytest.raises(InvokeError, match="未收到 response.completed"):
        list(
            _adapter()._handle_stream(
                model="gpt-5.6-sol",
                credentials={},
                response=Response(),
                prompt_messages=[],
            )
        )


def test_responses_stream_accepts_completed_terminal_event() -> None:
    class Response:
        def iter_lines(self, decode_unicode=False):
            assert decode_unicode is False
            yield b'data: {"type":"response.output_text.delta","delta":"complete"}'
            yield (
                b'data: {"type":"response.completed","response":'
                b'{"id":"response-1","model":"gpt-5.6-sol","usage":{}}}'
            )

    chunks = list(
        _adapter()._handle_stream(
            model="gpt-5.6-sol",
            credentials={},
            response=Response(),
            prompt_messages=[],
        )
    )

    assert len(chunks) == 2


def test_responses_stream_rejects_incomplete_terminal_event() -> None:
    class Response:
        def iter_lines(self, decode_unicode=False):
            assert decode_unicode is False
            yield b'data: {"type":"response.output_text.delta","delta":"partial"}'
            yield (
                b'data: {"type":"response.incomplete","response":'
                b'{"id":"response-1","incomplete_details":{"reason":"max_output_tokens"}}}'
            )

    with pytest.raises(UpstreamResponseIncompleteError, match="max_output_tokens"):
        list(
            _adapter()._handle_stream(
                model="gpt-5.6-sol",
                credentials={},
                response=Response(),
                prompt_messages=[],
            )
        )


def test_responses_request_retries_server_error_once(monkeypatch) -> None:
    class Response:
        def __init__(self, status_code, text=""):
            self.status_code = status_code
            self.text = text
            self.headers = {}
            self.closed = False

        def close(self):
            self.closed = True

    class InvocationLog:
        def __init__(self):
            self.response = {}
            self.events = []

        def set_response(self, **fields):
            self.response.update(fields)

        def event(self, name, **fields):
            self.events.append({"name": name, **fields})

    first = Response(503, "temporarily unavailable")
    second = Response(200)
    responses = iter([first, second])
    monkeypatch.setattr(
        "models.llm.native.openai_responses.requests.post",
        lambda *_args, **_kwargs: next(responses),
    )
    invocation_log = InvocationLog()

    result = _adapter()._open_response(
        model="gpt-5.6-sol",
        credentials={},
        request_body={"model": "gpt-5.6-sol"},
        stream=True,
        invocation_log=invocation_log,
    )

    assert result is second
    assert first.closed is True
    assert invocation_log.events[0]["name"] == "request_retry"
    assert invocation_log.events[0]["reason"] == "upstream_server_error"


def test_responses_request_does_not_retry_bad_request(monkeypatch) -> None:
    class Response:
        status_code = 400
        text = "bad request"
        headers = {}

    calls = 0

    def post(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return Response()

    monkeypatch.setattr("models.llm.native.openai_responses.requests.post", post)

    with pytest.raises(UpstreamBadRequestError):
        _adapter()._open_response(
            model="gpt-5.6-sol",
            credentials={},
            request_body={"model": "gpt-5.6-sol"},
            stream=True,
        )

    assert calls == 1


def test_responses_non_stream_failed_server_response_retries_once(monkeypatch) -> None:
    class Response:
        def __init__(self, payload):
            self.payload = payload
            self.status_code = 200
            self.text = ""
            self.headers = {}
            self.closed = False

        def json(self):
            return self.payload

        def close(self):
            self.closed = True

    first = Response({"status": "failed", "error": {"code": "server_error"}})
    second = Response(
        {
            "status": "completed",
            "model": "gpt-5.6-sol",
            "output": [{"content": [{"type": "output_text", "text": "complete"}]}],
            "usage": {},
        }
    )
    monkeypatch.setattr(
        "models.llm.native.openai_responses.requests.post",
        lambda *_args, **_kwargs: second,
    )

    result = _adapter()._handle_response_with_retry(
        model="gpt-5.6-sol",
        credentials={},
        initial_response=first,
        request_body={"model": "gpt-5.6-sol", "stream": False},
        prompt_messages=[],
    )

    assert result.message.content == "complete"
    assert first.closed is True


def test_responses_stream_retries_once_when_no_chunk_was_yielded(monkeypatch) -> None:
    class Response:
        def __init__(self, lines):
            self.lines = lines
            self.headers = {}
            self.status_code = 200
            self.text = ""
            self.closed = False

        def iter_lines(self, decode_unicode=False):
            assert decode_unicode is False
            yield from self.lines

        def close(self):
            self.closed = True

    class InvocationLog:
        def __init__(self):
            self.response = {}
            self.events = []

        def set_response(self, **fields):
            self.response.update(fields)

        def event(self, name, **fields):
            self.events.append({"name": name, **fields})

    first = Response(
        [
            b'data: {"type":"response.created","response":{"id":"response-first"}}',
            b'data: {"error":{"message":"upstream closed"}}',
        ]
    )
    second = Response(
        [
            b'data: {"type":"response.created","response":{"id":"response-second"}}',
            b'data: {"type":"response.output_text.delta","delta":"complete"}',
            b'data: {"type":"response.completed","response":{"id":"response-second","usage":{}}}',
        ]
    )
    posted = []

    def post(*_args, **kwargs):
        posted.append(kwargs)
        return second

    monkeypatch.setattr("models.llm.native.openai_responses.requests.post", post)
    invocation_log = InvocationLog()
    request_body = {"model": "gpt-5.6-sol", "stream": True}

    chunks = list(
        _adapter()._handle_stream_with_retry(
            model="gpt-5.6-sol",
            credentials={},
            initial_response=first,
            request_body=request_body,
            prompt_messages=[],
            invocation_log=invocation_log,
        )
    )

    assert len(chunks) == 2
    assert first.closed is True
    assert len(posted) == 1
    assert posted[0]["json"] is request_body
    assert invocation_log.response["response_id"] == "response-second"
    assert invocation_log.events == [
        {
            "name": "stream_retry",
            "attempt": 1,
            "reason": (
                "OpenAI Responses 流式响应提前结束：未收到 response.completed"
                "（last_event=unknown, events={'response.created': 1, 'unknown': 1}）"
            ),
            "response_id": "response-first",
        }
    ]


def test_responses_stream_does_not_retry_after_yielding_text(monkeypatch) -> None:
    class Response:
        def iter_lines(self, decode_unicode=False):
            assert decode_unicode is False
            yield b'data: {"type":"response.created","response":{"id":"response-first"}}'
            yield b'data: {"type":"response.output_text.delta","delta":"partial"}'
            yield b'data: {"error":{"message":"upstream closed"}}'

    def unexpected_post(*_args, **_kwargs):
        raise AssertionError("stream with visible output must not be retried")

    monkeypatch.setattr("models.llm.native.openai_responses.requests.post", unexpected_post)

    with pytest.raises(InvokeError, match="未收到 response.completed"):
        list(
            _adapter()._handle_stream_with_retry(
                model="gpt-5.6-sol",
                credentials={},
                initial_response=Response(),
                request_body={"model": "gpt-5.6-sol", "stream": True},
                prompt_messages=[],
            )
        )
