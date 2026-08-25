from types import SimpleNamespace

import pytest

from models.llm.errors import UpstreamBadRequestError, UpstreamResponseIncompleteError
from models.llm.native.base import model_family
from models.llm.native.openai_responses import OpenAIResponsesAdapter
from dify_plugin.entities.model.llm import LLMUsage
from dify_plugin.entities.model.message import (
    DocumentPromptMessageContent,
    ImagePromptMessageContent,
    SystemPromptMessage,
    UserPromptMessage,
)
from dify_plugin.errors.model import InvokeError
from models.llm.llm import FlyfusLargeLanguageModel


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


def test_ling_responses_uses_output_budget_and_explicit_none_effort() -> None:
    body = _adapter()._build_body(
        model="ling-3.0-flash",
        credentials={},
        prompt_messages=[SystemPromptMessage(content="You are helpful.")],
        model_parameters={
            "max_completion_tokens": 32768,
            "reasoning_effort": "none",
            "response_format": "json_object",
        },
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert model_family("ling-3.0-flash") == "openai_responses"
    assert body["max_output_tokens"] == 32768
    assert body["reasoning"] == {"effort": "none"}
    assert body["text"]["format"] == {"type": "json_object"}


@pytest.mark.parametrize("model", ["gpt-5.5", "gpt-5.6-sol", "gpt-5.6-terra"])
def test_codex_auth_models_drop_parameters_stripped_by_upstream(model: str) -> None:
    from models.llm.parameter_conversion import normalize_generation_parameters

    parameters = {
        "max_tokens": 128000,
        "max_completion_tokens": 128000,
        "temperature": 0.4,
        "top_p": 0.8,
        "presence_penalty": 1,
        "frequency_penalty": 1,
        "reasoning_effort": "max",
    }

    normalize_generation_parameters(model, parameters)

    assert parameters == {"reasoning_effort": "max"}


def test_muse_uses_responses_with_official_capabilities() -> None:
    assert model_family("muse-spark-1.2-contributor") == "openai_responses"

    body = _adapter()._build_body(
        model="muse-spark-1.2-contributor",
        credentials={},
        prompt_messages=[SystemPromptMessage(content="You are helpful.")],
        model_parameters={
            "max_completion_tokens": 128000,
            "reasoning_effort": "xhigh",
            "enable_web_search": True,
            "response_format": "json_schema",
            "json_schema": {
                "name": "probe",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"status": {"type": "string"}},
                    "required": ["status"],
                    "additionalProperties": False,
                },
            },
        },
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert body["max_output_tokens"] == 128000
    assert body["reasoning"] == {"effort": "xhigh"}
    assert body["tools"] == [{"type": "web_search"}]
    assert body["text"]["format"]["type"] == "json_schema"


def test_grok_46_accepts_public_image_url() -> None:
    image_url = "https://example.com/image.jpg"
    body = _adapter()._build_body(
        model="grok-4.6",
        credentials={},
        prompt_messages=[
            UserPromptMessage(
                content=[
                    ImagePromptMessageContent(
                        format="url",
                        url=image_url,
                        mime_type="image/jpeg",
                    )
                ]
            )
        ],
        model_parameters={},
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert body["input"] == [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "image_url": image_url}],
        }
    ]


@pytest.mark.parametrize(
    ("filename", "mime_type"),
    [("notes.txt", "text/plain"), ("data.json", "application/json")],
)
def test_muse_accepts_non_pdf_document_file_data(
    monkeypatch, filename: str, mime_type: str
) -> None:
    adapter = _adapter()
    monkeypatch.setattr(
        adapter,
        "_upload_file",
        lambda *_args: pytest.fail("Muse file_data must not use the Files API"),
    )

    body = adapter._build_body(
        model="muse-spark-1.2-contributor",
        credentials={},
        prompt_messages=[
            UserPromptMessage(
                content=[
                    DocumentPromptMessageContent(
                        format="base64",
                        base64_data="dGVzdA==",
                        mime_type=mime_type,
                        filename=filename,
                    )
                ]
            )
        ],
        model_parameters={},
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert body["input"][0]["content"] == [
        {
            "type": "input_file",
            "filename": filename,
            "file_data": f"data:{mime_type};base64,dGVzdA==",
        }
    ]


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
    request_ids = []

    def post(*_args, **kwargs):
        request_ids.append(kwargs["headers"]["x-request-id"])
        return next(responses)

    monkeypatch.setattr("models.llm.native.openai_responses.requests.post", post)
    invocation_log = InvocationLog()
    adapter = _adapter()
    model = object.__new__(FlyfusLargeLanguageModel)
    adapter._request_headers = model._request_headers

    result = adapter._open_response(
        model="gpt-5.6-sol",
        credentials={"_flyfus_log_id": "log-123"},
        request_body={"model": "gpt-5.6-sol"},
        stream=True,
        invocation_log=invocation_log,
    )

    assert result is second
    assert first.closed is True
    assert [event["name"] for event in invocation_log.events] == [
        "model_request_started",
        "request_retry",
        "model_request_started",
    ]
    assert invocation_log.events[1]["reason"] == "upstream_server_error"
    assert request_ids[0] != request_ids[1]
    assert [
        event["x_request_id"]
        for event in invocation_log.events
        if event["name"] == "model_request_started"
    ] == request_ids


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


def test_responses_request_retries_media_origin_fetch_failure_three_times(
    monkeypatch,
) -> None:
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

    error_text = (
        "failed to download media at input[40].content[9]: "
        "URL fetch failed before receiving an origin response"
    )
    failed_responses = [Response(400, error_text) for _ in range(3)]
    success = Response(200)
    responses = iter([*failed_responses, success])
    calls = 0

    def post(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr("models.llm.native.openai_responses.requests.post", post)
    invocation_log = InvocationLog()

    result = _adapter()._open_response(
        model="muse-spark-1.2-contributor",
        credentials={},
        request_body={"model": "muse-spark-1.2-contributor"},
        stream=True,
        invocation_log=invocation_log,
    )

    retries = [
        event for event in invocation_log.events if event["name"] == "request_retry"
    ]
    assert result is success
    assert calls == 4
    assert all(response.closed for response in failed_responses)
    assert [event["retry"] for event in retries] == [1, 2, 3]
    assert all(event["max_retries"] == 3 for event in retries)


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
    assert invocation_log.response.get("response_id") is None
    assert invocation_log.events == [
        {
            "name": "stream_retry",
            "attempt": 1,
            "reason": (
                "OpenAI Responses 流式响应提前结束：未收到 response.completed"
                "（last_event=unknown, events={'response.created': 1, 'unknown': 1}）"
            ),
        },
        {
            "name": "model_request_started",
            "adapter": "openai_responses",
            "x_request_id": None,
        },
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
