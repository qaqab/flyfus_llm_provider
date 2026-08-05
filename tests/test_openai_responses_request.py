from types import SimpleNamespace

import pytest

from models.llm.native.openai_responses import OpenAIResponsesAdapter
from dify_plugin.entities.model.message import SystemPromptMessage
from dify_plugin.errors.model import InvokeError


def _adapter() -> OpenAIResponsesAdapter:
    return OpenAIResponsesAdapter(
        endpoint_url=lambda _credentials, _path: "https://example.test/",
        request_headers=lambda _credentials: {},
        normalize_model_parameters=lambda _model, parameters: parameters,
        calc_response_usage=lambda *_args: None,
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

    with pytest.raises(InvokeError, match="max_output_tokens"):
        list(
            _adapter()._handle_stream(
                model="gpt-5.6-sol",
                credentials={},
                response=Response(),
                prompt_messages=[],
            )
        )
