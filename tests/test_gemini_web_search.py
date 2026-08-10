import pytest
import requests

from models.llm.errors import (
    GeminiSafetyBlockedError,
    GeminiStreamIncompleteError,
    UpstreamTimeoutError,
)
from models.llm.native.gemini import DEFAULT_THOUGHT_SIGNATURE, GeminiNativeDocumentAdapter
from models.llm.agent_context import _extract_context_urls, inject_context_from_tool_messages
from dify_plugin.entities.model.llm import LLMUsage
from dify_plugin.entities.model.message import AssistantPromptMessage, ToolPromptMessage, UserPromptMessage
from dify_plugin.errors.model import InvokeError


def _adapter() -> GeminiNativeDocumentAdapter:
    return GeminiNativeDocumentAdapter(
        endpoint_url=lambda _credentials, _path: "https://example.test/",
        normalize_model_parameters=lambda _model, parameters: parameters,
        calc_response_usage=lambda *_args: LLMUsage.empty_usage(),
    )


class _InvocationLog:
    def __init__(self) -> None:
        self.response = {}
        self.events = []

    def set_response(self, **fields) -> None:
        self.response.update(fields)

    def set_replay_request(self, **fields) -> None:
        pass

    def event(self, name, **fields) -> None:
        self.events.append({"name": name, **fields})


def test_native_gemini_request_retries_server_error_once(monkeypatch) -> None:
    class Response:
        def __init__(self, status_code, payload=None, text=""):
            self.status_code = status_code
            self.payload = payload or {}
            self.text = text
            self.headers = {}
            self.closed = False

        def json(self):
            return self.payload

        def close(self):
            self.closed = True

    first = Response(503, text="temporarily unavailable")
    second = Response(
        200,
        payload={
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {"parts": [{"text": "complete"}]},
                }
            ],
            "usageMetadata": {},
        },
    )
    responses = iter([first, second])
    monkeypatch.setattr(
        "models.llm.native.gemini.requests.post",
        lambda *_args, **_kwargs: next(responses),
    )
    invocation_log = _InvocationLog()

    result = _adapter().invoke(
        model="gemini-3.6-flash",
        credentials={"api_key": "key"},
        prompt_messages=[UserPromptMessage(content="hello")],
        model_parameters={},
        tools=None,
        stop=None,
        stream=False,
        user=None,
        invocation_log=invocation_log,
    )

    assert result.message.content == "complete"
    assert first.closed is True
    assert invocation_log.events[0]["name"] == "request_retry"
    assert invocation_log.events[0]["reason"] == "upstream_server_error"


def test_native_gemini_records_non_stream_response_id() -> None:
    class Response:
        def json(self):
            return {
                "responseId": "gemini-response-1",
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": "complete"}]},
                    }
                ],
            }

    invocation_log = _InvocationLog()

    _adapter()._handle_response(
        model="gemini-3.6-flash",
        credentials={},
        response=Response(),
        invocation_log=invocation_log,
    )

    assert invocation_log.response["response_id"] == "gemini-response-1"


def test_native_gemini_records_stream_response_id() -> None:
    class Response:
        def iter_lines(self, decode_unicode=False):
            assert decode_unicode is False
            yield (
                b'data: {"responseId":"gemini-response-2","candidates":[{"finishReason":"STOP",'
                b'"content":{"parts":[{"text":"complete"}]}}]}'
            )

    invocation_log = _InvocationLog()

    list(
        _adapter()._handle_stream(
            model="gemini-3.6-flash",
            credentials={},
            response=Response(),
            invocation_log=invocation_log,
        )
    )

    assert invocation_log.response["response_id"] == "gemini-response-2"


def test_native_gemini_uses_rest_google_search_tool() -> None:
    body = _adapter().build_body(
        model="gemini-3.6-flash",
        prompt_messages=[UserPromptMessage(content="Search the web")],
        model_parameters={"enable_web_search": True},
        tools=None,
        stop=None,
    )

    assert body["tools"] == [{"googleSearch": {}}]


def test_native_gemini_empty_event_diagnostic_keeps_finish_and_safety_details() -> None:
    event = {
        "promptFeedback": {"blockReason": "SAFETY"},
        "usageMetadata": {"promptTokenCount": 123},
        "candidates": [
            {
                "finishReason": "MALFORMED_FUNCTION_CALL",
                "finishMessage": "invalid arguments",
                "safetyRatings": [{"category": "HARM_CATEGORY_DANGEROUS_CONTENT"}],
                "content": {"parts": []},
            }
        ],
    }

    diagnostic = GeminiNativeDocumentAdapter._stream_event_diagnostic(
        event,
        raw_event='{"candidates":[]}',
        sequence=3,
    )

    assert diagnostic["sequence"] == 3
    assert diagnostic["prompt_feedback"] == {"blockReason": "SAFETY"}
    assert diagnostic["usage_metadata"] == {"promptTokenCount": 123}
    assert diagnostic["candidates"][0]["finish_reason"] == "MALFORMED_FUNCTION_CALL"
    assert diagnostic["candidates"][0]["safety_ratings"] == [
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT"}
    ]


def test_native_gemini_empty_stream_raises_instead_of_returning_stop() -> None:
    class Response:
        def iter_lines(self, decode_unicode=False):
            assert decode_unicode is False
            yield b'data: {"candidates":[{"finishReason":"SAFETY"}]}'

    with pytest.raises(GeminiSafetyBlockedError, match="安全策略"):
        list(
            _adapter()._handle_stream(
                model="gemini-3.6-flash",
                credentials={},
                response=Response(),
            )
        )


def test_native_gemini_partial_stream_requires_finish_reason() -> None:
    class Response:
        def iter_lines(self, decode_unicode=False):
            assert decode_unicode is False
            yield b'data: {"candidates":[{"content":{"parts":[{"text":"partial"}]}}]}'

    with pytest.raises(GeminiStreamIncompleteError, match="未收到 finishReason"):
        list(
            _adapter()._handle_stream(
                model="gemini-3.6-flash",
                credentials={},
                response=Response(),
            )
        )


def test_native_gemini_prompt_feedback_safety_block_has_specific_type() -> None:
    class Response:
        def iter_lines(self, decode_unicode=False):
            assert decode_unicode is False
            yield b'data: {"promptFeedback":{"blockReason":"SAFETY"}}'

    with pytest.raises(GeminiSafetyBlockedError, match="SAFETY"):
        list(
            _adapter()._handle_stream(
                model="gemini-3.6-flash",
                credentials={},
                response=Response(),
            )
        )


def test_native_gemini_stream_read_timeout_has_specific_type() -> None:
    class Response:
        def iter_lines(self, decode_unicode=False):
            assert decode_unicode is False
            raise requests.ReadTimeout("timed out")
            yield

    with pytest.raises(UpstreamTimeoutError):
        list(
            _adapter()._handle_stream(
                model="gemini-3.6-flash",
                credentials={},
                response=Response(),
            )
        )


def test_native_gemini_stream_accepts_stop_finish_reason() -> None:
    class Response:
        def iter_lines(self, decode_unicode=False):
            assert decode_unicode is False
            yield (
                b'data: {"candidates":[{"finishReason":"STOP",'
                b'"content":{"parts":[{"text":"complete"}]}}]}'
            )

    chunks = list(
        _adapter()._handle_stream(
            model="gemini-3.6-flash",
            credentials={},
            response=Response(),
        )
    )

    assert chunks[-1].delta.finish_reason == "STOP"


def test_native_gemini_omits_search_when_disabled() -> None:
    body = _adapter().build_body(
        model="gemini-3.5-flash",
        prompt_messages=[UserPromptMessage(content="Do not search")],
        model_parameters={"enable_web_search": False},
        tools=None,
        stop=None,
    )

    assert "tools" not in body


def test_native_gemini_preserves_server_side_tools_with_function_calls() -> None:
    body = _adapter().build_body(
        model="gemini-3.5-flash",
        prompt_messages=[UserPromptMessage(content="Search the web")],
        model_parameters={"enable_web_search": True},
        tools=[
            {
                "function": {
                    "name": "lookup",
                    "description": "Looks up a value",
                    "parameters": {"type": "object", "properties": {}},
                }
            }
        ],
        stop=None,
    )

    assert body["toolConfig"] == {"includeServerSideToolInvocations": True}


def test_native_gemini_normalizes_agent_tool_schemas() -> None:
    body = _adapter().build_body(
        model="gemini-3.5-flash",
        prompt_messages=[UserPromptMessage(content="Retrieve a tool schema")],
        model_parameters={},
        tools=[
            {
                "function": {
                    "name": "batch_call",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tools": {
                                "type": ["null", "array"],
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "toolName": {"type": "string"},
                                        "params": True,
                                    },
                                    "additionalProperties": False,
                                },
                            }
                        },
                    },
                }
            }
        ],
        stop=None,
    )

    parameters = body["tools"][0]["functionDeclarations"][0]["parameters"]
    assert parameters == {
        "type": "object",
        "properties": {
            "tools": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "toolName": {"type": "string"},
                        "params": {"type": "object"},
                    },
                },
            }
        },
    }


def test_native_gemini_normalizes_false_schemas() -> None:
    body = _adapter().build_body(
        model="gemini-3.5-flash",
        prompt_messages=[UserPromptMessage(content="Use the supplied tool")],
        model_parameters={},
        tools=[
            {
                "function": {
                    "name": "probe",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "disabled": False,
                            "values": {"type": "array", "items": False},
                        },
                        "required": ["disabled", "values"],
                    },
                }
            }
        ],
        stop=None,
    )

    parameters = body["tools"][0]["functionDeclarations"][0]["parameters"]
    assert parameters == {
        "type": "object",
        "properties": {
            "values": {"type": "array", "maxItems": 0},
        },
        "required": ["values"],
    }


def test_native_gemini_uses_url_from_plain_text_context() -> None:
    image_url = "https://m.media-amazon.com/images/I/81TZvhKFX9L._AC_SL1500_.jpg"
    prompt_messages = [
        ToolPromptMessage(
            name="read_file",
            tool_call_id="read-file-1",
            content=f"<FLYFUS_CONTEXT>Reference image: {image_url}</FLYFUS_CONTEXT>",
        )
    ]
    inject_context_from_tool_messages(prompt_messages, include_files=False)

    body = _adapter().build_body(
        model="gemini-3.5-flash",
        prompt_messages=prompt_messages,
        model_parameters={},
        tools=None,
        stop=None,
    )

    assert body["contents"][-1]["parts"][-1] == {
        "fileData": {"mimeType": "image/jpeg", "fileUri": image_url}
    }


def test_native_gemini_uses_url_from_json_context() -> None:
    image_url = "https://example.com/a.jpg"
    prompt_messages = [
        ToolPromptMessage(
            name="read_file",
            tool_call_id="read-file-1",
            content=f'<FLYFUS_CONTEXT>{{"data":["{image_url}"]}}</FLYFUS_CONTEXT>',
        )
    ]

    inject_context_from_tool_messages(prompt_messages, include_files=False)

    body = _adapter().build_body(
        model="gemini-3.5-flash",
        prompt_messages=prompt_messages,
        model_parameters={},
        tools=None,
        stop=None,
    )

    assert body["contents"][-1]["parts"][-1] == {
        "fileData": {"mimeType": "image/jpeg", "fileUri": image_url}
    }


def test_context_regex_classifies_image_and_file_urls() -> None:
    image_url = "https://example.com/a.jpg?signature=abc"
    file_url = "https://example.com/report.pdf"

    context_urls = _extract_context_urls(
        f"<FLYFUS_CONTEXT>image: {image_url}\nfile: {file_url}</FLYFUS_CONTEXT>"
    )

    assert context_urls == [(image_url, "image"), (file_url, "file")]


def test_context_regex_ignores_url_without_hostname() -> None:
    assert _extract_context_urls("<FLYFUS_CONTEXT>http:///a.jpg</FLYFUS_CONTEXT>") == []


def test_context_and_file_tags_are_supported_and_urls_are_deduplicated() -> None:
    first_image_url = "https://example.com/a.jpg"
    second_image_url = "https://example.com/b.png"
    file_url = "https://example.com/report.pdf"
    prompt_messages = [
        ToolPromptMessage(
            name="read_file",
            tool_call_id="read-file-1",
            content=(
                f"<FLYFUS_CONTEXT>{first_image_url} {second_image_url}</FLYFUS_CONTEXT>"
                f"<FLYFUS_FILE>{file_url}</FLYFUS_FILE>"
                f"<FLYFUS_CONTEXT>{first_image_url}</FLYFUS_CONTEXT>"
            ),
        )
    ]

    inject_context_from_tool_messages(prompt_messages, include_files=True)

    assert len(prompt_messages) == 2
    assert len(prompt_messages[-1].content) == 4


def test_native_gemini_omits_empty_history_messages() -> None:
    body = _adapter().build_body(
        model="gemini-3.5-flash",
        prompt_messages=[
            UserPromptMessage(content="First question"),
            AssistantPromptMessage(content=""),
            UserPromptMessage(content="Second question"),
        ],
        model_parameters={},
        tools=None,
        stop=None,
    )

    assert body["contents"] == [
        {
            "role": "user",
            "parts": [{"text": "First question"}, {"text": "Second question"}],
        },
    ]


def test_native_gemini_round_trips_function_calls_and_responses() -> None:
    tool_call = AssistantPromptMessage.ToolCall(
        id="gemini-time-1",
        type="function",
        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
            name="current_time",
            arguments='{"reason":"answer the date"}',
        ),
    )
    body = _adapter().build_body(
        model="gemini-3.5-flash",
        prompt_messages=[
            UserPromptMessage(content="What date is it?"),
            AssistantPromptMessage(content="", tool_calls=[tool_call]),
            ToolPromptMessage(
                name="current_time",
                tool_call_id="gemini-time-1",
                content='{"current_time":"2026-07-23 08:00:00"}',
            ),
        ],
        model_parameters={},
        tools=None,
        stop=None,
    )

    assert body["contents"] == [
        {"role": "user", "parts": [{"text": "What date is it?"}]},
        {
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "name": "current_time",
                        "args": {"reason": "answer the date"},
                        "id": "gemini-time-1",
                    },
                    "thoughtSignature": DEFAULT_THOUGHT_SIGNATURE,
                }
            ],
        },
        {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": "current_time",
                        "response": {"response": '{"current_time":"2026-07-23 08:00:00"}'},
                        "id": "gemini-time-1",
                    }
                }
            ],
        },
    ]


def test_native_gemini_preserves_upstream_function_call_id() -> None:
    tool_calls = _adapter()._extract_tool_calls(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "id": "gemini-time-1",
                                    "name": "current_time",
                                    "args": {},
                                }
                            }
                        ]
                    }
                }
            ]
        }
    )

    assert tool_calls[0].id == "gemini-time-1"


def test_native_gemini_merges_consecutive_tool_responses() -> None:
    body = _adapter().build_body(
        model="gemini-3.5-flash",
        prompt_messages=[
            UserPromptMessage(content="Use both tools"),
            AssistantPromptMessage(
                content="",
                tool_calls=[
                    AssistantPromptMessage.ToolCall(
                        id="call-1",
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name="first_tool", arguments="{}"
                        ),
                    ),
                    AssistantPromptMessage.ToolCall(
                        id="call-2",
                        type="function",
                        function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                            name="second_tool", arguments="{}"
                        ),
                    ),
                ],
            ),
            ToolPromptMessage(name="first_tool", tool_call_id="call-1", content="first"),
            ToolPromptMessage(name="second_tool", tool_call_id="call-2", content="second"),
        ],
        model_parameters={},
        tools=None,
        stop=None,
    )

    assert len(body["contents"]) == 3
    assert len(body["contents"][-1]["parts"]) == 2


def test_native_gemini_rejects_invalid_historical_function_arguments() -> None:
    message = AssistantPromptMessage(
        content="",
        tool_calls=[
            AssistantPromptMessage.ToolCall(
                id="call-1",
                type="function",
                function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                    name="current_time", arguments="not-json"
                ),
            )
        ],
    )

    with pytest.raises(InvokeError, match="有效 JSON"):
        _adapter().build_body(
            model="gemini-3.5-flash",
            prompt_messages=[message],
            model_parameters={},
            tools=None,
            stop=None,
        )
