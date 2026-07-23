from models.llm.native.gemini import GeminiNativeDocumentAdapter
from models.llm.agent_context import inject_context_from_tool_messages
from dify_plugin.entities.model.message import AssistantPromptMessage, ToolPromptMessage, UserPromptMessage


def _adapter() -> GeminiNativeDocumentAdapter:
    return GeminiNativeDocumentAdapter(
        endpoint_url=lambda _credentials, _path: "https://example.test/",
        normalize_model_parameters=lambda _model, parameters: parameters,
        calc_response_usage=lambda *_args: None,
    )


def test_native_gemini_uses_rest_google_search_tool() -> None:
    body = _adapter().build_body(
        model="gemini-3.5-flash",
        prompt_messages=[UserPromptMessage(content="Search the web")],
        model_parameters={"enable_web_search": True},
        tools=None,
        stop=None,
    )

    assert body["tools"] == [{"googleSearch": {}}]


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


def test_native_gemini_uses_public_url_from_read_file_context() -> None:
    image_url = "https://m.media-amazon.com/images/I/81TZvhKFX9L._AC_SL1500_.jpg"
    prompt_messages = [
        ToolPromptMessage(
            name="read_file",
            tool_call_id="read-file-1",
            content=(
                '<FLYFUS_CONTEXT>{"version":1,"type":"flyfus_context","urls":['
                f'"{image_url}"'
                "]}</FLYFUS_CONTEXT>"
            ),
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
        {"role": "user", "parts": [{"text": "First question"}]},
        {"role": "user", "parts": [{"text": "Second question"}]},
    ]
