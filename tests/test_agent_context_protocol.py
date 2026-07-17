import json

from dify_plugin.entities.model.message import (
    DocumentPromptMessageContent,
    ImagePromptMessageContent,
    PromptMessageContentType,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)

from models.llm.agent_context import inject_context_from_tool_messages
from models.llm.llm import FlypowerLargeLanguageModel
from models.llm.native.gemini import GeminiNativeDocumentAdapter
from models.llm.native.openai_responses import OpenAIResponsesAdapter
from models.llm.parameter_conversion import build_web_search_tool, normalize_generation_parameters, normalize_max_tokens


class FakeResponse:
    def __init__(self, *, status_code: int = 200, content: bytes = b"", headers: dict | None = None, payload: dict | None = None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self._payload = payload or {}
        self.text = json.dumps(self._payload)

    def json(self) -> dict:
        return self._payload

    def close(self) -> None:
        pass


def test_context_injection_adds_images_and_files_for_responses() -> None:
    tool_output = _context_output(
        {
            "images": [{"url": "https://cdn.example.com/a.png", "detail": "high"}],
            "files": [{"url": "https://cdn.example.com/report.xlsx", "filename": "report.xlsx"}],
        }
    )["output"]
    prompt_messages = [ToolPromptMessage(content=tool_output, tool_call_id="call_1", name="read_file")]

    inject_context_from_tool_messages(prompt_messages, include_files=True)

    injected = prompt_messages[-1]
    assert isinstance(injected, UserPromptMessage)
    assert [part.type for part in injected.content] == [
        PromptMessageContentType.TEXT,
        PromptMessageContentType.IMAGE,
        PromptMessageContentType.DOCUMENT,
    ]
    assert injected.content[1].url == "https://cdn.example.com/a.png"
    assert injected.content[2].url == "https://cdn.example.com/report.xlsx"


def test_context_injection_handles_dify_json_wrapped_observation() -> None:
    observation = (
        '{"read_files":"{\\"result\\":\\"<DIFY_CONTEXT>{\\\\\\"version\\\\\\":1,'
        '\\\\\\"type\\\\\\":\\\\\\"dify_context\\\\\\",\\\\\\"images\\\\\\":[],'
        '\\\\\\"files\\\\\\":[{\\\\\\"url\\\\\\":\\\\\\"https://cdn.example.com/report.xlsx\\\\\\",'
        '\\\\\\"filename\\\\\\":\\\\\\"report.xlsx\\\\\\",'
        '\\\\\\"mime_type\\\\\\":\\\\\\"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet\\\\\\"}]}'
        '</DIFY_CONTEXT>\\"}"}'
    )
    prompt_messages = [ToolPromptMessage(content=observation, tool_call_id="call_1", name="read_files")]

    inject_context_from_tool_messages(prompt_messages, include_files=True)

    injected = prompt_messages[-1]
    assert isinstance(injected, UserPromptMessage)
    assert [part.type for part in injected.content] == [
        PromptMessageContentType.TEXT,
        PromptMessageContentType.DOCUMENT,
    ]
    assert injected.content[1].url == "https://cdn.example.com/report.xlsx"


def test_context_injection_reads_protocol_from_user_text_for_responses() -> None:
    user_text = (
        "请读取这些上下文。\n"
        + _context_output(
            {
                "images": [{"url": "https://cdn.example.com/a.png", "detail": "high"}],
                "files": [{"url": "https://cdn.example.com/report.xlsx", "filename": "report.xlsx"}],
            }
        )["output"]
    )
    prompt_messages = [UserPromptMessage(content=user_text)]

    inject_context_from_tool_messages(prompt_messages, include_files=True)

    injected = prompt_messages[-1]
    assert isinstance(injected, UserPromptMessage)
    assert [part.type for part in injected.content] == [
        PromptMessageContentType.TEXT,
        PromptMessageContentType.IMAGE,
        PromptMessageContentType.DOCUMENT,
    ]
    assert injected.content[1].url == "https://cdn.example.com/a.png"
    assert injected.content[2].url == "https://cdn.example.com/report.xlsx"


def test_context_injection_skips_files_for_chat_models() -> None:
    tool_output = _context_output(
        {
            "images": [{"url": "https://cdn.example.com/a.png"}],
            "files": [{"url": "https://cdn.example.com/report.xlsx"}],
        }
    )["output"]
    prompt_messages = [ToolPromptMessage(content=tool_output, tool_call_id="call_1", name="read_file")]

    inject_context_from_tool_messages(prompt_messages, include_files=False)

    injected = prompt_messages[-1]
    assert [part.type for part in injected.content] == [
        PromptMessageContentType.TEXT,
        PromptMessageContentType.IMAGE,
    ]


def test_context_injection_ignores_internal_and_data_file_urls() -> None:
    tool_output = _context_output(
        {
            "images": [{"url": "data:image/png;base64,AAAA"}],
            "files": [
                {"url": "files/upload/report.xlsx"},
                {"url": "http://localhost:5001/files/report.xlsx"},
                {"url": "data:application/pdf;base64,AAAA", "filename": "report.pdf"},
            ],
        }
    )["output"]
    prompt_messages = [ToolPromptMessage(content=tool_output, tool_call_id="call_1", name="read_file")]

    inject_context_from_tool_messages(prompt_messages, include_files=True)

    injected = prompt_messages[-1]
    assert [part.type for part in injected.content] == [
        PromptMessageContentType.TEXT,
        PromptMessageContentType.IMAGE,
    ]
    assert injected.content[1].url == "data:image/png;base64,AAAA"


def test_legacy_image_protocol_is_not_injected() -> None:
    prompt_messages = [
        ToolPromptMessage(
            content=(
                '<DIFY_IMAGE_CONTEXT>{"type":"dify_image_context",'
                '"images":[{"url":"https://cdn.example.com/a.png"}]}</DIFY_IMAGE_CONTEXT>'
            ),
            tool_call_id="call_1",
            name="read_files",
        )
    ]

    inject_context_from_tool_messages(prompt_messages, include_files=True)

    assert len(prompt_messages) == 1


def test_responses_adapter_sends_context_file_url_as_input_file() -> None:
    adapter = OpenAIResponsesAdapter(
        endpoint_url=lambda credentials, path: f"https://api.openai.com/v1/{path}",
        request_headers=lambda credentials: {"Authorization": "Bearer test"},
        normalize_model_parameters=lambda model, params: params,
        calc_response_usage=lambda *args: None,
        create_final_chunk=lambda **kwargs: None,
    )
    message = UserPromptMessage(
        content=[
            TextPromptMessageContent(data="read file"),
            DocumentPromptMessageContent(
                format="url",
                url="https://cdn.example.com/report.xlsx",
                filename="report.xlsx",
                mime_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ),
        ]
    )

    body = adapter._build_body(
        model="gpt-5.1",
        credentials={},
        prompt_messages=[message],
        model_parameters={},
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert body["input"][0]["content"] == [
        {"type": "input_text", "text": "read file"},
        {
            "type": "input_file",
            "file_url": "https://cdn.example.com/report.xlsx",
        },
    ]


def test_web_search_enabled_models_omit_attachments_and_enable_web_search() -> None:
    adapter = OpenAIResponsesAdapter(
        endpoint_url=lambda credentials, path: f"https://api.openai.com/v1/{path}",
        request_headers=lambda credentials: {"Authorization": "Bearer test"},
        normalize_model_parameters=lambda model, params: params,
        calc_response_usage=lambda *args: None,
        create_final_chunk=lambda **kwargs: None,
    )
    message = UserPromptMessage(
        content=[
            TextPromptMessageContent(data="search this"),
            ImagePromptMessageContent(
                format="url",
                url="https://cdn.example.com/image.png",
                mime_type="image/png",
            ),
            DocumentPromptMessageContent(
                format="url",
                url="https://cdn.example.com/report.pdf",
                filename="report.pdf",
                mime_type="application/pdf",
            ),
        ]
    )

    body = adapter._build_body(
        model="grok-4.5",
        credentials={},
        prompt_messages=[message],
        model_parameters={"enable_web_search": True},
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert body["input"][0]["content"] == [{"type": "input_text", "text": "search this"}]
    assert body["tools"] == [{"type": "web_search"}]

    gpt_body = adapter._build_body(
        model="gpt-5.4",
        credentials={},
        prompt_messages=[message],
        model_parameters={"enable_web_search": True},
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )
    assert gpt_body["tools"] == [{"type": "web_search"}]

    disabled_body = adapter._build_body(
        model="gpt-5.4",
        credentials={},
        prompt_messages=[message],
        model_parameters={"enable_web_search": False},
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )
    assert "tools" not in disabled_body

    unsupported_body = adapter._build_body(
        model="low",
        credentials={},
        prompt_messages=[message],
        model_parameters={"enable_web_search": True},
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )
    assert "tools" not in unsupported_body
    assert build_web_search_tool("high", {"enable_web_search": True}) is None
    assert build_web_search_tool("gpt-5.4", {}) is None
    assert build_web_search_tool("gpt-5.4", {"enable_web_search": True}) == {"type": "web_search"}
    assert build_web_search_tool("gemini-3-flash-preview", {"enable_web_search": True}) == {"google_search": {}}


def test_gemini_web_search_uses_native_google_search_tool() -> None:
    adapter = GeminiNativeDocumentAdapter(
        endpoint_url=lambda credentials, path: f"https://example.com/{path}",
        normalize_model_parameters=lambda model, params: params,
        calc_response_usage=lambda *args: None,
    )
    message = UserPromptMessage(content="search current news")

    enabled_body = adapter.build_body(
        model="gemini-3-flash-preview",
        prompt_messages=[message],
        model_parameters={"enable_web_search": True},
        tools=None,
        stop=None,
    )
    disabled_body = adapter.build_body(
        model="gemini-3-flash-preview",
        prompt_messages=[message],
        model_parameters={"enable_web_search": False},
        tools=None,
        stop=None,
    )

    assert enabled_body["tools"] == [{"google_search": {}}]
    assert "tools" not in disabled_body


def test_generation_parameters_are_normalized_at_the_shared_parameter_boundary() -> None:
    parameters = {"temperature": 0.2}
    normalize_generation_parameters("gpt-5.5", parameters)
    assert parameters["temperature"] == 1

    non_gpt_parameters = {"temperature": 0.2, "top_p": 0.8, "response_format": "json_object"}
    normalize_generation_parameters("grok-4.5", non_gpt_parameters)
    assert non_gpt_parameters["temperature"] == 0.2
    assert non_gpt_parameters["top_p"] == 0.8
    assert non_gpt_parameters["response_format"] == "json_object"

    invalid_format_parameters = {"response_format": "xml"}
    normalize_generation_parameters("grok-4.5", invalid_format_parameters)
    assert "response_format" not in invalid_format_parameters

    token_parameters = {"max_tokens": 8192}
    normalize_max_tokens(token_parameters, "max_completion_tokens")
    assert token_parameters == {"max_completion_tokens": 8192}


def test_user_text_protocol_reaches_responses_body_as_input_image() -> None:
    adapter = OpenAIResponsesAdapter(
        endpoint_url=lambda credentials, path: f"https://api.openai.com/v1/{path}",
        request_headers=lambda credentials: {"Authorization": "Bearer test"},
        normalize_model_parameters=lambda model, params: params,
        calc_response_usage=lambda *args: None,
        create_final_chunk=lambda **kwargs: None,
    )
    prompt_messages = [
        UserPromptMessage(
            content=(
                "inspect image "
                '<DIFY_CONTEXT>{"version":1,"type":"dify_context",'
                '"images":[{"url":"data:image/png;base64,AAAA","detail":"high"}],'
                '"files":[]}</DIFY_CONTEXT>'
            )
        )
    ]
    inject_context_from_tool_messages(prompt_messages, include_files=True)

    body = adapter._build_body(
        model="gpt-5.1",
        credentials={},
        prompt_messages=prompt_messages,
        model_parameters={},
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert body["input"][-1]["content"] == [
        {
            "type": "input_text",
            "text": "External context refreshed by Dify tool output. Use the attached image(s) when answering.",
        },
        {
            "type": "input_image",
            "image_url": "data:image/png;base64,AAAA",
            "detail": "high",
        },
    ]


def test_tool_prompt_output_is_replaced_when_exact_protocol(monkeypatch) -> None:
    monkeypatch.setattr(
        FlypowerLargeLanguageModel,
        "_render_geo_prompt_text",
        classmethod(lambda cls, text, credentials: text.replace("{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}", "Rendered diagnosis skill")),
    )
    prompt_messages = [
        SystemPromptMessage(content="Base system prompt"),
        ToolPromptMessage(
            content='{"listing_diagnosis_tool_prompt":"{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}"}',
            tool_call_id="call_1",
            name="flyfus_skills",
        ),
        ToolPromptMessage(
            content=json.dumps(
                {
                    "flyfus_skills": json.dumps(
                        {
                            "listing_diagnosis_tool_prompt": (
                                "{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}"
                            )
                        }
                    )
                }
            ),
            tool_call_id="call_2",
            name="flyfus_skills",
        ),
        ToolPromptMessage(
            content=json.dumps(
                {
                    "flyfus_skills": json.dumps(
                        {
                            "listing_optimization_tool_prompt": (
                                "{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}"
                            )
                        }
                    )
                }
            ),
            tool_call_id="call_3",
            name="flyfus_skills",
        ),
    ]

    FlypowerLargeLanguageModel._replace_tool_prompt_outputs(prompt_messages, {})

    assert len(prompt_messages) == 4
    assert isinstance(prompt_messages[0], SystemPromptMessage)
    assert prompt_messages[0].content == "Base system prompt"
    assert prompt_messages[1].content == "Rendered diagnosis skill"
    assert prompt_messages[2].content == "Rendered diagnosis skill"
    assert prompt_messages[3].content == "Rendered diagnosis skill"


def test_tool_prompt_requires_exact_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        FlypowerLargeLanguageModel,
        "_render_geo_prompt_text",
        classmethod(lambda cls, text, credentials: "Rendered skill"),
    )
    prompt_messages = [
        SystemPromptMessage(content="Base system prompt"),
        ToolPromptMessage(
            content="{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}",
            tool_call_id="call_0",
            name="flyfus_skills",
        ),
        ToolPromptMessage(
            content='{"skill_prompt":"{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}"}',
            tool_call_id="call_1",
            name="flyfus_skills",
        ),
        ToolPromptMessage(
            content='{"tool_prompt":"{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}","extra":"no"}',
            tool_call_id="call_2",
            name="flyfus_skills",
        ),
        ToolPromptMessage(
            content="prefix {{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}",
            tool_call_id="call_3",
            name="flyfus_skills",
        ),
        ToolPromptMessage(
            content="{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}} suffix",
            tool_call_id="call_4",
            name="flyfus_skills",
        ),
        ToolPromptMessage(
            content=json.dumps(
                {
                    "flyfus_skills": json.dumps(
                        {
                            "listing_diagnosis_tool_prompt": (
                                "{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}"
                            ),
                            "extra": "no",
                        }
                    )
                }
            ),
            tool_call_id="call_5",
            name="flyfus_skills",
        ),
    ]

    FlypowerLargeLanguageModel._replace_tool_prompt_outputs(prompt_messages, {})

    assert prompt_messages[0].content == "Base system prompt"
    assert prompt_messages[1].content == "{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}"
    assert prompt_messages[2].content == '{"skill_prompt":"{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}"}'
    assert prompt_messages[3].content == '{"tool_prompt":"{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}","extra":"no"}'
    assert prompt_messages[4].content == "prefix {{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}"
    assert prompt_messages[5].content == "{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}} suffix"
    assert "extra" in prompt_messages[6].content


def test_reasoning_effort_is_read_only_from_set_next_step_tool() -> None:
    prompt_messages = [
        ToolPromptMessage(
            content='{"reasoning_effort":"high","next_objective":"Investigate the failure."}',
            tool_call_id="call_1",
            name="set_next_step",
        ),
        ToolPromptMessage(
            content='{"reasoning_effort":"xhigh"}',
            tool_call_id="call_2",
            name="other_tool",
        ),
    ]

    effort = FlypowerLargeLanguageModel._reasoning_effort_from_tool_messages(prompt_messages)

    assert effort == "high"
    assert "next_objective" in prompt_messages[0].content


def test_reasoning_effort_supports_dify_wrapped_workflow_output() -> None:
    output = '{"reasoning_effort":"xhigh","next_objective":"Prepare the final answer."}'
    prompt_messages = [
        ToolPromptMessage(
            content=json.dumps({"set_next_step": json.dumps({"result": output})}),
            tool_call_id="call_1",
            name="set_next_step",
        )
    ]

    assert FlypowerLargeLanguageModel._reasoning_effort_from_tool_messages(prompt_messages) == "xhigh"


def test_reasoning_effort_rejects_invalid_tool_output() -> None:
    prompt_messages = [
        ToolPromptMessage(
            content='{"reasoning_effort":"maximum"}',
            tool_call_id="call_1",
            name="set_next_step",
        )
    ]

    assert FlypowerLargeLanguageModel._reasoning_effort_from_tool_messages(prompt_messages) is None


def test_gemini_thought_parts_use_dify_think_tags() -> None:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"thought": True, "text": "reasoning"},
                        {"text": "answer"},
                    ]
                }
            }
        ]
    }

    assert GeminiNativeDocumentAdapter._extract_text(payload) == "<think>\nreasoning</think>\nanswer"


def test_gemini_function_calls_are_converted_for_dify() -> None:
    payload = {
        "candidates": [
            {"content": {"parts": [{"functionCall": {"name": "lookup_code", "args": {"id": 7}}}]}}
        ]
    }

    tool_calls = GeminiNativeDocumentAdapter._extract_tool_calls(payload)

    assert len(tool_calls) == 1
    assert tool_calls[0].function.name == "lookup_code"
    assert tool_calls[0].function.arguments == '{"id": 7}'


def _context_output(payload: dict) -> dict:
    context = {
        "version": 1,
        "type": "dify_context",
        "images": payload.get("images", []),
        "files": payload.get("files", []),
    }
    return {
        "output": "<DIFY_CONTEXT>"
        + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
        + "</DIFY_CONTEXT>"
    }


def _extract_context_from_output(output: str) -> dict:
    payload = output.split("<DIFY_CONTEXT>", 1)[1].split("</DIFY_CONTEXT>", 1)[0]
    return json.loads(payload)
