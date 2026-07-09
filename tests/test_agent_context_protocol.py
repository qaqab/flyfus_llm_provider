import json

from dify_plugin.entities.model.message import (
    DocumentPromptMessageContent,
    PromptMessageContentType,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)

from models.llm.agent_context import inject_context_from_tool_messages
from models.llm.llm import FlypowerLargeLanguageModel
from models.llm.native.openai_responses import OpenAIResponsesAdapter
from workflow_tools.read_files import main as read_files_main


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


def test_read_files_keeps_public_urls_and_returns_reusable_index() -> None:
    result = read_files_main(
        json.dumps(
            {
                "images": [
                    {"url": "https://cdn.example.com/a.png", "detail": "high"},
                    {"url": "https://cdn.example.com/a.png", "detail": "high"},
                ],
                "files": [
                    {
                        "url": "https://cdn.example.com/report.xlsx",
                        "filename": "report.xlsx",
                        "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    }
                ],
            }
        )
    )

    output = result["output"]
    assert output.startswith("已保存 URL 文件上下文")
    assert output.endswith("</DIFY_CONTEXT>")
    assert "<DIFY_IMAGE_CONTEXT>" not in output
    assert "1. 图片: a.png" in output
    assert "2. 文件: report.xlsx" in output

    payload = _extract_context_from_output(output)
    assert payload == {
        "version": 1,
        "type": "dify_context",
        "images": [
            {
                "url": "https://cdn.example.com/a.png",
                "filename": "a.png",
                "mime_type": "image/png",
                "detail": "high",
            }
        ],
        "files": [
            {
                "url": "https://cdn.example.com/report.xlsx",
                "filename": "report.xlsx",
                "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            }
        ],
    }


def test_read_files_accepts_plain_url_and_url_list() -> None:
    plain_result = read_files_main("https://cdn.example.com/a.png")
    plain_payload = _extract_context_from_output(plain_result["output"])
    assert plain_payload["images"] == [
        {
            "url": "https://cdn.example.com/a.png",
            "filename": "a.png",
            "mime_type": "image/png",
            "detail": "high",
        }
    ]

    list_result = read_files_main(json.dumps(["https://cdn.example.com/report.xlsx"]))
    list_payload = _extract_context_from_output(list_result["output"])
    assert list_payload["files"] == [
        {
            "url": "https://cdn.example.com/report.xlsx",
            "filename": "report.xlsx",
            "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }
    ]


def test_read_files_ignores_local_files_and_internal_urls() -> None:
    result = read_files_main(
        json.dumps(
            {
                "images": [{"url": "http://localhost:5001/files/a.png"}],
                "files": [
                    {"path": "/tmp/report.xlsx", "filename": "report.xlsx"},
                    {"url": "files/upload/report.xlsx", "filename": "report.xlsx"},
                ],
            }
        )
    )
    output = result["output"]
    assert output.startswith("没有可用的 URL 文件上下文。")
    payload = _extract_context_from_output(output)
    assert payload == {"version": 1, "type": "dify_context", "images": [], "files": []}


def test_context_injection_adds_images_and_files_for_responses() -> None:
    tool_output = _context_output(
        {
            "images": [{"url": "https://cdn.example.com/a.png", "detail": "high"}],
            "files": [{"url": "https://cdn.example.com/report.xlsx", "filename": "report.xlsx"}],
        }
    )["output"]
    prompt_messages = [ToolPromptMessage(content=tool_output, tool_call_id="call_1", name="read_files")]

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
    prompt_messages = [ToolPromptMessage(content=tool_output, tool_call_id="call_1", name="read_files")]

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
    prompt_messages = [ToolPromptMessage(content=tool_output, tool_call_id="call_1", name="read_files")]

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
        classmethod(lambda cls, text: text.replace("{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}", "Rendered diagnosis skill")),
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

    FlypowerLargeLanguageModel._replace_tool_prompt_outputs(prompt_messages)

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
        classmethod(lambda cls, text: "Rendered skill"),
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

    FlypowerLargeLanguageModel._replace_tool_prompt_outputs(prompt_messages)

    assert prompt_messages[0].content == "Base system prompt"
    assert prompt_messages[1].content == "{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}"
    assert prompt_messages[2].content == '{"skill_prompt":"{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}"}'
    assert prompt_messages[3].content == '{"tool_prompt":"{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}","extra":"no"}'
    assert prompt_messages[4].content == "prefix {{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}"
    assert prompt_messages[5].content == "{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}} suffix"
    assert "extra" in prompt_messages[6].content


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
