from unittest.mock import Mock

import requests

from dify_plugin.entities.model.message import (
    ToolPromptMessage,
    UserPromptMessage,
)

from models.llm.ai_mode import apply_ai_mode
from models.llm.flyfus_settings import extract_flyfus_settings


def _credentials() -> dict:
    return {
        "geo_prompt_render_url": "https://geo.example/api/geo/v2",
        "geo_prompt_api_key": "test-key",
    }


def test_last_ai_mode_in_multiple_settings_routes_model_and_keeps_context(monkeypatch) -> None:
    response = Mock(status_code=200)
    response.json.return_value = {
        "code": 200,
        "message": "success",
        "data": {
            "agent_name": "listing_analysis",
            "mode_name": "fast",
            "config": {
                "model": "gpt-5.5",
                "parameters": {
                    "max_tokens": 65536,
                    "temperature": 0.7,
                    "reasoning_effort": "high",
                },
            },
        },
    }
    post = Mock(return_value=response)
    monkeypatch.setattr("models.llm.ai_mode.requests.post", post)
    messages = [
        UserPromptMessage(
            content=(
                "<FLYFUS_CONTEXT>image1: https://example.com/a.jpg</FLYFUS_CONTEXT>\n"
                '<FLYFUS_SETTING>{"type":"ai_mode","reference":'
                '"{{dify_admin:ai_mode.other.deep}}"}</FLYFUS_SETTING>'
            )
        ),
        ToolPromptMessage(
            tool_call_id="call-1",
            content=(
                '<FLYFUS_SETTING>{"type":"ai_mode","reference":'
                '"{{dify_admin:ai_mode.listing_analysis.fast}}"}</FLYFUS_SETTING>'
            ),
        ),
    ]

    settings = extract_flyfus_settings(messages)
    result = apply_ai_mode(
        "medium",
        {"temperature": 1},
        settings.ai_mode_reference,
        _credentials(),
    )

    assert result.applied is True
    assert settings.ai_mode_name == "fast"
    assert result.model == "gpt-5.5"
    assert result.parameters == {
        "max_tokens": 65536,
        "temperature": 0.7,
        "reasoning_effort": "high",
    }
    post.assert_called_once_with(
        "https://geo.example/api/geo/v2/dify_admin/ai_mode/resolve_reference",
        headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"},
        json={"reference": "{{dify_admin:ai_mode.listing_analysis.fast}}"},
        timeout=(10, 60),
    )
    assert "https://example.com/a.jpg" in messages[0].content
    assert "ai_mode" not in messages[0].content
    assert messages[1].content == ""


def test_ai_mode_request_error_falls_back_to_original_model(monkeypatch) -> None:
    monkeypatch.setattr(
        "models.llm.ai_mode.requests.post",
        Mock(side_effect=requests.ConnectTimeout("timed out")),
    )
    message = UserPromptMessage(
        content=(
            " \n分析这个 Listing\n\n"
            '<FLYFUS_SETTING>{"type":"ai_mode","reference":'
            '"{{dify_admin:ai_mode.listing_analysis.fast}}"}</FLYFUS_SETTING>\n '
        ),
    )

    settings = extract_flyfus_settings([message])
    result = apply_ai_mode(
        "medium",
        {"max_tokens": 1000},
        settings.ai_mode_reference,
        _credentials(),
    )

    assert result.applied is False
    assert result.model == "medium"
    assert result.parameters == {"max_tokens": 1000}
    assert message.content == "分析这个 Listing"


def test_ai_mode_rejects_non_object_response(monkeypatch) -> None:
    response = Mock(status_code=200)
    response.json.return_value = []
    monkeypatch.setattr("models.llm.ai_mode.requests.post", Mock(return_value=response))
    message = UserPromptMessage(
        content=(
            '<FLYFUS_SETTING>{"type":"ai_mode","reference":'
            '"{{dify_admin:ai_mode.listing_analysis.fast}}"}</FLYFUS_SETTING>'
        )
    )

    settings = extract_flyfus_settings([message])
    result = apply_ai_mode("medium", {}, settings.ai_mode_reference, _credentials())

    assert result.applied is False
    assert message.content == ""


def test_log_context_is_extracted_from_user_and_missing_fields_are_empty() -> None:
    message = UserPromptMessage(
        content=(
            "分析这个 Listing\n"
            '<FLYFUS_SETTING>{"type":"log_context","user_id":" user-1 ",'
            '"workflow_id":"workflow-1","workflow_run_id":"run-1",'
            '"conversation_id":"conversation-1"}</FLYFUS_SETTING>'
        )
    )

    settings = extract_flyfus_settings([message])

    assert settings.log_context() == {
        "user_id": "user-1",
        "app_id": "",
        "workflow_id": "workflow-1",
        "workflow_run_id": "run-1",
        "conversation_id": "conversation-1",
    }
    assert message.content == "分析这个 Listing"


def test_log_context_from_tool_is_removed_but_not_recorded() -> None:
    message = ToolPromptMessage(
        tool_call_id="call-1",
        content=(
            '<FLYFUS_SETTING>{"type":"log_context","user_id":"tool-user",'
            '"app_id":"tool-app"}</FLYFUS_SETTING>'
        ),
    )

    settings = extract_flyfus_settings([message])

    assert settings.log_context() == {
        "user_id": "",
        "app_id": "",
        "workflow_id": "",
        "workflow_run_id": "",
        "conversation_id": "",
    }
    assert message.content == ""


def test_log_context_from_user_history_is_not_reused() -> None:
    messages = [
        UserPromptMessage(
            content=(
                '<FLYFUS_SETTING>{"type":"log_context","user_id":"old-user",'
                '"workflow_run_id":"old-run"}</FLYFUS_SETTING>'
            )
        ),
        UserPromptMessage(content="new request"),
    ]

    settings = extract_flyfus_settings(messages)

    assert settings.log_context() == {
        "user_id": "",
        "app_id": "",
        "workflow_id": "",
        "workflow_run_id": "",
        "conversation_id": "",
    }
    assert messages[0].content == ""
    assert messages[1].content == "new request"


def test_non_json_setting_is_removed_without_legacy_ai_mode_parsing() -> None:
    message = UserPromptMessage(
        content=(
            "request\n"
            "<FLYFUS_SETTING>{{dify_admin:ai_mode.listing_analysis.fast}}</FLYFUS_SETTING>"
        )
    )

    settings = extract_flyfus_settings([message])

    assert settings.ai_mode_reference is None
    assert message.content == "request"
