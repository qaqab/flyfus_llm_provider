from unittest.mock import Mock

import requests

from dify_plugin.entities.model.message import (
    ToolPromptMessage,
    UserPromptMessage,
)

from models.llm.ai_mode import apply_ai_mode


def _credentials() -> dict:
    return {
        "geo_prompt_render_url": "https://geo.example/api/geo/v2",
        "geo_prompt_api_key": "test-key",
    }


def test_last_ai_mode_in_multiple_contexts_routes_model_and_keeps_urls(monkeypatch) -> None:
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
                "<FLYFUS_CONTEXT>{{dify_admin:ai_mode.other.deep}}</FLYFUS_CONTEXT>"
            )
        ),
        ToolPromptMessage(
            tool_call_id="call-1",
            content="<FLYFUS_CONTEXT>\n{{dify_admin:ai_mode.listing_analysis.fast}}\n</FLYFUS_CONTEXT>",
        ),
    ]

    result = apply_ai_mode("medium", {"temperature": 1}, messages, _credentials())

    assert result.applied is True
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
            "<FLYFUS_CONTEXT>{{dify_admin:ai_mode.listing_analysis.fast}}</FLYFUS_CONTEXT>\n "
        ),
    )

    result = apply_ai_mode("medium", {"max_tokens": 1000}, [message], _credentials())

    assert result.applied is False
    assert result.model == "medium"
    assert result.parameters == {"max_tokens": 1000}
    assert message.content == "分析这个 Listing"


def test_ai_mode_rejects_non_object_response(monkeypatch) -> None:
    response = Mock(status_code=200)
    response.json.return_value = []
    monkeypatch.setattr("models.llm.ai_mode.requests.post", Mock(return_value=response))
    message = UserPromptMessage(
        content="<FLYFUS_CONTEXT>{{dify_admin:ai_mode.listing_analysis.fast}}</FLYFUS_CONTEXT>"
    )

    result = apply_ai_mode("medium", {}, [message], _credentials())

    assert result.applied is False
    assert message.content == ""
