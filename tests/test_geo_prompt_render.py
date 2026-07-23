from unittest.mock import Mock

import pytest
import requests

from dify_plugin.entities.model.message import SystemPromptMessage
from dify_plugin.errors.model import InvokeError
from models.llm.geo_prompt import render_geo_prompt_references, render_geo_prompt_text


def test_geo_prompt_render_retries_network_errors(monkeypatch) -> None:
    response = Mock(status_code=200)
    response.json.return_value = {"data": {"rendered_text": "rendered prompt"}}
    post = Mock(
        side_effect=[
            requests.ConnectTimeout("first timeout"),
            requests.ConnectTimeout("second timeout"),
            response,
        ]
    )
    sleep = Mock()
    monkeypatch.setattr("models.llm.geo_prompt.requests.post", post)
    monkeypatch.setattr("models.llm.geo_prompt.time.sleep", sleep)

    rendered_text = render_geo_prompt_text(
        "{{dify_admin:agent.prompt}}",
        {
            "geo_prompt_render_url": "https://geo.example/api/geo/v2",
            "geo_prompt_api_key": "test-key",
        },
    )

    assert rendered_text == "rendered prompt"
    assert post.call_count == 3
    assert post.call_args.kwargs["timeout"] == (10, 60)
    assert sleep.call_args_list == [((10,), {}), ((10,), {})]


def test_geo_prompt_render_stops_after_three_network_errors(monkeypatch) -> None:
    post = Mock(side_effect=requests.ConnectTimeout("connection timed out"))
    sleep = Mock()
    monkeypatch.setattr("models.llm.geo_prompt.requests.post", post)
    monkeypatch.setattr("models.llm.geo_prompt.time.sleep", sleep)

    with pytest.raises(InvokeError, match="已尝试 3 次"):
        render_geo_prompt_text(
            "{{dify_admin:agent.prompt}}",
            {"geo_prompt_render_url": "https://geo.example", "geo_prompt_api_key": "test-key"},
        )

    assert post.call_count == 3
    assert sleep.call_args_list == [((10,), {}), ((10,), {})]


def test_geo_prompt_references_render_individually_and_keep_failures(monkeypatch) -> None:
    calls = []

    def post(*args, **kwargs):
        reference = kwargs["json"]["text"]
        calls.append(reference)
        if reference == "{{dify_admin:agent.failed}}":
            return Mock(status_code=503, text="Geo unavailable")
        response = Mock(status_code=200)
        response.json.return_value = {
            "data": {
                "rendered_text": {
                    "{{dify_admin:agent.first}}": "first prompt",
                    "{{dify_admin:agent.second}}": "second prompt",
                }[reference]
            }
        }
        return response

    monkeypatch.setattr("models.llm.geo_prompt.requests.post", post)
    prompt_message = SystemPromptMessage(
        content=(
            "before {{dify_admin:agent.first}} middle {{dify_admin:agent.failed}} "
            "after {{dify_admin:agent.first}} and {{dify_admin:agent.second}}"
        )
    )

    render_geo_prompt_references(
        [prompt_message],
        {"geo_prompt_render_url": "https://geo.example", "geo_prompt_api_key": "test-key"},
    )

    assert set(calls) == {
        "{{dify_admin:agent.first}}",
        "{{dify_admin:agent.failed}}",
        "{{dify_admin:agent.second}}",
    }
    assert len(calls) == 3
    assert prompt_message.content == (
        "before first prompt middle {{dify_admin:agent.failed}} "
        "after first prompt and second prompt"
    )
