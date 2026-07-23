from unittest.mock import Mock

import pytest
import requests

from dify_plugin.errors.model import InvokeError
from models.llm.llm import FlyfusLargeLanguageModel


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
    monkeypatch.setattr("models.llm.llm.requests.post", post)
    monkeypatch.setattr("models.llm.llm.time.sleep", sleep)

    rendered_text = FlyfusLargeLanguageModel._render_geo_prompt_text(
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
    monkeypatch.setattr("models.llm.llm.requests.post", post)
    monkeypatch.setattr("models.llm.llm.time.sleep", sleep)

    with pytest.raises(InvokeError, match="已尝试 3 次"):
        FlyfusLargeLanguageModel._render_geo_prompt_text(
            "{{dify_admin:agent.prompt}}",
            {"geo_prompt_render_url": "https://geo.example", "geo_prompt_api_key": "test-key"},
        )

    assert post.call_count == 3
    assert sleep.call_args_list == [((10,), {}), ((10,), {})]
