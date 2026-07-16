from types import SimpleNamespace

from dify_plugin.entities.model.message import SystemPromptMessage, TextPromptMessageContent, UserPromptMessage

from models.llm.usage_reporting import (
    extract_usage_context,
    normalize_usage,
    post_token_usage,
    report_token_usage,
)


def test_normalize_responses_usage() -> None:
    result = normalize_usage(
        "request-1",
        "gpt-5.5",
        {
            "input_tokens": 4389,
            "input_tokens_details": {"cache_write_tokens": 0, "cached_tokens": 3840},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 4394,
        },
    )

    assert result == {
        "request_id": "request-1",
        "model": "gpt-5.5",
        "input_tokens": 4389,
        "cached_tokens": 3840,
        "cache_write_tokens": 0,
        "output_tokens": 5,
        "reasoning_tokens": 0,
        "total_tokens": 4394,
    }


def test_normalize_chat_completions_usage() -> None:
    result = normalize_usage(
        "request-2",
        "deepseek-v4-pro",
        {
            "prompt_tokens": 7,
            "completion_tokens": 28,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 26},
            "prompt_cache_hit_tokens": 0,
            "total_tokens": 35,
        },
    )

    assert result["input_tokens"] == 7
    assert result["cached_tokens"] == 0
    assert result["output_tokens"] == 28
    assert result["reasoning_tokens"] == 26
    assert result["total_tokens"] == 35


def test_normalize_missing_optional_usage() -> None:
    result = normalize_usage(
        "request-3",
        "qwen3.6-plus",
        {"prompt_tokens": 15, "completion_tokens": 1},
    )

    assert result["cached_tokens"] is None
    assert result["cache_write_tokens"] is None
    assert result["reasoning_tokens"] is None
    assert result["total_tokens"] == 16


def test_normalize_sdk_usage_object() -> None:
    result = normalize_usage(
        "request-object",
        "gpt-5.5",
        SimpleNamespace(prompt_tokens=8, completion_tokens=3, total_tokens=11),
    )

    assert result["input_tokens"] == 8
    assert result["output_tokens"] == 3
    assert result["total_tokens"] == 11


def test_post_token_usage_uses_dedicated_configuration(monkeypatch) -> None:
    calls = []

    class Response:
        def raise_for_status(self) -> None:
            return None

    response = Response()
    monkeypatch.setattr(
        "models.llm.usage_reporting.requests.post",
        lambda *args, **kwargs: calls.append((args, kwargs)) or response,
    )
    payload = normalize_usage("request-4", "gpt-5.5", {})

    credentials = {
        "token_usage_url": "https://usage.example.com/token-usage",
        "token_usage_api_key": "usage-key",
    }

    result = post_token_usage(payload, credentials)

    assert result is None
    assert calls == []


def test_extract_usage_context_removes_tag_from_string_and_list_content() -> None:
    context = '{"usage_owner_type":"report_chat_session","usage_owner_id":"12345"}'
    messages = [
        SystemPromptMessage(content=f"system\n<FP_USAGE_CONTEXT>{context}</FP_USAGE_CONTEXT>"),
        UserPromptMessage(
            content=[TextPromptMessageContent(data="question <FP_USAGE_CONTEXT>invalid</FP_USAGE_CONTEXT>")]
        ),
    ]

    result = extract_usage_context(messages)

    assert result == {"usage_owner_type": "report_chat_session", "usage_owner_id": "12345"}
    assert messages[0].content == "system\n"
    assert messages[1].content[0].data == "question "


def test_report_token_usage_adds_owner_and_keeps_request_id(monkeypatch) -> None:
    payloads = []
    monkeypatch.setattr(
        "models.llm.usage_reporting.post_token_usage",
        lambda payload, credentials: payloads.append((payload, credentials)),
    )

    reported = report_token_usage(
        "invocation-123",
        "gpt-5.5",
        {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        {"usage_owner_type": "report_chat_session", "usage_owner_id": "42"},
        {"token_usage_url": "https://usage.example.com/token-usage", "token_usage_api_key": "usage-key"},
    )

    assert reported is True
    assert payloads == [
        (
            {
                "request_id": "invocation-123",
                "model": "gpt-5.5",
                "input_tokens": 10,
                "cached_tokens": None,
                "cache_write_tokens": None,
                "output_tokens": 2,
                "reasoning_tokens": None,
                "total_tokens": 12,
                "usage_owner_type": "report_chat_session",
                "usage_owner_id": "42",
            },
            {"token_usage_url": "https://usage.example.com/token-usage", "token_usage_api_key": "usage-key"},
        )
    ]
