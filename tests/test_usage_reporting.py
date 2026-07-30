import requests

from models.llm import usage_reporting


def test_report_token_usage_posts_all_context_fields_without_user_format_filter(monkeypatch) -> None:
    captured = {}

    def post_token_usage(payload, credentials):
        captured["payload"] = payload
        captured["credentials"] = credentials

    monkeypatch.setattr(usage_reporting, "post_token_usage", post_token_usage)

    reported = usage_reporting.report_token_usage(
        "request-1",
        "gpt-5.6-sol",
        {"input_tokens": 10, "output_tokens": 5},
        "550e8400-e29b-41d4-a716-446655440000",
        {
            "user_id": "123:web_chat:session-1",
            "app_id": "app-1",
            "workflow_id": "workflow-1",
            "workflow_run_id": "run-1",
            "conversation_id": "conversation-1",
        },
        {"geo_url": "https://geo.example"},
    )

    assert reported is True
    assert captured["payload"] == {
        "request_id": "request-1",
        "model": "gpt-5.6-sol",
        "input_tokens": 10,
        "cached_tokens": None,
        "cache_write_tokens": None,
        "output_tokens": 5,
        "reasoning_tokens": None,
        "total_tokens": 15,
        "user_id": "123:web_chat:session-1",
        "app_id": "app-1",
        "workflow_id": "workflow-1",
        "workflow_run_id": "run-1",
        "conversation_id": "conversation-1",
        "user": "550e8400-e29b-41d4-a716-446655440000",
    }


def test_report_token_usage_posts_empty_values(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(
        usage_reporting,
        "post_token_usage",
        lambda payload, credentials: captured.update(payload=payload),
    )

    reported = usage_reporting.report_token_usage(
        "request-1",
        "gpt-5.6-sol",
        None,
        None,
        {},
        {},
    )

    assert reported is True
    assert captured["payload"]["user_id"] == ""
    assert captured["payload"]["app_id"] == ""
    assert captured["payload"]["workflow_id"] == ""
    assert captured["payload"]["workflow_run_id"] == ""
    assert captured["payload"]["conversation_id"] == ""
    assert captured["payload"]["user"] == ""


def test_report_token_usage_failure_does_not_fail_model_call(monkeypatch) -> None:
    def fail_post_token_usage(payload, credentials):
        raise requests.ConnectTimeout()

    monkeypatch.setattr(usage_reporting, "post_token_usage", fail_post_token_usage)

    assert usage_reporting.report_token_usage("request-1", "gpt-5.6-sol", None, "user", {}, {}) is False
