from types import SimpleNamespace

from models.llm.usage_reporting import (
    format_usage_currency,
    normalize_usage,
    normalize_upstream_usage,
    post_token_usage,
    report_token_usage,
    to_geo_payload,
)


def test_normalized_upstream_usage_has_no_transport_fields() -> None:
    assert normalize_upstream_usage({"input_tokens": 3, "output_tokens": 2}) == {
        "input_tokens": 3,
        "cached_tokens": None,
        "cache_write_tokens": None,
        "output_tokens": 2,
        "reasoning_tokens": None,
        "total_tokens": 5,
    }
    assert to_geo_payload("provider-request-1", "gpt-5.5", {"input_tokens": 3, "output_tokens": 2})[
        "request_id"
    ] == "provider-request-1"


def test_format_usage_currency_contains_complete_token_breakdown() -> None:
    assert format_usage_currency(
        {
            "input_tokens": 1200,
            "input_tokens_details": {"cached_tokens": 300, "cache_write_tokens": 0},
            "output_tokens": 450,
            "output_tokens_details": {"reasoning_tokens": 100},
            "total_tokens": 1650,
        }
    ) == (
        "输入 Token: 1200；缓存命中 Token: 300；缓存写入 Token: 0；"
        "输出 Token: 450；推理 Token: 100；总 Token: 1650"
    )


def test_format_usage_currency_appends_the_sls_log_id() -> None:
    assert format_usage_currency({}, log_id="invocation-123").endswith(" | log_id=invocation-123")


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
        "geo_url": "https://geo.example.com/api/geo/v2/",
        "geo_key": "geo-key",
    }

    result = post_token_usage(payload, credentials)

    assert result is response
    assert calls == [
        (
            ("https://geo.example.com/api/geo/v2/dify_llm/token-usage",),
            {
                "headers": {
                    "Authorization": "Bearer geo-key",
                    "Content-Type": "application/json",
                },
                "json": payload,
                "timeout": (3, 10),
            },
        )
    ]


def test_report_token_usage_adds_dify_user_and_keeps_request_id(monkeypatch) -> None:
    payloads = []
    monkeypatch.setattr(
        "models.llm.usage_reporting.post_token_usage",
        lambda payload, credentials: payloads.append((payload, credentials)),
    )

    reported = report_token_usage(
        "invocation-123",
        "gpt-5.5",
        {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        "100000:report_chat:session_abc",
        {"geo_url": "https://geo.example.com/api/geo/v2", "geo_key": "geo-key"},
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
                "user": "100000:report_chat:session_abc",
            },
            {"geo_url": "https://geo.example.com/api/geo/v2", "geo_key": "geo-key"},
        )
    ]


def test_report_token_usage_skips_missing_dify_user(monkeypatch) -> None:
    monkeypatch.setattr(
        "models.llm.usage_reporting.post_token_usage",
        lambda payload, credentials: (_ for _ in ()).throw(AssertionError("must not post")),
    )

    assert not report_token_usage(
        "invocation-123",
        "gpt-5.5",
        {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        None,
        {"geo_url": "https://geo.example.com/api/geo/v2", "geo_key": "geo-key"},
    )


def test_report_token_usage_skips_users_outside_the_allowlist(monkeypatch) -> None:
    monkeypatch.setattr(
        "models.llm.usage_reporting.post_token_usage",
        lambda payload, credentials: (_ for _ in ()).throw(AssertionError("must not post")),
    )

    assert not report_token_usage(
        "invocation-123",
        "gpt-5.5",
        {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
        "100000:workflow:other-session",
        {"geo_url": "https://geo.example.com/api/geo/v2", "geo_key": "geo-key"},
    )
