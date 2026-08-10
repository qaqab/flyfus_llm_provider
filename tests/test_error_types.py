import pytest
import requests

from models.llm.errors import (
    UpstreamAuthError,
    UpstreamBadRequestError,
    UpstreamConnectionError,
    UpstreamRateLimitedError,
    UpstreamServerError,
    UpstreamTimeoutError,
    http_error,
    request_error,
    response_failed_error,
    retry_delay_seconds,
)


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (400, UpstreamBadRequestError),
        (401, UpstreamAuthError),
        (403, UpstreamAuthError),
        (429, UpstreamRateLimitedError),
        (500, UpstreamServerError),
        (503, UpstreamServerError),
    ],
)
def test_http_errors_have_stable_types(status_code, error_type) -> None:
    assert isinstance(http_error("provider", status_code, "response body"), error_type)


def test_request_errors_distinguish_timeout_and_connection() -> None:
    timeout = request_error("provider", requests.ReadTimeout("timed out"))
    connection = request_error("provider", requests.ConnectionError("disconnected"))

    assert isinstance(timeout, UpstreamTimeoutError)
    assert isinstance(connection, UpstreamConnectionError)
    assert timeout.flyfus_internal_retryable is True
    assert connection.flyfus_internal_retryable is True


def test_failed_response_uses_specific_type_when_upstream_provides_reason() -> None:
    rate_limited = response_failed_error("provider", {"code": "rate_limit_exceeded"})
    invalid = response_failed_error("provider", {"code": "invalid_request_error"})

    assert isinstance(rate_limited, UpstreamRateLimitedError)
    assert isinstance(invalid, UpstreamBadRequestError)


def test_rate_limit_retry_delay_uses_header_with_safe_bounds() -> None:
    assert retry_delay_seconds(429, {"Retry-After": "3"}) == 3
    assert retry_delay_seconds(429, {"Retry-After": "60"}) == 10
    assert retry_delay_seconds(429, {}) == 1
    assert retry_delay_seconds(503, {"Retry-After": "3"}) == 0
