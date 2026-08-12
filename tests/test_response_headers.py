from models.llm.llm import FlyfusLargeLanguageModel


def test_request_id_header_is_unique_per_http_request() -> None:
    invocation_id = "invocation-123"
    model = object.__new__(FlyfusLargeLanguageModel)

    first = model._request_headers({"_flyfus_invocation_id": invocation_id})
    second = model._request_headers({"_flyfus_invocation_id": invocation_id})

    assert first["x-request-id"] != second["x-request-id"]
    assert first["X-Client-Request-Id"] == invocation_id
    assert second["X-Client-Request-Id"] == invocation_id
    assert first["X-Flyfus-Invocation-Id"] == invocation_id
    assert second["X-Flyfus-Invocation-Id"] == invocation_id
