from models.llm.llm import FlyfusLargeLanguageModel


def test_request_id_headers_use_invocation_id() -> None:
    invocation_id = "invocation-123"
    model = object.__new__(FlyfusLargeLanguageModel)

    headers = model._request_headers({"_flyfus_invocation_id": invocation_id})

    assert headers["x-request-id"] == invocation_id
    assert headers["X-Client-Request-Id"] == invocation_id
    assert headers["X-Flyfus-Invocation-Id"] == invocation_id
