import threading
from types import SimpleNamespace

import pytest

from dify_plugin.entities.model.llm import LLMUsage
from dify_plugin.entities.model.message import (
    ImagePromptMessageContent,
    UserPromptMessage,
)
from dify_plugin.errors.model import InvokeError

from models.llm.native.openai_responses import OpenAIResponsesAdapter


def _adapter() -> OpenAIResponsesAdapter:
    return OpenAIResponsesAdapter(
        endpoint_url=lambda _credentials, _path: "https://example.test/",
        request_headers=lambda _credentials: {},
        normalize_model_parameters=lambda _model, parameters: parameters,
        calc_response_usage=lambda *_args: LLMUsage.empty_usage(),
        create_final_chunk=lambda *_args, **_kwargs: SimpleNamespace(
            delta=SimpleNamespace(usage=None)
        ),
    )


def test_muse_mirrors_external_images_concurrently_and_preserves_order(
    monkeypatch,
) -> None:
    source_one = "https://m.media-amazon.com/images/I/one.jpg"
    source_two = "https://example.com/two.png?token=secret"
    trusted = "https://o1.flyfus.com/I/already.png"
    barrier = threading.Barrier(2, timeout=2)
    posted_urls = []

    class InvocationLog:
        def __init__(self):
            self.events = []
            self.lock = threading.Lock()

        def event(self, name, **fields):
            with self.lock:
                self.events.append({"name": name, **fields})

    class Response:
        status_code = 200
        text = ""
        headers = {"x-request-id": "mirror-request-id"}

        def __init__(self, public_url):
            self.public_url = public_url

        def json(self):
            return {
                "data": {
                    "public_url": self.public_url,
                    "file_size": 123,
                    "file_ext": self.public_url.rsplit(".", 1)[-1],
                }
            }

    def post(_url, **kwargs):
        image_url = kwargs["json"]["image_url"]
        posted_urls.append(image_url)
        barrier.wait()
        suffix = "one.jpg" if image_url == source_one else "two.png"
        return Response(f"https://o1.flyfus.com/I/{suffix}")

    monkeypatch.setattr("models.llm.native.openai_responses.requests.post", post)
    invocation_log = InvocationLog()
    cache = {}

    body = _adapter()._build_body(
        model="muse-spark-1.2-contributor",
        credentials={
            "oss_api_base_url": "https://upload.example.test",
            "oss_api_token": "secret",
        },
        prompt_messages=[
            UserPromptMessage(
                content=[
                    ImagePromptMessageContent(
                        format="url", url=source_one, mime_type="image/jpeg"
                    ),
                    ImagePromptMessageContent(
                        format="url", url=trusted, mime_type="image/png"
                    ),
                    ImagePromptMessageContent(
                        format="url", url=source_two, mime_type="image/png"
                    ),
                    ImagePromptMessageContent(
                        format="url", url=source_one, mime_type="image/jpeg"
                    ),
                ]
            )
        ],
        model_parameters={},
        tools=None,
        stop=None,
        stream=False,
        user=None,
        invocation_log=invocation_log,
        mirrored_image_cache=cache,
    )

    assert sorted(posted_urls) == sorted([source_one, source_two])
    assert [part["image_url"] for part in body["input"][0]["content"]] == [
        "https://o1.flyfus.com/I/one.jpg",
        trusted,
        "https://o1.flyfus.com/I/two.png",
        "https://o1.flyfus.com/I/one.jpg",
    ]
    assert cache == {
        source_one: "https://o1.flyfus.com/I/one.jpg",
        source_two: "https://o1.flyfus.com/I/two.png",
    }
    item_events = [
        event
        for event in invocation_log.events
        if event["name"] == "image_mirror_item_completed"
    ]
    assert len(item_events) == 2
    assert all("token=secret" not in event["source_url"] for event in item_events)
    assert {event["request_id"] for event in item_events} == {
        "mirror-request-id"
    }


def test_muse_reuses_mirrored_image_cache_when_body_is_rebuilt(monkeypatch) -> None:
    source_url = "https://m.media-amazon.com/images/I/one.jpg"
    adapter = _adapter()
    cache = {}
    upload_calls = 0

    class Response:
        status_code = 200
        text = ""
        headers = {}

        def json(self):
            return {
                "data": {
                    "public_url": "https://o1.flyfus.com/I/one.jpg",
                    "file_size": 123,
                    "file_ext": "jpg",
                }
            }

    def post(*_args, **_kwargs):
        nonlocal upload_calls
        upload_calls += 1
        return Response()

    monkeypatch.setattr("models.llm.native.openai_responses.requests.post", post)
    message = UserPromptMessage(
        content=[
            ImagePromptMessageContent(
                format="url", url=source_url, mime_type="image/jpeg"
            )
        ]
    )
    build_kwargs = {
        "model": "muse-spark-1.2-contributor",
        "credentials": {
            "oss_api_base_url": "https://upload.example.test",
            "oss_api_token": "secret",
        },
        "prompt_messages": [message],
        "model_parameters": {},
        "tools": None,
        "stop": None,
        "stream": False,
        "user": None,
        "mirrored_image_cache": cache,
    }

    first_body = adapter._build_body(**build_kwargs)
    second_body = adapter._build_body(**build_kwargs)

    assert upload_calls == 1
    assert first_body["input"] == second_body["input"]


def test_responses_invoke_reuses_prebuilt_request_body(monkeypatch) -> None:
    adapter = _adapter()
    request_body = {
        "model": "muse-spark-1.2-contributor",
        "input": [],
        "stream": False,
    }
    opened = {}

    monkeypatch.setattr(
        adapter,
        "_build_body",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("a prebuilt request body must not be built again")
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_open_response",
        lambda **kwargs: opened.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        adapter,
        "_handle_response_with_retry",
        lambda **kwargs: kwargs["request_body"],
    )

    result = adapter.invoke(
        model="muse-spark-1.2-contributor",
        credentials={},
        prompt_messages=[],
        model_parameters={},
        tools=None,
        stop=None,
        stream=False,
        user=None,
        request_body=request_body,
    )

    assert result is request_body
    assert opened["request_body"] is request_body


def test_non_muse_model_keeps_external_image_url(monkeypatch) -> None:
    source_url = "https://m.media-amazon.com/images/I/one.jpg"

    monkeypatch.setattr(
        "models.llm.native.openai_responses.requests.post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("non-Muse models must not mirror image URLs")
        ),
    )

    body = _adapter()._build_body(
        model="gpt-5.4",
        credentials={},
        prompt_messages=[
            UserPromptMessage(
                content=[
                    ImagePromptMessageContent(
                        format="url", url=source_url, mime_type="image/jpeg"
                    )
                ]
            )
        ],
        model_parameters={},
        tools=None,
        stop=None,
        stream=False,
        user=None,
    )

    assert body["input"][0]["content"][0]["image_url"] == source_url


def test_muse_stops_when_image_mirroring_fails(monkeypatch) -> None:
    events = []

    class InvocationLog:
        def event(self, name, **fields):
            events.append({"name": name, **fields})

    class Response:
        status_code = 502
        text = "upstream unavailable"
        headers = {"x-fc-request-id": "mirror-failed-request-id"}

    monkeypatch.setattr(
        "models.llm.native.openai_responses.requests.post",
        lambda *_args, **_kwargs: Response(),
    )

    with pytest.raises(InvokeError, match="图片转存失败，状态码：502"):
        _adapter()._build_body(
            model="muse-spark-1.2-contributor",
            credentials={
                "oss_api_base_url": "https://upload.example.test",
                "oss_api_token": "secret",
            },
            prompt_messages=[
                UserPromptMessage(
                    content=[
                        ImagePromptMessageContent(
                            format="url",
                            url="https://example.com/image.jpg",
                            mime_type="image/jpeg",
                        )
                    ]
                )
            ],
            model_parameters={},
            tools=None,
            stop=None,
            stream=False,
            user=None,
            invocation_log=InvocationLog(),
        )

    item_failed = next(
        event for event in events if event["name"] == "image_mirror_item_failed"
    )
    assert item_failed["image_index"] == 1
    assert item_failed["source_url"] == "https://example.com/image.jpg"
    assert item_failed["status_code"] == 502
    assert item_failed["request_id"] == "mirror-failed-request-id"
    assert item_failed["response_body_head"] == "upstream unavailable"
    assert events[-1]["name"] == "image_mirror_batch_failed"
