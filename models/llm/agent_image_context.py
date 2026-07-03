import json
import mimetypes
import os
import re
from contextlib import suppress
from typing import Optional
from urllib.parse import urlparse, urlunparse

from dify_plugin.entities.model.message import (
    ImagePromptMessageContent,
    PromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)

_IMAGE_CONTEXT_PATTERN = re.compile(
    r"<DIFY_IMAGE_CONTEXT>(.*?)</DIFY_IMAGE_CONTEXT>",
    re.DOTALL,
)
_IMAGE_REF_URL_PATTERN = re.compile(r"https?://[^\s,\]\)\"'<>]+")


def inject_image_context_from_tool_messages(prompt_messages: list[PromptMessage]) -> None:
    """把 image_context_refresher 工具返回的 URL 协议注入为当前轮多模态图片输入。"""
    image_parts: list[ImagePromptMessageContent] = []
    seen_refs: set[str] = set()

    for prompt_message in prompt_messages:
        if not isinstance(prompt_message, ToolPromptMessage):
            continue
        if prompt_message.name and prompt_message.name != "image_context_refresher":
            continue
        if not isinstance(prompt_message.content, str):
            continue

        for payload in _extract_image_context_payloads(prompt_message.content):
            for image_context in _iter_image_context_items(payload):
                for ref in _expand_image_refs_from_context(image_context):
                    if ref in seen_refs:
                        continue
                    seen_refs.add(ref)
                    image_parts.append(
                        _image_ref_to_prompt_content(
                            image_ref=ref,
                            detail=_optional_string(image_context.get("detail")),
                        )
                    )

    if not image_parts:
        return

    prompt_messages.append(
        UserPromptMessage(
            content=[
                TextPromptMessageContent(
                    data="Image context refreshed by image_context_refresher. Analyze the attached image(s) when answering."
                ),
                *image_parts,
            ]
        )
    )


def _extract_image_context_payloads(text: str) -> list[dict]:
    payloads: list[dict] = []
    for match in _IMAGE_CONTEXT_PATTERN.finditer(text):
        raw_payload = match.group(1).strip()
        decoded_payloads = [raw_payload]
        with suppress(ValueError):
            decoded_payloads.append(json.loads(f'"{raw_payload}"'))

        for payload_text in decoded_payloads:
            try:
                payload = json.loads(payload_text)
            except ValueError:
                continue
            if isinstance(payload, dict) and payload.get("type") == "dify_image_context":
                payloads.append(payload)
                break
    return payloads


def _iter_image_context_items(payload: dict) -> list[dict]:
    images = payload.get("images")
    if isinstance(images, list):
        return [item for item in images if isinstance(item, dict)]
    image_refs = payload.get("image_refs")
    if image_refs is not None:
        return [{"image_refs": image_refs, "detail": payload.get("detail")}]
    image_ref = payload.get("image_ref") or payload.get("url")
    if image_ref:
        return [{"image_refs": image_ref, "detail": payload.get("detail")}]
    return []


def _expand_image_refs_from_context(image_context: dict) -> list[str]:
    refs: list[str] = []
    for key in ("url", "image_ref", "image_refs"):
        refs.extend(_expand_image_refs(image_context.get(key)))
    return refs


def _expand_image_refs(image_refs: object) -> list[str]:
    if image_refs is None:
        return []
    if isinstance(image_refs, list):
        refs: list[str] = []
        for item in image_refs:
            refs.extend(_expand_image_refs(item))
        return refs
    if isinstance(image_refs, dict):
        return _expand_image_refs(
            image_refs.get("url") or image_refs.get("image_ref") or image_refs.get("image_refs")
        )
    if not isinstance(image_refs, str):
        return []

    value = image_refs.strip()
    if not value:
        return []
    if value.startswith(("[", "{")):
        with suppress(ValueError):
            decoded = json.loads(value)
            if isinstance(decoded, (list, dict)):
                return _expand_image_refs(decoded)

    extracted_urls = _IMAGE_REF_URL_PATTERN.findall(value)
    if extracted_urls:
        return extracted_urls

    refs: list[str] = []
    for item in re.split(r"[\n,]+", value):
        item = item.strip().rstrip(",")
        if _is_supported_image_ref(item):
            refs.append(item)
    return refs


def _image_ref_to_prompt_content(
    *,
    image_ref: str,
    detail: Optional[str],
) -> ImagePromptMessageContent:
    resolved_url = _resolve_image_ref_url(image_ref)
    return ImagePromptMessageContent(
        format="url",
        url=resolved_url,
        mime_type=_guess_image_mime_type(resolved_url),
        detail=_image_detail(detail),
    )


def _resolve_image_ref_url(image_ref: str) -> str:
    value = image_ref.strip()
    if value.startswith("data:"):
        return value

    internal_base = _dify_internal_api_base().rstrip("/")
    if value.startswith("/"):
        return internal_base + value
    if value.startswith("files/"):
        return internal_base + "/" + value

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"} and parsed.hostname in {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "web",
        "nginx",
    }:
        internal = urlparse(internal_base)
        return urlunparse(
            (
                internal.scheme,
                internal.netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )
    return value


def _dify_internal_api_base() -> str:
    return os.getenv("PLUGIN_DIFY_INNER_API_URL") or os.getenv("DIFY_INNER_API_URL") or "http://api:5001"


def _is_supported_image_ref(value: str) -> bool:
    if not value:
        return False
    return value.startswith(("http://", "https://", "data:", "/", "files/"))


def _image_detail(detail: Optional[str]) -> ImagePromptMessageContent.DETAIL:
    if isinstance(detail, str) and detail.lower() == "high":
        return ImagePromptMessageContent.DETAIL.HIGH
    return ImagePromptMessageContent.DETAIL.LOW


def _guess_image_mime_type(url: str) -> str:
    if url.startswith("data:"):
        match = re.match(r"^data:([^;,]+)[;,]", url)
        if match and match.group(1).startswith("image/"):
            return match.group(1)
    guessed_type, _ = mimetypes.guess_type(urlparse(url).path)
    return guessed_type if guessed_type and guessed_type.startswith("image/") else "image/png"


def _optional_string(value: object) -> Optional[str]:
    return value if isinstance(value, str) and value else None
