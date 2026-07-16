import json
import mimetypes
import re
from contextlib import suppress
from typing import Optional
from urllib.parse import urlparse

from dify_plugin.entities.model.message import (
    DocumentPromptMessageContent,
    ImagePromptMessageContent,
    PromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
)

_CONTEXT_PATTERN = re.compile(r"<FLYPOWER_CONTEXT>(.*?)</FLYPOWER_CONTEXT>", re.DOTALL)


def inject_context_from_tool_messages(
    prompt_messages: list[PromptMessage],
    *,
    include_files: bool,
) -> None:
    """Inject URL context found in tool outputs or user text into the current model call."""

    parts: list[object] = []
    image_count = 0
    file_count = 0
    seen_images: set[str] = set()
    seen_files: set[str] = set()

    for prompt_message in prompt_messages:
        for payload in _extract_context_payloads_from_message(prompt_message):
            for image_context in _context_items(payload, "images"):
                url = _image_url(image_context)
                if not url or url in seen_images:
                    continue
                seen_images.add(url)
                image_count += 1
                parts.append(_image_url_to_prompt_content(url, _optional_string(image_context.get("detail"))))

            if not include_files:
                continue

            for file_context in _context_items(payload, "files"):
                url = _file_url(file_context)
                if not url or url in seen_files:
                    continue
                seen_files.add(url)
                file_count += 1
                parts.append(_file_url_to_prompt_content(url, file_context))

    if not parts:
        return

    prompt_messages.append(
        UserPromptMessage(
            content=[
                TextPromptMessageContent(
                    data=_context_instruction(image_count=image_count, file_count=file_count)
                ),
                *parts,
            ]
        )
    )


def _extract_context_payloads_from_message(prompt_message: PromptMessage) -> list[dict]:
    if isinstance(prompt_message, ToolPromptMessage) and isinstance(prompt_message.content, str):
        return _extract_context_payloads(prompt_message.content)

    if isinstance(prompt_message, UserPromptMessage):
        payloads: list[dict] = []
        for text in _user_message_texts(prompt_message):
            payloads.extend(_extract_context_payloads(text))
        return payloads

    return []


def _user_message_texts(prompt_message: UserPromptMessage) -> list[str]:
    if isinstance(prompt_message.content, str):
        return [prompt_message.content]

    if not isinstance(prompt_message.content, list):
        return []

    texts: list[str] = []
    for part in prompt_message.content:
        if isinstance(part, TextPromptMessageContent):
            texts.append(part.data)
    return texts


def _extract_context_payloads(text: str) -> list[dict]:
    payloads: list[dict] = []
    for match in _CONTEXT_PATTERN.finditer(text):
        raw_payload = match.group(1).strip()
        for payload_text in _payload_text_candidates(raw_payload):
            try:
                payload = json.loads(payload_text)
            except ValueError:
                continue
            normalized_payload = _normalize_context_payload(payload)
            if normalized_payload is not None:
                payloads.append(normalized_payload)
                break
    return payloads


def _payload_text_candidates(raw_payload: str) -> list[str]:
    candidates = [raw_payload]
    current = raw_payload
    for _ in range(3):
        with suppress(ValueError):
            decoded = json.loads(f'"{current}"')
            if isinstance(decoded, str) and decoded not in candidates:
                candidates.append(decoded)
                current = decoded
                continue
        break
    return candidates


def _context_items(payload: dict, key: str) -> list[dict]:
    items = payload.get(key)
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _normalize_context_payload(payload: object) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None

    if payload.get("type") != "flypower_context":
        return None

    images: list[dict] = []
    files: list[dict] = []
    seen_urls: set[str] = set()
    raw_urls = payload.get("urls")
    if not isinstance(raw_urls, list):
        return {"images": images, "files": files}

    for raw_url in raw_urls:
        url = _optional_string(raw_url)
        if not url or url in seen_urls or not _is_public_url(url, allow_data=False):
            continue
        seen_urls.add(url)
        mime_type = _guess_mime_type(url, default="application/octet-stream")
        if mime_type.startswith("image/"):
            images.append({"url": url, "mime_type": mime_type, "detail": "high"})
        else:
            files.append(
                {
                    "url": url,
                    "mime_type": mime_type,
                    "filename": _filename_from_url(url) or "document",
                }
            )

    return {"images": images, "files": files}


def _image_url(item: dict) -> Optional[str]:
    url = _optional_string(item.get("url"))
    if not url or not _is_public_url(url, allow_data=True):
        return None
    return url


def _file_url(item: dict) -> Optional[str]:
    url = _optional_string(item.get("url"))
    if not url or not _is_public_url(url, allow_data=False):
        return None
    return url


def _image_url_to_prompt_content(
    image_url: str,
    detail: Optional[str],
) -> ImagePromptMessageContent:
    return ImagePromptMessageContent(
        format="url",
        url=image_url,
        mime_type=_guess_mime_type(image_url, default="image/png", prefix="image/"),
        detail=_image_detail(detail),
    )


def _file_url_to_prompt_content(file_url: str, file_context: dict) -> DocumentPromptMessageContent:
    filename = _optional_string(file_context.get("filename")) or _filename_from_url(file_url) or "document"
    mime_type = (
        _optional_string(file_context.get("mime_type"))
        or _guess_mime_type(file_url, default="application/octet-stream")
    )
    return DocumentPromptMessageContent(
        format="url",
        url=file_url,
        mime_type=mime_type,
        filename=filename,
    )


def _context_instruction(*, image_count: int, file_count: int) -> str:
    labels: list[str] = []
    if image_count:
        labels.append("image(s)")
    if file_count:
        labels.append("file(s)")
    attachment_label = " and ".join(labels) or "context"
    return f"External context refreshed by Flypower tool output. Use the attached {attachment_label} when answering."


def _is_public_url(value: str, *, allow_data: bool) -> bool:
    parsed = urlparse(value.strip())
    if allow_data and parsed.scheme == "data":
        return True
    if parsed.scheme not in {"http", "https"}:
        return False
    return parsed.hostname not in {"localhost", "127.0.0.1", "0.0.0.0", "web", "nginx", "api"}


def _image_detail(detail: Optional[str]) -> ImagePromptMessageContent.DETAIL:
    if isinstance(detail, str) and detail.lower() == "high":
        return ImagePromptMessageContent.DETAIL.HIGH
    return ImagePromptMessageContent.DETAIL.LOW


def _guess_mime_type(url: str, *, default: str, prefix: Optional[str] = None) -> str:
    if url.startswith("data:"):
        match = re.match(r"^data:([^;,]+)[;,]", url)
        if match and (prefix is None or match.group(1).startswith(prefix)):
            return match.group(1)
    guessed_type, _ = mimetypes.guess_type(urlparse(url).path)
    if guessed_type and (prefix is None or guessed_type.startswith(prefix)):
        return guessed_type
    return default


def _filename_from_url(url: str) -> str:
    path = urlparse(url).path
    filename = path.rsplit("/", 1)[-1]
    return filename.strip()


def _optional_string(value: object) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() else None
