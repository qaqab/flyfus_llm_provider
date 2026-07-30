import mimetypes
import re
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

from models.llm.context_tags import FlyfusContextTag

# Public context and file tags use the same URL extraction rules. The backreference
# keeps an opening tag from matching a different closing tag.
_ATTACHMENT_TAGS = (FlyfusContextTag.CONTEXT, FlyfusContextTag.FILE)
_CONTEXT_PATTERN = re.compile(
    rf"<(?P<tag>{'|'.join(_ATTACHMENT_TAGS)})>(?P<content>.*?)</(?P=tag)>",
    re.DOTALL,
)
_IMAGE_URL_SUFFIXES = (".png", ".jpg", ".jpeg")
_FILE_URL_SUFFIXES = (".pdf", ".md", ".xlsx", ".csv", ".txt", ".html")
# Context is intentionally treated as plain text. This also finds URLs inside JSON,
# lists, and prose without depending on a field name or JSON parsing.
_CONTEXT_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+?\.(?:png|jpe?g|pdf|md|xlsx|csv|txt|html)"
    r"(?:[?#][^\s<>\"']*)?(?=$|[\s<>\"'),.;:!?\]}])",
    re.IGNORECASE,
)


def inject_context_from_tool_messages(
    prompt_messages: list[PromptMessage],
    *,
    include_files: bool,
) -> None:
    """Inject URL context found in tool outputs or user text into the current model call."""

    parts: list[object] = []
    image_count = 0
    file_count = 0
    seen_urls: set[str] = set()

    for prompt_message in prompt_messages:
        for url, context_kind in _extract_context_urls_from_message(prompt_message):
            if url in seen_urls or (context_kind == "file" and not include_files):
                continue
            seen_urls.add(url)

            if context_kind == "image":
                image_count += 1
                parts.append(_image_url_to_prompt_content(url))
            else:
                file_count += 1
                parts.append(_file_url_to_prompt_content(url))

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


def _extract_context_urls_from_message(prompt_message: PromptMessage) -> list[tuple[str, str]]:
    if isinstance(prompt_message, ToolPromptMessage) and isinstance(prompt_message.content, str):
        return _extract_context_urls(prompt_message.content)

    if isinstance(prompt_message, UserPromptMessage):
        urls: list[tuple[str, str]] = []
        for text in _user_message_texts(prompt_message):
            urls.extend(_extract_context_urls(text))
        return urls

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


def _extract_context_urls(text: str) -> list[tuple[str, str]]:
    context_urls: list[tuple[str, str]] = []
    for match in _CONTEXT_PATTERN.finditer(text):
        context_urls.extend(_extract_urls(match.group("content")))
    return context_urls


def _extract_urls(context_text: str) -> list[tuple[str, str]]:
    urls: list[tuple[str, str]] = []
    # JSON may escape forward slashes. Normalizing them lets the same URL regex
    # handle ordinary text and serialized JSON without parsing either format.
    normalized_text = context_text.replace(r"\/", "/")
    for raw_url in _CONTEXT_URL_PATTERN.findall(normalized_text):
        # A URL in prose may be followed by punctuation that is not part of it.
        url = raw_url.rstrip("),.;:!?]}")
        if not _is_public_url(url):
            continue
        context_kind = _url_context_kind(url)
        if context_kind:
            urls.append((url, context_kind))
    return urls


def _url_context_kind(url: str) -> Optional[str]:
    path = urlparse(url).path.lower()
    if path.endswith(_IMAGE_URL_SUFFIXES):
        return "image"
    if path.endswith(_FILE_URL_SUFFIXES):
        return "file"
    return None


def _image_url_to_prompt_content(image_url: str) -> ImagePromptMessageContent:
    return ImagePromptMessageContent(
        format="url",
        url=image_url,
        mime_type=_guess_mime_type(image_url, default="image/png", prefix="image/"),
        detail=ImagePromptMessageContent.DETAIL.HIGH,
    )


def _file_url_to_prompt_content(file_url: str) -> DocumentPromptMessageContent:
    filename = _filename_from_url(file_url) or "document"
    mime_type = _guess_mime_type(file_url, default="application/octet-stream")
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
    return f"External context refreshed by Flyfus tool output. Use the attached {attachment_label} when answering."


def _is_public_url(value: str) -> bool:
    parsed = urlparse(value.strip())
    if parsed.scheme not in {"http", "https"}:
        return False
    return bool(parsed.hostname) and parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "web",
        "nginx",
        "api",
    }


def _guess_mime_type(url: str, *, default: str, prefix: Optional[str] = None) -> str:
    guessed_type, _ = mimetypes.guess_type(urlparse(url).path)
    if guessed_type and (prefix is None or guessed_type.startswith(prefix)):
        return guessed_type
    return default


def _filename_from_url(url: str) -> str:
    path = urlparse(url).path
    filename = path.rsplit("/", 1)[-1]
    return filename.strip()
