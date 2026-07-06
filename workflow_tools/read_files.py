import json
import mimetypes
import os
from urllib.parse import urlparse


def main(context_refs: str = "") -> dict:
    payload = _normalize_context_refs(context_refs)

    items = _iter_url_items(payload)
    context = {
        "version": 1,
        "type": "dify_context",
        "images": [],
        "files": [],
    }
    for item in items:
        if item["mime_type"].startswith("image/"):
            context["images"].append(
                {
                    "url": item["url"],
                    "filename": item["filename"],
                    "mime_type": item["mime_type"],
                    "detail": item.get("detail") or "high",
                }
            )
        else:
            context["files"].append(
                {
                    "url": item["url"],
                    "filename": item["filename"],
                    "mime_type": item["mime_type"],
                }
            )

    output = _visible_file_index(items)
    output += "\n\n<DIFY_CONTEXT>" + json.dumps(context, ensure_ascii=False, separators=(",", ":")) + "</DIFY_CONTEXT>"
    return {"output": output}


def _normalize_context_refs(context_refs: object) -> dict:
    if isinstance(context_refs, dict):
        return context_refs

    if isinstance(context_refs, list):
        return {"files": context_refs}

    text = _string(context_refs)
    if not text:
        return {}

    try:
        parsed = json.loads(text)
    except ValueError:
        return _payload_from_url_text(text)

    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"files": parsed}
    if isinstance(parsed, str):
        return _payload_from_url_text(parsed)
    return {}


def _payload_from_url_text(text: str) -> dict:
    urls = _extract_urls(text)
    return {"files": [{"url": url} for url in urls]}


def _extract_urls(text: str) -> list[str]:
    candidates = text.replace(",", "\n").splitlines()
    urls: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        url = _supported_url(candidate.strip().strip('"').strip("'"))
        if not url or url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _iter_url_items(payload: dict) -> list[dict]:
    items: list[dict] = []
    seen: set[str] = set()
    for key in ("images", "files"):
        value = payload.get(key)
        if not isinstance(value, list):
            continue
        for raw_item in value:
            if isinstance(raw_item, str):
                raw_item = {"url": raw_item}
            if not isinstance(raw_item, dict):
                continue
            url = _supported_url(raw_item.get("url"))
            if not url or url in seen:
                continue
            seen.add(url)
            filename = _string(raw_item.get("filename")) or _filename_from_url(url)
            mime_type = _string(raw_item.get("mime_type")) or mimetypes.guess_type(urlparse(url).path)[0]
            if key == "images" and not (mime_type or "").startswith("image/"):
                mime_type = "image/png"
            items.append(
                {
                    "url": url,
                    "filename": filename or "file",
                    "mime_type": mime_type or "application/octet-stream",
                    "detail": raw_item.get("detail"),
                }
            )
    return items


def _supported_url(value: object) -> str:
    url = _string(value)
    if not url:
        return ""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https", "data"}:
        return ""
    if parsed.scheme == "data":
        return url
    if parsed.hostname in {"localhost", "127.0.0.1", "0.0.0.0", "web", "nginx", "api"}:
        return ""
    return url


def _filename_from_url(url: str) -> str:
    if url.startswith("data:"):
        return "image"
    return os.path.basename(urlparse(url).path) or "file"


def _visible_file_index(items: list[dict]) -> str:
    if not items:
        return "没有可用的 URL 文件上下文。"

    lines = ["已保存 URL 文件上下文，后续问题可以继续按文件名、序号或 URL 引用："]
    for index, item in enumerate(items, start=1):
        kind = "图片" if item["mime_type"].startswith("image/") else "文件"
        lines.extend(
            [
                f"{index}. {kind}: {item['filename']}",
                f"   url: {item['url']}",
                f"   mime_type: {item['mime_type']}",
            ]
        )
    return "\n".join(lines)


def _string(value: object) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""
