from __future__ import annotations

import json
import time
from typing import Any

from aliyun.log import LogClient, LogItem, PutLogsRequest


SLS_LOGSTORE = "flyfus-dify-llm-log"


def write_invocation_log(credentials: dict[str, Any], event: dict[str, Any]) -> None:
    endpoint = str(credentials.get("sls_endpoint") or "").strip()
    project = str(credentials.get("sls_project") or "").strip()
    access_key_id = str(credentials.get("sls_access_key_id") or "").strip()
    access_key_secret = str(credentials.get("sls_access_key_secret") or "").strip()
    if not endpoint or not project or not access_key_id or not access_key_secret:
        return

    contents = [
        ("log_id", str(event.get("invocation_id") or "")),
        ("event", "llm_invocation"),
        ("source", "flypower_llm_provider"),
        ("model", str(event.get("model") or "")),
        ("status", str(event.get("status") or "unknown")),
        ("duration_ms", str(event.get("duration_ms") or 0)),
        ("event_json", json.dumps(event, ensure_ascii=False, default=str)),
    ]
    try:
        log_item = LogItem()
        log_item.set_time(int(time.time()))
        log_item.set_contents(contents)
        LogClient(endpoint, access_key_id, access_key_secret).put_logs(
            PutLogsRequest(project, SLS_LOGSTORE, "flypower-llm-provider", "", [log_item])
        )
    except Exception:
        return
