from __future__ import annotations

import json
import logging
import time
from typing import Any

from aliyun.log import LogClient, LogItem, PutLogsRequest


SLS_LOGSTORE = "flyfus-dify-llm-log"
logger = logging.getLogger(__name__)


def write_invocation_log(credentials: dict[str, Any], event: dict[str, Any]) -> None:
    endpoint = str(credentials.get("sls_endpoint") or "").strip()
    project = str(credentials.get("sls_project") or "").strip()
    access_key_id = str(credentials.get("sls_access_key_id") or "").strip()
    access_key_secret = str(credentials.get("sls_access_key_secret") or "").strip()
    if not endpoint or not project or not access_key_id or not access_key_secret:
        logger.warning(
            "Flyfus LLM log skipped: SLS credentials are incomplete",
            extra={"invocation_id": event.get("invocation_id")},
        )
        return

    metrics = ((event.get("input") or {}).get("metrics") or {})
    contents = [
        ("log_id", str(event.get("invocation_id") or "")),
        ("user", str(event.get("user") or "")),
        ("user_id", str(event.get("user_id") or "")),
        ("app_id", str(event.get("app_id") or "")),
        ("workflow_id", str(event.get("workflow_id") or "")),
        ("workflow_run_id", str(event.get("workflow_run_id") or "")),
        ("conversation_id", str(event.get("conversation_id") or "")),
        ("latest_user_message_md5", str(metrics.get("latest_user_message_md5") or "")),
        ("output_text_md5", str(((event.get("output") or {}).get("text_md5") or ""))),
        ("event", "llm_invocation"),
        ("source", "flyfus_llm_provider"),
        ("model", str(event.get("model") or "")),
        ("configured_model", str(event.get("configured_model") or "")),
        ("routed_model", str(event.get("routed_model") or event.get("model") or "")),
        ("model_route_applied", "true" if event.get("model_route_applied") else "false"),
        ("status", str(event.get("status") or "unknown")),
        ("duration_ms", str(event.get("duration_ms") or 0)),
        ("event_json", json.dumps(event, ensure_ascii=False, default=str)),
    ]
    try:
        log_item = LogItem()
        log_item.set_time(int(time.time()))
        log_item.set_contents(contents)
        LogClient(endpoint, access_key_id, access_key_secret).put_logs(
            PutLogsRequest(project, SLS_LOGSTORE, "flyfus-llm-provider", "", [log_item])
        )
    except Exception as error:
        logger.warning(
            "Flyfus LLM log delivery failed",
            extra={
                "invocation_id": event.get("invocation_id"),
                "exception_type": type(error).__name__,
            },
        )
        return
