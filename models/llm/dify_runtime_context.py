import json
from functools import wraps
from typing import Any

from dify_plugin.core.plugin_executor import PluginExecutor


DIFY_RUNTIME_CONTEXT_KEY = "_flyfus_dify_runtime_context"
_PATCH_MARKER = "_flyfus_session_context_probe"


def install_dify_session_context_probe() -> None:
    """Expose the model plugin Session fields for invocation diagnostics."""
    original_invoke_llm = PluginExecutor.invoke_llm
    if getattr(original_invoke_llm, _PATCH_MARKER, False):
        return

    @wraps(original_invoke_llm)
    def invoke_llm_with_context(self, session, data):
        runtime_context = _session_context(session)
        print(
            "[flyfus-runtime-probe] " + json.dumps(runtime_context, ensure_ascii=True),
            flush=True,
        )
        credentials = dict(data.credentials)
        credentials[DIFY_RUNTIME_CONTEXT_KEY] = runtime_context
        data.credentials = credentials
        return original_invoke_llm(self, session, data)

    setattr(invoke_llm_with_context, _PATCH_MARKER, True)
    PluginExecutor.invoke_llm = invoke_llm_with_context


def _session_context(session: Any) -> dict[str, Any]:
    return {
        "plugin_session_id": getattr(session, "session_id", None),
        "conversation_id": getattr(session, "conversation_id", None),
        "message_id": getattr(session, "message_id", None),
        "app_id": getattr(session, "app_id", None),
        "workflow_run_id": getattr(session, "workflow_run_id", None),
        "trace_id": getattr(session, "trace_id", None),
    }
