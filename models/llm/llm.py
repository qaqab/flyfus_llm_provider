import json
from copy import deepcopy
import re
from contextvars import ContextVar
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from typing import Generator, Optional, Union
from urllib.parse import urljoin

import requests
import yaml

from dify_plugin.entities.model.llm import LLMMode, LLMResult, LLMResultChunk, LLMResultChunkDelta
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    AudioPromptMessageContent,
    DocumentPromptMessageContent,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageFunction,
    PromptMessageRole,
    PromptMessageTool,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
    VideoPromptMessageContent,
)
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeError
from dify_plugin.interfaces.model.openai_compatible.llm import OAICompatLargeLanguageModel

from models.llm.agent_context import inject_context_from_tool_messages
from models.llm.invocation_logging import (
    http_response_summary,
    InvocationLog,
    failure_output_text,
    llm_result_summary,
    prompt_messages_metrics,
    prompt_messages_summary,
    tools_summary,
    upstream_openai_compatible_request_summary,
    wrap_stream_with_invocation_log,
)
from models.llm.parameter_conversion import normalize_generation_parameters, normalize_max_tokens
from models.llm.native.base import model_family
from models.llm.native.gemini import GeminiNativeDocumentAdapter
from models.llm.native.openai_responses import OpenAIResponsesAdapter
from models.llm.usage_reporting import format_usage_currency, normalize_upstream_usage, report_token_usage


_ACTIVE_INVOCATION_LOG: ContextVar[Optional[InvocationLog]] = ContextVar(
    "flyfus_active_invocation_log",
    default=None,
)


class FlyfusLargeLanguageModel(OAICompatLargeLanguageModel):
    """Flyfus LLM 调用适配器。

    当前插件主要复用 Dify SDK 的 OpenAI 兼容基类；OpenAI 系列模型单独走
    Responses API，其他模型继续走 Chat Completions 兼容路径。这里保留少量
    适配逻辑，用来处理不同上游对“兼容 OpenAI”理解不完全一致的地方：

    - OpenAI 系列走 Responses，包含文件、结构化输出和工具调用。
    - 非 OpenAI 系列固定走 chat 模式。
    - 默认使用新版工具调用 tool_call。
    - 转换图片、文档、音频、视频输入。
    - 统一使用 max_completion_tokens 参数。
    - 按模型 YAML 可选适配 thinking/reasoning 私有参数。
    - 用轻量请求校验供应商凭据。

    Gemini 和其他国产/兼容模型暂不走文件特殊路径，后续可按模型族单独拆分。
    """

    _THINK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
    _GEO_PROMPT_REFERENCE_PATTERN = re.compile(
        r"\{\{dify_admin:(?P<name>[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+)}}"
    )
    _GEO_PROMPT_TOKEN_PATTERN = re.compile(r"\{\{dify_admin:[^}]*}}")
    _REASONING_EFFORT_TOOL_NAME = "set_next_step"
    _REASONING_EFFORT_VALUES = {"low", "medium", "high", "xhigh"}

    @classmethod
    def _render_geo_prompt_text(cls, text: str, credentials: dict) -> str:
        """渲染一段包含 Geo Prompt 引用的文本。"""
        normalized_text = cls._normalize_geo_prompt_references(text, credentials)
        if normalized_text == text and not cls._GEO_PROMPT_TOKEN_PATTERN.search(text):
            return text

        geo_base_url = str(credentials.get("geo_prompt_render_url") or "").strip().rstrip("/")
        invocation_log = _ACTIVE_INVOCATION_LOG.get()
        if invocation_log is not None:
            invocation_log.event(
                "geo_prompt_render_request",
                endpoint=f"{geo_base_url}/dify_admin/render",
                reference_count=len(cls._GEO_PROMPT_TOKEN_PATTERN.findall(text)),
            )
        try:
            response = requests.post(
                f"{geo_base_url}/dify_admin/render",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {str(credentials.get('geo_prompt_api_key') or '').strip()}",
                },
                json={"text": normalized_text},
                timeout=(10, 60),
            )
        except Exception as error:
            raise InvokeError(f"Geo Prompt 渲染请求失败：{error}") from error

        if invocation_log is not None:
            invocation_log.event(
                "geo_prompt_render_response",
                status_code=response.status_code,
            )

        if response.status_code != 200:
            raise InvokeError(
                f"Geo Prompt 渲染失败，状态码：{response.status_code}，响应：{response.text}"
            )

        try:
            payload = response.json()
        except ValueError as error:
            raise InvokeError("Geo Prompt 渲染接口返回的不是 JSON。") from error

        rendered_text = payload.get("data", {}).get("rendered_text")
        if not isinstance(rendered_text, str):
            raise InvokeError("Geo Prompt 渲染接口返回格式缺少 data.rendered_text。")

        return rendered_text

    @classmethod
    def _normalize_geo_prompt_references(cls, text: str, credentials: dict) -> str:
        geo_prompt_tokens = cls._GEO_PROMPT_TOKEN_PATTERN.findall(text)
        if not geo_prompt_tokens:
            return text
        for token in geo_prompt_tokens:
            reference = cls._GEO_PROMPT_REFERENCE_PATTERN.fullmatch(token)
            if reference is None:
                raise InvokeError(
                    "Geo Prompt 引用必须使用 {{dify_admin:agent.prompt}} 格式。"
                )
        return text

    def _wrap_thinking_by_reasoning_content(self, delta: dict, is_reasoning: bool) -> tuple[str, bool]:
        """把上游 reasoning 字段转换成 Dify 能识别的思考块。

        很多 OpenAI-compatible 网关不会直接流式输出 ``<think>``，而是把思考
        token 放在 ``reasoning`` 或旧版 ``reasoning_content`` 字段里。Dify 后续
        展示和过滤思考内容时识别的是 ``<think>...</think>``，所以这里在流式
        响应阶段补上开始/结束标签。
        """
        reasoning_piece = delta.get("reasoning") or delta.get("reasoning_content") or ""
        content_piece = delta.get("content") or ""
        output = ""

        if reasoning_piece:
            if not is_reasoning:
                output += f"<think>\n{reasoning_piece}"
                is_reasoning = True
            else:
                output += str(reasoning_piece)

        if is_reasoning:
            if not reasoning_piece and not content_piece:
                is_reasoning = False
                output += "\n</think>"
            if content_piece:
                is_reasoning = False
                output += f"\n</think>{content_piece}"
        elif content_piece:
            output += content_piece

        return output, is_reasoning

    def _normalize_credentials(self, model: str, credentials: dict) -> dict:
        """补齐运行时凭据默认值。

        Dify 的 OpenAI 兼容基类会根据 credentials["mode"] 决定调用
        /chat/completions 还是 /completions。本插件所有预设模型都是聊天模型，
        所以统一使用 chat。
        """
        normalized_credentials = dict(credentials)
        normalized_credentials["mode"] = "chat"
        normalized_credentials.setdefault("function_calling_type", "tool_call")
        normalized_credentials.setdefault("stream_function_calling", "supported")
        normalized_credentials.setdefault("endpoint_model_name", model)
        return normalized_credentials

    def get_model_schema(self, model: str, credentials=None):
        """Keep schema lookup compatible with credentials saved before ``mode`` existed."""
        normalized_credentials = self._normalize_credentials(model, dict(credentials or {}))
        return super().get_model_schema(model, normalized_credentials)

    def _request_headers(self, credentials: dict) -> dict:
        """构造上游请求头。"""
        headers = {"Content-Type": "application/json"}
        if credentials.get("api_key"):
            headers["Authorization"] = f"Bearer {credentials['api_key']}"
        invocation_id = credentials.get("_flyfus_invocation_id")
        if invocation_id:
            headers["X-Client-Request-Id"] = invocation_id
            headers["X-Flyfus-Invocation-Id"] = invocation_id
        return headers

    def _endpoint_url(self, credentials: dict, path: str) -> str:
        """拼接上游 OpenAI-compatible 请求地址。"""
        endpoint_url = credentials["endpoint_url"]
        if not endpoint_url.endswith("/"):
            endpoint_url += "/"
        return urljoin(endpoint_url, path)

    def _list_available_models(self, credentials: dict) -> set[str]:
        """读取上游 /models 返回的模型 ID。"""
        response = requests.get(
            self._endpoint_url(credentials, "models"),
            headers=self._request_headers(credentials),
            timeout=(10, 60),
        )
        if response.status_code != 200:
            raise CredentialsValidateFailedError(
                f"模型列表接口校验失败，状态码：{response.status_code}，响应：{response.text}"
            )

        try:
            payload = response.json()
        except ValueError as error:
            raise CredentialsValidateFailedError("模型列表接口返回的不是 JSON。") from error

        raw_models = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(raw_models, list):
            raise CredentialsValidateFailedError("模型列表接口返回格式不符合 OpenAI-compatible /models 结构。")

        model_ids: set[str] = set()
        for item in raw_models:
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                model_ids.add(item["id"])
        return model_ids

    def _normalize_model_parameters(self, model: str, model_parameters: dict) -> dict:
        """生成一次调用专用的参数副本，并按固定顺序完成所有转换。

        原始 ``model_parameters`` 可能会在 Dify 的后续节点、重试或日志中继续使用，
        因此这里必须先复制，绝不能原地修改调用方对象。转换顺序也有含义：

        1. 规范化温度、Top P 和回复格式等通用页面参数；
        2. 根据 YAML 决定输出 token 字段应为 ``max_tokens`` 还是
           ``max_completion_tokens``；
        3. 根据 YAML 的 ``extra.thinking`` 把统一的思考控制转换成供应商字段。

        返回值是即将发送给原生适配器的参数字典。网络搜索不在这里写入，因为它在
        Responses 请求体构建阶段作为 ``tools`` 数组的一部分处理。
        """
        normalized_parameters = dict(model_parameters)
        normalize_generation_parameters(model, normalized_parameters)
        normalize_max_tokens(normalized_parameters, self._token_param_name(model))

        self._apply_thinking_parameters(model, normalized_parameters)
        return normalized_parameters

    @classmethod
    def _token_param_name(cls, model: str) -> str:
        """从模型 YAML 选择上游输出 token 上限字段名。

        Dify 前端只展示 ``max_tokens``，但部分 OpenAI 兼容模型要求使用
        ``max_completion_tokens``。YAML 可通过 ``extra.token_param_name`` 显式声明
        其中之一；只有这两个白名单值才会被采用，缺失、拼写错误或未知值均回退到
        ``max_tokens``。回退策略是为了让新模型在未添加额外配置时仍保持标准兼容
        行为，而不是把无效配置直接传给上游。
        """
        configured_name = cls._load_model_extra(model).get("token_param_name")
        if configured_name in {"max_tokens", "max_completion_tokens"}:
            return configured_name
        return "max_tokens"

    def _apply_thinking_parameters(self, model: str, model_parameters: dict) -> None:
        """按模型 YAML 的 ``extra.thinking`` 映射非标准思考/推理参数。

        这里不要按模型名字写硬编码分支，而是让每个 YAML 自己声明映射模式。
        这样后面新增模型时，只需要在模型配置里选择对应模式：

        - ``top_level``：上游直接接收 ``enable_thinking`` / ``thinking_budget``。
        - ``deepseek`` / ``zhipu``：上游接收 ``thinking: {"type": "enabled"}``。
        - ``gemini``：上游接收 ``thinking_config``。
        - ``minimax``：上游接收 Anthropic 风格的 ``thinking.budget_tokens``。
        - ``openrouter``：上游接收 ``reasoning`` 对象。
        - ``chat_template_kwargs``：上游运行时从模板参数读取思考开关。

        前端公共参数与上游参数不是一一对应关系。这个入口先取出公共参数，随后
        分别交给三个独立方法处理：

        - ``_apply_enable_thinking``：是否让模型执行思考过程；
        - ``_apply_thinking_budget``：思考过程允许消耗的 token 预算；
        - ``_apply_reasoning_effort``：推理深度/强度档位。

        这样模型 YAML 只负责声明 ``mode``，而具体的字段转换集中在这里，避免
        调用路径到处出现 ``if model == ...``。函数会先 ``pop`` 前端专用字段，
        以保证未在 YAML 中声明支持方式的模型不会把未知参数原样传给上游。

        ``thinking_level`` 与 ``include_thoughts`` 是 Gemini 专属参数：前者决定
        推理级别，后者决定是否在响应中返回思考内容。它们不是公共三项的一部分，
        因为“开启思考”不等同于“展示思考内容”。
        """
        enable_thinking = model_parameters.pop("enable_thinking", None)
        thinking = model_parameters.pop("thinking", None)
        if enable_thinking is None:
            enable_thinking = thinking
        thinking_budget = model_parameters.pop("thinking_budget", None)
        thinking_level = model_parameters.pop("thinking_level", None)
        include_thoughts = model_parameters.pop("include_thoughts", None)
        thinking_config = self._load_model_extra(model).get("thinking", {})
        if not isinstance(thinking_config, dict):
            return

        mode = thinking_config.get("mode", "none")
        thinking_enabled = self._apply_enable_thinking(
            model_parameters, mode, enable_thinking, thinking_budget
        )
        self._apply_thinking_budget(model_parameters, mode, thinking_enabled, thinking_budget)

        if thinking_level is not None:
            if mode == "gemini":
                model_parameters.setdefault("thinking_config", {})["thinking_level"] = thinking_level
            elif mode == "openrouter":
                model_parameters.setdefault("reasoning", {})["effort"] = str(thinking_level).lower()

        if include_thoughts is not None and mode == "gemini":
            model_parameters.setdefault("thinking_config", {})["include_thoughts"] = bool(include_thoughts)

        exclude_reasoning_tokens = model_parameters.pop("exclude_reasoning_tokens", None)
        if exclude_reasoning_tokens is not None and mode == "openrouter":
            model_parameters.setdefault("reasoning", {})["exclude"] = bool(exclude_reasoning_tokens)

        self._apply_reasoning_effort(model_parameters, mode, thinking_config)

    def _apply_enable_thinking(
        self, model_parameters: dict, mode: str, enable_thinking: object, thinking_budget: object
    ) -> Optional[bool]:
        """将统一的“开启思考”参数转换为各模型实际接受的字段。

        Dify 页面统一使用 ``enable_thinking: bool``。不同上游对同一语义使用的
        请求格式如下：

        - ``top_level``：GLM/Qwen/Kimi 一类，直接发送 ``enable_thinking``；
        - ``deepseek`` / ``zhipu``：发送 ``thinking: {"type": "enabled"}``；
        - ``openrouter``：发送 ``reasoning: {"enabled": true}``；
        - ``chat_template_kwargs``：把值放入兼容层模板参数；
        - ``minimax``：构造带 ``budget_tokens`` 的 Anthropic 风格 ``thinking``。

        返回归一化后的布尔值，供 ``_apply_thinking_budget`` 判断。例如用户明确
        关闭思考时，MiniMax 不应再因为有预算值而被重新开启。``mode == 'none'``
        没有分支，意味着参数会被消费但不会转发，防止不支持该参数的模型报错。
        """
        if enable_thinking is None:
            return None

        thinking_enabled = self._to_bool(enable_thinking)
        if mode == "top_level":
            model_parameters["enable_thinking"] = thinking_enabled
        elif mode in {"deepseek", "zhipu"}:
            model_parameters["thinking"] = {"type": "enabled" if thinking_enabled else "disabled"}
        elif mode == "openrouter":
            model_parameters.setdefault("reasoning", {})["enabled"] = thinking_enabled
        elif mode == "chat_template_kwargs":
            template_kwargs = model_parameters.setdefault("chat_template_kwargs", {})
            template_kwargs["enable_thinking"] = thinking_enabled
            template_kwargs["thinking"] = thinking_enabled
        elif mode == "minimax":
            minimax_thinking = self._minimax_thinking_payload(thinking_enabled, thinking_budget)
            if minimax_thinking:
                model_parameters["thinking"] = minimax_thinking
        return thinking_enabled

    def _apply_thinking_budget(
        self, model_parameters: dict, mode: str, thinking_enabled: Optional[bool], thinking_budget: object
    ) -> None:
        """将统一的“思考预算”参数转换为各模型实际接受的字段。

        页面参数 ``thinking_budget`` 表示“模型内部推理最多可用多少 token”，不等同
        于普通输出上限 ``max_tokens``。各模式的转换为：

        - ``top_level``：发送 ``thinking_budget``；
        - ``openrouter``：发送 ``reasoning.max_tokens``；
        - ``gemini``：发送 ``thinking_config.thinking_budget``；
        - ``minimax``：并入 ``thinking.budget_tokens``。

        MiniMax 有一个额外约束：当用户显式关闭 ``enable_thinking`` 时，预算不能
        单独生成 ``thinking`` 对象，否则“关闭思考”会被预算反向打开。对于没有
        思考映射的 ``mode``，预算在这里被丢弃，避免把供应商私有字段误传出去。
        """
        if thinking_budget is None:
            return
        if mode == "top_level":
            model_parameters["thinking_budget"] = thinking_budget
        elif mode == "openrouter":
            model_parameters.setdefault("reasoning", {})["max_tokens"] = thinking_budget
        elif mode == "gemini":
            model_parameters.setdefault("thinking_config", {})["thinking_budget"] = thinking_budget
        elif mode == "minimax" and thinking_enabled is not False and "thinking" not in model_parameters:
            model_parameters["thinking"] = self._minimax_thinking_payload(True, thinking_budget)

    @staticmethod
    def _apply_reasoning_effort(model_parameters: dict, mode: str, thinking_config: dict) -> None:
        """将统一的“推理强度”参数转换为各模型实际接受的字段。

        ``reasoning_effort`` 是前端的档位参数，例如 ``low``、``medium``、``high``。
        它与布尔的 ``enable_thinking`` 不同：前者调整已开启推理的深度，后者决定
        是否执行推理。实际字段由 YAML 决定：

        - ``reasoning_effort_target: chat_template_kwargs``：兼容 Chat Completions
          模型时放入 ``chat_template_kwargs.reasoning_effort``；
        - ``mode: openrouter``：转换为 ``reasoning.effort``，并只接受已确认的
          ``high``、``medium``、``low``、``minimal``、``none``，非法值不发送。

        未声明上述映射的模型会保留原有参数处理路径。例如 GPT/Grok 的 Responses
        适配器会直接读取 ``reasoning_effort``，并生成
        ``reasoning: {"effort": ...}``，这里不能提前 ``pop``，否则会导致它们的
        推理强度失效。
        """
        if thinking_config.get("reasoning_effort_target") == "chat_template_kwargs":
            reasoning_effort = model_parameters.get("reasoning_effort")
            if reasoning_effort is not None:
                model_parameters.setdefault("chat_template_kwargs", {})["reasoning_effort"] = reasoning_effort
        elif mode == "openrouter":
            reasoning_effort = model_parameters.pop("reasoning_effort", None)
            if reasoning_effort in {"high", "medium", "low", "minimal", "none"}:
                model_parameters.setdefault("reasoning", {})["effort"] = reasoning_effort

    @staticmethod
    def _to_bool(value: object) -> bool:
        """把页面、脚本或旧配置传入的开关值安全归一化为真正的布尔值。

        标准 Dify 页面会根据 YAML 的 ``type: boolean`` 传入 ``True`` 或 ``False``，
        此时直接返回，避免把布尔值意外当成字符串处理。实际运行中也会遇到三种
        非标准来源：旧版本保存的字符串、调试脚本的字符串，以及其他插件转交的
        数字/对象。

        对字符串先 ``strip().lower()``，再把 ``""``、``"0"``、``"false"``、
        ``"no"``、``"off"``、``"disabled"`` 统一认定为关闭；其它非空字符串
        认定为开启。之所以不能直接使用 Python 的 ``bool(value)``，是因为
        ``bool("false")`` 和 ``bool("0")`` 都是 ``True``，会把用户关闭思考的
        配置反向打开。非字符串值最终使用 Python 原生真值规则，保证数值 ``0``
        为关闭、非零数值为开启。
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in {"", "0", "false", "no", "off", "disabled"}
        return bool(value)

    @staticmethod
    def _minimax_thinking_payload(enabled: bool, thinking_budget: Optional[int]) -> Optional[dict]:
        """构造 MiniMax 的 Anthropic-compatible ``thinking`` 请求对象。

        多数模型把“是否思考”和“思考预算”拆成顶层字段；MiniMax 的兼容接口要求
        把两者合并为 ``thinking: {"type": "enabled", "budget_tokens": N}``。
        因此该方法是唯一应构造这个供应商私有对象的位置，调用方不需要知道其字段
        细节。

        ``enabled=False`` 时返回 ``None``，调用方据此完全省略 ``thinking`` 字段，
        而不是发送 ``{"type": "disabled"}``。这是该兼容协议的关闭语义，也避免
        上游把带预算的对象仍当作开启思考。开启时，未填预算使用 1024，低于 1024
        的输入也抬升到 1024；该下限保护来自 Anthropic-compatible thinking 对最小
        预算的常见要求。数值转换在这里集中处理，确保页面传入的 int 与脚本传入的
        可转换数值走同一条路径。
        """
        if not enabled:
            return None
        budget_tokens = max(1024, int(thinking_budget or 1024))
        return {"type": "enabled", "budget_tokens": budget_tokens}

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """校验供应商凭据。

        这里自己发一个最小聊天请求，避免保存凭据时因为 token 参数名不兼容而失败。
        """
        normalized_credentials = self._normalize_credentials(model, credentials)
        available_models = self._list_available_models(normalized_credentials)
        predefined_models = self._load_predefined_chat_models()
        matched_models = predefined_models & available_models
        if not matched_models:
            raise CredentialsValidateFailedError(
                "该 API Key 的 /models 没有返回本插件支持的聊天模型，请检查 API 地址或密钥。"
            )

        validation_model = model if model in matched_models else sorted(matched_models)[0]
        request_url = self._endpoint_url(normalized_credentials, "chat/completions")

        request_body = {
            "model": validation_model,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
        }
        request_body[self._token_param_name(validation_model)] = 16

        try:
            response = requests.post(
                request_url,
                headers=self._request_headers(normalized_credentials),
                json=request_body,
                timeout=(10, 300),
            )
        except Exception as error:
            raise CredentialsValidateFailedError(f"凭据校验请求失败：{error}") from error

        if response.status_code != 200:
            raise CredentialsValidateFailedError(
                f"凭据校验失败，状态码：{response.status_code}，响应：{response.text}"
            )

    def _convert_prompt_message_to_dict(
        self, message: PromptMessage, credentials: Optional[dict] = None
    ) -> dict:
        """转换 Dify 消息为 OpenAI-compatible 消息。

        模型 YAML 声明对应能力后，Dify 才会传入图片或文档。
        这里只负责把已收到的内容转换成 OpenAI-compatible 请求结构。

        这里沿用官方 openai_api_compatible 插件的形态：

        - 图片：``image_url``。
        - 视频/音频：仍放进 ``image_url.url``，由兼容网关按 data URI 识别。
        - 文档：``file.file_data`` 直接使用 Dify SDK 提供的 data URI。

        这段逻辑不判断模型是否真正支持某种模态；能力开关由模型 YAML 的
        ``features`` 控制。这样可以把“模型能力声明”和“请求格式转换”分开。
        """
        if not isinstance(message, UserPromptMessage) or not isinstance(message.content, list):
            return super()._convert_prompt_message_to_dict(message, credentials)

        content_parts: list[dict] = []
        for content in message.content:
            if content.type == PromptMessageContentType.TEXT:
                content_parts.append({"type": "text", "text": content.data})
            elif content.type == PromptMessageContentType.IMAGE:
                image_content: ImagePromptMessageContent = content
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_content.data,
                            "detail": image_content.detail.value,
                        },
                    }
                )
            elif content.type == PromptMessageContentType.VIDEO:
                video_content: VideoPromptMessageContent = content
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": video_content.data},
                    }
                )
            elif content.type == PromptMessageContentType.AUDIO:
                audio_content: AudioPromptMessageContent = content
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": audio_content.data},
                    }
                )
            elif content.type == PromptMessageContentType.DOCUMENT:
                document_content: DocumentPromptMessageContent = content
                content_parts.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": document_content.filename or "document",
                            "file_data": document_content.data,
                        },
                    }
                )

        message_dict: dict = {"role": "user", "content": content_parts}
        if message.name:
            message_dict["name"] = message.name
        return message_dict

    def _openai_responses_adapter(self) -> OpenAIResponsesAdapter:
        return OpenAIResponsesAdapter(
            endpoint_url=self._endpoint_url,
            request_headers=self._request_headers,
            normalize_model_parameters=self._normalize_model_parameters,
            calc_response_usage=self._calc_response_usage,
            build_dify_usage=self._build_dify_usage,
            create_final_chunk=self._create_final_llm_result_chunk,
        )

    def _gemini_native_adapter(self) -> GeminiNativeDocumentAdapter:
        """构造 Gemini 原生适配器，避免它退回 OpenAI-compatible 路径。"""
        return GeminiNativeDocumentAdapter(
            endpoint_url=self._endpoint_url,
            normalize_model_parameters=self._normalize_model_parameters,
            calc_response_usage=self._calc_response_usage,
            build_dify_usage=self._build_dify_usage,
        )

    def _build_dify_usage(self, model: str, credentials: dict, raw_usage: dict):
        """Build Dify's fixed usage object from the upstream usage source of truth."""
        normalized = normalize_upstream_usage(raw_usage)
        usage = self._calc_response_usage(
            model,
            credentials,
            normalized["input_tokens"] or 0,
            normalized["output_tokens"] or 0,
        )
        if normalized["total_tokens"] is not None:
            usage.total_tokens = normalized["total_tokens"]
        invocation_log = _ACTIVE_INVOCATION_LOG.get()
        log_id = invocation_log.invocation_id if invocation_log else credentials.get("_flyfus_invocation_id")
        usage.currency = format_usage_currency(raw_usage, log_id=str(log_id) if log_id else None)
        return usage

    @classmethod
    def _drop_analyze_channel(cls, prompt_messages: list[PromptMessage]) -> None:
        """移除历史 assistant 消息里的思考内容。

        Dify 会把上一轮 assistant 回复继续放进下一轮上下文。如果历史回复里
        已经带有 ``<think>...</think>``，下一次请求会把旧思考链也发给模型，
        既浪费 token，也容易污染下一轮回答。因此只在历史 assistant 文本里
        做轻量清理，不改用户原始输入。
        """
        for prompt_message in prompt_messages:
            if not isinstance(prompt_message, AssistantPromptMessage):
                continue
            if not isinstance(prompt_message.content, str):
                continue
            if "<think>" not in prompt_message.content:
                continue
            prompt_message.content = cls._THINK_PATTERN.sub("", prompt_message.content)

    @staticmethod
    def _apply_json_schema_prompt(model_parameters: dict, prompt_messages: list[PromptMessage]) -> None:
        """把 Dify 的 json_schema 参数补成兼容端更容易遵循的系统提示。

        Dify 会把 ``response_format=json_schema`` 和 ``json_schema`` 传进模型
        参数，但并不是每个 OpenAI-compatible 网关都原生支持这个字段。官方插件
        也采用了向 system prompt 注入 schema 的兼容策略。这里保留原参数不删，
        让原生支持 json_schema 的上游仍有机会使用，同时用系统提示兜底。
        """
        if model_parameters.get("response_format") != "json_schema":
            return

        json_schema = model_parameters.get("json_schema")
        if not json_schema:
            return

        structured_output_prompt = (
            "Your response must be a JSON object that validates against the following JSON schema, and nothing else.\n"
            f"JSON Schema: ```json\n{json_schema}\n```"
        )
        existing_system_prompt = next(
            (p for p in prompt_messages if p.role == PromptMessageRole.SYSTEM),
            None,
        )
        if existing_system_prompt:
            existing_system_prompt.content = (
                structured_output_prompt + "\n\n" + existing_system_prompt.content
            )
        else:
            prompt_messages.insert(0, SystemPromptMessage(content=structured_output_prompt))

    @classmethod
    def _render_geo_prompt_references(cls, prompt_messages: list[PromptMessage], credentials: dict) -> None:
        """渲染 system prompt 中的 Geo Prompt 引用。"""
        for prompt_message in prompt_messages:
            if prompt_message.role != PromptMessageRole.SYSTEM:
                continue
            if not isinstance(prompt_message.content, str):
                continue
            prompt_message.content = cls._render_geo_prompt_text(prompt_message.content, credentials)

    @classmethod
    def _reasoning_effort_from_tool_messages(cls, prompt_messages: list[PromptMessage]) -> Optional[str]:
        """Read the next reasoning effort from the dedicated workflow tool only."""
        reasoning_effort = None
        for prompt_message in prompt_messages:
            if not isinstance(prompt_message, ToolPromptMessage):
                continue
            if prompt_message.name != cls._REASONING_EFFORT_TOOL_NAME:
                continue
            if not isinstance(prompt_message.content, str):
                continue
            parsed_effort = cls._extract_reasoning_effort(prompt_message.content)
            if parsed_effort:
                reasoning_effort = parsed_effort
        return reasoning_effort

    @classmethod
    def _extract_reasoning_effort(cls, content: str) -> Optional[str]:
        payload = cls._try_parse_json(content.strip())
        # Dify wraps Workflow tool output as tool name -> result -> output.
        for _ in range(4):
            if not isinstance(payload, dict):
                return None

            effort = payload.get("reasoning_effort")
            if isinstance(effort, str) and effort.lower() in cls._REASONING_EFFORT_VALUES:
                return effort.lower()

            if len(payload) != 1:
                return None
            wrapped_value = next(iter(payload.values()))
            if not isinstance(wrapped_value, str):
                return None
            payload = cls._try_parse_json(wrapped_value.strip())
        return None

    @staticmethod
    def _try_parse_json(text: str):
        if not text or text[0] not in "{[":
            return None
        try:
            return json.loads(text)
        except ValueError:
            return None

    def _invoke(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        effective_model_parameters = dict(model_parameters)
        reasoning_effort = self._reasoning_effort_from_tool_messages(prompt_messages)
        if reasoning_effort:
            effective_model_parameters["reasoning_effort"] = reasoning_effort

        invocation_log = InvocationLog.from_credentials(
            model=model,
            credentials=credentials,
            stream=stream,
            user=user,
        )
        def usage_reporter(usage):
            raw_usage = invocation_log.response.get("upstream_usage") or usage
            provider_request_id = (
                invocation_log.response.get("provider_request_id")
                or invocation_log.response.get("response_id")
                or invocation_log.invocation_id
            )
            try:
                reported = report_token_usage(
                    provider_request_id,
                    model,
                    raw_usage,
                    user,
                    credentials,
                )
            except Exception as error:
                invocation_log.event(
                    "token_usage_report",
                    status="error",
                    request_id=provider_request_id,
                    error_type=type(error).__name__,
                )
                return False
            invocation_log.event(
                "token_usage_report",
                status="success" if reported else "skipped_or_failed",
                request_id=provider_request_id,
                has_usage=raw_usage is not None,
            )
            return reported

        invocation_log.set_request(
            model=model,
            stream=stream,
            user=user,
            stop=stop,
            model_parameters=effective_model_parameters,
            prompt_metrics_initial=prompt_messages_metrics(prompt_messages),
            prompt_messages_initial=prompt_messages_summary(prompt_messages),
            tools=tools_summary(tools),
        )
        invocation_log.event(
            "invoke_started",
            model_parameters=effective_model_parameters,
            tools_count=len(tools or []),
            stop=stop,
            prompt_metrics=prompt_messages_metrics(prompt_messages),
        )
        normalized_credentials = self._normalize_credentials(model, credentials)
        normalized_credentials["_flyfus_invocation_id"] = invocation_log.invocation_id
        active_log_token = _ACTIVE_INVOCATION_LOG.set(invocation_log)
        family = model_family(model)
        invocation_log.set_request(model_family=family)
        try:
            with invocation_log.step("context_injection", include_files=family == "openai_responses"):
                inject_context_from_tool_messages(
                    prompt_messages,
                    include_files=family == "openai_responses",
                )
            with invocation_log.step("geo_prompt_render"):
                self._render_geo_prompt_references(prompt_messages, credentials)
            with invocation_log.step("analyze_channel_drop"):
                with suppress(Exception):
                    self._drop_analyze_channel(prompt_messages)
            invocation_log.set_request(
                prompt_metrics_final=prompt_messages_metrics(prompt_messages),
                prompt_messages_final=prompt_messages_summary(prompt_messages),
            )
            if family == "gemini":
                gemini_adapter = self._gemini_native_adapter()
                upstream_request_body = gemini_adapter.build_body(
                    model=model,
                    prompt_messages=prompt_messages,
                    model_parameters=dict(effective_model_parameters),
                    tools=tools,
                    stop=stop,
                )
                invocation_log.set_request(
                    model_parameters_final=self._normalize_model_parameters(model, dict(effective_model_parameters)),
                    adapter="gemini_native",
                    upstream_request={
                        "endpoint": self._endpoint_url(normalized_credentials, f"models/{model}:generateContent"),
                        "body_summary": {
                            "content_count": len(upstream_request_body.get("contents") or []),
                            "tool_count": len(upstream_request_body.get("tools") or []),
                            "has_google_search": {"google_search": {}} in (upstream_request_body.get("tools") or []),
                            "has_generation_config": bool(upstream_request_body.get("generationConfig")),
                        },
                    },
                )
                with invocation_log.step("upstream_request", adapter="gemini_native"):
                    result = gemini_adapter.invoke(
                        model=model,
                        credentials=normalized_credentials,
                        prompt_messages=prompt_messages,
                        model_parameters=effective_model_parameters,
                        tools=tools,
                        stop=stop,
                        stream=stream,
                        user=user,
                        invocation_log=invocation_log,
                    )
                if stream:
                    return wrap_stream_with_invocation_log(
                        result,
                        invocation_log,
                        usage_reporter,
                        self._failure_stream_chunk_factory(model, prompt_messages),
                    )
                result_summary = llm_result_summary(result)
                invocation_log.set_response(**result_summary)
                invocation_log.success(result_type=type(result).__name__, output_text=result_summary.get("output_text"))
                usage_reporter(getattr(result, "usage", None))
                return result

            if family == "openai_responses":
                responses_adapter = self._openai_responses_adapter()
                upstream_request_body = responses_adapter._build_body(
                    model=model,
                    credentials=normalized_credentials,
                    prompt_messages=prompt_messages,
                    model_parameters=dict(effective_model_parameters),
                    tools=tools,
                    stop=stop,
                    stream=stream,
                    user=user,
                )
                invocation_log.set_request(
                    model_parameters_final=effective_model_parameters,
                    adapter="openai_responses",
                    upstream_request={
                        "endpoint": self._endpoint_url(normalized_credentials, "responses"),
                        "headers": {
                            key: value
                            for key, value in self._request_headers(normalized_credentials).items()
                            if key.lower() != "authorization"
                        },
                        "body_summary": {
                            "model": upstream_request_body.get("model"),
                            "stream": upstream_request_body.get("stream"),
                            "temperature": upstream_request_body.get("temperature"),
                            "max_output_tokens": upstream_request_body.get("max_output_tokens"),
                            "input_count": len(upstream_request_body.get("input") or []),
                            "tool_count": len(upstream_request_body.get("tools") or []),
                            "tool_choice": upstream_request_body.get("tool_choice"),
                            "has_text_format": bool(upstream_request_body.get("text")),
                        },
                        "input_count": len(upstream_request_body.get("input") or []),
                        "tool_count": len(upstream_request_body.get("tools") or []),
                    },
                )
                with invocation_log.step("upstream_request", adapter="openai_responses"):
                    result = responses_adapter.invoke(
                        model=model,
                        credentials=normalized_credentials,
                        prompt_messages=prompt_messages,
                        model_parameters=effective_model_parameters,
                        tools=tools,
                        stop=stop,
                        stream=stream,
                        user=user,
                        invocation_log=invocation_log,
                    )
                if stream:
                    return wrap_stream_with_invocation_log(
                        result,
                        invocation_log,
                        usage_reporter,
                        self._failure_stream_chunk_factory(model, prompt_messages),
                    )
                result_summary = llm_result_summary(result)
                invocation_log.set_response(**result_summary)
                invocation_log.success(result_type=type(result).__name__, output_text=result_summary.get("output_text"))
                usage_reporter(getattr(result, "usage", None))
                return result

            with invocation_log.step("json_schema_prompt_apply"):
                self._apply_json_schema_prompt(effective_model_parameters, prompt_messages)
            with invocation_log.step("model_parameters_normalize"):
                normalized_model_parameters = self._normalize_model_parameters(model, effective_model_parameters)
            replay_body = self._build_openai_compatible_replay_body(
                model=model,
                credentials=normalized_credentials,
                prompt_messages=prompt_messages,
                model_parameters=normalized_model_parameters,
                tools=tools,
                stop=stop,
                stream=stream,
                user=user,
            )
            invocation_log.set_replay_request(
                endpoint=self._endpoint_url(normalized_credentials, "chat/completions"),
                body=replay_body,
            )
            invocation_log.set_request(
                model_parameters_final=normalized_model_parameters,
                prompt_messages_final=prompt_messages_summary(prompt_messages),
                adapter="openai_compatible",
                upstream_request=upstream_openai_compatible_request_summary(
                    model=model,
                    credentials=normalized_credentials,
                    prompt_messages=prompt_messages,
                    model_parameters=normalized_model_parameters,
                    tools=tools,
                    stop=stop,
                    stream=stream,
                    user=user,
                    convert_message=self._convert_prompt_message_to_dict,
                    headers={
                        key: value
                        for key, value in self._request_headers(normalized_credentials).items()
                        if key.lower() != "authorization"
                    },
                ),
            )
            with invocation_log.step("upstream_request", adapter="openai_compatible"):
                result = super()._invoke(
                    model=model,
                    credentials=normalized_credentials,
                    prompt_messages=prompt_messages,
                    model_parameters=normalized_model_parameters,
                    tools=tools,
                    stop=stop,
                    stream=stream,
                    user=user,
                )
            if stream:
                return wrap_stream_with_invocation_log(
                    result,
                    invocation_log,
                    usage_reporter,
                    self._failure_stream_chunk_factory(model, prompt_messages),
                )
            result_summary = llm_result_summary(result)
            invocation_log.set_response(**result_summary)
            invocation_log.success(result_type=type(result).__name__, output_text=result_summary.get("output_text"))
            usage_reporter(getattr(result, "usage", None))
            return result
        except Exception as error:
            invocation_log.failure(error)
            if stream:
                invocation_log.flush()
            failure_text = failure_output_text(invocation_log, error)
            if stream:
                return iter([self._failure_stream_chunk_factory(model, prompt_messages)(failure_text, 0)])
            return self._failure_result(model, credentials, prompt_messages, failure_text)
        finally:
            _ACTIVE_INVOCATION_LOG.reset(active_log_token)
            if not stream:
                invocation_log.flush()

    @staticmethod
    def _failure_stream_chunk_factory(model: str, prompt_messages: list[PromptMessage]):
        def create_failure_chunk(content: str, index: int) -> LLMResultChunk:
            return LLMResultChunk(
                model=model,
                prompt_messages=prompt_messages,
                delta=LLMResultChunkDelta(
                    index=index,
                    message=AssistantPromptMessage(content=content),
                    finish_reason="stop",
                ),
            )

        return create_failure_chunk

    def _build_openai_compatible_replay_body(
        self,
        *,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]],
        stop: Optional[list[str]],
        stream: bool,
        user: Optional[str],
    ) -> dict:
        """Mirror Dify's OpenAI-compatible request serialization for replay logs."""
        parameters = deepcopy(model_parameters)
        response_format = parameters.get("response_format")
        if response_format == "json_schema":
            json_schema = parameters.get("json_schema")
            if json_schema:
                parameters.pop("json_schema", None)
                parameters["response_format"] = {
                    "type": "json_schema",
                    "json_schema": json.loads(json_schema),
                }
        elif response_format:
            parameters["response_format"] = {"type": response_format}
        else:
            parameters.pop("json_schema", None)

        body = {
            "model": credentials.get("endpoint_model_name", model),
            "stream": stream,
            **parameters,
            "messages": [
                self._convert_prompt_message_to_dict(message, credentials)
                for message in prompt_messages
            ],
        }
        if tools:
            if credentials.get("function_calling_type", "no_call") == "function_call":
                body["functions"] = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                    for tool in tools
                ]
            elif credentials.get("function_calling_type", "no_call") == "tool_call":
                body["tool_choice"] = "auto"
                body["tools"] = [PromptMessageFunction(function=tool).model_dump() for tool in tools]
        if stop:
            body["stop"] = stop
        if user:
            body["user"] = user
        return body

    def _failure_result(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        content: str,
    ) -> LLMResult:
        return LLMResult(
            model=model,
            prompt_messages=prompt_messages,
            message=AssistantPromptMessage(content=content),
            usage=self._calc_response_usage(
                model,
                credentials,
                self._num_tokens_from_messages(prompt_messages, credentials=credentials),
                self._num_tokens_from_string(content),
            ),
        )

    def _handle_generate_response(
        self,
        model: str,
        credentials: dict,
        response: requests.Response,
        prompt_messages: list[PromptMessage],
    ) -> LLMResult:
        """处理非流式响应里 tool_calls 存在但 message.content 缺失的兼容端返回。"""
        active_log = _ACTIVE_INVOCATION_LOG.get()
        if active_log is not None:
            active_log.set_response(http=http_response_summary(response))
        response_json: dict = response.json()
        completion_type = LLMMode.value_of(credentials.get("mode", "chat"))
        choices = response_json.get("choices") or []
        if not choices:
            raise InvokeError("LLM response returned no choices")

        output = choices[0]
        message_id = response_json.get("id")
        response_content = ""
        tool_calls = None
        function_calling_type = credentials.get("function_calling_type", "no_call")

        if completion_type is LLMMode.CHAT:
            message = output.get("message") or {}
            raw_content = message.get("content")
            if isinstance(raw_content, str):
                response_content = raw_content
            elif raw_content is None:
                response_content = ""
            else:
                response_content = str(raw_content)

            if function_calling_type == "tool_call":
                tool_calls = message.get("tool_calls")
            elif function_calling_type == "function_call":
                tool_calls = message.get("function_call")
        elif completion_type is LLMMode.COMPLETION:
            raw_text = output.get("text", "")
            response_content = raw_text if isinstance(raw_text, str) else str(raw_text or "")

        assistant_message = AssistantPromptMessage(content=response_content, tool_calls=[])
        if tool_calls:
            if function_calling_type == "tool_call":
                assistant_message.tool_calls = self._extract_response_tool_calls(tool_calls)
            elif function_calling_type == "function_call":
                function_call = self._extract_response_function_call(tool_calls)
                assistant_message.tool_calls = [function_call] if function_call else []

        usage = response_json.get("usage")
        if usage:
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
        else:
            prompt_tokens = self._num_tokens_from_messages(prompt_messages, credentials=credentials)
            completion_tokens = self._num_tokens_from_string(assistant_message.content or "")

        return LLMResult(
            id=message_id,
            model=response_json.get("model", model),
            message=assistant_message,
            usage=self._calc_response_usage(model, credentials, prompt_tokens, completion_tokens),
        )

    @classmethod
    @lru_cache(maxsize=1)
    def _load_predefined_chat_models(cls) -> set[str]:
        """从模型 YAML 读取预定义聊天模型名，避免 Python 列表和 YAML 重复维护。"""
        model_names: set[str] = set()
        for payload in cls._load_model_configs().values():
            if (
                payload.get("model_type") == "llm"
                and payload.get("model_properties", {}).get("mode") == "chat"
            ):
                model_name = payload.get("model")
                if isinstance(model_name, str) and model_name:
                    model_names.add(model_name)
        return model_names

    @classmethod
    @lru_cache(maxsize=1)
    def _load_model_configs(cls) -> dict[str, dict]:
        """按模型名读取全部模型 YAML 配置。"""
        models_dir = Path(__file__).resolve().parent
        configs: dict[str, dict] = {}
        for model_file in models_dir.glob("*.yaml"):
            if model_file.name.startswith("_"):
                continue
            with model_file.open("r", encoding="utf-8") as file:
                payload = yaml.safe_load(file) or {}
            model_name = payload.get("model")
            if isinstance(model_name, str) and model_name:
                configs[model_name] = payload
        return configs

    @classmethod
    def _load_model_extra(cls, model: str) -> dict:
        """读取模型 YAML 的 extra 配置。"""
        extra = cls._load_model_configs().get(model, {}).get("extra", {})
        return extra if isinstance(extra, dict) else {}
