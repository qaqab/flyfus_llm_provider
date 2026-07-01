from functools import lru_cache
from pathlib import Path
from typing import Generator, Optional, Union
from urllib.parse import urljoin

import requests
import yaml

from dify_plugin.entities.model.llm import LLMMode, LLMResult
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    AudioPromptMessageContent,
    DocumentPromptMessageContent,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageTool,
    UserPromptMessage,
    VideoPromptMessageContent,
)
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeError
from dify_plugin.interfaces.model.openai_compatible.llm import OAICompatLargeLanguageModel


class FlypowerLargeLanguageModel(OAICompatLargeLanguageModel):
    """Flypower LLM 调用适配器。

    当前插件只接入 OpenAI Chat Completions 兼容模型，所以主体能力直接复用
    Dify SDK 的 OpenAI 兼容基类。这里保留少量适配逻辑：

    - 固定走 chat 模式。
    - 默认使用新版工具调用 tool_call。
    - 转换图片和文档输入。
    - 统一使用 max_completion_tokens 参数。
    - 按模型 YAML 可选适配 thinking 开关。
    - 用轻量请求校验供应商凭据。

    文档按 OpenAI Chat Completions 的原生 file content part 传递。
    插件不把文档解码成普通文本，避免伪装成模型原生文档能力。
    """

    def _wrap_thinking_by_reasoning_content(self, delta: dict, is_reasoning: bool) -> tuple[str, bool]:
        """把兼容端返回的 reasoning/reasoning_content 包成 Dify 可识别的 <think> 块。"""
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

    def _request_headers(self, credentials: dict) -> dict:
        """构造上游请求头。"""
        headers = {"Content-Type": "application/json"}
        if credentials.get("api_key"):
            headers["Authorization"] = f"Bearer {credentials['api_key']}"
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
        """整理模型调用参数。

        Dify 页面上使用 max_tokens 这个通用参数名。调用上游时统一转换为
        max_completion_tokens，保持请求参数一致。
        """
        normalized_parameters = dict(model_parameters)
        if (
            "max_completion_tokens" not in normalized_parameters
            and "max_tokens" in normalized_parameters
        ):
            normalized_parameters["max_completion_tokens"] = normalized_parameters.pop("max_tokens")

        self._apply_thinking_parameters(model, normalized_parameters)
        return normalized_parameters

    def _apply_thinking_parameters(self, model: str, model_parameters: dict) -> None:
        """按模型 YAML 中的 extra.thinking 映射非标准思考参数。"""
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
        if enable_thinking is not None:
            enabled = bool(enable_thinking)
            if mode == "top_level":
                model_parameters["enable_thinking"] = enabled
            elif mode == "deepseek":
                model_parameters["thinking"] = {"type": "enabled" if enabled else "disabled"}
            elif mode == "openrouter":
                model_parameters.setdefault("reasoning", {})["enabled"] = enabled
            elif mode == "zhipu":
                model_parameters["thinking"] = {"type": "enabled" if enabled else "disabled"}
            elif mode == "chat_template_kwargs":
                template_kwargs = model_parameters.setdefault("chat_template_kwargs", {})
                template_kwargs["enable_thinking"] = enabled
                template_kwargs["thinking"] = enabled
            elif mode == "minimax":
                minimax_thinking = self._minimax_thinking_payload(enabled, thinking_budget)
                if minimax_thinking:
                    model_parameters["thinking"] = minimax_thinking

        if thinking_budget is not None:
            if mode == "top_level":
                model_parameters["thinking_budget"] = thinking_budget
            elif mode == "openrouter":
                model_parameters.setdefault("reasoning", {})["max_tokens"] = thinking_budget
            elif mode == "gemini":
                model_parameters.setdefault("thinking_config", {})["thinking_budget"] = thinking_budget
            elif mode == "minimax" and "thinking" not in model_parameters:
                model_parameters["thinking"] = self._minimax_thinking_payload(True, thinking_budget)

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

        if thinking_config.get("reasoning_effort_target") == "chat_template_kwargs":
            reasoning_effort = model_parameters.get("reasoning_effort")
            if reasoning_effort is not None:
                model_parameters.setdefault("chat_template_kwargs", {})["reasoning_effort"] = reasoning_effort
        elif mode == "openrouter":
            reasoning_effort = model_parameters.pop("reasoning_effort", None)
            if reasoning_effort in {"high", "medium", "low", "minimal", "none"}:
                model_parameters.setdefault("reasoning", {})["effort"] = reasoning_effort

    @staticmethod
    def _minimax_thinking_payload(enabled: bool, thinking_budget: Optional[int]) -> Optional[dict]:
        """构造 MiniMax Anthropic-compatible thinking 参数。"""
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
            "max_completion_tokens": 16,
            "stream": False,
        }

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
        return super()._invoke(
            model=model,
            credentials=self._normalize_credentials(model, credentials),
            prompt_messages=prompt_messages,
            model_parameters=self._normalize_model_parameters(model, model_parameters),
            tools=tools,
            stop=stop,
            stream=stream,
            user=user,
        )

    def _handle_generate_response(
        self,
        model: str,
        credentials: dict,
        response: requests.Response,
        prompt_messages: list[PromptMessage],
    ) -> LLMResult:
        """处理非流式响应里 tool_calls 存在但 message.content 缺失的兼容端返回。"""
        response_json: dict = response.json()
        completion_type = LLMMode.value_of(credentials["mode"])
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
