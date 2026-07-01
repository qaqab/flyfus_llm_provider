from functools import lru_cache
from pathlib import Path
from typing import Generator, Optional, Union
from urllib.parse import urljoin

import requests
import yaml

from dify_plugin.entities.model.llm import LLMResult
from dify_plugin.entities.model.message import (
    DocumentPromptMessageContent,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageTool,
    UserPromptMessage,
)
from dify_plugin.errors.model import CredentialsValidateFailedError
from dify_plugin.interfaces.model.openai_compatible.llm import OAICompatLargeLanguageModel


class FlypowerLargeLanguageModel(OAICompatLargeLanguageModel):
    """Flypower LLM 调用适配器。

    当前插件只接入 OpenAI Chat Completions 兼容模型，所以主体能力直接复用
    Dify SDK 的 OpenAI 兼容基类。这里保留少量适配逻辑：

    - 固定走 chat 模式。
    - 默认使用新版工具调用 tool_call。
    - 转换图片和文档输入。
    - 为 GPT 5 系列使用 max_completion_tokens 参数。
    - 用轻量请求校验供应商凭据。

    文档按 OpenAI Chat Completions 的原生 file content part 传递。
    插件不把文档解码成普通文本，避免伪装成模型原生文档能力。
    """

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

    def _uses_max_completion_tokens(self, model: str) -> bool:
        """判断当前模型是否需要 OpenAI 新版 token 参数。"""
        return model.lower().startswith("gpt-5")

    def _normalize_model_parameters(self, model: str, model_parameters: dict) -> dict:
        """整理模型调用参数。

        Dify 页面上使用 max_tokens 这个通用参数名。GPT 5 系列使用
        max_completion_tokens，其他 OpenAI-compatible 模型保留 max_tokens。
        """
        normalized_parameters = dict(model_parameters)
        if (
            self._uses_max_completion_tokens(model)
            and "max_completion_tokens" not in normalized_parameters
            and "max_tokens" in normalized_parameters
        ):
            normalized_parameters["max_completion_tokens"] = normalized_parameters.pop("max_tokens")
        return normalized_parameters

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

        token_parameter = "max_completion_tokens" if self._uses_max_completion_tokens(validation_model) else "max_tokens"
        request_body = {
            "model": validation_model,
            "messages": [{"role": "user", "content": "ping"}],
            token_parameter: 16,
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

    @classmethod
    @lru_cache(maxsize=1)
    def _load_predefined_chat_models(cls) -> set[str]:
        """从模型 YAML 读取预定义聊天模型名，避免 Python 列表和 YAML 重复维护。"""
        models_dir = Path(__file__).resolve().parent
        model_names: set[str] = set()
        for model_file in models_dir.glob("*.yaml"):
            if model_file.name.startswith("_"):
                continue
            with model_file.open("r", encoding="utf-8") as file:
                payload = yaml.safe_load(file) or {}
            if (
                payload.get("model_type") == "llm"
                and payload.get("model_properties", {}).get("mode") == "chat"
            ):
                model_name = payload.get("model")
                if isinstance(model_name, str) and model_name:
                    model_names.add(model_name)
        return model_names
