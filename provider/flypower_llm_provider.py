from collections.abc import Mapping

from dify_plugin import ModelProvider


class FlypowerLLMProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """校验供应商级凭据。"""
        if not credentials.get("endpoint_url"):
            raise ValueError("请填写 API 地址")
        if not credentials.get("api_key"):
            raise ValueError("请填写 API Key")
