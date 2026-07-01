from collections.abc import Mapping

from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import CredentialsValidateFailedError
from dify_plugin import ModelProvider


class FlypowerLLMProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """校验供应商级凭据。"""
        if not credentials.get("endpoint_url"):
            raise CredentialsValidateFailedError("请填写 API 地址")
        if not credentials.get("api_key"):
            raise CredentialsValidateFailedError("请填写 API Key")

        try:
            model_instance = self.get_model_instance(ModelType.LLM)
            model_instance.validate_credentials("gpt-5.4", dict(credentials))
        except CredentialsValidateFailedError:
            raise
        except Exception as error:
            raise CredentialsValidateFailedError(f"凭据校验失败：{error}") from error
