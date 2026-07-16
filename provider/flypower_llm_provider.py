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
        required_credentials = {
            "sls_endpoint": "请填写 SLS 地址",
            "sls_project": "请填写 SLS 项目",
            "sls_access_key_id": "请填写 SLS AccessKey ID",
            "sls_access_key_secret": "请填写 SLS AccessKey Secret",
            "geo_prompt_render_url": "请填写 Geo Prompt 渲染地址",
            "geo_prompt_api_key": "请填写 Geo Prompt API Key",
        }
        for name, error_message in required_credentials.items():
            if not str(credentials.get(name) or "").strip():
                raise CredentialsValidateFailedError(error_message)

        try:
            model_instance = self.get_model_instance(ModelType.LLM)
            model_instance.validate_credentials("gpt-5.4", dict(credentials))
        except CredentialsValidateFailedError:
            raise
        except Exception as error:
            raise CredentialsValidateFailedError(f"凭据校验失败：{error}") from error
