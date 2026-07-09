import base64

from dify_plugin.errors.model import InvokeError


def model_family(model: str) -> str:
    model_name = model.lower()
    if model_name == "high" or model_name.startswith("gpt-"):
        return "openai_responses"
    if model_name.startswith("gemini-"):
        return "gemini"
    return "openai_compatible"


def file_bytes(content: object) -> bytes:
    base64_data = getattr(content, "base64_data", "")
    if not base64_data:
        raise InvokeError("文件上传需要 Dify 提供 base64_data，URL 文件输入暂未实现。")
    return base64.b64decode(base64_data)
