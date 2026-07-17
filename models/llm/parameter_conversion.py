"""生成参数的公共规范化与工具转换。

Dify 的模型页面希望使用同一套易理解的参数名，但中转站和各上游 API 对参数的
接受范围并不一致。本模块只处理所有调用路径共享的转换规则；模型特有的思考
参数转换位于 ``models.llm.llm.FlypowerLargeLanguageModel``，避免在适配器中
重复同一份分支逻辑。
"""

from typing import Optional


WEB_SEARCH_MODELS = frozenset(
    {
        "grok-4.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.5",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
    }
)


def normalize_generation_parameters(model: str, parameters: dict) -> None:
    """在路由到 Chat Completions 或 Responses API 前规范化通用参数。

    ``parameters`` 是调用方传入字典的副本，会被原地修改。调用顺序固定为温度、
    Top P、回复格式：三者当前互不依赖，但集中在此入口可以保证所有上游路径都
    获得一致结果。模型专属字段，例如思考预算、联网搜索和工具调用，不在本方法
    内处理，原因是它们需要知道后续选择的协议和模型 YAML 配置。
    """
    normalize_temperature(model, parameters)
    normalize_top_p(model, parameters)
    normalize_response_format(parameters)


def normalize_temperature(model: str, parameters: dict) -> None:
    """处理 GPT 系列对 ``temperature`` 的代理兼容要求。

    页面允许用户调节温度，但当前中转站的 GPT Responses 路由只接受 ``1``；传入
    其他数值会被上游拒绝。因此只要模型 ID 以 ``gpt-`` 开头且调用方传了温度，
    就强制改为 ``1``。非 GPT 模型保留调用方给出的数值，也不主动添加缺失字段，
    让供应商自己的默认值继续生效。
    """
    if model.lower().startswith("gpt-") and "temperature" in parameters:
        parameters["temperature"] = 1


def normalize_top_p(model: str, parameters: dict) -> None:
    """保留 ``top_p`` 的公共转换入口，当前不改变其数值。

    目前已确认的模型可直接接收 ``top_p``，因此这里是有意的 no-op。保留独立方法
    而不是在入口直接跳过，目的是将来出现某个模型要求范围裁剪、字段改名或与温度
    互斥时，只需要在此处添加规则，不必修改所有调用协议。参数不存在时不创建它。
    ``model`` 当前未使用，但与其他通用转换方法保持相同签名，方便后续加入模型
    特例而不改变调用方。
    """
    if "top_p" in parameters:
        parameters["top_p"] = parameters["top_p"]


def normalize_response_format(parameters: dict) -> None:
    """限制回复格式为插件当前能安全转换的三个公共值。

    Dify 会把页面选择传为 ``response_format``。本插件只支持 ``text``、
    ``json_object`` 和 ``json_schema``：Responses 适配器会将后两者转换为
    ``text.format``，Gemini 适配器会转换为 ``responseMimeType`` / ``responseSchema``。
    若收到其它值，直接移除字段并使用上游默认文本输出，避免把非法枚举传到中转站
    导致整次请求失败。JSON Schema 的具体内容由 ``json_schema`` 参数承载，不在
    此处解析或校验。
    """
    response_format = parameters.get("response_format")
    if response_format in {"text", "json_object", "json_schema"}:
        return
    if response_format is not None:
        parameters.pop("response_format")


def normalize_max_tokens(parameters: dict, target_parameter_name: str) -> None:
    """把页面统一的 ``max_tokens`` 按模型要求改名。

    前端始终使用 ``max_tokens``，以免每个模型显示不同名称。大多数上游也接受该
    字段；只有模型 YAML 的 ``extra.token_param_name`` 明确要求
    ``max_completion_tokens`` 时才移动字段。目标字段若已存在则尊重它，不覆盖调用
    方或其他转换步骤已设置的值。此函数只改字段名，不改变 token 数值，也不推断
    思考 token 与输出 token 的分配。
    """
    if target_parameter_name != "max_tokens" and "max_tokens" in parameters and target_parameter_name not in parameters:
        parameters[target_parameter_name] = parameters.pop("max_tokens")


def build_web_search_tool(model: str, parameters: dict) -> Optional[dict]:
    """根据显式开关为支持的 Responses 模型构造联网搜索工具。

    ``WEB_SEARCH_MODELS`` 是已通过中转站验证的白名单。只有模型在白名单内且
    ``enable_web_search is True`` 时才返回 ``{"type": "web_search"}``；``False``、
    参数缺失、字符串 ``"true"`` 或不支持的模型一律返回 ``None``。严格比较真正
    的布尔 ``True`` 可避免页面默认值、旧配置或脚本字符串意外开启外部搜索。

    返回值由 Responses 适配器追加到请求的 ``tools`` 数组中。此方法不执行搜索，
    也不把任意 URL 自动抓取；是否调用工具仍由上游模型在 ``tool_choice: auto``
    下根据问题自行决定。
    """
    if model.lower() not in WEB_SEARCH_MODELS or parameters.get("enable_web_search") is not True:
        return None
    return {"type": "web_search"}
