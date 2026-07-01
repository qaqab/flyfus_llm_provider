# Flypower 大模型供应商插件

这是一个 Dify LLM Provider 插件，用于接入 OpenAI-compatible
`/chat/completions` 聚合接口。

## 当前能力

- 单个供应商、单套 `API 地址` 和 `API Key`。
- 只支持预定义模型，不开放自定义模型。
- 统一走 `/chat/completions`。
- 保存凭据时会先读取上游 `/models`，确认密钥至少包含一个本插件支持的聊天模型。
- 默认开启流式输出。
- 默认使用新版工具调用 `tool_call`。
- 不支持旧版函数调用 `function_call`。
- 图片和文档能力由各模型 YAML 的 `features` 声明控制。
- 文档只走 OpenAI Chat Completions 原生 `file_data` 协议，不做插件内抽文本兜底。
- 支持 Dify 结构化输出参数：`response_format`、`json_schema`。

## 预设模型

模型定义在 `models/llm/`。当前按 Flyposter `/models` 返回结果整理，
排除了 `gpt-image-*` 图片模型和 `codex-auto-review`。

当前 LLM 模型：

- `MiniMax-M2.7`
- `deepseek-v4-flash`
- `deepseek-v4-pro`
- `gemini-2.5-flash`
- `gemini-2.5-pro`
- `gemini-3-flash-preview`
- `gemini-3-pro-preview`
- `gemini-3.1-pro-preview`
- `gemini-3.5-flash`
- `glm-5`
- `glm-5.1`
- `glm-5.2`
- `gpt-5.4`
- `gpt-5.4-mini`
- `gpt-5.5`
- `kimi-k2.5`
- `kimi-k2.7-code`
- `minimax-m2.5`
- `qwen3-coder-next`
- `qwen3-max`
- `qwen3-vl-flash`
- `qwen3.5-flash`
- `qwen3.5-plus`
- `qwen3.6-flash`
- `qwen3.6-max-preview`
- `qwen3.6-plus`
- `qwen3.7-max`
- `qwen3.7-plus`

## 凭据

在 Dify 的“管理凭据”里配置：

- `API 地址`：OpenAI-compatible Base URL，例如 `https://openapi.flyposter.ai/v1`
- `API Key`：聚合接口密钥

凭据保存时，插件会调用：

```text
GET {API 地址}/models
```

如果返回列表里没有任何本插件支持的聊天模型，保存会失败。

注意：Dify 的预定义模型列表是静态 YAML。即使某个 key 只开放部分模型，
页面仍可能显示全部预定义模型；调用缺失模型时，上游会返回错误。若要按
key 动态只展示可用模型，需要改成动态模型/自定义模型方案。

## 图片和文档输入

模型 YAML 可以通过 `features` 声明支持：

- `vision`：图片
- `document`：文档

文档会转换成 OpenAI Chat Completions 原生文件输入格式：

```json
{
  "type": "file",
  "file": {
    "filename": "document.pdf",
    "file_data": "data:application/pdf;base64,..."
  }
}
```

插件只负责把 Dify 传入的 `base64_data` 整理成 `file_data`。
插件不解析文档、不抽取文本、不把文件正文拼进 prompt。
如果某个上游模型不支持文档输入，删除对应模型 YAML 里的 `document` 即可。

## 开发约定

- 新增模型时，优先新增 `models/llm/*.yaml` 文件。
- 如果模型也是 Chat Completions 兼容，一般不需要改 `models/llm/llm.py`。
- 如果模型不支持原生 `json_schema`，后续可以按模型增加结构化输出策略。
- 注释、报错和文档使用中文；代码标识符使用英文。


