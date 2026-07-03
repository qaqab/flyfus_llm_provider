# Flypower 大模型供应商插件

这是一个 Dify LLM Provider 插件，用于接入 OpenAI-compatible
`/chat/completions` 聚合接口。

## 当前能力

- 单个供应商、单套 `API 地址` 和 `API Key`。
- 只支持预定义模型，不开放自定义模型。
- 统一走 `/chat/completions`。
- 保存凭据时会先读取上游 `/models`，确认密钥至少包含一个本插件支持的聊天模型。
- 支持流式工具调用；普通调用是否流式由 Dify 调用场景决定。
- 默认使用新版工具调用 `tool_call`。
- 不支持旧版函数调用 `function_call`。
- 图片和文档能力由各模型 YAML 的 `features` 声明控制。
- 文档只走 OpenAI Chat Completions 原生 `file_data` 协议，不做插件内抽文本兜底。
- 支持 Dify 结构化输出参数：`response_format`、`json_schema`。
- 支持 `reasoning_effort` 推理强度参数；非标准 thinking 开关按模型 YAML 显式配置。
- 支持 Agent 工具返回的图片 URL 上下文协议，只在工具返回固定协议时注入图片输入。

## 0.0.25 更新内容

- 新增 Agent 图片 URL 上下文协议支持：识别 `image_context_refresher` 工具返回的
  `<DIFY_IMAGE_CONTEXT>...</DIFY_IMAGE_CONTEXT>`。
- 支持单张图片和多张图片：`url`、`image_ref`、`image_refs`、`images` 列表都可以解析。
- 支持工具输入里的多种 URL 写法：单个 URL、多行 URL、逗号分隔 URL、JSON URL 列表、
  JSON 对象数组、Markdown 链接。
- 插件会把解析到的图片 URL 注入为 Dify `ImagePromptMessageContent`，再交给
  OpenAI-compatible 模型接口处理。
- 插件不下载图片、不把图片转 base64；图片仍由上游多模态模型或聚合网关按 URL 读取。
- 只处理 `ToolPromptMessage` 里的固定协议，不影响普通聊天、普通 LLM 节点、其他工具返回、
  文档输入或已有多模态输入。
- 增加 URL 过滤和去重：裸文件名、空值、非法格式不会传给模型，避免上游报
  `invalid image_url`。
- Agent 图片上下文逻辑已拆到 `models/llm/agent_image_context.py`，`llm.py` 只保留调用入口。

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

## 多模态输入

模型 YAML 可以通过 `features` 声明支持：

- `vision`：图片
- `video`：视频
- `audio`：音频
- `document`：文档

当前插件只在已核对到上游能力的模型上暴露视频/音频：

| 模型 | 多模态能力 |
| --- | --- |
| `gemini-2.5-flash` | 图片、文档、视频、音频 |
| `gemini-2.5-pro` | 图片、文档、视频、音频 |
| `gemini-3-flash-preview` | 图片、文档、视频、音频 |
| `gemini-3-pro-preview` | 图片、文档、视频、音频 |
| `gemini-3.1-pro-preview` | 图片、文档、视频、音频 |
| `gemini-3.5-flash` | 图片、文档、视频、音频 |
| `qwen3-vl-flash` | 图片、视频 |
| `gpt-5.4` / `gpt-5.4-mini` / `gpt-5.5` | 图片、文档；不声明视频/音频 |

不要仅因为模型支持图片就顺手加 `video` 或 `audio`。新增模型时先核对上游
模型表；没有明确写音频输入时，不要暴露 `audio`。

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

插件只负责把 Dify 传入的 `data` 透传为 `file_data`。
插件不解析文档、不抽取文本、不把文件正文拼进 prompt。
如果某个上游模型不支持文档输入，删除对应模型 YAML 里的 `document` 即可。

视频和音频会按 OpenAI-compatible 聚合端常见写法放入：

```json
{
  "type": "image_url",
  "image_url": {
    "url": "data:video/mp4;base64,..."
  }
}
```

如果上游模型或聚合端不支持视频/音频输入，删除对应模型 YAML 里的
`video` 或 `audio` 即可。

## 开发约定

- 新增模型时，优先新增 `models/llm/*.yaml` 文件。
- 如果模型也是 Chat Completions 兼容，一般不需要改 `models/llm/llm.py`。
- 如果模型不支持原生 `json_schema`，后续可以按模型增加结构化输出策略。
- 注释、报错和文档使用中文；代码标识符使用英文。

## 思考与推理参数

- `agent-thought` 只声明模型支持思考/推理内容展示，不会额外向上游发送私有参数。
- `reasoning_effort` 是 reasoning 风格参数，界面文案按模型语境显示为推理强度或思考强度。
- `enable_thinking`、`thinking`、`thinking_budget`、`thinking_level`、`include_thoughts`
  都是非标准参数，只有模型 YAML 显式声明 `extra.thinking.mode` 时才会发送。
- 如果某个模型确实需要 thinking 开关，在该模型 YAML 里增加 `extra.thinking`：

```yaml
extra:
  thinking:
    mode: top_level
```

当前支持的 `mode`：

- `top_level`：发送 `enable_thinking: true/false`
- `deepseek`：发送 `thinking: {"type": "enabled"|"disabled"}`
- `gemini`：发送 `thinking_config.thinking_budget`、`thinking_config.thinking_level`、`thinking_config.include_thoughts`
- `minimax`：发送 `thinking: {"type": "enabled", "budget_tokens": N}`
- `openrouter`：发送 `reasoning.enabled`、`reasoning.max_tokens`、`reasoning.effort`、`reasoning.exclude`
- `zhipu`：发送 `thinking: {"type": "enabled"|"disabled"}`
- `chat_template_kwargs`：发送 `chat_template_kwargs.enable_thinking` 和 `chat_template_kwargs.thinking`

如果某个模型还要求把 `reasoning_effort` 同步写入 `chat_template_kwargs`：

```yaml
extra:
  thinking:
    mode: chat_template_kwargs
    reasoning_effort_target: chat_template_kwargs
```

当前预设模型暴露的参数分组和中文语义：

- `gpt-5.*`：`reasoning_effort`，显示为推理强度
- `deepseek-v4-*`：`thinking` 显示为思考模式，`reasoning_effort` 显示为思考强度
- `gemini-2.5-*`：`thinking_budget` 显示为思考预算，`include_thoughts` 显示为返回思考过程
- `gemini-3*`：`thinking_level` 显示为思考层级，`include_thoughts` 显示为返回思考过程
- `qwen3*`、`glm-5*`、`kimi-k2.5`：`enable_thinking` 显示为思考模式，`thinking_budget` 显示为思考预算
- `minimax-m2.5`、`MiniMax-M2.7`：`enable_thinking` 显示为思考模式，`thinking_budget` 显示为思考预算

注意：这里说的是本插件当前 YAML 暴露的接口参数，不是断言模型能力本质上
只有“思考”或只有“推理”。聚合商或 OpenRouter 风格模型可能同时暴露
`enable_thinking`、`reasoning_budget`、`reasoning_effort`、
`exclude_reasoning_tokens` 等参数；新增这类模型时应按实际上游接口单独配置。

如果上游流式响应里返回 `reasoning` 或 `reasoning_content` 字段，
插件会包装为：

```text
<think>
...
</think>
```

这样 Dify 可以按思考内容处理和展示。非流式响应里，如果模型返回
`tool_calls` 但省略 `message.content`，插件会把 content 当作空字符串处理，
避免兼容端触发 `KeyError('content')`。
