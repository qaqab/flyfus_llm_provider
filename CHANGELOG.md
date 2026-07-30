# Changelog

## 0.0.15

- Normalize boolean `false` schemas before Gemini function calls.

## 0.0.14

- Remove `additionalProperties` from Gemini function schemas for compatibility with Gemini 3.1 Pro.

## 0.0.13

- Remove the unavailable `gemini-3-pro-preview` model from the catalog.

## 0.0.12

- Preserve `additionalProperties` in Gemini function schemas after compatibility testing.

## 0.0.11

- Normalize runtime Agent tool schemas before Gemini native function calls, including nullable array types and boolean parameter schemas.

## 0.0.3

- 调用日志新增可重放的上游请求 Body，覆盖 Responses、Gemini 和 OpenAI-compatible Chat Completions。
- 重放记录不保存 API Key；使用当前凭据重发即可。

## 0.0.1

- Initial Flyfus LLM Provider release.
- Includes OpenAI-compatible, Responses API, Gemini, usage reporting, invocation logging, and Flyfus URL context support.
- Published as a new plugin; configure credentials in Dify from scratch.

## 0.0.36

- 兼容 Dify Workflow 工具的 `tool name -> result -> output` 三层结果包装。

## 0.0.35

- 修复旧凭据缺少 `mode` 时读取模型 schema 失败，导致 Agent 不能选择 Flyfus 模型的问题。
- `log_context` 新增 `conversation_id`，并与 `log_id` 同级写入 SLS。
- 将 Dify 模型调用的 `user` 参数与 `log_id` 同级写入 SLS。
- Token 用量上报新增完整日志上下文字段；所有字段缺失时传空字符串，不再按用户格式过滤。

## 0.0.34

- 支持 `set_next_step` 工作流工具返回 `reasoning_effort`，并将其仅用于该工具调用后的下一次模型请求。
- 将 AI Mode 改为通过 JSON `FLYFUS_SETTING` 解析，并在模型调用前删除整个 Setting 块。
- 支持从最新 User 消息的 `log_context` Setting 解析 `user_id`、`app_id`、`workflow_id` 和 `workflow_run_id`。
- 将四个日志字段与 `log_id` 同级写入 SLS；缺失字段记录为空字符串。
- 调用日志新增 Dify Session 运行时上下文，便于核对消息声明字段与实际工作流字段。

## 0.0.27

- GPT/OpenAI 系列模型改走 Responses API，Gemini 和其他模型继续走 chat 路径。
- GPT/Responses 支持 Dify 原生文档附件：本地上传文档会先上传到 OpenAI Files，再通过 `input_file.file_id` 传入模型。
- 新增统一 Agent URL 上下文协议 `<DIFY_CONTEXT>...</DIFY_CONTEXT>`，用于工作流 Agent 节点或工具产物把外部图片/文件 URL 传给模型。
- `read_files` 简化为纯 URL 上下文工具，只接收可被模型服务访问的 URL，不接收文件内容、不下载、不上传文件。
- `read_files` 兼容单个 URL 字符串和 URL 字符串数组，降低 Agent 工具调用参数格式错误率。
- 工具 URL 图片转为 `input_image.image_url`；工具 URL 文件只在 GPT/Responses 路径转为 `input_file.file_url`。
- Chat App 用户直接上传图片/文档时优先走 Dify 原生附件能力，不需要调用 `read_files`。
- 收紧 URL 协议：文件只接受公网 `http/https`，图片接受公网 `http/https` 或 `data:image/...`，忽略 Dify 内部地址和本地地址。
- 重构 Agent 上下文逻辑到 `models/llm/agent_context.py`，删除旧的图片专用协议实现。
- 精简 README 和上下文协议文档，保留当前架构和发布前验证方式。
- GitHub 发布脚本会自动读取对应版本的 `CHANGELOG.md` 作为 Release Notes。

## 0.0.26

- GitHub Release 过渡版本。
- 后续版本从 `CHANGELOG.md` 读取发布说明。

## 0.0.25

- 新增 Agent 图片 URL 上下文协议支持。
- 支持工具返回 `<DIFY_IMAGE_CONTEXT>...</DIFY_IMAGE_CONTEXT>` 后，自动把其中的图片 URL 注入为当前轮多模态图片输入。
- 支持单张图片和多张图片，并对重复 URL 做去重。
- 插件只透传 URL，不下载图片、不转 base64。
- 协议只在 Agent 工具返回固定标记时生效，不影响普通 LLM 调用和其他工具。
- 将 Agent 图片上下文逻辑拆分到 `models/llm/agent_image_context.py`，降低 `llm.py` 复杂度。

## 0.0.24

- 修复 Dify Cloud 插件运行时不支持 `use_template: enable_thinking` 导致的模型 schema 解析失败。
- 将 thinking 相关参数改为普通自定义参数，并按模型 YAML 的 `extra.thinking.mode` 决定是否发送给上游。
