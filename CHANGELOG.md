# Changelog

## 0.0.36

- 兼容 Dify Workflow 工具的 `tool name -> result -> output` 三层结果包装。

## 0.0.35

- 修复旧凭据缺少 `mode` 时读取模型 schema 失败，导致 Agent 不能选择 Flypower 模型的问题。

## 0.0.34

- 支持 `set_next_step` 工作流工具返回 `reasoning_effort`，并将其仅用于该工具调用后的下一次模型请求。

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
