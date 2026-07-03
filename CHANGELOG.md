# Changelog

## 0.0.25

- 新增 Agent 图片 URL 上下文协议支持。
- 支持 `image_context_refresher` 工具返回 `<DIFY_IMAGE_CONTEXT>...</DIFY_IMAGE_CONTEXT>` 后，
  自动把其中的图片 URL 注入为当前轮多模态图片输入。
- 支持单 URL、多 URL、JSON 列表、对象数组、多行文本、逗号分隔文本和 Markdown 链接。
- 支持多图片一次注入，并对重复 URL 做去重。
- 插件只透传 URL，不下载图片、不转 base64，由上游多模态模型或聚合网关读取图片。
- 协议只在 Agent 工具返回固定标记时生效，不影响普通 LLM 调用和其他工具。
- 过滤裸文件名和非法图片引用，减少上游 `invalid image_url` 报错。
- 将 Agent 图片上下文逻辑拆分到 `models/llm/agent_image_context.py`，降低 `llm.py` 复杂度。

## 0.0.24

- 修复 Dify Cloud 插件运行时不支持 `use_template: enable_thinking` 导致的模型 schema 解析失败。
- 将 thinking 相关参数改为普通自定义参数，并按模型 YAML 的 `extra.thinking.mode` 决定是否发送给上游。
