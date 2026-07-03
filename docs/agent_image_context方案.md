# Agent 图片 URL 上下文方案

## 目标

让 Dify Agent 在多轮对话中按需重新读取图片 URL，而不是把图片摘要写死到历史里，也不在每一轮都重复携带图片。

## 当前实现

需要改动两个位置：

- LLM Provider 插件：识别工具返回的图片上下文协议，并注入为 `ImagePromptMessageContent`。
- Dify 工具或工作流工具：把图片 URL 包装成固定协议返回给 Agent。

LLM 插件只在 `ToolPromptMessage` 里发现下面的固定标记时生效：

```text
<DIFY_IMAGE_CONTEXT>{"version":1,"type":"dify_image_context","images":[{"url":"https://example.com/a.png","detail":"auto"}]}</DIFY_IMAGE_CONTEXT>
```

推荐多图片格式：

```json
{
  "version": 1,
  "type": "dify_image_context",
  "images": [
    {
      "url": "https://example.com/a.png",
      "mime_type": "image/png",
      "detail": "auto"
    },
    {
      "url": "https://example.com/b.jpg",
      "mime_type": "image/jpeg",
      "detail": "auto"
    }
  ]
}
```

也兼容 `url`、`image_ref`、`image_refs`，以及 Markdown 链接、多行 URL、逗号分隔 URL 和 JSON URL 列表。

## 运行流程

1. 用户上传图片，或工作流提前把文件转换成模型可访问的 URL。
2. Agent 判断需要看图时，调用 `image_context_refresher` 工具。
3. 工具返回 `<DIFY_IMAGE_CONTEXT>...</DIFY_IMAGE_CONTEXT>`。
4. LLM 插件在调用模型前解析该协议，把图片 URL 注入成 `ImagePromptMessageContent`。
5. 上游多模态模型或聚合网关按 URL 读取图片并回答。

## 边界

- 插件不下载图片。
- 插件不把图片转 base64。
- 插件不保存图片。
- 插件不处理普通文本里的 URL，必须有 `<DIFY_IMAGE_CONTEXT>` 固定标记。
- 裸文件名会被忽略，因为模型接口无法直接读取本地文件名。

## 本地测试

1. 打包并安装 `0.0.25` 版本插件。
2. 在 Dify 里创建或导入图片上下文工具，工具输出固定协议。
3. Agent 节点选择支持 vision 的模型，例如 `gpt-5.5` 或 `qwen3-vl-flash`。
4. 首轮输入两张图片 URL 和问题。
5. 第二轮追问“这些图片都有品牌吗”。
6. 正常结果是 Agent 会调用工具刷新图片上下文，模型能重新读取图片并回答每张图的细节。

## 期望效果

- 首轮能识别图片内容。
- 多轮追问时，不依赖历史摘要，仍能重新看原图。
- 多张图片会作为多张图片传入模型，不会被拼成一个非法 URL。
- 普通聊天和非图片工具不受影响。
