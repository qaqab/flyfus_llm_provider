# Agent URL Context

`read_files` 是纯 URL 上下文工具，用来把工作流 Agent 节点或其他工具产出的图片/文件 URL 传给 GPT/Responses。

Chat App 里用户直接上传的图片和文档走 Dify 原生附件能力，不需要调用这个工具。

## 协议

工具输出里必须包含：

```text
<DIFY_CONTEXT>{"version":1,"type":"dify_context","images":[],"files":[]}</DIFY_CONTEXT>
```

字段：

- `type` 必须是 `dify_context`。
- `images` 和 `files` 是对象数组。
- 每个对象必须有 `url`。
- 图片可选 `filename`、`mime_type`、`detail`。
- 文件可选 `filename`、`mime_type`。

示例：

```json
{
  "version": 1,
  "type": "dify_context",
  "images": [
    {
      "url": "https://example.com/a.png",
      "detail": "high"
    }
  ],
  "files": [
    {
      "url": "https://example.com/a.xlsx",
      "filename": "a.xlsx",
      "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    }
  ]
}
```

`read_files` 推荐接收上面的 JSON 对象。为了降低 Agent 调用失败率，也兼容单个 URL 字符串和 URL 字符串数组。

## URL 规则

- 图片支持公网 `http://`、`https://`，也支持 `data:image/...`。
- 文件只支持公网 `http://`、`https://`。
- 不支持本地文件、Dify 内部 `files/...`、`localhost`、`api`、`web`、`nginx` 等内部地址。
- 插件不下载、不上传、不保存 URL 内容。

## 模型路由

- GPT/OpenAI 系列走 Responses API。
- 工具里的图片 URL 会转成 `input_image.image_url`。
- 工具里的文件 URL 只在 GPT/Responses 路径注入，并转成 `input_file.file_url`。
- Gemini 和其他 chat 模型不注入文件 URL。
