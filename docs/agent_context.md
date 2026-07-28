# Agent URL Context

`read_files` 是纯 URL 上下文工具，用来把工作流 Agent 节点或其他工具产出的图片/文件 URL 传给 GPT/Responses。

Chat App 里用户直接上传的图片和文档走 Dify 原生附件能力，不需要调用这个工具。

## 协议

工具输出里必须包含：

```text
<FLYFUS_CONTEXT>
image1: "https://example.com/a.png"
file1: "https://example.com/a.xlsx"
</FLYFUS_CONTEXT>
```

字段：

- 支持 `FLYFUS_CONTEXT`、`FLYFUS_FILE`、`FLYFUS_COMPONENT` 三种标签。
- 每种标签内部都按正则扫描公网 URL，不要求 JSON 或字段名。
- 标签内部可以是普通文本、JSON、列表或它们的组合。
- 图片支持 PNG、JPG、JPEG；文件支持 PDF、MD、XLSX、CSV、TXT、HTML。
- 同一次模型调用中，相同 URL 只会作为一个附件注入。

示例：

```text
<FLYFUS_CONTEXT>{"data":["https://example.com/a.png","https://example.com/a.xlsx"]}</FLYFUS_CONTEXT>
```

`read_file` 工具接收逗号或换行分隔的 URL，并输出标签内的换行 URL 列表。

## URL 规则

- 图片和文件都只支持公网 `http://`、`https://` URL。
- 不支持本地文件、Dify 内部 `files/...`、`localhost`、`api`、`web`、`nginx` 等内部地址。
- 插件不下载、不上传、不保存 URL 内容。

## 模型路由

- GPT/OpenAI 系列走 Responses API。
- 工具里的图片 URL 会转成 `input_image.image_url`。
- 工具里的文件 URL 只在 GPT/Responses 路径注入，并转成 `input_file.file_url`。
- Gemini 和其他 chat 模型不注入文件 URL。
