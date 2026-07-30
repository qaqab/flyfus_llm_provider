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

- `FLYFUS_CONTEXT` 和 `FLYFUS_FILE` 返回给前端，同时保留在模型内容中。
- `FLYFUS_INTERNAL_CONTEXT` 预留给不返回前端、但保留在模型内容中的场景；插件暂不从中提取附件。
- `FLYFUS_SETTING` 不作为模型内容，只用于模型参数和日志字段解析。
- `FLYFUS_CONTEXT` 和 `FLYFUS_FILE` 内部都按正则扫描公网 URL，不要求 JSON 或字段名。
- 这两个附件标签内部可以是普通文本、JSON、列表或它们的组合。
- 图片支持 PNG、JPG、JPEG；文件支持 PDF、MD、XLSX、CSV、TXT、HTML。
- 同一次模型调用中，相同 URL 只会作为一个附件注入。

示例：

```text
<FLYFUS_CONTEXT>{"data":["https://example.com/a.png","https://example.com/a.xlsx"]}</FLYFUS_CONTEXT>
```

`read_file` 工具接收逗号或换行分隔的 URL，并输出标签内的换行 URL 列表。

## AI Mode

User 或 Tool 消息使用 `type: "ai_mode"` 选择 AI Mode：

```text
<FLYFUS_SETTING>
{"type":"ai_mode","reference":"{{dify_admin:ai_mode.listing_analysis.fast}}"}
</FLYFUS_SETTING>
```

插件使用最后出现的引用，请求 `POST /dify_admin/ai_mode/resolve_reference`，请求体为
`{"reference":"{{dify_admin:ai_mode.listing_analysis.fast}}"}`。接口返回的
模型和生成参数取自 `data.config`，整个 `FLYFUS_SETTING` 块在发送给大模型前会被删除。
即使解析接口失败，`FLYFUS_SETTING` 也会被删除，并继续使用原模型。
包含 Setting 的消息在删除标签块后，还会清理整条消息首尾的空格和换行。

## 日志上下文

User 消息使用 `type: "log_context"` 写入调用日志：

```text
<FLYFUS_SETTING>
{"type":"log_context","user_id":"user-1","app_id":"app-1","workflow_id":"workflow-1","workflow_run_id":"run-1","conversation_id":"conversation-1"}
</FLYFUS_SETTING>
```

`user_id`、`app_id`、`workflow_id`、`workflow_run_id`、`conversation_id` 会在 SLS 中与 `log_id`
同级记录，并同时写入 `event_json` 顶层。缺少或不是字符串的字段记录为空字符串。
模型调用原有的 Dify `user` 参数也会与 `log_id` 同级记录，与 `user_id` 相互独立。
Tool 消息中的 `log_context` 会被删除，但不会写入日志。

模型调用成功完成后，Token 用量接口还会收到 `user_id`、`app_id`、`workflow_id`、
`workflow_run_id`、`conversation_id` 和 Dify `user`。这些字段不做格式判断，缺失时传空字符串。

## URL 规则

- 图片和文件都只支持公网 `http://`、`https://` URL。
- 不支持本地文件、Dify 内部 `files/...`、`localhost`、`api`、`web`、`nginx` 等内部地址。
- 插件不下载、不上传、不保存 URL 内容。

## 模型路由

- GPT/OpenAI 系列走 Responses API。
- 工具里的图片 URL 会转成 `input_image.image_url`。
- 工具里的文件 URL 只在 GPT/Responses 路径注入，并转成 `input_file.file_url`。
- Gemini 和其他 chat 模型不注入文件 URL。
