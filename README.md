# Flypower LLM Provider

Dify LLM Provider 插件，用于接入 Flypower/OpenAI-compatible 聚合接口。

## 当前结构

- `models/llm/llm.py`：Dify Provider 入口，负责凭据校验、模型路由和 chat 兼容路径。
- `models/llm/native/openai_responses.py`：GPT/OpenAI 系列 Responses API 适配。
- `models/llm/native/base.py`：模型族判断和文件 bytes 辅助函数。
- `models/llm/agent_context.py`：解析工具返回的 `<DIFY_CONTEXT>` URL 上下文。
- `workflow_tools/read_files.py`：纯 URL 上下文工具代码，用于同步到 Dify 工作流工具。

## 模型路由

- `gpt-*`、`o*`：走 Responses API。
- `gemini-*`：继续走 chat 路径。
- 其他模型：继续走 OpenAI-compatible chat 路径。

## 文件和图片

Chat App 用户直接上传的图片/文档优先走 Dify 原生附件能力，不需要 `read_files`。

`read_files` 只用于工作流 Agent 节点或工具产物已经有外部 URL 的情况：

```json
{
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

推荐传上面的 JSON 对象；如果只有一个 URL，也可以直接传 URL 字符串。URL 字符串数组也可以接受。

工具会返回可见文件索引和 `<DIFY_CONTEXT>...</DIFY_CONTEXT>`。插件只识别这个固定协议，不解析普通文本里的 URL。

URL 规则：

- 图片支持公网 `http://`、`https://`，也支持 `data:image/...`。
- 文件只支持公网 `http://`、`https://`。
- 不接收本地文件，不上传文件，不下载 URL。
- 不支持 `files/...`、`localhost`、`api`、`web`、`nginx` 等内部地址。

Responses 转换规则：

- 图片 URL -> `input_image.image_url`
- URL 文件 -> `input_file.file_url`
- Dify 原生上传文档 -> 先上传 OpenAI `/files`，再用 `input_file.file_id`

## 本地开发

```bash
cd /Users/walker/code_base/dify_demo/llm_provider_plugins/flypower_llm_provider_plugins
.venv/bin/python -m py_compile models/llm/llm.py models/llm/agent_context.py models/llm/native/openai_responses.py workflow_tools/read_files.py
.venv/bin/python -m pytest tests/test_agent_context_protocol.py -q
```

打包并上传到本地 Dify：

```bash
/usr/bin/env bash /Users/walker/code_base/dify_demo/scripts/plugins/打包并上传插件.sh
```

本地 Dify：

```text
http://localhost:18080
```

未登录访问 `/apps` 正常应返回 307 到 `/auth/refresh...`。
