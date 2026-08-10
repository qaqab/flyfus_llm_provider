# 模型接入测试规范

本文定义 Flyfus LLM Provider 新增或更新模型前必须执行的兼容性测试。目标不是确认请求能返回
HTTP 200，而是确认 Dify 暴露的每项参数和能力都产生了可观察、可重复的效果。

每个模型测试完成后，按照 [模型测试结果文档模板](model-report-template.md) 返回结果。

已完成的实例记录：

- [MiniMax M3](model-tests/minimax-m3.md)
- [Muse Spark 1.2 Contributor](model-tests/muse-spark-1.2-contributor.md)

新增模型时复制实例表格结构并替换测试证据。

## 基本原则

- 先测试上游接口，再修改模型 YAML。
- “接口接受参数”和“参数实际生效”必须分别验证。
- 模型列表出现某个 ID，只说明模型已注册，不说明推理、附件或工具能力可用。
- HTTP 200 但模型忽略输入，按不支持处理。
- 503、超时和 DNS 错误属于测试未完成，不得据此声明支持或不支持。
- 每项关键测试最多尝试 3 次；结果不一致时标记为不稳定，不在 YAML 中声明。
- API Key 只通过环境变量传入，不写入脚本、文档、测试产物或日志。
- 只声明有直接证据的能力。无法确认的能力默认不声明。

## 测试环境

使用实际生产中转地址和待发布的虚拟密钥：

```bash
export LLM_BASE_URL="https://example.com/v1"
export LLM_API_KEY="<virtual-key>"
export LLM_MODEL="<model-id>"
```

测试记录必须包含：

- 测试日期和时区；
- 实际模型 ID；
- 接口路径，例如 `/chat/completions` 或 `/responses`；
- 是否流式；
- 参数名和测试值；
- HTTP 状态码、`finish_reason`、响应关键字段和用量；
- 判定：`pass`、`fail`、`unstable` 或 `not-tested`；
- 失败原因，但不得记录密钥和完整敏感提示词。

## 必测项目

### 1. 模型发现

请求 `/models`，确认返回的 `data[].id` 精确包含目标模型 ID。模型 YAML 的 `model` 必须与该 ID
完全一致，包括大小写、点号和连字符。

通过标准：

- HTTP 200；
- 响应符合 OpenAI-compatible 模型列表结构；
- 目标 ID 唯一存在。

### 2. 基础对话

分别测试非流式和流式 Chat Completions。提示模型返回一个随机测试标记，例如
`MODEL_PROBE_6F29C8`，避免把通用回答误判为成功。

通过标准：

- 非流式返回准确标记和 `finish_reason`；
- 流式至少返回 role、正文增量、结束原因和 `[DONE]`；
- 声明 `agent-thought` 时，确认 `reasoning` 或 `reasoning_content` 能被插件转换；
- 用量存在时，确认 prompt、completion 和 total token 结构可解析。

### 3. 生成参数

每个参数先单独测试，再组合测试。至少覆盖默认值、建议最小值和建议最大值：

- `temperature`；
- `top_p`；
- `max_tokens` 或 `max_completion_tokens`；
- `presence_penalty`；
- `frequency_penalty`；
- 模型私有的推理参数。

判定规则：

- 返回 2xx 只代表参数被接受；
- `max_tokens` 必须通过截断或 token 用量确认效果；
- 采样和惩罚参数需要多轮对照才能确认效果，无法统计确认时记录为“接受，效果未确认”；
- YAML 的范围不得超过已测试边界；
- 不得通过一次超大 `max_tokens` 请求被接受，就推断模型真实上下文窗口。

上下文长度优先从有权限的模型详情接口或供应商文档取得。无法取得时使用保守值，并在测试记录中说明来源。

### 4. Stop 序列

提示模型输出包含唯一停止标记的固定文本，并设置对应 `stop`。至少使用两个不同标记重复测试。

通过标准：响应正文在停止标记前终止。仅返回 `finish_reason: stop`，但正文仍包含停止标记，判定失败。

### 5. 思考能力

分别测试：

- 不传思考参数；
- `enable_thinking=true`；
- `enable_thinking=false`；
- 最小和最大 `thinking_budget`；
- 支持时测试各档 `reasoning_effort`。

通过标准：

- 开关前后的 `reasoning_content` 或等效字段有稳定差异；
- 关闭后仍持续输出 reasoning，说明模型可能默认强制思考，不得暴露 `enable_thinking`；
- 接受 `thinking_budget` 但无法观察预算效果时，不得宣称预算可控；
- YAML `extra.thinking.mode` 必须与实际发送字段一致。

### 6. 结构化输出

分别测试 `text`、`json_object` 和 `json_schema`。Schema 必须包含 required 字段和
`additionalProperties: false`，并使用 JSON 解析器和 Schema 校验器验证响应。

通过标准：

- `json_object` 返回可直接解析的 JSON，不得带 Markdown 代码围栏；
- `json_schema` 返回纯 JSON，并满足完整 Schema；
- 仅“不报错”但格式不合规，判定失败；
- YAML 只列出通过的 response format 选项。

### 7. 工具调用

依次测试：

- 强制单工具调用；
- `tool_choice: auto`；
- 两个独立工具的并行调用；
- 流式工具调用；
- 将工具结果回传后生成最终答案。

通过标准：

- `tool_calls[].id`、name 和 JSON arguments 完整；
- 多工具调用返回多个独立 call；
- 流式参数分片能被插件合并为有效 JSON；
- 结束原因为 `tool_calls`；
- 完整工具往返能够结束，而不是重复调用或丢失 tool call ID。

YAML 能力映射：

- 单工具通过后才声明 `tool-call`；
- 多工具通过后才声明 `multi-tool-call`；
- 流式工具通过后才声明 `stream-tool-call`。

### 8. 联网搜索

搜索问题必须使用测试时刚产生、基础模型无法预知的事实，并保留真实答案用于核对。分别测试供应商私有搜索开关和
标准搜索工具格式。

通过标准：

- 服务端实际执行搜索并返回正确的新事实；
- 响应提供结果内容或可核验引用；
- 只返回一个未执行的 `web_search` function call，不算搜索成功；
- 模型声称可以使用 curl、浏览器或外部工具，但没有实际工具结果，不算成功；
- 只有端到端搜索通过后，才声明 `enable_web_search`。

如果模型只支持普通 function calling，应在 Dify Agent 中配置真实搜索工具，不要伪装成模型内置搜索。

### 9. 图片输入

至少测试两种输入：

- 本地已知图片转成 base64 data URI；
- 公网 HTTPS 图片 URL。

图片应包含可验证的主体、颜色、文字或数量。不要仅询问“是否看到图片”。

通过标准：

- 模型准确描述只有查看图片才能得出的信息；
- base64 与 URL 至少一种格式与插件实际发送格式一致并通过；
- 外链下载失败要与模型不支持区分；
- 通过后才在 YAML `features` 中加入 `vision`。

### 10. 文件输入

为 TXT 和真实 PDF 分别生成随机标记，例如 `FILE_PROBE_6F29C8`。按目标接口支持的格式测试：

- Chat Completions 的 `file.file_data`；
- Chat Completions 的 `input_file`；
- Responses API 的 `input_file.file_data` 或 `file_url`；
- Dify 实际上传后插件产生的最终请求格式。

通过标准：模型必须准确返回文件内部随机标记。

以下情况都判定失败：

- HTTP 200，但模型表示没有附件；
- 模型根据文件名或提示猜测内容；
- TXT 成功但 PDF 未测，却笼统声明 document；
- 只有把文件内容预先提取成文本后才能回答。

只有原始附件链路通过后，才声明 `document`。如果只能先提取文本，应记录为工作流能力，而不是模型附件能力。

### 11. 错误与重试

至少检查：

- 无效模型；
- 无效参数；
- 上游 4xx；
- 上游 5xx；
- 流在正文前中断；
- 流在已有可见正文后中断。

插件必须保留根因分类和可追踪 ID。不得在已有可见输出后自动重试并产生重复答案。

## YAML 声明规则

新增 `models/llm/<model>.yaml` 时：

- `features` 只写通过端到端测试的能力；
- `parameter_rules` 只写不会报错且具有明确用途的参数；
- 默认值必须位于已验证范围内，并避免默认使用极端资源上限；
- 不稳定、被忽略或语义不完整的参数不展示在 Dify 页面；
- `response_format` 只列出通过严格解析的选项；
- `document` 与 `vision` 分开验证，不能相互推断；
- `_position.yaml` 中模型 ID 必须唯一；
- 价格未知时不得编造价格，并在发布记录中说明。

## 自动化验收

模型接入后至少完成：

1. 使用 `AIModelEntity.model_validate` 解析 YAML；
2. 测试模型 ID、features、参数名、范围和默认值；
3. 测试 `_position.yaml` 中模型只出现一次；
4. 运行完整测试：

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

5. 打包插件，并检查模型 YAML 确实进入产物：

```bash
UV_CACHE_DIR=/tmp/dify-plugin-uv-cache \
  ../../local_dify/bin/dify plugin package . -o /tmp/provider.difypkg
unzip -l /tmp/provider.difypkg
```

6. 在 Dify 安装新版本，确认模型可见、参数页面可保存，并执行一次真实调用。

## 测试记录模板

每个模型建议在工单或发布记录中保存以下表格：

```markdown
| 项目 | 请求格式/测试值 | HTTP | 观察结果 | 判定 |
|---|---|---:|---|---|
| 模型发现 | /models | 200 | 找到 model-id | pass |
| 非流式 | exact marker | 200 | 返回准确标记 | pass |
| 流式 | stream=true | 200 | delta、usage、DONE 完整 | pass |
| temperature | min/default/max | 200 | 接受，效果待统计 | not-tested（效果） |
| max_tokens | min/default/max | 200 | 截断与用量符合 | pass |
| 思考关闭 | enable_thinking=false | 200 | 仍返回 reasoning | fail |
| 搜索 | enable_web_search=true | 200 | 仅返回未执行 tool call | fail |
| 图片 | base64 PNG | 200 | 准确识别主体 | pass |
| 文件 | TXT/PDF marker | 200 | 未读取附件 | fail |
```

最终模型 YAML 必须以该记录中的通过项为准，不能以模型名称、供应商宣传或其他版本模型的能力类推。
