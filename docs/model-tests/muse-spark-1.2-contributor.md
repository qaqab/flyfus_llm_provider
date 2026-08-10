# Muse Spark 1.2 Contributor 接入测试记录

## 基本信息

- 模型 ID：`muse-spark-1.2-contributor`
- 接口：Flyfus LiteLLM OpenAI-compatible API
- 主要路径：`/v1/models`、`/v1/chat/completions`、`/v1/responses`
- 测试时间：2026-08-10，Asia/Shanghai
- 密钥：虚拟密钥，仅在请求头使用，未写入仓库
- 上下文窗口：模型列表未提供；实测 40013 输入 token 成功，插件保守配置为 65536

## 测试结果

| 项目 | 请求格式/测试值 | HTTP | 观察结果 | 判定 |
|---|---|---:|---|---|
| 模型发现 | `/v1/models` | 200 | `data[].id` 包含目标模型 | pass |
| 长上下文 | 40000 次 `probe` + 指令 | 200 | `prompt_tokens=40013`，准确返回 `LONG_CONTEXT_OK` | pass |
| 非流式，64 token | 固定标记 | 200 | 61 reasoning token，正文为空，`finish_reason=length` | fail |
| 非流式，512 token | 固定标记 | 200 | 准确返回标记；176 completion、161 reasoning token | pass |
| 流式，64 token | 固定标记 | 200 | 直接 stop，0 completion token，无正文 | fail |
| 流式，512 token | 固定标记 | 200 | 正文、finish reason、usage 和 `[DONE]` 完整 | pass |
| temperature=0 | 与低边界参数组合 | 200 | 准确返回标记 | pass |
| temperature=1 | 单独测试 | 200 | 准确返回标记 | pass |
| temperature=1.5 | 单独测试 | 200 | 准确返回标记 | pass |
| temperature=1.99 | 单独测试，2048 token | 200 | 约 90 秒后长度耗尽，正文为空 | fail |
| temperature=2 | 单独测试，2048 token | 200 | 长度耗尽，正文为空 | fail |
| top_p | `0.01`、`1` | 200 | 接口接受；采样效果未做统计检验 | pass（接受） |
| frequency_penalty | `-2`、`2` | 200 | 均准确返回标记；惩罚效果未做统计检验 | pass（接受） |
| presence_penalty | `-2`、`2` | 200 | 均准确返回标记；惩罚效果未做统计检验 | pass（接受） |
| max_tokens | `64`、`512`、`128000` | 200 | 64 不足；512 可完成；128000 被接受 | pass（接受） |
| max_completion_tokens | `512` | 200 | 准确返回标记 | pass |
| stop | `stop=["THREE"]` | 400 | 上游明确报告 stop 不支持 | fail |
| 关闭思考 | `enable_thinking=false` | 400 | 未知参数 | fail |
| 推理强度 | `reasoning_effort=low` | 400 | 不支持该参数 | fail |
| text format | `response_format.type=text` | 200 | 返回普通文本 | pass |
| JSON object | `response_format.type=json_object` | 200 | 返回可直接解析的纯 JSON | pass |
| JSON Schema | strict object schema | 400 | 缺少 structured output adapter mode | fail |
| 指定工具 | named function `tool_choice` | 400 | 只支持 `tool_choice=auto` | fail |
| 单工具 auto | `get_weather` | 200 | name、arguments、call ID 和 finish reason 完整 | pass |
| 多工具 auto | 两个独立函数 | 200 | 一次返回两个完整 tool calls | pass |
| 流式工具 | `stream=true` | 200 | name 与 arguments 分片可合并，结束原因为 tool_calls | pass |
| 工具结果回传 | assistant tool call + tool result | 200 | 正确使用工具结果生成最终答案 | pass |
| 搜索开关 | `enable_web_search=true` | 400 | 未知参数 | fail |
| 标准搜索工具 | `tools=[{"type":"web_search"}]` | 400 | 工具类型不受支持 | fail |
| 小图标 data URI | base64 PNG | 200 | 识别图标外形、颜色和中心符号 | pass |
| 已知图形 data URI | 5 圆房屋图 | 200 | 准确回答 5 圆、4 蓝 1 橙、房屋形连接线 | pass |
| 图片 URL | 公网 JPEG | 400 | 上游下载源站返回 400，无法判断 URL 格式 | not-tested |
| TXT file_data | 随机标记 TXT | 200 | 模型表示没有附件，没有返回标记 | fail |
| PDF file_data | 含随机标记的真实 PDF | 200 | 准确返回 `MUSE_FILE_PROBE_8A41D2` | pass |
| Chat input_file | PDF data URI | 500 | user message 格式无效 | fail |
| Responses input_file | PDF data URI | 200 | 准确返回随机标记 | pass |

## 重要行为

模型存在较高的隐藏推理消耗：

- 简单固定回答可能消耗 100 至 300 个 reasoning token；
- 图片请求消耗约 500 至 800 个 reasoning token；
- `max_tokens=64` 可能没有任何可见正文；
- 流式 usage 中 `reasoning_tokens` 可能大于 `completion_tokens`，用量处理不能假设两者包含关系；
- 温度接近 2 时可能长时间运行并耗尽 2048 token，仍无正文。

因此默认输出预算建议不低于 `4096`，温度范围建议限制为 `0..1.5`，默认值为 `1`。

模型列表没有声明上下文窗口，但 40013 输入 token 的直连请求已成功。插件使用
`context_size=65536`，避免 32768 配置在默认 4096 输出预算下把 31503 token 的请求
误判为超限。该测试只证明 40013 token 可用，不代表已经验证完整 65536 token 边界。

## 接入建议

建议能力：

```yaml
features:
- agent-thought
- tool-call
- multi-tool-call
- stream-tool-call
- vision
```

文件能力需要谨慎处理：

- PDF 通过 Chat `file.file_data` 和 Responses `input_file` 均读取成功；
- TXT `file.file_data` 被忽略；
- 如果 Dify 页面无法按扩展名限制为 PDF，不应直接声明通用 `document`；
- 可以先增加 PDF-only 校验，再声明 `document`。

建议参数：

- `temperature`：默认 `1`，范围 `0..1.5`；
- `top_p`：范围 `0.01..1`；
- `max_tokens`：默认至少 `4096`，上限 `128000` 只确认被接受，未确认真实输出上限；
- `frequency_penalty`、`presence_penalty`：`-2..2` 被接受，实际效果仍需统计检验；
- `response_format`：只提供 `text` 和 `json_object`。

不要提供：

- `stop`；
- `enable_thinking`、`thinking_budget`、`reasoning_effort`；
- `enable_web_search` 或内置 web search；
- `json_schema`；
- named、none 或 required `tool_choice`。

插件现有 OpenAI-compatible Chat 路径可支持对话、图片和工具调用。若需要 Responses PDF 输入，需要为该模型增加明确路由并补齐回归测试；不能仅因 `/responses` 单次成功就修改模型族路由。
