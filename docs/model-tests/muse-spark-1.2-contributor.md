# Muse Spark 1.2 Contributor 接入测试记录

## 基本信息

- 模型 ID：`muse-spark-1.2-contributor`
- 接口：Flyfus LiteLLM OpenAI-compatible API
- 主要路径：`/v1/models`、`/v1/chat/completions`、`/v1/responses`
- 测试时间：2026-08-10，Asia/Shanghai
- 密钥：虚拟密钥，仅在请求头使用，未写入仓库
- 官方上下文窗口：`1,048,576` tokens，输入与输出共享
- 中转实测输入：最高 `150029` tokens，并准确读取尾部校验码
- 官方价格：输入 `$0.10/M`、缓存输入 `$0.002/M`、输出 `$0.20/M`

官方资料：

- [Muse Spark 1.2 模型页](https://developer.meta.com/ai/models/muse-spark/)
- [Meta Model API 模型规格](https://dev.meta.ai/docs/models)
- [Chat Completions 参数](https://dev.meta.ai/docs/protocols/chat-completions)
- [Responses API](https://dev.meta.ai/docs/protocols/responses)
- [文件处理](https://dev.meta.ai/docs/file-handling)
- [联网搜索](https://dev.meta.ai/docs/search-grounding)

## 测试结果

| 项目 | 请求格式/测试值 | HTTP | 观察结果 | 判定 |
|---|---|---:|---|---|
| 模型发现 | `/v1/models` | 200 | `data[].id` 包含目标模型 | pass |
| 长上下文 | 40000 次 `probe` + 指令 | 200 | `prompt_tokens=40013`，准确返回 `LONG_CONTEXT_OK` | pass |
| 长上下文 | 60000 次 `probe` + 尾部校验码 | 200 | `prompt_tokens=60027`，准确返回尾部校验码 | pass |
| 长上下文 | 100000 次 `probe` + 尾部校验码 | 200 | `prompt_tokens=100028`，准确返回尾部校验码 | pass |
| 长上下文 | 120000 次 `probe` + 尾部校验码 | 200 | `prompt_tokens=120028`，准确返回尾部校验码 | pass |
| 长上下文 | 128000 次 `probe` + 尾部校验码 | 200 | `prompt_tokens=128028`，准确返回尾部校验码 | pass |
| 长上下文 | 150000 次 `probe` + 尾部校验码 | 200 | `prompt_tokens=150029`，准确返回尾部校验码 | pass |
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
| Responses max_output_tokens | `128000` | 200 | 参数被接受并准确返回标记；未验证实际生成 128000 token | pass（接受） |
| stop | `stop=["THREE"]` | 400 | 上游明确报告 stop 不支持 | fail |
| 关闭思考 | `enable_thinking=false` | 400 | 未知参数 | fail |
| Chat 推理强度 | `reasoning_effort=low` | 400 | LiteLLM Chat 路径未正确透传 | fail（路径） |
| Responses 推理强度 | `reasoning.effort=low`、`xhigh` | 200 | 两档均返回 reasoning item 和准确标记 | pass |
| text format | `response_format.type=text` | 200 | 返回普通文本 | pass |
| JSON object | `response_format.type=json_object` | 200 | 返回可直接解析的纯 JSON | pass |
| Chat JSON Schema | strict object schema | 400 | LiteLLM Chat 路径缺少 structured output adapter mode | fail（路径） |
| Responses JSON Schema | `text.format.type=json_schema` | 200 | 返回满足 strict Schema 的纯 JSON | pass |
| 指定工具 | named function `tool_choice` | 400 | 只支持 `tool_choice=auto` | fail |
| 单工具 auto | `get_weather` | 200 | name、arguments、call ID 和 finish reason 完整 | pass |
| 多工具 auto | 两个独立函数 | 200 | 一次返回两个完整 tool calls | pass |
| 流式工具 | `stream=true` | 200 | name 与 arguments 分片可合并，结束原因为 tool_calls | pass |
| 工具结果回传 | assistant tool call + tool result | 200 | 正确使用工具结果生成最终答案 | pass |
| Chat 搜索开关 | `enable_web_search=true` | 400 | 不是 Meta 官方请求字段 | fail（格式） |
| Chat 标准搜索工具 | `tools=[{"type":"web_search"}]` | 400 | 官网明确 Chat 不提供搜索 | fail（端点） |
| Responses 联网搜索 | `tools=[{"type":"web_search"}]` | 200 | 返回多个 `web_search_call`、最终正文和 URL citation | pass |
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

因此输出预算下限建议不低于 `512`。官网规定温度范围为 `0..2`、默认值为 `1`，并建议一般保持默认值；
高温虽然不报 API 错误，但实测可能耗尽输出预算而没有可见正文。

Meta 官方明确声明上下文窗口为 `1,048,576` tokens。插件使用该精确值；直连中转请求目前
只实测到 150029 输入 token，因此更长请求是否被 LiteLLM 中转完整放行仍应以上游实际响应为准。
官网未公布固定最大输出值；`128000` 只确认被 Responses API 接受，不能写成“已实测生成 128K”。

## 接入建议

建议能力：

```yaml
features:
- agent-thought
- vision
- document
- tool-call
- multi-tool-call
- stream-tool-call
```

文件与多模态能力：

- 官网声明 text、image、video、audio 和 PDF 输入；模型输出为 text；
- PDF 通过 Chat `file.file_data` 和 Responses `input_file` 均读取成功，插件声明 `document`；
- TXT 在推理请求中被忽略，与官网“TXT/JSON 主要用于 batch 等其他用途”的说明一致；
- 当前插件 Responses 适配器尚未转换 Dify 的 video/audio 内容，因此暂不声明 `video`、`audio`。

建议参数：

- `temperature`：默认 `1`，官网范围 `0..2`；
- `top_p`：范围 `0.01..1`；
- `max_tokens`：Dify 页面名称；发送时转换为 Chat 的 `max_completion_tokens` 或 Responses 的 `max_output_tokens`；
- `reasoning_effort`：`minimal`、`low`、`medium`、`high`、`xhigh`；Muse 不支持 `none`；
- `enable_web_search`：默认关闭；开启时经 Responses API 添加 `web_search` 工具；
- `frequency_penalty`、`presence_penalty`：`-2..2` 被接受，实际效果仍需统计检验；
- `response_format`：提供 `text`、`json_object` 和经 Responses 实测的 `json_schema`。

不要提供：

- `stop`；
- `enable_thinking`、`thinking_budget`；
- `reasoning_effort=none`；
- named、none 或 required `tool_choice`。

插件将该模型明确路由到 Responses API，以支持 reasoning、搜索、JSON Schema、PDF 和工具调用；
普通 Dify 参数 `max_tokens` 会在请求构造时转换成官网协议对应字段。
