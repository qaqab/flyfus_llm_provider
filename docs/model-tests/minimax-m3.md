# MiniMax M3 接入测试记录

## 基本信息

- 模型 ID：`minimax-m3`
- 接口：Flyfus LiteLLM OpenAI-compatible API
- 主要路径：`/v1/models`、`/v1/chat/completions`、`/v1/responses`
- 测试时间：2026-08-06 至 2026-08-10，Asia/Shanghai
- 密钥：虚拟密钥，仅在请求头使用，未写入仓库

## 测试结果

| 项目 | 请求格式/测试值 | HTTP | 观察结果 | 判定 |
|---|---|---:|---|---|
| 模型发现 | `/v1/models` | 200 | `data[].id` 包含 `minimax-m3` | pass |
| 非流式对话 | 返回固定标记 `M3_OK` | 200 | 正文准确，包含 `reasoning_content` 和 usage | pass |
| 流式对话 | `stream=true`、`include_usage=true` | 200 | role、reasoning、正文、finish reason、usage 和 `[DONE]` 完整 | pass |
| temperature | `0`、`0.3`、`0.5`、`2` | 200 | 全部接受；随机性效果未做统计检验 | pass（接受） |
| top_p | `0.01`、`0.8`、`1` | 200 | 全部接受；采样效果未做统计检验 | pass（接受） |
| max_tokens | `32` 至 `128000` | 200 | 小上限限制 reasoning 与正文总输出；大上限被接受 | pass |
| frequency_penalty | `-2`、`0.2`、`2` | 200 | 全部接受；惩罚效果未做统计检验 | pass（接受） |
| presence_penalty | `-2`、`0.2`、`2` | 200 | 全部接受；惩罚效果未做统计检验 | pass（接受） |
| stop | 两个不同停止标记 | 200 | 正文仍包含停止标记，没有截断 | fail |
| text format | `response_format.type=text` | 200 | 返回普通文本 | pass |
| JSON object | `response_format.type=json_object` | 200 | 返回 Markdown JSON 代码围栏，不是纯 JSON | fail |
| JSON Schema | strict object schema | 200 | 返回可解析且符合 Schema 的纯 JSON | pass |
| 强制工具调用 | 指定 `get_weather` | 200 | 返回正确 name、arguments、call ID 和 `tool_calls` finish reason | pass |
| 多工具调用 | 两个独立函数、`parallel_tool_calls=true` | 200 | 一次返回两个完整 tool calls | pass |
| 流式工具调用 | `stream=true`、`tool_choice=auto` | 200 | name 与 arguments 分片可合并，结束原因为 `tool_calls` | pass |
| 关闭思考 | `enable_thinking=false` | 200 | 仍返回 `reasoning_content` | fail |
| 思考预算 | `thinking_budget=1`、`1024` | 200 | 请求被接受，但无法观察预算控制效果 | not-tested（效果） |
| 搜索开关 | `enable_web_search=true` | 200 | 只返回未执行的 `web_search` function call，没有搜索结果 | fail |
| 标准搜索工具 | `tools=[{"type":"web_search"}]` | 200 | 模型表示没有可用搜索工具 | fail |
| 图片 URL | 公网 JPEG | 503 | 上游暂时不可用，无法判定 URL 格式 | not-tested |
| 图片 data URI | 本地 PNG base64 | 200 | 准确识别终端图标的主体、提示符和颜色 | pass |
| TXT file_data | 随机标记 TXT | 200 | 模型表示没有附件，没有返回标记 | fail |
| PDF file_data | 含随机标记的真实 PDF | 200 | 模型表示没有 PDF，没有返回标记 | fail |
| Chat input_file | PDF data URI | 500 | 上游报告 user message 无效 | fail |
| Responses input_file | PDF data URI | 503 | 路由暂时不可用，不能作为文件支持证据 | not-tested |

## YAML 决策

允许声明的 features：

```yaml
features:
- agent-thought
- tool-call
- multi-tool-call
- stream-tool-call
- vision
```

禁止声明：

- `document`：TXT/PDF 原始附件均未被读取；
- 内置联网搜索：只产生未执行的函数调用；
- 可关闭思考：`enable_thinking=false` 无效。

参数规则：

- 可提供 `temperature`、`top_p`、`max_tokens`；
- penalty 参数只确认接口接受，业务使用前仍应做多轮统计检验；
- 可提供通过严格验证的 `json_schema`；
- 不提供 `json_object`、`stop`、`enable_thinking`、`thinking_budget` 和 `enable_web_search`。

文件工作流必须先用 Dify 文档提取器或文件读取工具转换为文本；扫描 PDF 可先转为图片。上述方式属于工作流预处理，不能据此为模型声明 `document`。
