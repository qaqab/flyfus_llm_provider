# Muse Spark 1.3 Contributor 接入测试记录

## 基本信息

- 模型 ID：`muse-spark-1.3-contributor`
- 接口：Flyfus LiteLLM OpenAI-compatible API
- 测试路径：`/v1/models`、`/v1/model/info`、`/v1/chat/completions`、`/v1/responses`
- 测试时间：2026-09-03，Asia/Shanghai
- 密钥：仅在请求头使用，未写入仓库
- LiteLLM 输入上限：`1,000,000` tokens
- LiteLLM 价格：输入 `$0.10/M`、缓存输入 `$0.002/M`、输出 `$0.20/M`

## 测试结果

| 项目 | 请求格式/测试值 | HTTP | 观察结果 | 判定 |
|---|---|---:|---|---|
| 模型发现 | `/v1/models` | 200 | `data[].id` 包含目标模型 | pass |
| Chat Completions | 固定标记，非流式 | 200 | 准确返回固定标记 | pass |
| Responses | 固定标记，非流式 | 200 | 返回 reasoning item 和准确标记 | pass |
| 生成参数 | temperature、Top P、penalties、xhigh、128000 输出预算 | 200 | 参数被接受并准确返回固定标记 | pass（接受） |
| JSON Schema | strict object schema | 200 | 返回满足 Schema 的纯 JSON | pass |
| 单工具 auto | `get_weather` | 200 | 返回完整 name、arguments 和 call ID | pass |
| 多工具 auto | 两个独立函数 | 200 | 一次返回两个完整 function call | pass |
| Responses 流式 | 固定标记 | 200 | 收到正文和 `response.completed` | pass |
| 联网搜索 | `tools=[{"type":"web_search"}]` | 200 | 返回 4 个 `web_search_call` 和 URL citation | pass |
| TXT 文档 | `input_file.file_data` | 200 | 准确返回文件内随机标记 | pass |
| PNG 图片 | 5 圆房屋图 data URI | 200 | 准确回答 5 圆、4 蓝 1 橙 | pass |

## 接入结论

模型明确路由到 Responses API。插件沿用 Muse 1.2 已验证的参数与输入转换，并将
1.3 纳入 Muse 外链图片转存和文档 `file_data` 直传分支。`context_size` 使用
LiteLLM `/v1/model/info` 返回的 `max_input_tokens=1000000`，不外推更大窗口。
