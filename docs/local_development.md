# 本地开发、日志和发布规范

这份文档固定本插件的本地调试流程，避免每次排查都重新摸一遍。

## 目录

- 插件目录：`/Users/walker/code_base/dify_demo/llm_provider_plugins/flypower_llm_provider_plugins`
- Dify Docker 目录：`/Users/walker/code_base/dify_demo/local_dify/dify_docker`
- 本地 Dify：`http://localhost:18080`
- 测试 App：`http://localhost:18080/app/f42dc1f1-0454-4289-ba3d-44cd715b3576/configuration`

## 开发检查

修改插件代码后，先在插件目录执行：

```bash
cd /Users/walker/code_base/dify_demo/llm_provider_plugins/flypower_llm_provider_plugins
.venv/bin/python -m py_compile models/llm/llm.py models/llm/agent_context.py models/llm/native/openai_responses.py workflow_tools/read_files.py
.venv/bin/python -m pytest tests/test_agent_context_protocol.py -q
```

只改文档时不需要跑测试。

## 本地 Dify 检查

```bash
cd /Users/walker/code_base/dify_demo/local_dify/dify_docker
docker compose ps
docker compose config
curl -I http://localhost:18080/apps
```

`/apps` 未登录时正常应返回 `307`，跳转到 `/auth/refresh...`。如果返回 `500`，优先检查 web/API 配置。

Dify 1.15.0 的 web 需要：

```yaml
SERVER_CONSOLE_API_URL: http://api:5001
```

`.env` 里的 `CONSOLE_API_URL`、`SERVICE_API_URL`、`APP_API_URL` 保持空值，避免 web 容器请求错误的 `localhost`。

## 本地打包和安装

```bash
/usr/bin/env bash /Users/walker/code_base/dify_demo/scripts/plugins/打包并上传插件.sh
```

这个脚本会：

- 打包插件到 `/Users/walker/code_base/dify_demo/local_dify/packages/flypower_llm_provider.difypkg`
- 复制最新包到 `/Users/walker/code_base/dify_demo/local_dify/packages/flypower_llm_provider-latest.difypkg`
- 上传并安装到本地 Dify

如果安装失败，先看 `plugin_daemon` 日志里是否有：

- `local runtime ready`
- `instance_error`
- `PluginInvokeError`
- `model schema`
- `no available node, plugin runtime not found`

## GitHub 发布

发布前先确认：

- `CHANGELOG.md` 已写目标版本说明。
- `manifest.yaml` 版本号符合预期。
- 本地语法检查和测试通过。
- 本地打包安装没问题。

发布脚本：

```bash
/usr/bin/env bash /Users/walker/code_base/dify_demo/scripts/plugins/打包并上传到GitHub.sh --publish
```

指定版本：

```bash
/usr/bin/env bash /Users/walker/code_base/dify_demo/scripts/plugins/打包并上传到GitHub.sh --publish --version 0.0.31
```

脚本会从 `CHANGELOG.md` 读取对应版本说明，作为 GitHub Release notes。以后每个版本都必须补 changelog。

## 日志查询

常用日志：

```bash
cd /Users/walker/code_base/dify_demo/local_dify/dify_docker
docker compose logs -f plugin_daemon
docker compose logs -f api worker
```

干净测试前先写一个时间标记：

```bash
cd /Users/walker/code_base/dify_demo/local_dify/dify_docker
date -u +"%Y-%m-%dT%H:%M:%SZ" > .clean-test-logs-since
```

之后只看这次测试之后的日志：

```bash
cd /Users/walker/code_base/dify_demo/local_dify/dify_docker
SINCE="$(cat .clean-test-logs-since)"
docker compose logs --since "$SINCE" plugin_daemon
docker compose logs --since "$SINCE" api worker
```

过滤重点信息：

```bash
SINCE="$(cat .clean-test-logs-since)"
docker compose logs --since "$SINCE" plugin_daemon \
  | rg -i 'flypower|responses|geo prompt|PluginInvokeError|InvokeError|instance_error|failed|error|dispatch/llm/invoke'

docker compose logs --since "$SINCE" api worker \
  | rg -i 'flypower|responses|geo prompt|PluginInvokeError|InvokeError|failed|error'
```

## 数据库辅助查询

查看最近消息：

```bash
cd /Users/walker/code_base/dify_demo/local_dify/dify_docker
docker compose exec -T db_postgres psql -U postgres -d dify -P pager=off -c "
select id, query, left(answer, 1000) as answer_head, error, created_at
from messages
where app_id='f42dc1f1-0454-4289-ba3d-44cd715b3576'
order by created_at desc
limit 5;
"
```

查看最近工具调用：

```bash
cd /Users/walker/code_base/dify_demo/local_dify/dify_docker
docker compose exec -T db_postgres psql -U postgres -d dify -P pager=off -c "
select message_id, tool, left(tool_input,800) as tool_input_head,
       left(observation,1500) as observation_head, created_at
from message_agent_thoughts
where tool ilike '%flyfus%' or tool ilike '%read_files%' or observation ilike '%tool_prompt%'
order by created_at desc
limit 10;
"
```

## 插件日志写法

不要在插件运行时直接使用：

```python
print(...)
```

也不要把业务 debug 打到 stdout/stderr。Dify plugin_daemon 会读取插件子进程 stdout；如果输出不符合插件协议，可能被记录成 `instance_error`，严重时会导致 runtime shutdown 或安装失败。

推荐规则：

- 默认不要写业务日志。
- 需要临时 debug 时，用环境变量开关。
- debug 内容写到 `/tmp/...log` 文件。
- 不记录 API Key、完整提示词、完整用户文件内容。
- 日志只写摘要：模型、开关、数量、长度、状态码、错误类型。

Responses 路径当前已有 debug 开关：

```bash
FLYPOWER_RESPONSES_DEBUG=1
```

开启后会写：

```text
/tmp/flypower_responses_debug.log
```

查询：

```bash
cd /Users/walker/code_base/dify_demo/local_dify/dify_docker
docker compose exec -T plugin_daemon sh -lc 'cat /tmp/flypower_responses_debug.log 2>/dev/null || true'
```

清理：

```bash
cd /Users/walker/code_base/dify_demo/local_dify/dify_docker
docker compose exec -T plugin_daemon sh -lc 'rm -f /tmp/flypower_responses_debug.log'
```

## 提示词替换验证

工具返回如果是 Dify 外层包装：

```json
{"flyfus_skills": "{\"diagnosis_tool_prompt\":\"{{geo_prompt:flyfus-agent.flyfus-skill-listing-diagnosis@dev}}\"}"}
```

插件会忽略外层工具名，只检查内层：

- 必须是 JSON object。
- 只能有一个字段。
- 字段名必须以 `_tool_prompt` 结尾。
- 字段值必须完全匹配 `{{geo_prompt:category.prompt@env}}`。

如果渲染接口被调用，说明替换逻辑已经命中。Dify 前端仍可能显示原始工具 observation，这是前端展示存储的工具结果，不等于发给模型的最终 Tool message。

## 常见判断

- `OpenAI Responses 请求失败` 且是 `SSLEOFError` / `RemoteDisconnected`：通常是上游网关或网络断开，不是工具协议问题。
- `Mutually exclusive parameters: file_id or filename`：Responses `input_file` 里不能同时给互斥字段，URL 文件只传 `file_url`。
- `Invalid context_refs JSON`：工具参数没有按 `read_files` 要求传 JSON 对象；推荐传 `{"files":[...],"images":[...]}`。
- `plugin stdout reader exiting instance_error=...`：插件向 stdout/stderr 打了不该打的内容，先删除 `print` / stderr debug，再重新打包安装。
