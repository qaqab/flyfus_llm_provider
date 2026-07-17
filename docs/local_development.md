# 本地开发

在插件目录中运行最小语法检查：

```bash
.venv/bin/python -m py_compile models/llm/llm.py models/llm/parameter_conversion.py
```

参数转换相关测试：

```bash
.venv/bin/python -m pytest tests/test_agent_context_protocol.py -q -k 'temperature or top_p or max_tokens or web_search or thinking or reasoning'
```

本地安装、日志、GitHub 发布与远程调试统一见仓库根目录的 [`docs/插件运维.md`](../../../docs/插件运维.md)。

插件运行时不要使用 `print` 写业务日志。需要记录插件内部事件时，按 [Dify 插件日志文档](https://docs.dify.ai/en/develop-plugin/features-and-specs/plugin-types/plugin-logging) 使用 SDK 的 `plugin_logger_handler`，不要输出 API Key、完整提示词或用户文件内容。
