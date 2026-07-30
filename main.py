from dify_plugin import DifyPluginEnv, Plugin

from models.llm.dify_runtime_context import install_dify_session_context_probe


install_dify_session_context_probe()
plugin = Plugin(DifyPluginEnv())


if __name__ == "__main__":
    plugin.run()
