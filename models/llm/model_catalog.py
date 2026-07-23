"""模型 YAML 配置读取。"""

from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(maxsize=1)
def load_model_configs() -> dict[str, dict]:
    """按模型名读取全部模型 YAML 配置。"""
    models_dir = Path(__file__).resolve().parent
    configs: dict[str, dict] = {}
    for model_file in models_dir.glob("*.yaml"):
        if model_file.name.startswith("_"):
            continue
        with model_file.open("r", encoding="utf-8") as file:
            payload = yaml.safe_load(file) or {}
        model_name = payload.get("model")
        if isinstance(model_name, str) and model_name:
            configs[model_name] = payload
    return configs


@lru_cache(maxsize=1)
def load_predefined_chat_models() -> set[str]:
    """读取插件声明的聊天模型名。"""
    return {
        model_name
        for model_name, payload in load_model_configs().items()
        if payload.get("model_type") == "llm"
        and payload.get("model_properties", {}).get("mode") == "chat"
    }


def load_model_extra(model: str) -> dict:
    """读取模型 YAML 的 extra 配置。"""
    extra = load_model_configs().get(model, {}).get("extra", {})
    return extra if isinstance(extra, dict) else {}
