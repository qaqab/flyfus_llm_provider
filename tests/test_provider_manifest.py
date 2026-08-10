from pathlib import Path

import yaml
from dify_plugin.entities.model.schema import AIModelEntity


def test_provider_credential_form_types_are_supported_by_dify() -> None:
    provider_file = Path(__file__).parents[1] / "provider" / "flyfus_llm_provider.yaml"
    provider = yaml.safe_load(provider_file.read_text(encoding="utf-8"))
    schemas = provider["provider_credential_schema"]["credential_form_schemas"]
    supported_types = {"text-input", "secret-input", "select", "radio", "switch"}

    assert all(schema["type"] in supported_types for schema in schemas)
    assert all(
        isinstance(schema.get("default"), str)
        for schema in schemas
        if schema["type"] == "switch" and "default" in schema
    )


def test_minimax_m3_exposes_only_verified_parameters() -> None:
    plugin_root = Path(__file__).parents[1]
    model_file = plugin_root / "models" / "llm" / "MiniMax-M3.yaml"
    model = yaml.safe_load(model_file.read_text(encoding="utf-8"))
    schema = AIModelEntity.model_validate(model)
    rules = {rule["name"]: rule for rule in model["parameter_rules"]}

    assert schema.model == "minimax-m3"
    assert model["features"] == [
        "agent-thought",
        "tool-call",
        "multi-tool-call",
        "stream-tool-call",
        "vision",
    ]
    assert "document" not in model["features"]
    assert set(rules) == {
        "temperature",
        "top_p",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
        "response_format",
        "json_schema",
    }
    assert rules["response_format"]["options"] == ["text", "json_schema"]
    assert rules["temperature"]["min"] == 0
    assert rules["temperature"]["max"] == 2
    assert rules["top_p"]["min"] == 0.01
    assert rules["top_p"]["max"] == 1
    assert rules["max_tokens"]["max"] == 128000
    assert rules["frequency_penalty"]["min"] == -2
    assert rules["frequency_penalty"]["max"] == 2
    assert rules["presence_penalty"]["min"] == -2
    assert rules["presence_penalty"]["max"] == 2

    position = yaml.safe_load((plugin_root / "models" / "llm" / "_position.yaml").read_text(encoding="utf-8"))
    assert position.count("minimax-m3") == 1


def test_muse_spark_contributor_exposes_only_verified_capabilities() -> None:
    plugin_root = Path(__file__).parents[1]
    model_file = plugin_root / "models" / "llm" / "muse-spark-1.2-contributor.yaml"
    model = yaml.safe_load(model_file.read_text(encoding="utf-8"))
    schema = AIModelEntity.model_validate(model)
    rules = {rule["name"]: rule for rule in model["parameter_rules"]}

    assert schema.model == "muse-spark-1.2-contributor"
    assert model["features"] == [
        "agent-thought",
        "vision",
        "document",
        "tool-call",
        "multi-tool-call",
        "stream-tool-call",
    ]
    assert model["model_properties"]["context_size"] == 1048576
    assert set(rules) == {
        "temperature",
        "top_p",
        "max_tokens",
        "frequency_penalty",
        "presence_penalty",
        "response_format",
        "json_schema",
        "reasoning_effort",
        "enable_web_search",
    }
    assert rules["temperature"]["min"] == 0
    assert rules["temperature"]["max"] == 2
    assert rules["top_p"]["min"] == 0.01
    assert rules["top_p"]["max"] == 1
    assert rules["max_tokens"]["default"] == 128000
    assert rules["max_tokens"]["min"] == 512
    assert rules["max_tokens"]["max"] == 128000
    assert rules["frequency_penalty"]["min"] == -2
    assert rules["frequency_penalty"]["max"] == 2
    assert rules["presence_penalty"]["min"] == -2
    assert rules["presence_penalty"]["max"] == 2
    assert rules["response_format"]["options"] == ["text", "json_object", "json_schema"]
    assert rules["reasoning_effort"]["options"] == ["minimal", "low", "medium", "high", "xhigh"]
    assert rules["reasoning_effort"]["default"] == "high"
    assert rules["enable_web_search"]["default"] is False
    assert model["pricing"]["input"] == "0.10"
    assert model["pricing"]["output"] == "0.20"
    assert model["extra"]["token_param_name"] == "max_completion_tokens"

    position = yaml.safe_load((plugin_root / "models" / "llm" / "_position.yaml").read_text(encoding="utf-8"))
    assert position.count("muse-spark-1.2-contributor") == 1
