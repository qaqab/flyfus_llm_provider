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


def test_gpt_terra_exposes_only_effective_parameters() -> None:
    plugin_root = Path(__file__).parents[1]
    model_file = plugin_root / "models" / "llm" / "gpt-5.6-terra.yaml"
    model = yaml.safe_load(model_file.read_text(encoding="utf-8"))
    schema = AIModelEntity.model_validate(model)
    rules = {rule["name"]: rule for rule in model["parameter_rules"]}

    assert schema.model == "gpt-5.6-terra"
    assert model["model_properties"]["context_size"] == 360000
    assert set(rules) == {
        "response_format",
        "json_schema",
        "reasoning_effort",
        "enable_web_search",
    }
    assert rules["reasoning_effort"]["options"] == [
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    assert "extra" not in model


def test_gpt_sol_exposes_only_effective_parameters() -> None:
    plugin_root = Path(__file__).parents[1]
    model_file = plugin_root / "models" / "llm" / "gpt-5.6-sol.yaml"
    model = yaml.safe_load(model_file.read_text(encoding="utf-8"))
    schema = AIModelEntity.model_validate(model)
    rules = {rule["name"]: rule for rule in model["parameter_rules"]}

    assert schema.model == "gpt-5.6-sol"
    assert model["model_properties"]["context_size"] == 921000
    assert set(rules) == {
        "response_format",
        "json_schema",
        "reasoning_effort",
        "enable_web_search",
    }
    assert rules["reasoning_effort"]["options"] == [
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    assert rules["enable_web_search"]["default"] is True
    assert "extra" not in model


def test_gpt_55_exposes_only_effective_parameters() -> None:
    plugin_root = Path(__file__).parents[1]
    model_file = plugin_root / "models" / "llm" / "gpt-5.5.yaml"
    model = yaml.safe_load(model_file.read_text(encoding="utf-8"))
    schema = AIModelEntity.model_validate(model)
    rules = {rule["name"]: rule for rule in model["parameter_rules"]}

    assert schema.model == "gpt-5.5"
    assert model["model_properties"]["context_size"] == 260000
    assert set(rules) == {
        "response_format",
        "json_schema",
        "reasoning_effort",
        "enable_web_search",
    }
    assert rules["reasoning_effort"]["options"] == [
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
    ]
    assert rules["enable_web_search"]["default"] is True
    assert "extra" not in model


def test_gemini_flash_models_expose_verified_parameters() -> None:
    plugin_root = Path(__file__).parents[1]
    expectations = {
        "gemini-3.5-flash": {
            "context_size": 1000000,
            "parameters": {
                "temperature",
                "top_p",
                "response_format",
                "json_schema",
                "thinking_level",
            },
            "default_level": "High",
            "web_search": False,
        },
        "gemini-3.6-flash": {
            "context_size": 750000,
            "parameters": {
                "temperature",
                "top_p",
                "response_format",
                "json_schema",
                "thinking_level",
                "enable_web_search",
            },
            "default_level": "Low",
            "web_search": True,
        },
    }

    for model_name, expected in expectations.items():
        model_file = plugin_root / "models" / "llm" / f"{model_name}.yaml"
        model = yaml.safe_load(model_file.read_text(encoding="utf-8"))
        schema = AIModelEntity.model_validate(model)
        rules = {rule["name"]: rule for rule in model["parameter_rules"]}

        assert schema.model == model_name
        assert model["model_properties"]["context_size"] == expected["context_size"]
        assert set(rules) == expected["parameters"]
        assert rules["thinking_level"]["options"] == ["Minimal", "Low", "Medium", "High"]
        assert rules["thinking_level"]["default"] == expected["default_level"]
        assert ("enable_web_search" in rules) is expected["web_search"]


def test_gemini_37_flash_exposes_only_verified_capabilities() -> None:
    plugin_root = Path(__file__).parents[1]
    model_file = plugin_root / "models" / "llm" / "gemini-3.7-flash.yaml"
    model = yaml.safe_load(model_file.read_text(encoding="utf-8"))
    schema = AIModelEntity.model_validate(model)
    rules = {rule["name"]: rule for rule in model["parameter_rules"]}

    assert schema.model == "gemini-3.7-flash"
    assert model["features"] == ["agent-thought", "vision", "document", "tool-call"]
    assert model["model_properties"]["context_size"] == 1047000
    assert set(rules) == {"thinking_level", "enable_web_search"}
    assert rules["thinking_level"]["options"] == ["low", "medium", "high"]
    assert rules["thinking_level"]["default"] == "medium"
    assert rules["enable_web_search"]["default"] is False

    position = yaml.safe_load(
        (plugin_root / "models" / "llm" / "_position.yaml").read_text(encoding="utf-8")
    )
    assert position.count("gemini-3.7-flash") == 1


def test_ling_flash_exposes_only_verified_capabilities() -> None:
    plugin_root = Path(__file__).parents[1]
    model_file = plugin_root / "models" / "llm" / "ling-3.0-flash.yaml"
    model = yaml.safe_load(model_file.read_text(encoding="utf-8"))
    schema = AIModelEntity.model_validate(model)
    rules = {rule["name"]: rule for rule in model["parameter_rules"]}

    assert schema.model == "ling-3.0-flash"
    assert model["features"] == [
        "agent-thought",
        "tool-call",
        "multi-tool-call",
        "stream-tool-call",
    ]
    assert model["model_properties"]["context_size"] == 262144
    assert set(rules) == {
        "temperature",
        "top_p",
        "frequency_penalty",
        "presence_penalty",
        "max_tokens",
        "response_format",
        "reasoning_effort",
    }
    assert rules["response_format"]["options"] == ["text", "json_object"]
    assert rules["max_tokens"]["default"] == 32768
    assert rules["max_tokens"]["max"] == 32768
    assert rules["reasoning_effort"]["options"] == [
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
    ]
    assert "enable_thinking" not in rules
    assert model["extra"]["token_param_name"] == "max_completion_tokens"

    position = yaml.safe_load(
        (plugin_root / "models" / "llm" / "_position.yaml").read_text(encoding="utf-8")
    )
    assert position.count("ling-3.0-flash") == 1


def test_grok_46_exposes_only_verified_capabilities() -> None:
    plugin_root = Path(__file__).parents[1]
    model_file = plugin_root / "models" / "llm" / "grok-4.6.yaml"
    model = yaml.safe_load(model_file.read_text(encoding="utf-8"))
    schema = AIModelEntity.model_validate(model)
    rules = {rule["name"]: rule for rule in model["parameter_rules"]}

    assert schema.model == "grok-4.6"
    assert model["features"] == ["agent-thought", "vision", "tool-call"]
    assert model["model_properties"]["context_size"] == 450000
    assert set(rules) == {"reasoning_effort"}
    assert rules["reasoning_effort"]["options"] == ["low", "medium", "high", "xhigh"]

    position = yaml.safe_load(
        (plugin_root / "models" / "llm" / "_position.yaml").read_text(encoding="utf-8")
    )
    assert position.count("grok-4.6") == 1


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
