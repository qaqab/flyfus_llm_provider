from pathlib import Path

import yaml


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
