import pytest

from models.llm.llm import FlyfusLargeLanguageModel
from models.llm.native.gemini import GeminiNativeDocumentAdapter
from dify_plugin.entities.model.message import UserPromptMessage


@pytest.mark.parametrize(
    ("reasoning_effort", "thinking_level"),
    [
        ("low", "Low"),
        ("medium", "Medium"),
        ("high", "High"),
        ("xhigh", "High"),
    ],
)
def test_set_next_step_effort_maps_to_gemini_thinking_level(
    reasoning_effort: str, thinking_level: str
) -> None:
    parameters = {"reasoning_effort": reasoning_effort, "thinking_config": {"thinking_level": "Low"}}

    FlyfusLargeLanguageModel._apply_reasoning_effort(parameters, "gemini", {})

    body = GeminiNativeDocumentAdapter(
        endpoint_url=lambda _credentials, _path: "https://example.test/",
        normalize_model_parameters=lambda _model, value: value,
        calc_response_usage=lambda *_args: None,
    ).build_body(
        model="gemini-3.6-flash",
        prompt_messages=[UserPromptMessage(content="Reply")],
        model_parameters=parameters,
        tools=None,
        stop=None,
    )

    assert "reasoning_effort" not in parameters
    assert body["generationConfig"]["thinkingConfig"]["thinkingLevel"] == thinking_level
