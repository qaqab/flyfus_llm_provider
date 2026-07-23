from models.llm.native.openai_responses import OpenAIResponsesAdapter
from dify_plugin.entities.model.message import SystemPromptMessage


def _adapter() -> OpenAIResponsesAdapter:
    return OpenAIResponsesAdapter(
        endpoint_url=lambda _credentials, _path: "https://example.test/",
        request_headers=lambda _credentials: {},
        normalize_model_parameters=lambda _model, parameters: parameters,
        calc_response_usage=lambda *_args: None,
        create_final_chunk=lambda *_args: None,
    )


def test_responses_system_message_omits_type_and_request_omits_user() -> None:
    body = _adapter()._build_body(
        model="gpt-5",
        credentials={},
        prompt_messages=[SystemPromptMessage(content="You are helpful.")],
        model_parameters={},
        tools=None,
        stop=None,
        stream=False,
        user="af158386-224e-4dba-b382-5a3aa1209fde",
    )

    assert body["input"] == [{"role": "system", "content": "You are helpful."}]
    assert "user" not in body
