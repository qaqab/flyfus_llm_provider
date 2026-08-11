from dify_plugin.entities.model.message import ToolPromptMessage, UserPromptMessage

from models.llm.prompt_preprocessing import deduplicate_tool_responses


def test_deduplicates_equivalent_mcp_text_and_structured_outputs() -> None:
    payload = '{"toolsCallResults":[{"toolName":"lookup"}]}'
    messages = [
        ToolPromptMessage(
            name="batch_call",
            tool_call_id="call-1",
            content=f"tool response: {payload}.tool response: {payload}.",
        )
    ]

    removed_characters = deduplicate_tool_responses(messages)

    assert messages[0].content == f"tool response: {payload}."
    assert removed_characters > 0


def test_keeps_non_equivalent_or_non_tool_content() -> None:
    tool_message = ToolPromptMessage(
        name="batch_call",
        tool_call_id="call-1",
        content='tool response: {"value":1}.tool response: {"value":2}.',
    )
    user_message = UserPromptMessage(
        content='tool response: {"value":1}.tool response: {"value":1}.'
    )
    messages = [tool_message, user_message]

    removed_characters = deduplicate_tool_responses(messages)

    assert removed_characters == 0
    assert tool_message.content.endswith('{"value":2}.')
    assert user_message.content.count("tool response:") == 2


def test_keeps_duplicate_content_from_non_mcp_tool() -> None:
    payload = '{"value":1}'
    message = ToolPromptMessage(
        name="read_file",
        tool_call_id="call-1",
        content=f"tool response: {payload}.tool response: {payload}.",
    )

    removed_characters = deduplicate_tool_responses([message])

    assert removed_characters == 0
    assert message.content.count("tool response:") == 2
