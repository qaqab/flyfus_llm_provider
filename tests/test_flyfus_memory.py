from dify_plugin.entities.model.message import (
    SystemPromptMessage,
    TextPromptMessageContent,
    UserPromptMessage,
)

from models.llm.flyfus_memory import apply_flyfus_memory


def _memory(memory_type: str, content: str) -> str:
    return (
        '<FLYFUS_MEMORY>{"type":"'
        + memory_type
        + '","content":"'
        + content
        + '"}</FLYFUS_MEMORY>'
    )


def test_injects_system_and_current_user_memory_and_removes_transport_blocks() -> None:
    messages = [
        SystemPromptMessage(content="original system"),
        UserPromptMessage(content="first request\n" + _memory("system", "profile")),
        UserPromptMessage(content="current request\n" + _memory("user", "relevant")),
    ]

    apply_flyfus_memory(messages)

    assert messages[0].content == "original system\n\nprofile"
    assert messages[1].content == "first request"
    assert messages[2].content == "current request\n\nrelevant"
    assert all("FLYFUS_MEMORY" not in message.get_text_content() for message in messages)


def test_historical_user_memory_is_removed_but_not_reinjected() -> None:
    messages = [
        UserPromptMessage(content="old request\n" + _memory("user", "old relevant")),
        UserPromptMessage(content="current request\n" + _memory("user", "new relevant")),
    ]

    apply_flyfus_memory(messages)

    assert messages[0].content == "old request"
    assert messages[1].content == "current request\n\nnew relevant"


def test_uses_last_valid_payload_for_each_destination() -> None:
    messages = [
        UserPromptMessage(
            content=_memory("system", "old profile") + _memory("system", "new profile")
        ),
        UserPromptMessage(
            content=(
                "request\n"
                + _memory("user", "old relevant")
                + _memory("user", "new relevant")
            )
        ),
    ]

    apply_flyfus_memory(messages)

    assert isinstance(messages[0], SystemPromptMessage)
    assert messages[0].content == "new profile"
    assert messages[1].content == ""
    assert messages[2].content == "request\n\nnew relevant"


def test_invalid_unknown_and_empty_payloads_are_only_removed() -> None:
    messages = [
        UserPromptMessage(
            content=(
                "request\n"
                "<FLYFUS_MEMORY>not json</FLYFUS_MEMORY>"
                + _memory("unknown", "ignored")
                + _memory("system", "   ")
            )
        )
    ]

    apply_flyfus_memory(messages)

    assert len(messages) == 1
    assert messages[0].content == "request"


def test_processes_each_multimodal_text_part() -> None:
    message = UserPromptMessage(
        content=[
            TextPromptMessageContent(data="request"),
            TextPromptMessageContent(data=_memory("system", "profile")),
            TextPromptMessageContent(data=_memory("user", "relevant")),
        ]
    )
    messages = [message]

    apply_flyfus_memory(messages)

    assert isinstance(messages[0], SystemPromptMessage)
    assert messages[0].content == "profile"
    assert message.content[0].data == "request"
    assert message.content[1].data == ""
    assert message.content[2].data == "relevant"
