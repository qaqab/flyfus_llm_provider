"""Bound model input by removing complete, protocol-safe history blocks."""

from dataclasses import dataclass
from typing import Callable, Optional

from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
)


SOFT_LIMIT_RATIO = 0.80
TARGET_LIMIT_RATIO = 0.70
EMERGENCY_TARGET_RATIO = 0.60


class ContextGuardError(ValueError):
    """Raised when mandatory context cannot fit or message protocol is invalid."""


@dataclass(frozen=True)
class ContextGuardResult:
    initial_tokens: int
    final_tokens: int
    hard_budget: int
    soft_limit: int
    target_limit: int
    removed_message_count: int
    removed_block_count: int
    emergency: bool

    @property
    def trimmed(self) -> bool:
        return self.removed_message_count > 0


@dataclass(frozen=True)
class _Block:
    indices: tuple[int, ...]
    priority: int
    mandatory: bool = False


def guard_prompt_messages(
    prompt_messages: list[PromptMessage],
    *,
    token_counter: Callable[[list[PromptMessage]], int],
    context_size: int,
    max_output_tokens: int,
    extra_input_tokens: int = 0,
    emergency: bool = False,
    required_user_message: Optional[PromptMessage] = None,
) -> ContextGuardResult:
    """Trim whole history blocks in place while preserving required context."""

    validate_message_protocol(prompt_messages)
    safety_margin = min(32768, max(256, context_size // 50))
    hard_budget = context_size - max_output_tokens - safety_margin
    if hard_budget <= 0:
        raise ContextGuardError(
            "MANDATORY_CONTEXT_TOO_LARGE: output reservation and safety margin "
            "consume the model context window"
        )

    def count(messages: list[PromptMessage]) -> int:
        return token_counter(messages) + max(0, extra_input_tokens)

    initial_tokens = count(prompt_messages)
    soft_limit = int(hard_budget * SOFT_LIMIT_RATIO)
    target_ratio = EMERGENCY_TARGET_RATIO if emergency else TARGET_LIMIT_RATIO
    target_limit = int(hard_budget * target_ratio)
    if emergency:
        target_limit = min(target_limit, int(initial_tokens * 0.70))
    trigger_limit = target_limit if emergency else soft_limit

    if initial_tokens <= trigger_limit:
        return ContextGuardResult(
            initial_tokens=initial_tokens,
            final_tokens=initial_tokens,
            hard_budget=hard_budget,
            soft_limit=soft_limit,
            target_limit=target_limit,
            removed_message_count=0,
            removed_block_count=0,
            emergency=emergency,
        )

    blocks = _build_blocks(prompt_messages, required_user_message)
    candidates = sorted(
        (block for block in blocks if not block.mandatory),
        key=lambda block: (-block.priority, block.indices[0]),
    )
    removed_indices: set[int] = set()
    final_tokens = initial_tokens
    removed_block_count = 0

    for block in candidates:
        removed_indices.update(block.indices)
        remaining = [
            message
            for index, message in enumerate(prompt_messages)
            if index not in removed_indices
        ]
        final_tokens = count(remaining)
        removed_block_count += 1
        if final_tokens <= target_limit:
            break

    if removed_indices:
        prompt_messages[:] = [
            message
            for index, message in enumerate(prompt_messages)
            if index not in removed_indices
        ]

    validate_message_protocol(prompt_messages)
    final_tokens = count(prompt_messages)
    if final_tokens > hard_budget:
        raise ContextGuardError(
            "MANDATORY_CONTEXT_TOO_LARGE: required context uses "
            f"{final_tokens} tokens but the input budget is {hard_budget}"
        )

    return ContextGuardResult(
        initial_tokens=initial_tokens,
        final_tokens=final_tokens,
        hard_budget=hard_budget,
        soft_limit=soft_limit,
        target_limit=target_limit,
        removed_message_count=len(removed_indices),
        removed_block_count=removed_block_count,
        emergency=emergency,
    )


def validate_message_protocol(prompt_messages: list[PromptMessage]) -> None:
    """Reject orphaned or incomplete tool-call cycles."""

    pending_call_ids: set[str] = set()
    for message in prompt_messages:
        if pending_call_ids and not isinstance(message, ToolPromptMessage):
            raise ContextGuardError(
                "INVALID_MESSAGE_PROTOCOL: assistant tool calls must be followed "
                "by their tool results"
            )

        if isinstance(message, AssistantPromptMessage) and message.tool_calls:
            call_ids = {str(call.id) for call in message.tool_calls if call.id}
            if len(call_ids) != len(message.tool_calls):
                raise ContextGuardError(
                    "INVALID_MESSAGE_PROTOCOL: duplicate or empty tool_call_id"
                )
            pending_call_ids = call_ids
            continue

        if isinstance(message, ToolPromptMessage):
            call_id = str(message.tool_call_id or "")
            if call_id not in pending_call_ids:
                raise ContextGuardError(
                    f"INVALID_MESSAGE_PROTOCOL: orphan tool result {call_id or '<empty>'}"
                )
            pending_call_ids.remove(call_id)

    if pending_call_ids:
        raise ContextGuardError(
            "INVALID_MESSAGE_PROTOCOL: missing tool results for "
            + ", ".join(sorted(pending_call_ids))
        )


def is_context_window_error(error: BaseException) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "exceeds the context window",
            "context length exceeded",
            "maximum context length",
            "too many input tokens",
        )
    )


def _build_blocks(
    prompt_messages: list[PromptMessage],
    required_user_message: Optional[PromptMessage],
) -> list[_Block]:
    latest_user_index = _required_user_index(prompt_messages, required_user_message)
    if latest_user_index is None:
        return [
            _Block(indices=(index,), priority=0, mandatory=True)
            for index in range(len(prompt_messages))
        ]

    blocks: list[_Block] = []
    historical_turns = _historical_turns(prompt_messages, latest_user_index)
    recent_turn_start = max(0, len(historical_turns) - 2)
    for position, indices in enumerate(historical_turns):
        blocks.append(
            _Block(
                indices=indices,
                priority=1 if position >= recent_turn_start else 3,
            )
        )

    covered = {index for block in blocks for index in block.indices}
    for index, message in enumerate(prompt_messages[:latest_user_index]):
        if index in covered:
            continue
        blocks.append(
            _Block(
                indices=(index,),
                priority=0,
                mandatory=isinstance(message, SystemPromptMessage),
            )
        )

    blocks.append(_Block(indices=(latest_user_index,), priority=0, mandatory=True))
    current_blocks = _current_turn_blocks(prompt_messages, latest_user_index + 1)
    for block in current_blocks:
        blocks.append(
            _Block(
                indices=block.indices,
                priority=0,
                mandatory=True,
            )
        )

    return blocks


def _latest_user_index(prompt_messages: list[PromptMessage]) -> Optional[int]:
    for index in range(len(prompt_messages) - 1, -1, -1):
        if isinstance(prompt_messages[index], UserPromptMessage):
            return index
    return None


def _required_user_index(
    prompt_messages: list[PromptMessage],
    required_user_message: Optional[PromptMessage],
) -> Optional[int]:
    if required_user_message is not None:
        for index, message in enumerate(prompt_messages):
            if message is required_user_message:
                return index
    return _latest_user_index(prompt_messages)


def _historical_turns(
    prompt_messages: list[PromptMessage], latest_user_index: int
) -> list[tuple[int, ...]]:
    user_indices = [
        index
        for index, message in enumerate(prompt_messages[:latest_user_index])
        if isinstance(message, UserPromptMessage)
    ]
    turns: list[tuple[int, ...]] = []
    for position, start in enumerate(user_indices):
        end = (
            user_indices[position + 1]
            if position + 1 < len(user_indices)
            else latest_user_index
        )
        indices = tuple(
            index
            for index in range(start, end)
            if not isinstance(prompt_messages[index], SystemPromptMessage)
        )
        if indices:
            turns.append(indices)
    return turns


def _current_turn_blocks(
    prompt_messages: list[PromptMessage], start: int
) -> list[_Block]:
    blocks: list[_Block] = []
    index = start
    while index < len(prompt_messages):
        message = prompt_messages[index]
        if isinstance(message, AssistantPromptMessage) and message.tool_calls:
            call_ids = {str(call.id) for call in message.tool_calls if call.id}
            end = index + 1
            found_ids: set[str] = set()
            while end < len(prompt_messages) and isinstance(
                prompt_messages[end], ToolPromptMessage
            ):
                found_ids.add(str(prompt_messages[end].tool_call_id or ""))
                end += 1
                if found_ids == call_ids:
                    break
            blocks.append(_Block(indices=tuple(range(index, end)), priority=2))
            index = end
            continue

        blocks.append(_Block(indices=(index,), priority=2))
        index += 1
    return blocks
