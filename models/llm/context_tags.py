from enum import StrEnum


class FlyfusContextTag(StrEnum):
    """Flyfus message protocol tags and their intended consumers."""

    # Returned to the frontend and kept in model content.
    CONTEXT = "FLYFUS_CONTEXT"
    FILE = "FLYFUS_FILE"

    # Reserved for model-visible context that the application hides from the frontend.
    # The provider does not extract attachments from this tag yet.
    INTERNAL_CONTEXT = "FLYFUS_INTERNAL_CONTEXT"

    # Removed from model content after being consumed by model parameter parsing.
    SETTING = "FLYFUS_SETTING"


class FlyfusSettingType(StrEnum):
    AI_MODE = "ai_mode"
    LOG_CONTEXT = "log_context"
