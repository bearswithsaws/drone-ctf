"""Decode wire messages while retaining and reporting unknown payloads."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Mapping

from pydantic import ValidationError

from agent.transport.schema import (
    ActionCompletedMessage,
    ActionFailedMessage,
    ActionQueuedMessage,
    ActionRejectedMessage,
    BuildingActionCompletedMessage,
    BuildingActionQueuedMessage,
    InfoMessage,
    ParsedWireMessage,
    UnknownWireMessage,
    WarningMessage,
    WireMessage,
)

logger = logging.getLogger(__name__)

_MESSAGE_MODELS: dict[str, type[WireMessage]] = {
    "action_completed": ActionCompletedMessage,
    "action_failed": ActionFailedMessage,
    "action_queued": ActionQueuedMessage,
    "action_rejected": ActionRejectedMessage,
    "building_action_completed": BuildingActionCompletedMessage,
    "building_action_queued": BuildingActionQueuedMessage,
    "info": InfoMessage,
    "warning": WarningMessage,
}


def _preserve_unknown(payload: dict[str, Any]) -> UnknownWireMessage:
    message = UnknownWireMessage.model_validate(payload)
    message._raw_payload = deepcopy(payload)
    return message


def parse_wire_message(payload: Mapping[str, Any]) -> ParsedWireMessage:
    """Validate a server message, logging and preserving anything unrecognised."""

    raw = deepcopy(dict(payload))
    message_type = raw.get("message_type")
    model = _MESSAGE_MODELS.get(message_type) if isinstance(message_type, str) else None
    if model is None:
        logger.warning("Unknown wire message type %r; payload=%r", message_type, raw)
        return _preserve_unknown(raw)

    try:
        return model.model_validate(raw)
    except ValidationError:
        logger.warning(
            "Unrecognised %s wire payload; payload=%r",
            message_type,
            raw,
            exc_info=True,
        )
        return _preserve_unknown(raw)
