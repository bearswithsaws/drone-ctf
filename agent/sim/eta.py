"""Cycle-by-cycle forecasts for a drone's pending FIFO action queue.

Queued actions report how many cycles they require, but their position,
heading, and battery effects are not visible until completion.  This module
projects those completion effects without mutating the world or entity
simulators, giving the pipeliner and pathfinder a stable view of where a drone
will be when each queued action finishes.

The forecast assumes queued actions complete successfully and in the supplied
order.  Collision and other precondition checks belong to the planner; a
server-side failure should cause the caller to rebuild the forecast from the
new authoritative state and queue.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from agent.rules.costs import estimate_battery_cost
from agent.rules.hexmath import get_neighbor

HexCoordinate = tuple[int, int]


@dataclass(frozen=True, slots=True)
class EntityState:
    """The drone values needed to forecast its action queue.

    ``position`` is an absolute odd-q coordinate and ``heading`` uses the game
    direction convention (0=NE, 1=SE, 2=S, 3=SW, 4=NW, 5=N).  Weight is kept
    on the input state because drive battery cost depends on it even though it
    is not part of each output snapshot.
    """

    position: HexCoordinate
    heading: int
    battery: int
    total_weight: float = 0.0

    def __post_init__(self) -> None:
        if len(self.position) != 2:
            raise ValueError("position must contain q and r")
        if not 0 <= self.heading <= 5:
            raise ValueError("heading must be between 0 and 5")
        if self.battery < 0:
            raise ValueError("battery cannot be negative")
        if self.total_weight < 0:
            raise ValueError("total_weight cannot be negative")


@dataclass(frozen=True, slots=True)
class PendingAction:
    """Normalised input for one queued action."""

    subject: str
    cycles_required: int
    details: Mapping[str, Any] = field(default_factory=dict)
    action_id: str | None = None

    def __post_init__(self) -> None:
        if self.cycles_required < 1:
            raise ValueError("cycles_required must be positive")


@dataclass(frozen=True, slots=True)
class CyclePrediction:
    """Projected drone state after one future cycle has run."""

    cycle: int
    position: HexCoordinate
    heading: int
    battery: int
    action_id: str | None
    subject: str
    action_index: int
    action_cycle: int
    completes_action: bool
    total_weight: float = 0.0

    @property
    def state(self) -> EntityState:
        """Return the projected values as a new input state.

        Weight is invariant across this forecast, but retaining it lets the
        returned state seed a later forecast without changing drive costs.
        """

        return EntityState(self.position, self.heading, self.battery, self.total_weight)


def predict_pipeline(
    initial: EntityState,
    pending_actions: Iterable[PendingAction | Mapping[str, Any] | Any],
    *,
    start_cycle: int = 0,
    research_tiers: Mapping[str, int] | None = None,
) -> list[CyclePrediction]:
    """Predict position, heading, and battery after every queued future cycle.

    ``start_cycle`` names the state represented by ``initial``.  The first
    returned item is therefore ``start_cycle + 1``.  Actions may be explicit
    :class:`PendingAction` objects, raw queue/message dictionaries, or Pydantic
    wire models.  Raw values accept both observed message fields
    (``subject``, nested ``details``, ``cycles_required``) and common REST queue
    variants (``action_type``, ``parameters``, ``cycles_remaining``).

    Effects are applied on the final cycle of each FIFO action.  Earlier cycles
    retain the prior state, matching the game's completion-message semantics.
    """

    cycle = int(start_cycle)
    position = initial.position
    heading = initial.heading
    battery = initial.battery
    tiers = {
        str(key): max(0, int(value)) for key, value in (research_tiers or {}).items()
    }
    predictions: list[CyclePrediction] = []

    for action_index, raw_action in enumerate(pending_actions):
        action = normalise_action(raw_action)
        for action_cycle in range(1, action.cycles_required + 1):
            cycle += 1
            completes = action_cycle == action.cycles_required
            if completes:
                position, heading, battery = _apply_completion(
                    position,
                    heading,
                    battery,
                    action,
                    total_weight=initial.total_weight,
                    research_tiers=tiers,
                )
            predictions.append(
                CyclePrediction(
                    cycle=cycle,
                    position=position,
                    heading=heading,
                    battery=battery,
                    action_id=action.action_id,
                    subject=action.subject,
                    action_index=action_index,
                    action_cycle=action_cycle,
                    completes_action=completes,
                    total_weight=initial.total_weight,
                )
            )

    return predictions


def predict_queue(
    initial: EntityState,
    pending_actions: Iterable[PendingAction | Mapping[str, Any] | Any],
    *,
    start_cycle: int = 0,
    research_tiers: Mapping[str, int] | None = None,
) -> list[CyclePrediction]:
    """Alias for :func:`predict_pipeline` using the API's queue terminology."""

    return predict_pipeline(
        initial,
        pending_actions,
        start_cycle=start_cycle,
        research_tiers=research_tiers,
    )


def completion_predictions(
    predictions: Iterable[CyclePrediction],
) -> list[CyclePrediction]:
    """Select the snapshots at which queued actions complete."""

    return [prediction for prediction in predictions if prediction.completes_action]


def normalise_action(action: PendingAction | Mapping[str, Any] | Any) -> PendingAction:
    """Convert a queue action or queued wire message to :class:`PendingAction`."""

    if isinstance(action, PendingAction):
        return action

    payload = _as_mapping(action)
    details = _first_mapping(payload, "details", "parameters", "params", "payload")

    # A few queue builds expose action parameters at the top level.  Preserve
    # those when no nested value already provides them.
    for key in (
        "direction",
        "level",
        "target_q",
        "target_r",
        "height_change",
        "elevation_change",
    ):
        if key in payload and key not in details:
            details[key] = payload[key]

    subject = _first_string(payload, "subject", "action_type", "action", "name", "endpoint")
    if not subject:
        raise ValueError("queued action must include subject or action type")

    cycles = _cycles_from(payload, details)
    action_id = _first_string(payload, "action_id", "id") or None
    return PendingAction(
        subject=subject,
        cycles_required=cycles,
        details=details,
        action_id=action_id,
    )


def _apply_completion(
    position: HexCoordinate,
    heading: int,
    battery: int,
    action: PendingAction,
    *,
    total_weight: float,
    research_tiers: Mapping[str, int],
) -> tuple[HexCoordinate, int, int]:
    action_name = _normalise_subject(action.subject)

    if _is_turn(action_name):
        turn_direction = action.details.get("direction")
        if turn_direction is None:
            turn_direction = _direction_from_subject(action_name, turn=True)
        heading = (heading + _turn_delta(turn_direction)) % 6
    elif _is_drive(action_name):
        drive_direction = action.details.get("direction")
        if drive_direction is None:
            drive_direction = _direction_from_subject(action_name, turn=False)
        direction = heading
        if _is_reverse(drive_direction):
            direction = (direction + 3) % 6
        position = get_neighbor(*position, direction)

    completed_subject = _completed_subject(action.subject)
    if _is_turn(action_name):
        completed_subject = "turn completed"
    elif _is_drive(action_name):
        completed_subject = "drive completed"
    battery_cost = estimate_battery_cost(
        completed_subject,
        action.details,
        research_tiers=research_tiers,
        total_weight=total_weight,
    )
    return position, heading, max(0, battery - battery_cost)


def _cycles_from(payload: Mapping[str, Any], details: Mapping[str, Any]) -> int:
    # Remaining-cycle fields matter when forecasting a queue whose head action
    # is already in progress.  Fall back to the original required duration for
    # freshly queued messages and untouched tail actions.
    key_groups = (
        ("cycles_remaining", "remaining_cycles"),
        ("cycles_required", "cycles"),
    )
    for keys in key_groups:
        for source in (payload, details):
            for key in keys:
                if key not in source:
                    continue
                value = source[key]
                if isinstance(value, bool):
                    break
                try:
                    cycles = int(value)
                except (TypeError, ValueError):
                    break
                if cycles < 1:
                    raise ValueError(f"{key} must be positive")
                return cycles
    raise ValueError("queued action must include its required or remaining cycles")


def _normalise_subject(subject: str) -> str:
    return " ".join(subject.lower().replace("_", " ").replace("/", " ").split())


def _completed_subject(subject: str) -> str:
    normalised = _normalise_subject(subject)
    if "queued" in normalised:
        return normalised.replace("queued", "completed")
    if "completed" not in normalised:
        return f"{normalised} completed"
    return normalised


def _is_turn(action_name: str) -> bool:
    return "turn" in action_name.split()


def _is_drive(action_name: str) -> bool:
    return "drive" in action_name.split()


def _direction_from_subject(action_name: str, *, turn: bool) -> int:
    words = set(action_name.split())
    if turn:
        if words & {"right", "clockwise", "cw"}:
            return 1
        if words & {"left", "counterclockwise", "ccw"}:
            return -1
    else:
        if words & {"forward", "forwards"}:
            return 1
        if words & {"back", "backward", "backwards", "reverse"}:
            return -1
    kind = "turn" if turn else "drive"
    raise ValueError(f"{kind} action must include a direction")


def _turn_delta(direction: Any) -> int:
    if isinstance(direction, bool):
        raise ValueError("turn direction must be left/right or -1/1")
    if isinstance(direction, (int, float)):
        if direction == 1:
            return 1
        if direction == -1:
            return -1
    value = str(direction).strip().lower().replace("-", "_")
    if value in {"1", "right", "clockwise", "clock_wise", "cw"}:
        return 1
    if value in {"-1", "left", "counterclockwise", "counter_clockwise", "ccw"}:
        return -1
    raise ValueError(f"unsupported turn direction: {direction!r}")


def _is_reverse(direction: Any) -> bool:
    if isinstance(direction, bool):
        raise ValueError("drive direction must be forward/reverse or -1/1")
    if isinstance(direction, (int, float)):
        if direction == 1:
            return False
        if direction == -1:
            return True
    value = str(direction).strip().lower()
    if value in {"1", "forward", "forwards"}:
        return False
    if value in {"-1", "back", "backward", "backwards", "reverse"}:
        return True
    raise ValueError(f"unsupported drive direction: {direction!r}")


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        payload = model_dump()
        if isinstance(payload, Mapping):
            return dict(payload)
    raise TypeError("queued action must be a mapping, Pydantic model, or PendingAction")


def _first_mapping(payload: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _first_string(payload: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return ""


__all__ = [
    "CyclePrediction",
    "EntityState",
    "PendingAction",
    "completion_predictions",
    "normalise_action",
    "predict_pipeline",
    "predict_queue",
]
