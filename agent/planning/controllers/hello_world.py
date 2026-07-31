"""Minimal end-to-end autonomy loop used to prove the transport stack.

The controller deliberately knows nothing about the world model.  It verifies
a straight route from live status/scan results, drives three tiles forward,
and then reverses every successful outbound move.  Each command is submitted via
the :class:`~agent.transport.action_tracker.ActionTracker` and the next command
is not issued until the previous intent reaches a terminal tracked state.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Callable

from agent.rules.hexmath import get_neighbor
from agent.transport.action_tracker import ActionTracker, EntityKind, Intent, IntentState

log = logging.getLogger("agent.planning.hello_world")


@dataclass(frozen=True)
class HelloWorldResult:
    """Summary of one scan/out-and-back proof loop."""

    scan_state: IntentState
    outbound_moves: int
    return_moves: int

    @property
    def returned(self) -> bool:
        return self.outbound_moves == self.return_moves


class HelloWorldController:
    """Run the hardcoded E1.7 scan, three-tile drive, and return loop."""

    def __init__(
        self,
        tracker: ActionTracker,
        drone_id: str,
        *,
        move_count: int = 3,
        action_timeout_s: float = 120.0,
        poll_interval_s: float = 0.05,
        failure_backoff_s: float = 1.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if move_count < 1:
            raise ValueError("move_count must be positive")
        if action_timeout_s <= 0:
            raise ValueError("action_timeout_s must be positive")

        self._tracker = tracker
        self.drone_id = drone_id
        self.move_count = move_count
        self._action_timeout = action_timeout_s
        self._poll_interval = poll_interval_s
        self._failure_backoff = failure_backoff_s
        self._clock = clock

        self._sequence = 0
        self._loop_number = 0
        self._direction: int | None = None
        self._prepared_scan: Intent | None = None
        # Relative displacement along the drone's starting heading.  It is
        # changed only after a tracked COMPLETED action, so retry
        # preconditions remain safe when a command is definitely rejected.
        self._displacement = 0

    @property
    def displacement(self) -> int:
        return self._displacement

    async def prepare(self) -> bool:
        """Verify this drone has a known-clear three-tile straight route."""

        status = await self._submit_and_wait(
            label="route-status",
            action="status",
            payload={},
            precondition=lambda: self._displacement == 0,
        )
        direction = _status_direction(status)
        if status.state is not IntentState.COMPLETED or direction is None:
            log.warning("Route preflight rejected drone %s: no terminal status", self.drone_id)
            return False

        scan = await self._submit_and_wait(
            label="route-scan",
            action="sensors/scan",
            payload={"level": 1},
            precondition=lambda: self._displacement == 0,
        )
        if scan.state is not IntentState.COMPLETED or not _scan_clears_straight_route(
            scan, direction, self.move_count
        ):
            log.warning("Route preflight rejected drone %s: no clear route", self.drone_id)
            return False

        self._direction = direction
        self._prepared_scan = scan
        return True

    async def run_once(self) -> HelloWorldResult:
        """Complete one proof loop, returning only after every action is terminal."""

        self._loop_number += 1
        loop_number = self._loop_number
        log.info("Hello-world loop %d starting (drone=%s)", loop_number, self.drone_id)

        # A previous loop may have suffered a failed reverse move.  Restore the
        # invariant before starting another outbound leg.
        recovered = await self._return_home(loop_number, phase="recovery")
        if self._displacement:
            log.warning(
                "Hello-world loop %d could not recover the final %d tile(s)",
                loop_number,
                self._displacement,
            )
            return HelloWorldResult(IntentState.ABANDONED, 0, recovered)

        scan = self._prepared_scan
        self._prepared_scan = None
        if scan is None:
            scan = await self._submit_and_wait(
                label="scan",
                action="sensors/scan",
                payload={"level": 1},
                precondition=lambda: self._displacement == 0,
            )
        if scan.state is not IntentState.COMPLETED:
            log.warning("Hello-world loop %d scan ended as %s", loop_number, scan.state.value)
            return HelloWorldResult(scan.state, 0, 0)
        if self._direction is None or not _scan_clears_straight_route(
            scan, self._direction, self.move_count
        ):
            log.warning("Hello-world loop %d has no verified clear route", loop_number)
            return HelloWorldResult(IntentState.ABANDONED, 0, 0)

        outbound = 0
        for index in range(self.move_count):
            expected = self._displacement
            move = await self._submit_and_wait(
                label=f"out-{index + 1}",
                action="propulsion/drive",
                payload={"direction": 1, "level": 1},
                precondition=lambda expected=expected: self._displacement == expected,
            )
            if move.state is not IntentState.COMPLETED:
                log.warning(
                    "Hello-world loop %d outbound move %d ended as %s",
                    loop_number,
                    index + 1,
                    move.state.value,
                )
                break
            self._displacement += 1
            outbound += 1

        returned = await self._return_home(loop_number, phase="return")
        result = HelloWorldResult(scan.state, outbound, returned)
        log.info(
            "Hello-world loop %d terminal (outbound=%d returned=%d at_home=%s)",
            loop_number,
            result.outbound_moves,
            result.return_moves,
            self._displacement == 0,
        )
        return result

    async def run(self, *, duration_s: float | None = None) -> int:
        """Run proof loops indefinitely, or until ``duration_s`` has elapsed.

        The duration is checked between complete loops so a finite acceptance
        run never exits with a locally submitted action still non-terminal.
        Returns the number of loops attempted.
        """

        if duration_s is not None and duration_s < 0:
            raise ValueError("duration_s cannot be negative")

        loop = asyncio.get_running_loop()
        clock = self._clock or loop.time
        deadline = None if duration_s is None else clock() + duration_s
        attempted = 0

        while deadline is None or clock() < deadline:
            result = await self.run_once()
            attempted += 1
            if (result.scan_state is not IntentState.COMPLETED or not result.returned) and (
                deadline is None or clock() < deadline
            ):
                await asyncio.sleep(self._failure_backoff)
        return attempted

    async def _return_home(self, loop_number: int, *, phase: str) -> int:
        returned = 0
        while self._displacement > 0:
            expected = self._displacement
            move = await self._submit_and_wait(
                label=f"{phase}-{expected}",
                action="propulsion/drive",
                payload={"direction": -1, "level": 1},
                precondition=lambda expected=expected: self._displacement == expected,
            )
            if move.state is not IntentState.COMPLETED:
                log.warning(
                    "Hello-world loop %d %s move ended as %s",
                    loop_number,
                    phase,
                    move.state.value,
                )
                break
            self._displacement -= 1
            returned += 1
        return returned

    async def _submit_and_wait(
        self,
        *,
        label: str,
        action: str,
        payload: dict[str, int],
        precondition: Callable[[], bool],
    ) -> Intent:
        self._sequence += 1
        local_id = f"hello-world:{self.drone_id}:{self._sequence}:{label}"
        intent = await self._tracker.submit(
            local_id,
            EntityKind.DRONE,
            self.drone_id,
            action,
            payload=payload,
            distance=float(self._displacement),
            precondition=precondition,
        )
        if not intent.terminal:
            intent = await self._wait_for_terminal(local_id)

        log.info(
            "Action terminal (local_id=%s action_id=%s state=%s attempts=%d inferred=%s)",
            local_id,
            intent.action_id,
            intent.state.value,
            intent.attempts,
            intent.inferred_terminal,
        )
        return intent

    async def _wait_for_terminal(self, local_id: str) -> Intent:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._action_timeout
        while True:
            intent = self._tracker.get(local_id)
            if intent is None:
                raise RuntimeError(f"ActionTracker lost registered intent {local_id!r}")
            if intent.terminal:
                return intent

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(
                    f"action {local_id!r} did not reach a terminal state "
                    f"within {self._action_timeout:g}s"
                )
            await asyncio.sleep(min(self._poll_interval, remaining))


def _status_direction(intent: Intent) -> int | None:
    details = (intent.last_message or {}).get("details")
    status = details.get("status") if isinstance(details, dict) else None
    direction = status.get("direction") if isinstance(status, dict) else None
    return (
        direction
        if isinstance(direction, int) and not isinstance(direction, bool) and 0 <= direction < 6
        else None
    )


def _scan_clears_straight_route(intent: Intent, direction: int, move_count: int) -> bool:
    """Fail closed unless both possible odd-q path variants are known clear."""

    details = (intent.last_message or {}).get("details")
    scan_data = details.get("scan_data") if isinstance(details, dict) else None
    raw_tiles = scan_data.get("tiles") if isinstance(scan_data, dict) else None
    if not isinstance(raw_tiles, list):
        return False

    tiles: dict[tuple[int, int], dict[str, object]] = {}
    for tile in raw_tiles:
        coordinates = tile.get("coordinates") if isinstance(tile, dict) else None
        if (
            isinstance(coordinates, (list, tuple))
            and len(coordinates) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in coordinates)
        ):
            tiles[(coordinates[0], coordinates[1])] = tile

    for starting_q in (0, 1):
        q, r = starting_q, 0
        for _ in range(move_count):
            q, r = get_neighbor(q, r, direction)
            tile = tiles.get((q - starting_q, r))
            if tile is None or tile.get("terrain_type") not in {1, 2}:
                return False
            if tile.get("has_building") is not False or tile.get("has_drone") is not False:
                return False
    return True
