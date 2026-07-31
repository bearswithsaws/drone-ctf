"""Task declarations are immutable, typed descriptions of strategic work."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agent.planning.tasks import (
    DeployRepeater,
    Escort,
    LayMines,
    MineLoop,
    ProduceDrone,
    Research,
    ScoutSector,
    Strike,
    Task,
)


def test_task_families_capture_controller_inputs() -> None:
    tasks = [
        MineLoop("mine-iron", (2, 3), "silo-1", "titanium_ore"),
        ScoutSector("scout-north", (5, -2), radius=4),
        Escort("escort-miner", "miner-1"),
        Strike("strike-base", "enemy-cc"),
        LayMines("minefield-west", [(0, 1), (0, 2)]),  # type: ignore[arg-type]
        DeployRepeater("relay-ridge", (10, 4)),
        Research("research-drive", "propulsion", 2),
        ProduceDrone("produce-scout", ["propulsion", "sensor_array"]),  # type: ignore[arg-type]
    ]

    assert [task.kind for task in tasks] == [
        "mine_loop",
        "scout_sector",
        "escort",
        "strike",
        "lay_mines",
        "deploy_repeater",
        "research",
        "produce_drone",
    ]
    assert tasks[4].positions == ((0, 1), (0, 2))  # type: ignore[attr-defined]
    assert tasks[7].equipment == ("propulsion", "sensor_array")  # type: ignore[attr-defined]


def test_common_task_policy_is_validated_and_immutable() -> None:
    task = Task("hold", priority=2, max_assignees=3, metadata={"doctrine": "defend"})

    with pytest.raises(FrozenInstanceError):
        task.priority = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        task.metadata["doctrine"] = "raid"  # type: ignore[index]

    with pytest.raises(ValueError, match="task_id"):
        Task(" ")
    with pytest.raises(ValueError, match="priority"):
        Task("bad-priority", priority=0)
    with pytest.raises(ValueError, match="max_assignees"):
        Task("bad-capacity", max_assignees=0)


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: MineLoop("mine", (1, 2), ""), "dropoff_id"),
        (lambda: ScoutSector("scout", (1, 2), radius=-1), "radius"),
        (lambda: Escort("escort", ""), "target_id"),
        (lambda: Strike("strike", ""), "target"),
        (lambda: LayMines("mines", ()), "positions"),
        (lambda: DeployRepeater("relay", (1,)), "position"),  # type: ignore[arg-type]
        (lambda: Research("research", "drive", 0), "target_level"),
        (lambda: ProduceDrone("drone", ("",)), "equipment"),
    ],
)
def test_specific_tasks_reject_invalid_controller_inputs(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()
