from __future__ import annotations

from dataclasses import dataclass

import pytest

from agent.planning.controllers.base import (
    BuildOrder,
    ProductionController,
    RefinerController,
    ResearchController,
    ResearchStep,
)
from agent.planning.pipeliner import Pipeliner
from agent.transport.action_tracker import EntityKind


def test_refiner_plans_one_bounded_priority_batch() -> None:
    controller = RefinerController(
        "refiner-1",
        batch_size=4,
        ore_priority=("unobtanium_ore", "titanium_ore"),
        efficiency=0.8,
    )

    plan = controller.plan({"unobtanium_ore": 2, "titanium_ore": 20})

    assert plan.refiner_id == "refiner-1"
    assert plan.ore_type == "unobtanium_ore"
    assert plan.units == 2
    assert [action.action for action in plan.actions] == ["refiner/process"] * 2
    assert [action.payload for action in plan.actions] == [
        {"efficiency": 0.8, "ore_type": "unobtanium_ore"},
        {"efficiency": 0.8, "ore_type": "unobtanium_ore"},
    ]


def test_refiner_caps_large_batch_and_returns_empty_for_no_ore() -> None:
    controller = RefinerController("refiner-1", batch_size=3)

    assert controller.plan({"titanium_ore": 8}).units == 3
    empty = controller.plan({"titanium_ore": 0, "rare_earth_ore": 0})
    assert empty.empty
    assert empty.ore_type is None


def test_refiner_live_inventory_guards_retries() -> None:
    inventory = {"titanium_ore": 2}
    plan = RefinerController("refiner-1").plan(lambda: inventory)

    assert plan.actions[0].precondition()
    inventory["titanium_ore"] = 0
    assert not plan.actions[0].precondition()


def test_research_advances_through_live_tiers_and_never_exhausts() -> None:
    tiers = {"unlock_hover_tech": 0, "propulsion": 1}
    controller = ResearchController(
        "command-1",
        (
            ResearchStep("unlock_hover_tech", 1),
            ResearchStep("propulsion", 2),
        ),
        lambda: tiers,
        fallback_key="baseline_drone",
    )

    first = controller.next_action()
    assert first.payload["equipment_type"] == "unlock_hover_tech"

    tiers["unlock_hover_tech"] = 1
    second = controller.next_action()
    assert second.payload["equipment_type"] == "propulsion"

    tiers["propulsion"] = 2
    assert [controller.next_action().payload["equipment_type"] for _ in range(20)] == [
        "baseline_drone"
    ] * 20


def test_research_accepts_task_and_string_order_entries() -> None:
    from agent.planning.tasks import Research

    controller = ResearchController(
        "command-1",
        (Research("unlock", research_key="unlock_repeater"), "sensors"),
        {},
    )

    assert controller.research_order == (
        ResearchStep("unlock_repeater", 1),
        ResearchStep("sensors", 3),
    )


def test_production_expands_build_order_per_plant_in_order() -> None:
    controller = ProductionController("drone-plant", "equipment-plant", efficiency=0.75)
    order = BuildOrder.from_loadouts((("drill", "hopper"), ("sensors", "battery")))

    plan = controller.plan(order)

    assert len(plan.drone_actions) == 2
    assert [action.payload for action in plan.drone_actions] == [
        {"efficiency": 0.75},
        {"efficiency": 0.75},
    ]
    assert [action.payload["equipment_type"] for action in plan.equipment_actions] == [
        "drill",
        "hopper",
        "sensors",
        "battery",
    ]
    assert tuple(plan.outputs) == ("drone-plant", "equipment-plant")
    assert plan.actions_for("equipment-plant") == plan.equipment_actions
    assert plan.actions_for("unknown") == ()


def test_production_requires_the_plant_requested_by_build_order() -> None:
    with pytest.raises(ValueError, match="no drone plant"):
        ProductionController(None, "equipment-plant").plan(BuildOrder(drones=1))
    with pytest.raises(ValueError, match="no equipment plant"):
        ProductionController("drone-plant", None).plan(BuildOrder(equipment=("drill",)))


@dataclass
class _Result:
    data: dict
    ok: bool = True


class _Rest:
    async def get(self, path: str, params: dict | None = None) -> _Result:
        del path, params
        return _Result({"actions": []})

    async def delete(self, path: str, json: dict | None = None) -> _Result:
        del path, json
        return _Result({})


@dataclass
class _Intent:
    action_id: str


class _Tracker:
    def __init__(self) -> None:
        self.submissions: list[dict] = []

    async def submit(
        self,
        local_id: str,
        entity_kind: EntityKind | str,
        entity_id: str,
        action: str,
        **kwargs: object,
    ) -> _Intent:
        self.submissions.append(
            {
                "local_id": local_id,
                "entity_kind": EntityKind(entity_kind),
                "entity_id": entity_id,
                "action": action,
                **kwargs,
            }
        )
        return _Intent(f"action-{len(self.submissions)}")


@pytest.mark.asyncio
async def test_research_factory_refills_every_empty_queue_cycle() -> None:
    tracker = _Tracker()
    pipeline = Pipeliner(_Rest(), tracker, target_depth=4)
    research = ResearchController("command-1", ("sensors",), {})
    pipeline.register(EntityKind.BUILDING, "command-1", research.output)

    first = await pipeline.maintain(EntityKind.BUILDING, "command-1")
    second = await pipeline.maintain(EntityKind.BUILDING, "command-1")

    assert first.submitted == 4
    assert second.submitted == 4
    assert len(tracker.submissions) == 8
    assert {submission["action"] for submission in tracker.submissions} == {
        "command_center/research"
    }
    assert not first.exhausted
    assert not second.exhausted


@pytest.mark.parametrize(
    ("factory", "match"),
    [
        (lambda: RefinerController("r", batch_size=0), "batch_size"),
        (lambda: ResearchStep("sensors", 4), "cannot exceed 3"),
        (lambda: BuildOrder(drones=-1), "drones"),
        (lambda: ProductionController("same", "same"), "different ids"),
    ],
)
def test_base_controller_configuration_validation(factory: object, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        factory()  # type: ignore[operator]
