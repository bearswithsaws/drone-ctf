# Arming path — empirical calibration (HUNT-T1)

Probed live against `adam.dronebattles.ca` as the team player (`user1`) with
`tools/calibrate/probe_arming.py`. These findings finalize HUNT-T3–T5.

## Verdicts

| Question | Answer |
|---|---|
| `POST /drones/<id>/equipment {equipment_type}` player-usable? | **No — HTTP 404 `NOT_FOUND`.** The docs mark it "Public" but it is not reachable with a player token. Arming must go through the equipment plant. |
| Where does `production/manufacture` draw resources? | **The equipment plant's own `stored_resources`** — which starts **empty** (`{}`). With no stock it returns `building_action_failed` "Cannot build equipment" carrying `missing_resources` + `stored_resources`. ⇒ **we must deliver refined resources to the plant (HUNT-T4 is required).** |
| `equipment_plant/install` coordinate frame? | **Relative to the equipment-plant origin.** Absolute drone coords → HTTP 404 `DRONE_NOT_FOUND`. `(drone_abs − plant_origin)` → HTTP 201 queued, and the `building_action_queued` message resolved the correct `drone_id` from the position. |
| Adjacency enforced at queue time? | **Not at queue time** — install queued (201) with the drone 2 tiles from the plant's near edge and an empty plant. Completion behavior with a non-adjacent drone / empty plant is untested (the action fails later or is a no-op). **T5 still routes the drone adjacent before installing** — don't rely on the server's leniency. |

## Concrete numbers from this run

- Equipment plant: `origin=[4,2]`, `building_type=drone_equipment_production_plant`.
- Drone plant: `origin=[-4,2]`. Command center: `origin=[0,0]`.
- Probe drone `c66a5cfe…`: abs location `[4,0]`, `equipment_slots=6`, `available_equipment_slots=2` (propulsion+drill+sensors+hopper) → **room for 2 lasers, no removal needed**.
- Laser cost confirmed: `missing_resources={titanium_parts:18, battery_materials:12}` — matches `EQUIPMENT_COSTS["laser_cannon"]`.
- Install queued payload: `{cycles:25, efficiency:1.0, equipment_type:"laser_cannon"}`; wire subject `Building install_equipment queued`, message carries `building_id` + `drone_id`.

## Wire message shapes to model (feeds HUNT-T2, ties to #89)

```
building_action_failed "Cannot build equipment"
  details: {building_id, equipment_type, missing_resources:{...}, stored_resources:{...}}
building_action_queued "Building install_equipment queued"
  details: {cycles:25, efficiency, equipment_type}   + top-level building_id, drone_id
```
The `stored_resources` in the build-failed message is a usable signal for the
plant's current stock (the per-building status_report entry does **not** include
`stored_resources`).

## Implications for the plan

- **HUNT-T3** manufacture: emit `production/manufacture {efficiency, equipment_type}`
  at the equipment plant, affordability-gated on the plant's own stock (18 Ti +
  12 BM per laser).
- **HUNT-T4** (required, not optional): haul refined `titanium_parts` +
  `battery_materials` to the equipment plant before manufacture can succeed.
- **HUNT-T5** install: emit `equipment_plant/install` with `q,r = drone_abs −
  plant_origin` (building-origin-relative), after routing the drone adjacent to
  the plant footprint.
