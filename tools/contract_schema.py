#!/usr/bin/env python3
"""Generate or verify the commander contract's Pydantic JSON Schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "packages" / "contract" / "schema" / "contract-v1.schema.json"

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent.commander.contract import CONTRACT_VERSION, ContractV1  # noqa: E402


def render_schema() -> str:
    schema = ContractV1.model_json_schema(
        mode="validation",
        ref_template="#/$defs/{model}",
    )
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"https://drone-ctf.dev/schemas/commander-contract-{CONTRACT_VERSION}.json"
    schema["title"] = "Drone CTF Commander Contract v1"
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def check_schema(output: Path = DEFAULT_OUTPUT) -> bool:
    expected = render_schema()
    try:
        actual = output.read_text(encoding="utf-8")
    except FileNotFoundError:
        return False
    return actual == expected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the checked-in schema is stale",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output = args.output.resolve()
    if args.check:
        if check_schema(output):
            print(f"commander contract schema is current: {output}")
            return 0
        print(
            f"commander contract schema is stale: {output}\n"
            "run `npm run generate` in packages/contract",
            file=sys.stderr,
        )
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_schema(), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
