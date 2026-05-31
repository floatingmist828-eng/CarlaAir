from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_episode(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recorded CarlaAir active world episodes.")
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No episode files found in {args.input_dir}")

    total_steps = 0
    total_risk = 0.0
    total_episodes = 0

    for path in files:
        episode = load_episode(path)
        steps = episode.get("steps", [])
        total_episodes += 1
        total_steps += len(steps)
        for step in steps:
            total_risk += float(step.get("label", {}).get("risk_proxy", 0.0))

    avg_steps = total_steps / max(1, total_episodes)
    avg_risk = total_risk / max(1, total_steps)
    print(f"episodes={total_episodes}")
    print(f"avg_steps={avg_steps:.2f}")
    print(f"avg_risk={avg_risk:.4f}")


if __name__ == "__main__":
    main()
