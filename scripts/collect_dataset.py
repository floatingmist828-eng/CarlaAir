from __future__ import annotations

import argparse
from datetime import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carlaair_active_world.env import ActiveAirGroundEnv
from carlaair_active_world.policies import build_policy
from carlaair_active_world.recorder import EpisodeRecorder
from carlaair_active_world.scenario import ScenarioConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect CarlaAir active air-ground V0 episodes.")
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--policy", type=str, default="heuristic")
    parser.add_argument("--policy-index", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("recordings") / "active_world")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    scenario = ScenarioConfig.load(args.scenario)
    policy = build_policy(args.policy, index=args.policy_index, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for episode_idx in range(args.episodes):
        env = ActiveAirGroundEnv(scenario)
        recorder = EpisodeRecorder(
            output_path=args.output_dir / f"{scenario.name}_{policy.name}_{episode_idx:03d}.json",
            meta={
                "scenario": scenario.to_dict(),
                "policy": policy.name,
                "episode_index": episode_idx,
            },
        )
        observation = env.reset()
        done = False
        step_idx = 0
        try:
            while not done:
                action = policy.select(observation, scenario.candidate_offsets)
                result = env.step(action)
                observation = result.observation
                done = result.done
                step_idx += 1
                recorder.append(
                    {
                        "step": step_idx,
                        "action": int(action),
                        "observation": observation,
                        "label": result.label,
                        "info": result.info,
                    }
                )
        finally:
            env.close()
            saved = recorder.save()
            print(f"Saved episode {episode_idx}: {saved}")


if __name__ == "__main__":
    main()
