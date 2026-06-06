from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from carlaair_active_world.env import ActiveAirGroundEnv
from carlaair_active_world.policies import build_policy
from carlaair_active_world.recorder import EpisodeRecorder
from carlaair_active_world.scenario import ScenarioConfig
from carlaair_active_world.sensors import save_numpy_image


def show_ego_view(env: ActiveAirGroundEnv, window_name: str = "EgoVehicleRGB") -> bool:
    try:
        import cv2
    except Exception as exc:
        print(f"Viewer unavailable: OpenCV import failed: {exc}", file=sys.stderr)
        return False

    rig = getattr(getattr(env, "ego_driver", None), "sensor_rig", None)
    if rig is None:
        print("Viewer unavailable: ego driver has no vehicle camera rig.", file=sys.stderr)
        return False

    frames = rig.snapshot()
    rgb = frames.get("rgb")
    if rgb is None:
        return True

    try:
        cv2.imshow(window_name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
    except Exception as exc:
        print(f"Viewer unavailable: OpenCV display failed: {exc}", file=sys.stderr)
        return False
    return True


def load_scenario(path: Path) -> ScenarioConfig:
    return ScenarioConfig.load(path)


def save_step_artifacts(env: ActiveAirGroundEnv, step_idx: int, artifact_dir: Path) -> None:
    step_dir = artifact_dir / f"step_{step_idx:06d}"
    driver = getattr(env, "ego_driver", None)
    payload = {
        "step": int(step_idx),
        "ego_control": dict(getattr(driver, "last_diagnostics", {}) or {}),
    }

    rig = getattr(driver, "sensor_rig", None)
    if rig is not None:
        try:
            frames = rig.snapshot() or {}
            if frames.get("rgb") is not None:
                path = step_dir / "ego_rgb.png"
                save_numpy_image(path, frames["rgb"])
                payload["ego_rgb"] = str(path)
        except Exception as exc:
            payload["ego_rgb_error"] = f"{type(exc).__name__}: {exc}"

    uav_rig = getattr(env, "uav_sensors", None)
    if uav_rig is not None:
        try:
            frames = uav_rig.snapshot() or {}
            if frames.get("rgb") is not None:
                path = step_dir / "uav_rgb.png"
                save_numpy_image(path, frames["rgb"])
                payload["uav_rgb"] = str(path)
        except Exception as exc:
            payload["uav_rgb_error"] = f"{type(exc).__name__}: {exc}"

    step_dir.mkdir(parents=True, exist_ok=True)
    with (step_dir / "control.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CarlaAir active air-ground V0.")
    parser.add_argument("--scenario", type=Path, required=True, help="Path to a scenario JSON file.")
    parser.add_argument("--policy", type=str, default="heuristic", help="fixed|random|follow|intersection|heuristic")
    parser.add_argument("--policy-index", type=int, default=0, help="Fixed policy candidate index.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--record", type=Path, default=None, help="Optional output JSON file.")
    parser.add_argument("--viewer", action="store_true", help="Show the ego vehicle RGB camera in an OpenCV window.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=None,
        help="Save ego/UAV frames and control diagnostics for visual analysis.",
    )
    args = parser.parse_args()

    scenario = load_scenario(args.scenario)
    env = ActiveAirGroundEnv(scenario)
    policy = build_policy(args.policy, index=args.policy_index, seed=args.seed)
    recorder = None
    if args.record is not None:
        recorder = EpisodeRecorder(
            output_path=args.record,
            meta={
                "scenario": scenario.to_dict(),
                "policy": policy.name,
            },
        )

    observation = env.reset()
    viewer_enabled = bool(args.viewer)
    if viewer_enabled:
        viewer_enabled = show_ego_view(env)
    print(f"Scenario: {scenario.name}")
    print(f"Policy: {policy.name}")
    print(f"Candidates: {[c.name for c in scenario.candidate_offsets]}")

    done = False
    step_idx = 0
    try:
        while not done:
            action = policy.select(observation, scenario.candidate_offsets)
            result = env.step(action)
            observation = result.observation
            done = result.done
            if viewer_enabled:
                viewer_enabled = show_ego_view(env)
            step_idx += 1
            if args.artifact_dir is not None:
                save_step_artifacts(env, step_idx, args.artifact_dir)

            candidate_name = scenario.candidate_offsets[action].name
            print(
                f"step={step_idx:03d} t={observation['time']:.2f} "
                f"action={action}:{candidate_name} risk={result.label.get('risk_proxy', 0.0):.3f}"
            )

            if recorder is not None:
                recorder.append(
                    {
                        "step": step_idx,
                        "action": int(action),
                        "candidate_name": candidate_name,
                        "observation": observation,
                        "label": result.label,
                        "info": result.info,
                    }
                )
    finally:
        env.close()
        if recorder is not None:
            saved = recorder.save()
            print(f"Saved: {saved}")


if __name__ == "__main__":
    main()
