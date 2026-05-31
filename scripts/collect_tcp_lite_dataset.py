from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_distances(value: str) -> List[float]:
    distances = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not distances:
        raise ValueError("At least one trajectory distance is required.")
    return distances


def local_xy(ego_transform: carla.Transform, location: carla.Location) -> List[float]:
    ego_loc = ego_transform.location
    yaw = math.radians(float(ego_transform.rotation.yaw))
    dx = float(location.x - ego_loc.x)
    dy = float(location.y - ego_loc.y)
    local_x = dx * math.cos(-yaw) - dy * math.sin(-yaw)
    local_y = dx * math.sin(-yaw) + dy * math.cos(-yaw)
    return [float(local_x), float(local_y)]


def future_route_trajectory(
    world: Any,
    vehicle: Any,
    distances_m: Sequence[float],
) -> List[List[float]]:
    import carla

    ego_transform = vehicle.get_transform()
    fallback = [[float(distance), 0.0] for distance in distances_m]
    try:
        waypoint = world.get_map().get_waypoint(
            vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except Exception:
        return fallback

    points: List[List[float]] = []
    for distance in distances_m:
        try:
            candidates = waypoint.next(max(0.5, float(distance)))
        except Exception:
            candidates = []
        if not candidates:
            points.append(fallback[len(points)])
            continue
        best = None
        best_cost = None
        for candidate in candidates:
            xy = local_xy(ego_transform, candidate.transform.location)
            cost = abs(xy[1])
            if best_cost is None or cost < best_cost:
                best = xy
                best_cost = cost
        points.append(best if best is not None else fallback[len(points)])
    return points


def control_from_diagnostics(diagnostics: Dict[str, Any]) -> Optional[Dict[str, float]]:
    required = ("steer", "throttle", "brake")
    if not all(key in diagnostics for key in required):
        return None
    try:
        return {key: float(diagnostics[key]) for key in required}
    except (TypeError, ValueError):
        return None


def driver_decision_observation(driver: Any) -> Dict[str, Any]:
    observation = dict(getattr(driver, "last_observation", {}) or {})
    if observation.get("rgb") is not None:
        return observation

    sensor_rig = getattr(driver, "sensor_rig", None)
    if sensor_rig is None:
        return observation
    frames = dict(sensor_rig.snapshot() or {})
    if frames.get("rgb") is not None:
        observation["rgb"] = frames.get("rgb")
    return observation


def should_keep_tcp_lite_sample(diagnostics: Dict[str, Any], trajectory: Sequence[Sequence[float]]) -> bool:
    if bool(diagnostics.get("reverse", False)) or bool(diagnostics.get("recovery_active", False)):
        return False
    try:
        return all(float(point[0]) > 0.25 for point in trajectory)
    except (IndexError, TypeError, ValueError):
        return False


def vehicle_speed_mps(vehicle: carla.Actor) -> float:
    velocity = vehicle.get_velocity()
    return float(np.sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z))


def collect_tcp_lite_dataset(
    scenario_path: Path,
    output_dir: Path,
    duration_sec: Optional[float] = None,
    sample_hz: float = 2.0,
    command: str = "lane_follow",
    trajectory_distances: Iterable[float] = (2.0, 4.0, 6.0, 8.0),
    disable_uav: bool = True,
) -> Path:
    from carlaair_active_world.env import ActiveAirGroundEnv
    from carlaair_active_world.scenario import ScenarioConfig
    from carlaair_active_world.sensors import save_numpy_image

    scenario = ScenarioConfig.load(scenario_path)
    if duration_sec is not None:
        scenario.duration_sec = float(duration_sec)
    scenario.step_sec = 1.0 / max(0.1, float(sample_hz))
    scenario.vision_navigation_command = str(command)
    if disable_uav:
        scenario.uav_enabled = False

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / "samples.jsonl"
    meta_path = output_dir / "meta.json"
    distances = [float(distance) for distance in trajectory_distances]

    env = ActiveAirGroundEnv(scenario)
    observation = env.reset()
    done = False
    step_idx = 0
    saved = 0
    try:
        with samples_path.open("w", encoding="utf-8") as f:
            while not done:
                result = env.step(0)
                observation = result.observation
                done = result.done
                step_idx += 1

                if env.ego_driver is None or env.ego_vehicle is None or env.world is None:
                    continue
                decision_observation = driver_decision_observation(env.ego_driver)
                rgb = decision_observation.get("rgb")
                if rgb is None:
                    continue
                diagnostics = dict(getattr(env.ego_driver, "last_diagnostics", {}) or {})
                control = control_from_diagnostics(diagnostics)
                if control is None:
                    continue
                trajectory = future_route_trajectory(env.world, env.ego_vehicle, distances)
                if not should_keep_tcp_lite_sample(diagnostics, trajectory):
                    continue

                image_name = f"{saved:06d}.png"
                save_numpy_image(image_dir / image_name, np.asarray(rgb, dtype=np.uint8))
                sample = {
                    "rgb": f"images/{image_name}",
                    "speed_mps": float(decision_observation.get("speed_mps", vehicle_speed_mps(env.ego_vehicle))),
                    "command": str(decision_observation.get("navigation_command", command)),
                    "trajectory": trajectory,
                    "control": control,
                    "source": {
                        "scenario": str(scenario_path),
                        "expert": "vision_simple",
                        "step": int(step_idx),
                        "time": float(observation.get("time", 0.0)),
                        "ego_control": diagnostics,
                    },
                }
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                saved += 1
    finally:
        env.close()

    meta = {
        "scenario": scenario.to_dict(),
        "sample_hz": float(sample_hz),
        "command": str(command),
        "trajectory_distances_m": distances,
        "samples": int(saved),
        "samples_path": str(samples_path),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved {saved} TCP-Lite samples to {samples_path}")
    return samples_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect TCP-Lite imitation data from the rule vision expert.")
    parser.add_argument("--scenario", type=Path, default=ROOT / "configs" / "scenarios" / "town10hd_vision_simple.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--sample-hz", type=float, default=2.0)
    parser.add_argument("--command", type=str, default="lane_follow")
    parser.add_argument("--trajectory-distances", type=str, default="2,4,6,8")
    parser.add_argument("--enable-uav", action="store_true")
    args = parser.parse_args()
    collect_tcp_lite_dataset(
        scenario_path=args.scenario,
        output_dir=args.output_dir,
        duration_sec=args.duration_sec,
        sample_hz=args.sample_hz,
        command=args.command,
        trajectory_distances=parse_distances(args.trajectory_distances),
        disable_uav=not args.enable_uav,
    )


if __name__ == "__main__":
    main()
