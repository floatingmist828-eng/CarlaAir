from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SUMMARY_FIELDS = [
    "file",
    "scenario",
    "experiment_group",
    "scenario_stage",
    "steps",
    "duration_sec",
    "collision_rate",
    "lane_departure_rate",
    "off_road_rate",
    "lane_offset_mean_abs",
    "lane_offset_max_abs",
    "path_completion_rate",
    "path_distance_m",
    "net_displacement_m",
    "avg_speed_mps",
    "hard_brake_count",
    "steer_oscillation_count",
    "safety_gate_count",
    "scene_success",
]

SEGMENT_FIELDS = [
    "file",
    "scenario",
    "experiment_group",
    "scenario_stage",
    "segment_start_sec",
    "segment_end_sec",
    "path_distance_m",
]

TIMESERIES_FIELDS = [
    "file",
    "scenario",
    "experiment_group",
    "scenario_stage",
    "time_sec",
    "lane_offset_m",
    "steer",
    "speed_mps",
    "brake",
    "safety_gate_blocked",
    "uav_fusion_mode",
    "uav_steer_correction",
]


def load_episode(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _nested(mapping: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _position(step: Dict[str, Any]) -> Optional[tuple[float, float]]:
    obs = step.get("observation", {})
    position = _nested(obs, ["ego", "pose", "position"], {})
    x = _as_float(position.get("x") if isinstance(position, dict) else None)
    y = _as_float(position.get("y") if isinstance(position, dict) else None)
    if x is None or y is None:
        return None
    return x, y


def _speed(step: Dict[str, Any]) -> Optional[float]:
    obs = step.get("observation", {})
    speed = _as_float(_nested(obs, ["ego_control", "stabilization", "speed_mps"]))
    if speed is not None:
        return speed
    velocity = _nested(obs, ["ego", "velocity"], {})
    if not isinstance(velocity, dict):
        return None
    vx = _as_float(velocity.get("x"), 0.0) or 0.0
    vy = _as_float(velocity.get("y"), 0.0) or 0.0
    vz = _as_float(velocity.get("z"), 0.0) or 0.0
    return math.sqrt(vx * vx + vy * vy + vz * vz)


def _lane_offset(step: Dict[str, Any]) -> Optional[float]:
    obs = step.get("observation", {})
    value = _as_float(_nested(obs, ["ego_control", "stabilization", "lane_center_offset_m"]))
    if value is not None:
        return value
    return _as_float(_nested(obs, ["ego_control", "lane_center_offset_m"]))


def _control_value(step: Dict[str, Any], key: str) -> Optional[float]:
    return _as_float(_nested(step.get("observation", {}), ["ego_control", key]))


def _safety_gate_blocked(step: Dict[str, Any]) -> bool:
    gate = _nested(step.get("observation", {}), ["ego_control", "safety_gate"], {})
    return bool(isinstance(gate, dict) and gate.get("blocked", False))


def _has_collision(step: Dict[str, Any]) -> bool:
    label = step.get("label", {})
    if isinstance(label, dict):
        if bool(label.get("collision", False)):
            return True
        if _as_float(label.get("collision_count"), 0.0):
            return True
    info = step.get("info", {})
    return bool(isinstance(info, dict) and info.get("collision", False))


def _off_road(step: Dict[str, Any], lane_departed: bool) -> bool:
    label = step.get("label", {})
    if isinstance(label, dict) and "off_road" in label:
        return bool(label.get("off_road"))
    return lane_departed


def _path_distance(positions: List[Optional[tuple[float, float]]]) -> float:
    distance = 0.0
    previous = None
    for position in positions:
        if position is None:
            continue
        if previous is not None:
            distance += math.dist(previous, position)
        previous = position
    return distance


def _segments(steps: List[Dict[str, Any]], segment_sec: float = 30.0) -> List[Dict[str, float]]:
    segments: Dict[float, float] = {}
    previous_time = None
    previous_position = None
    for step in steps:
        obs = step.get("observation", {})
        time_value = _as_float(obs.get("time"))
        position = _position(step)
        if time_value is None or position is None:
            continue
        if previous_time is not None and previous_position is not None:
            segment_start = math.floor(previous_time / segment_sec) * segment_sec
            segments[segment_start] = segments.get(segment_start, 0.0) + math.dist(previous_position, position)
        previous_time = time_value
        previous_position = position
    return [
        {
            "segment_start_sec": float(start),
            "segment_end_sec": float(start + segment_sec),
            "path_distance_m": round(distance, 6),
        }
        for start, distance in sorted(segments.items())
    ]


def _uav_fusion(step: Dict[str, Any]) -> Dict[str, Any]:
    control = _nested(step.get("observation", {}), ["ego_control"], {})
    if not isinstance(control, dict):
        return {}
    fusion = control.get("uav_bev_fusion")
    if isinstance(fusion, dict):
        return fusion
    stabilization = control.get("stabilization")
    if isinstance(stabilization, dict) and isinstance(stabilization.get("uav_bev_fusion"), dict):
        return stabilization["uav_bev_fusion"]
    return {}


def _timeseries(path: Path, scenario: Dict[str, Any], steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for step in steps:
        obs = step.get("observation", {})
        fusion = _uav_fusion(step)
        rows.append(
            {
                "file": str(path),
                "scenario": str(scenario.get("name", path.stem)),
                "experiment_group": str(scenario.get("experiment_group", "")),
                "scenario_stage": str(scenario.get("scenario_stage", "")),
                "time_sec": _as_float(obs.get("time"), 0.0),
                "lane_offset_m": _lane_offset(step),
                "steer": _control_value(step, "steer"),
                "speed_mps": _speed(step),
                "brake": _control_value(step, "brake"),
                "safety_gate_blocked": _safety_gate_blocked(step),
                "uav_fusion_mode": str(fusion.get("mode", "")),
                "uav_steer_correction": _as_float(fusion.get("steer_correction"), 0.0),
            }
        )
    return rows


def evaluate_episode(path: Path) -> Dict[str, Any]:
    episode = load_episode(path)
    meta = episode.get("meta", {})
    scenario = meta.get("scenario", {}) if isinstance(meta, dict) else {}
    steps = episode.get("steps", [])
    if not isinstance(steps, list):
        steps = []

    times = [_as_float(step.get("observation", {}).get("time")) for step in steps]
    valid_times = [value for value in times if value is not None]
    positions = [_position(step) for step in steps]
    speeds = [value for value in (_speed(step) for step in steps) if value is not None]
    lane_offsets = [value for value in (_lane_offset(step) for step in steps) if value is not None]
    lane_abs = [abs(value) for value in lane_offsets]
    lane_departed = [abs(value) > 1.75 for value in lane_offsets]
    steer_values = [_control_value(step, "steer") for step in steps]

    oscillations = 0
    last_sign = 0
    for value in steer_values:
        if value is None or abs(value) <= 0.03:
            continue
        sign = 1 if value > 0.0 else -1
        if last_sign and sign != last_sign:
            oscillations += 1
        last_sign = sign

    path_distance = _path_distance(positions)
    first_position = next((position for position in positions if position is not None), None)
    last_position = next((position for position in reversed(positions) if position is not None), None)
    net_displacement = math.dist(first_position, last_position) if first_position and last_position else 0.0
    expected_distance = (
        _as_float(scenario.get("duration_sec"), 0.0) or 0.0
    ) * (_as_float(scenario.get("ego_target_speed_mps"), 0.0) or 0.0)
    path_completion = path_distance / expected_distance if expected_distance > 0.0 else 0.0
    path_completion = _clamp(path_completion, 0.0, 1.0)

    step_count = len(steps)
    collision = any(_has_collision(step) for step in steps)
    lane_departure_rate = sum(1 for item in lane_departed if item) / max(1, len(lane_departed))
    off_road_rate = sum(
        1 for step, departed in zip(steps, lane_departed) if _off_road(step, departed)
    ) / max(1, step_count)
    scene_success = 1.0 if not collision and off_road_rate == 0.0 and path_distance > 1.0 else 0.0

    return {
        "file": str(path),
        "scenario": str(scenario.get("name", path.stem)),
        "experiment_group": str(scenario.get("experiment_group", "")),
        "scenario_stage": str(scenario.get("scenario_stage", "")),
        "steps": step_count,
        "duration_sec": round((max(valid_times) - min(valid_times)) if len(valid_times) >= 2 else 0.0, 6),
        "collision_rate": 1.0 if collision else 0.0,
        "lane_departure_rate": round(lane_departure_rate, 6),
        "off_road_rate": round(off_road_rate, 6),
        "lane_offset_mean_abs": round(sum(lane_abs) / max(1, len(lane_abs)), 6),
        "lane_offset_max_abs": round(max(lane_abs) if lane_abs else 0.0, 6),
        "path_completion_rate": round(path_completion, 6),
        "path_distance_m": round(path_distance, 6),
        "net_displacement_m": round(net_displacement, 6),
        "avg_speed_mps": round(sum(speeds) / max(1, len(speeds)), 6),
        "hard_brake_count": sum(1 for step in steps if (_control_value(step, "brake") or 0.0) >= 0.5),
        "steer_oscillation_count": oscillations,
        "safety_gate_count": sum(1 for step in steps if _safety_gate_blocked(step)),
        "scene_success": scene_success,
        "segments": _segments(steps),
        "timeseries": _timeseries(path, scenario, steps),
    }


def _clamp(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def _write_summary_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in SUMMARY_FIELDS})


def _write_segments_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SEGMENT_FIELDS)
        writer.writeheader()
        for result in results:
            for segment in result.get("segments", []):
                writer.writerow(
                    {
                        "file": result.get("file", ""),
                        "scenario": result.get("scenario", ""),
                        "experiment_group": result.get("experiment_group", ""),
                        "scenario_stage": result.get("scenario_stage", ""),
                        "segment_start_sec": segment.get("segment_start_sec", ""),
                        "segment_end_sec": segment.get("segment_end_sec", ""),
                        "path_distance_m": segment.get("path_distance_m", ""),
                    }
                )


def _write_timeseries_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TIMESERIES_FIELDS)
        writer.writeheader()
        for result in results:
            for row in result.get("timeseries", []):
                writer.writerow({field: row.get(field, "") for field in TIMESERIES_FIELDS})


def _plot_metric(results: List[Dict[str, Any]], output_dir: Path, metric: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    labels = [f"{item.get('scenario_stage') or item.get('scenario')}\n{item.get('experiment_group')}" for item in results]
    values = [float(item.get(metric, 0.0) or 0.0) for item in results]
    if not values:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(6.0, len(values) * 1.2), 4.0))
    ax.bar(range(len(values)), values)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(metric)
    fig.tight_layout()
    fig.savefig(output_dir / f"{metric}.png")
    plt.close(fig)


def _plot_timeseries(results: List[Dict[str, Any]], output_dir: Path, field: str, title: str, ylabel: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    plotted = False
    for result in results:
        rows = result.get("timeseries", [])
        xs = [row.get("time_sec") for row in rows if row.get(field) is not None]
        ys = [row.get(field) for row in rows if row.get(field) is not None]
        if not xs or not ys:
            continue
        label = f"{result.get('scenario_stage') or result.get('scenario')}:{result.get('experiment_group')}"
        ax.plot(xs, ys, label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title(title)
    ax.set_xlabel("time_sec")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{field}_curve.png")
    plt.close(fig)


def _plot_segments(results: List[Dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    plotted = False
    for result in results:
        rows = result.get("segments", [])
        xs = [row.get("segment_start_sec") for row in rows]
        ys = [row.get("path_distance_m") for row in rows]
        if not xs or not ys:
            continue
        label = f"{result.get('scenario_stage') or result.get('scenario')}:{result.get('experiment_group')}"
        ax.plot(xs, ys, marker="o", label=label)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_title("30s Segment Path Distance")
    ax.set_xlabel("segment_start_sec")
    ax.set_ylabel("path_distance_m")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "segment_path_distance_curve.png")
    plt.close(fig)


def _write_plots(results: List[Dict[str, Any]], output_dir: Path) -> None:
    _plot_metric(results, output_dir, "collision_rate", "Collision Rate")
    _plot_metric(results, output_dir, "lane_offset_mean_abs", "Mean Lane Offset")
    _plot_metric(results, output_dir, "avg_speed_mps", "Average Speed")
    _plot_metric(results, output_dir, "path_distance_m", "Path Distance")
    _plot_metric(results, output_dir, "scene_success", "Scene Success")
    _plot_timeseries(results, output_dir, "lane_offset_m", "Lane Offset Curve", "lane_offset_m")
    _plot_timeseries(results, output_dir, "steer", "Steer Curve", "steer")
    _plot_timeseries(results, output_dir, "speed_mps", "Speed Curve", "speed_mps")
    _plot_segments(results, output_dir)


def evaluate_directory(input_dir: Path, output_dir: Path, make_plots: bool = True) -> List[Dict[str, Any]]:
    files = sorted(input_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"No episode files found in {input_dir}")
    results = [evaluate_episode(path) for path in files]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_summary_csv(results, output_dir / "summary.csv")
    _write_segments_csv(results, output_dir / "segments.csv")
    _write_timeseries_csv(results, output_dir / "timeseries.csv")
    if make_plots:
        _write_plots(results, output_dir / "plots")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CarlaAir UAV fusion benchmark recordings.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    results = evaluate_directory(args.input_dir, args.output_dir, make_plots=not args.no_plots)
    print(f"episodes={len(results)}")
    print(f"summary={args.output_dir / 'summary.csv'}")
    print(f"segments={args.output_dir / 'segments.csv'}")


if __name__ == "__main__":
    main()
