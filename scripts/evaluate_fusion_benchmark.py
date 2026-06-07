from __future__ import annotations

import argparse
import csv
import html
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
    "junction_lane_offset_mean_abs",
    "junction_lane_offset_max_abs",
    "path_completion_rate",
    "path_distance_m",
    "net_displacement_m",
    "avg_speed_mps",
    "hard_brake_count",
    "steer_oscillation_count",
    "junction_steer_oscillation_count",
    "safety_gate_count",
    "safety_gate_rate",
    "uav_available_rate",
    "uav_applied_rate",
    "uav_mean_abs_steer_correction",
    "meeting_scene_success",
    "pedestrian_scene_success",
    "scene_success",
]

COMPARISON_FIELDS = [
    "scenario_stage",
    "comparison_group",
    "baseline_group",
    "collision_rate_reduction",
    "lane_departure_rate_reduction",
    "off_road_rate_reduction",
    "lane_offset_mean_abs_reduction",
    "lane_offset_max_abs_reduction",
    "steer_oscillation_reduction",
    "junction_steer_oscillation_reduction",
    "safety_gate_count_reduction",
    "path_completion_rate_gain",
    "net_displacement_m_gain",
    "avg_speed_mps_gain",
    "scene_success_gain",
    "meeting_scene_success_gain",
    "pedestrian_scene_success_gain",
    "uav_helped",
    "uav_help_reason",
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
    control = _nested(step.get("observation", {}), ["ego_control"], {})
    if not isinstance(control, dict):
        return None
    value = _as_float(control.get(key))
    if value is not None:
        return value
    nested_control = control.get("control")
    if isinstance(nested_control, dict):
        value = _as_float(nested_control.get(key))
        if value is not None:
            return value
    vehicle_control = control.get("vehicle_control")
    if isinstance(vehicle_control, dict):
        return _as_float(vehicle_control.get(key))
    return None


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
    lane_offset = _lane_offset(step)
    if lane_offset is not None:
        return abs(lane_offset) > 3.5
    return lane_departed


def _in_junction(step: Dict[str, Any]) -> bool:
    obs = step.get("observation", {})
    if bool(_nested(obs, ["waypoint", "is_junction"], False)):
        return True
    if bool(_nested(obs, ["ego_control", "stabilization", "in_junction"], False)):
        return True
    label = step.get("label", {})
    return bool(isinstance(label, dict) and _nested(label, ["ego", "junction"], False))


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


def _count_steer_oscillations(values: Iterable[Optional[float]], threshold: float = 0.03) -> int:
    oscillations = 0
    last_sign = 0
    for value in values:
        if value is None or abs(value) <= threshold:
            continue
        sign = 1 if value > 0.0 else -1
        if last_sign and sign != last_sign:
            oscillations += 1
        last_sign = sign
    return oscillations


def _mean_abs(values: Iterable[float]) -> float:
    items = [abs(value) for value in values]
    return sum(items) / max(1, len(items))


def _interaction_success(scenario: Dict[str, Any], scene_success: float, interaction: str) -> Any:
    stage = str(scenario.get("scenario_stage", "")).lower()
    complexity = scenario.get("scenario_complexity", [])
    if isinstance(complexity, str):
        complexity = [complexity]
    labels = {str(item).lower() for item in complexity}
    if interaction in stage or interaction in labels:
        return scene_success
    return ""


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
    junction_steps = [step for step in steps if _in_junction(step)]
    junction_lane_offsets = [value for value in (_lane_offset(step) for step in junction_steps) if value is not None]
    junction_lane_abs = [abs(value) for value in junction_lane_offsets]
    junction_steer_values = [_control_value(step, "steer") for step in junction_steps]

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
    scene_success = 1.0 if not collision and off_road_rate == 0.0 and path_completion >= 0.5 else 0.0
    fusion_items = [_uav_fusion(step) for step in steps]
    fusion_available = [item for item in fusion_items if bool(item.get("available", False))]
    fusion_applied = [item for item in fusion_items if bool(item.get("applied", False))]
    fusion_corrections = [
        abs(value)
        for value in (_as_float(item.get("steer_correction")) for item in fusion_items)
        if value is not None
    ]

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
        "junction_lane_offset_mean_abs": round(sum(junction_lane_abs) / max(1, len(junction_lane_abs)), 6),
        "junction_lane_offset_max_abs": round(max(junction_lane_abs) if junction_lane_abs else 0.0, 6),
        "path_completion_rate": round(path_completion, 6),
        "path_distance_m": round(path_distance, 6),
        "net_displacement_m": round(net_displacement, 6),
        "avg_speed_mps": round(sum(speeds) / max(1, len(speeds)), 6),
        "hard_brake_count": sum(1 for step in steps if (_control_value(step, "brake") or 0.0) >= 0.5),
        "steer_oscillation_count": _count_steer_oscillations(steer_values),
        "junction_steer_oscillation_count": _count_steer_oscillations(junction_steer_values),
        "safety_gate_count": sum(1 for step in steps if _safety_gate_blocked(step)),
        "safety_gate_rate": round(
            sum(1 for step in steps if _safety_gate_blocked(step)) / max(1, step_count),
            6,
        ),
        "uav_available_rate": round(len(fusion_available) / max(1, step_count), 6),
        "uav_applied_rate": round(len(fusion_applied) / max(1, step_count), 6),
        "uav_mean_abs_steer_correction": round(sum(fusion_corrections) / max(1, len(fusion_corrections)), 6),
        "meeting_scene_success": _interaction_success(scenario, scene_success, "meeting"),
        "pedestrian_scene_success": _interaction_success(scenario, scene_success, "pedestrian"),
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


def _as_metric_float(result: Dict[str, Any], key: str) -> Optional[float]:
    value = result.get(key)
    if value == "":
        return None
    return _as_float(value)


def _metric_delta(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    key: str,
    higher_is_better: bool = False,
) -> str:
    baseline_value = _as_metric_float(baseline, key)
    candidate_value = _as_metric_float(candidate, key)
    if baseline_value is None or candidate_value is None:
        return ""
    delta = candidate_value - baseline_value if higher_is_better else baseline_value - candidate_value
    if key.endswith("_count") and float(delta).is_integer():
        return str(int(delta))
    return str(round(delta, 6))


def _comparison_rows(results: List[Dict[str, Any]], baseline_group: str = "no_uav") -> List[Dict[str, Any]]:
    by_stage: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for result in results:
        stage = str(result.get("scenario_stage") or result.get("scenario") or "")
        group = str(result.get("experiment_group") or "")
        if not stage or not group:
            continue
        by_stage.setdefault(stage, {})[group] = result

    rows: List[Dict[str, Any]] = []
    for stage, groups in sorted(by_stage.items()):
        baseline = groups.get(baseline_group)
        if baseline is None:
            continue
        for group, candidate in sorted(groups.items()):
            if group == baseline_group:
                continue
            row = {
                "scenario_stage": stage,
                "comparison_group": group,
                "baseline_group": baseline_group,
                "collision_rate_reduction": _metric_delta(baseline, candidate, "collision_rate"),
                "lane_departure_rate_reduction": _metric_delta(baseline, candidate, "lane_departure_rate"),
                "off_road_rate_reduction": _metric_delta(baseline, candidate, "off_road_rate"),
                "lane_offset_mean_abs_reduction": _metric_delta(baseline, candidate, "lane_offset_mean_abs"),
                "lane_offset_max_abs_reduction": _metric_delta(baseline, candidate, "lane_offset_max_abs"),
                "steer_oscillation_reduction": _metric_delta(baseline, candidate, "steer_oscillation_count"),
                "junction_steer_oscillation_reduction": _metric_delta(
                    baseline,
                    candidate,
                    "junction_steer_oscillation_count",
                ),
                "safety_gate_count_reduction": _metric_delta(baseline, candidate, "safety_gate_count"),
                "path_completion_rate_gain": _metric_delta(
                    baseline,
                    candidate,
                    "path_completion_rate",
                    higher_is_better=True,
                ),
                "net_displacement_m_gain": _metric_delta(
                    baseline,
                    candidate,
                    "net_displacement_m",
                    higher_is_better=True,
                ),
                "avg_speed_mps_gain": _metric_delta(baseline, candidate, "avg_speed_mps", higher_is_better=True),
                "scene_success_gain": _metric_delta(baseline, candidate, "scene_success", higher_is_better=True),
                "meeting_scene_success_gain": _metric_delta(
                    baseline,
                    candidate,
                    "meeting_scene_success",
                    higher_is_better=True,
                ),
                "pedestrian_scene_success_gain": _metric_delta(
                    baseline,
                    candidate,
                    "pedestrian_scene_success",
                    higher_is_better=True,
                ),
            }
            key_values = [
                _as_float(row.get("collision_rate_reduction"), 0.0) or 0.0,
                _as_float(row.get("lane_offset_mean_abs_reduction"), 0.0) or 0.0,
                _as_float(row.get("junction_steer_oscillation_reduction"), 0.0) or 0.0,
                _as_float(row.get("scene_success_gain"), 0.0) or 0.0,
            ]
            row["uav_helped"] = "true" if all(value >= 0.0 for value in key_values) and any(value > 0.0 for value in key_values) else "false"
            row["uav_help_reason"] = (
                "reduced risk/offset/oscillation or improved success"
                if row["uav_helped"] == "true"
                else "no positive aggregate improvement over no_uav"
            )
            rows.append(row)
    return rows


def _write_comparisons_csv(results: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for row in _comparison_rows(results):
            writer.writerow({field: row.get(field, "") for field in COMPARISON_FIELDS})


def _svg_text(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _write_svg_bar_chart(results: List[Dict[str, Any]], output_path: Path, metric: str, title: str) -> None:
    values = [float(item.get(metric, 0.0) or 0.0) for item in results]
    if not values:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = max(720, 120 * len(values))
    height = 420
    margin_left = 70
    margin_bottom = 110
    plot_width = width - margin_left - 30
    plot_height = height - 70 - margin_bottom
    max_value = max(max(values), 1.0)
    bar_width = max(18, int(plot_width / max(1, len(values)) * 0.55))
    gap = plot_width / max(1, len(values))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{_svg_text(title)}</text>',
        f'<line x1="{margin_left}" y1="50" x2="{margin_left}" y2="{50 + plot_height}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{50 + plot_height}" x2="{margin_left + plot_width}" y2="{50 + plot_height}" stroke="#333"/>',
    ]
    for idx, (result, value) in enumerate(zip(results, values)):
        x = margin_left + idx * gap + (gap - bar_width) * 0.5
        bar_height = plot_height * (value / max_value)
        y = 50 + plot_height - bar_height
        label = f"{result.get('scenario_stage') or result.get('scenario')} {result.get('experiment_group')}"
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="#2f80ed"/>')
        lines.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.3g}</text>'
        )
        lines.append(
            f'<text x="{x + bar_width / 2:.1f}" y="{50 + plot_height + 20}" text-anchor="end" transform="rotate(-35 {x + bar_width / 2:.1f} {50 + plot_height + 20})" font-family="Arial" font-size="11">{_svg_text(label)}</text>'
        )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _line_points(xs: List[float], ys: List[float], bounds: Dict[str, float]) -> str:
    min_x = bounds["min_x"]
    max_x = bounds["max_x"]
    min_y = bounds["min_y"]
    max_y = bounds["max_y"]
    width = bounds["plot_width"]
    height = bounds["plot_height"]
    left = bounds["left"]
    top = bounds["top"]
    x_span = max(max_x - min_x, 1e-9)
    y_span = max(max_y - min_y, 1e-9)
    points = []
    for x_value, y_value in zip(xs, ys):
        x = left + width * ((x_value - min_x) / x_span)
        y = top + height - height * ((y_value - min_y) / y_span)
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _write_svg_line_chart(
    series: List[Dict[str, Any]],
    output_path: Path,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    cleaned = []
    all_x = []
    all_y = []
    for item in series:
        xs = [float(value) for value in item.get("xs", [])]
        ys = [float(value) for value in item.get("ys", [])]
        if not xs or not ys:
            continue
        cleaned.append({"label": item.get("label", ""), "xs": xs, "ys": ys})
        all_x.extend(xs)
        all_y.extend(ys)
    if not cleaned:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = 860
    height = 430
    bounds = {
        "left": 70.0,
        "top": 50.0,
        "plot_width": 620.0,
        "plot_height": 300.0,
        "min_x": min(all_x),
        "max_x": max(all_x),
        "min_y": min(all_y),
        "max_y": max(all_y),
    }
    if bounds["min_y"] == bounds["max_y"]:
        bounds["min_y"] -= 1.0
        bounds["max_y"] += 1.0
    colors = ["#2f80ed", "#27ae60", "#eb5757", "#9b51e0", "#f2994a", "#00a7a7"]
    axis_bottom = bounds["top"] + bounds["plot_height"]
    axis_right = bounds["left"] + bounds["plot_width"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="18">{_svg_text(title)}</text>',
        f'<line x1="{bounds["left"]}" y1="{bounds["top"]}" x2="{bounds["left"]}" y2="{axis_bottom}" stroke="#333"/>',
        f'<line x1="{bounds["left"]}" y1="{axis_bottom}" x2="{axis_right}" y2="{axis_bottom}" stroke="#333"/>',
        f'<text x="{(bounds["left"] + axis_right) / 2:.1f}" y="405" text-anchor="middle" font-family="Arial" font-size="12">{_svg_text(x_label)}</text>',
        f'<text x="18" y="{(bounds["top"] + axis_bottom) / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {(bounds["top"] + axis_bottom) / 2:.1f})" font-family="Arial" font-size="12">{_svg_text(y_label)}</text>',
    ]
    for idx, item in enumerate(cleaned):
        color = colors[idx % len(colors)]
        points = _line_points(item["xs"], item["ys"], bounds)
        legend_y = 65 + idx * 20
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{points}"/>')
        lines.append(f'<line x1="715" y1="{legend_y}" x2="745" y2="{legend_y}" stroke="{color}" stroke-width="2"/>')
        lines.append(
            f'<text x="752" y="{legend_y + 4}" font-family="Arial" font-size="11">{_svg_text(item["label"])}</text>'
        )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _plot_metric(results: List[Dict[str, Any]], output_dir: Path, metric: str, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        _write_svg_bar_chart(results, output_dir / f"{metric}.svg", metric, title)
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
        series = []
        for result in results:
            rows = result.get("timeseries", [])
            xs = [row.get("time_sec") for row in rows if row.get(field) is not None]
            ys = [row.get(field) for row in rows if row.get(field) is not None]
            series.append(
                {
                    "label": f"{result.get('scenario_stage') or result.get('scenario')}:{result.get('experiment_group')}",
                    "xs": xs,
                    "ys": ys,
                }
            )
        _write_svg_line_chart(series, output_dir / f"{field}_curve.svg", title, "time_sec", ylabel)
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
        series = []
        for result in results:
            rows = result.get("segments", [])
            series.append(
                {
                    "label": f"{result.get('scenario_stage') or result.get('scenario')}:{result.get('experiment_group')}",
                    "xs": [row.get("segment_start_sec") for row in rows],
                    "ys": [row.get("path_distance_m") for row in rows],
                }
            )
        _write_svg_line_chart(
            series,
            output_dir / "segment_path_distance_curve.svg",
            "30s Segment Path Distance",
            "segment_start_sec",
            "path_distance_m",
        )
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
    _write_comparisons_csv(results, output_dir / "comparisons.csv")
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
