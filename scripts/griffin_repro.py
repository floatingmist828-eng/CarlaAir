#!/usr/bin/env python3
"""Utilities for the isolated Griffin paper reproduction package."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPRO_ROOT = REPO_ROOT / "griffin_repro"
OFFICIAL_ROOT = REPRO_ROOT / "official"
MANIFEST_PATH = REPRO_ROOT / "manifest.json"
RESULTS_CSV = OFFICIAL_ROOT / "docs" / "detailed_results.csv"

DATASETS = {
    "50scenes_25m": {
        "dataset_prefix": "griffin_50scenes_25m",
        "scene_count": 47,
        "altitude": "25 +/- 2 m",
    },
    "50scenes_40m": {
        "dataset_prefix": "griffin_50scenes_40m",
        "scene_count": 54,
        "altitude": "40 +/- 2 m",
    },
    "50scenes_55m": {
        "dataset_prefix": "griffin_50scenes_55m",
        "scene_count": 50,
        "altitude": "55 +/- 2 m",
    },
    "100scenes_random": {
        "dataset_prefix": "griffin_100scenes_random",
        "scene_count": 104,
        "altitude": "20-60 m",
    },
}

FUSION_METHODS = [
    "0-no fusion",
    "1-early fusion",
    "2a1-v2x-vit",
    "2a2-where2comm",
    "2b1-cooptrack",
    "2b2-univ2x",
    "3-late fusion",
]

METRICS = {
    "AP": "3D detection average precision from the Griffin benchmark table.",
    "AMOTA": "Tracking accuracy metric used as the main multi-object tracking score.",
    "BPS": "Communication bytes per sample, computed by the official analysis tooling when feature payloads exist.",
    "FPS": "Runtime throughput reported by the paper for efficiency comparison.",
}

ROBUSTNESS = {
    "communication_latency_ms": [100, 200, 300, 400],
    "packet_loss": [0.1, 0.2, 0.3, 0.4, 0.5],
    "translation_error_m": [0.5, 1.0, 1.5, 2.0, 2.5],
    "rotation_error_deg": [1, 2, 3, 4, 5],
}


def load_manifest() -> dict[str, Any]:
    with MANIFEST_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_results() -> list[dict[str, str]]:
    with RESULTS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def is_zero_condition(row: dict[str, str]) -> bool:
    return all(
        float(row[name]) == 0.0
        for name in (
            "communication latency",
            "packet loss",
            "translation error",
            "rotation error",
        )
    )


def config_files() -> list[Path]:
    projects = OFFICIAL_ROOT / "projects"
    if not projects.exists():
        return []
    files = []
    for path in projects.rglob("*.py"):
        if any(part.startswith("configs_griffin_") for part in path.parts):
            files.append(path)
    return sorted(files)


def result_summary() -> dict[str, Any]:
    rows = load_results()
    baseline = []
    for row in rows:
        if is_zero_condition(row):
            baseline.append(
                {
                    "dataset": row["dataset"],
                    "method": row["methods"],
                    "AP": float(row["AP"]),
                    "AMOTA": float(row["AMOTA"]),
                }
            )

    robustness = {
        "communication_latency": sorted(
            {float(row["communication latency"]) for row in rows if float(row["communication latency"]) > 0}
        ),
        "packet_loss": sorted({float(row["packet loss"]) for row in rows if float(row["packet loss"]) > 0}),
        "translation_error": sorted(
            {float(row["translation error"]) for row in rows if float(row["translation error"]) > 0}
        ),
        "rotation_error": sorted({float(row["rotation error"]) for row in rows if float(row["rotation error"]) > 0}),
    }
    return {"baseline": baseline, "robustness": robustness, "rows": len(rows)}


def verify_layout() -> dict[str, Any]:
    manifest = load_manifest()
    rows = load_results() if RESULTS_CSV.exists() else []
    missing = []
    required = [
        OFFICIAL_ROOT / "README.md",
        OFFICIAL_ROOT / "tools" / "dist_eval.sh",
        OFFICIAL_ROOT / "tools" / "analysis_tools" / "compute_BPS.py",
        OFFICIAL_ROOT / "projects" / "mmdet3d_plugin" / "datasets" / "griffin_dataset.py",
    ]
    for path in required:
        if not path.exists():
            missing.append(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))
    return {
        "official_exists": OFFICIAL_ROOT.exists(),
        "manifest_source": manifest["source"]["repository"],
        "config_files": len(config_files()),
        "detailed_result_rows": len(rows),
        "baseline_rows": sum(1 for row in rows if is_zero_condition(row)),
        "experiment_profiles": len(manifest["profiles"]),
        "missing": missing,
    }


def dataset_prefix(dataset: str) -> str:
    if dataset not in DATASETS:
        raise SystemExit(f"Unknown dataset {dataset!r}")
    return DATASETS[dataset]["dataset_prefix"]


def profile_payload(profile_name: str) -> dict[str, Any]:
    manifest = load_manifest()
    profiles = manifest["profiles"]
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise SystemExit(f"Unknown profile {profile_name!r}. Available profiles: {available}")
    profile = profiles[profile_name]
    command = (
        f"cd griffin_repro/official && "
        f"CUDA_VISIBLE_DEVICES=${{CUDA_VISIBLE_DEVICES:-0}} "
        f"./tools/dist_eval.sh {profile['config']} {profile['checkpoint']} {profile['gpus']}"
    )
    return {
        "profile": profile_name,
        "description": profile["description"],
        "dataset": profile["dataset"],
        "method": profile["method"],
        "config": profile["config"],
        "checkpoint": profile["checkpoint"],
        "expected": profile["expected"],
        "commands": [command],
        "asset_checks": [f"griffin_repro/official/{path}" for path in profile.get("required_paths", [])],
    }


def list_profiles() -> dict[str, Any]:
    manifest = load_manifest()
    profiles = {}
    for name, profile in sorted(manifest["profiles"].items()):
        config = OFFICIAL_ROOT / profile["config"]
        checkpoint = OFFICIAL_ROOT / profile["checkpoint"]
        checks = check_assets(name)["checks"]
        profiles[name] = {
            "dataset": profile["dataset"],
            "method": profile["method"],
            "description": profile["description"],
            "config": profile["config"],
            "checkpoint": profile["checkpoint"],
            "config_exists": config.exists(),
            "checkpoint_exists": checkpoint.exists(),
            "ready": all(item["exists"] for item in checks),
            "expected": profile["expected"],
        }
    return {"profiles": profiles}


def check_assets(profile_name: str) -> dict[str, Any]:
    payload = profile_payload(profile_name)
    checks = []
    for rel_path in payload["asset_checks"]:
        path = REPO_ROOT / rel_path
        checks.append({"path": rel_path, "exists": path.exists()})
    return {"profile": profile_name, "checks": checks, "ready": all(item["exists"] for item in checks)}


def check_path_list(paths: list[str]) -> dict[str, Any]:
    checks = [{"path": path, "exists": (REPO_ROOT / path).exists()} for path in paths]
    return {"checks": checks, "ready": all(item["exists"] for item in checks)}


def paper_matrix() -> dict[str, Any]:
    manifest = load_manifest()
    rows = load_results()
    baseline = {(row["dataset"], row["methods"]) for row in rows if is_zero_condition(row)}
    expected_baseline = {(dataset, method) for dataset in DATASETS for method in FUSION_METHODS}
    return {
        "source": manifest["source"],
        "datasets": DATASETS,
        "fusion_methods": FUSION_METHODS,
        "metrics": METRICS,
        "robustness": ROBUSTNESS,
        "result_rows": len(rows),
        "baseline_rows": len(baseline),
        "baseline_complete": expected_baseline <= baseline,
        "missing_baselines": [
            {"dataset": dataset, "method": method}
            for dataset, method in sorted(expected_baseline - baseline)
        ],
    }


def partial_run_plan(profile_name: str) -> dict[str, Any]:
    payload = profile_payload(profile_name)
    prefix = dataset_prefix(payload["dataset"])
    drone_checkpoint = f"ckpts/{prefix}/drone-side/{Path(payload['checkpoint']).name}"
    commands = [
        "cd griffin_repro/official",
        f"bash tools/griffin_converter.sh {prefix}",
        (
            "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ./tools/dist_eval.sh "
            f"projects/configs_{prefix}/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval_train.py "
            f"{drone_checkpoint} 1"
        ),
        (
            "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ./tools/dist_eval.sh "
            f"projects/configs_{prefix}/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval.py "
            f"{drone_checkpoint} 1"
        ),
        payload["commands"][0].split("&&", 1)[1].strip(),
    ]
    preprocess_assets = [
        f"griffin_repro/official/datasets/{prefix}/griffin-release/vehicle-side",
        f"griffin_repro/official/datasets/{prefix}/griffin-release/drone-side",
        f"griffin_repro/official/data/split_datas/{prefix}.json",
        f"griffin_repro/official/{drone_checkpoint}",
        f"griffin_repro/official/{payload['checkpoint']}",
    ]
    evaluation_assets = [
        *payload["asset_checks"],
    ]
    return {
        "profile": profile_name,
        "dataset": payload["dataset"],
        "dataset_prefix": prefix,
        "method": payload["method"],
        "expected": payload["expected"],
        "commands": commands,
        "preprocess_assets": list(dict.fromkeys(preprocess_assets)),
        "evaluation_assets": list(dict.fromkeys(evaluation_assets)),
        "required_assets": list(dict.fromkeys([*preprocess_assets, *evaluation_assets])),
    }


def mobaxterm_script(profile_name: str) -> str:
    plan = partial_run_plan(profile_name)
    preprocess_assets = "\n".join(f'  "{path}"' for path in plan["preprocess_assets"])
    evaluation_assets = "\n".join(f'  "{path}"' for path in plan["evaluation_assets"])
    _, convert_command, drone_train_command, drone_val_command, final_eval_command = plan["commands"]
    return f"""#!/usr/bin/env bash
set -euo pipefail

cd "${{GRIFFIN_REPRO_ROOT:-/home/fp/CARLA/CarlaAir-v0.1.7/code}}"

check_assets() {{
  local group_name="$1"
  shift
  missing_assets=0
  for asset in "$@"; do
    if [ ! -e "$asset" ]; then
      echo "MISSING ($group_name): $asset" >&2
      missing_assets=1
    fi
  done
  if [ "$missing_assets" -ne 0 ]; then
    echo "Install the listed Griffin assets before running $group_name." >&2
    exit 2
  fi
}}

preprocess_assets=(
{preprocess_assets}
)
check_assets "preprocess" "${{preprocess_assets[@]}}"

cd griffin_repro/official
{convert_command}
{drone_train_command}
{drone_val_command}
cd ../..

evaluation_assets=(
{evaluation_assets}
)
check_assets "evaluation" "${{evaluation_assets[@]}}"

cd griffin_repro/official
{final_eval_command}
"""


def write_mobaxterm_script(profile_name: str, out: str) -> dict[str, Any]:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mobaxterm_script(profile_name), encoding="utf-8", newline="\n")
    return {"profile": profile_name, "path": str(path), "bytes": path.stat().st_size}


def check_partial_assets(profile_name: str) -> dict[str, Any]:
    plan = partial_run_plan(profile_name)
    stages = []
    for stage_name, key in (("preprocess", "preprocess_assets"), ("evaluation", "evaluation_assets")):
        stage = check_path_list(plan[key])
        stage["stage"] = stage_name
        stages.append(stage)
    return {
        "profile": profile_name,
        "dataset": plan["dataset"],
        "method": plan["method"],
        "stages": stages,
        "ready": all(stage["ready"] for stage in stages),
    }


def run_profile(profile_name: str, dry_run: bool) -> int:
    payload = profile_payload(profile_name)
    print(json.dumps(payload, indent=2))
    if dry_run:
        return 0

    assets = check_assets(profile_name)
    if not assets["ready"]:
        print(json.dumps(assets, indent=2), file=sys.stderr)
        return 2

    command = payload["commands"][0].split("&&", 1)[1].strip()
    return subprocess.call(command, cwd=OFFICIAL_ROOT, shell=True)


def emit(payload: dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify-layout")
    verify_parser.add_argument("--json", action="store_true")

    summary_parser = subparsers.add_parser("summarize-results")
    summary_parser.add_argument("--json", action="store_true")

    paper_parser = subparsers.add_parser("paper-matrix")
    paper_parser.add_argument("--json", action="store_true")

    profiles_parser = subparsers.add_parser("list-profiles")
    profiles_parser.add_argument("--json", action="store_true")

    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument("--profile", required=True)
    matrix_parser.add_argument("--json", action="store_true")

    partial_parser = subparsers.add_parser("plan-partial-run")
    partial_parser.add_argument("--profile", required=True)
    partial_parser.add_argument("--json", action="store_true")

    script_parser = subparsers.add_parser("write-mobaxterm-script")
    script_parser.add_argument("--profile", required=True)
    script_parser.add_argument("--out", required=True)
    script_parser.add_argument("--json", action="store_true")

    assets_parser = subparsers.add_parser("check-assets")
    assets_parser.add_argument("--profile", required=True)
    assets_parser.add_argument("--json", action="store_true")

    partial_assets_parser = subparsers.add_parser("check-partial-assets")
    partial_assets_parser.add_argument("--profile", required=True)
    partial_assets_parser.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser("run-profile")
    run_parser.add_argument("--profile", required=True)
    run_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "verify-layout":
        emit(verify_layout(), args.json)
    elif args.command == "summarize-results":
        emit(result_summary(), args.json)
    elif args.command == "paper-matrix":
        emit(paper_matrix(), args.json)
    elif args.command == "list-profiles":
        emit(list_profiles(), args.json)
    elif args.command == "matrix":
        emit(profile_payload(args.profile), args.json)
    elif args.command == "plan-partial-run":
        emit(partial_run_plan(args.profile), args.json)
    elif args.command == "write-mobaxterm-script":
        emit(write_mobaxterm_script(args.profile, args.out), args.json)
    elif args.command == "check-assets":
        emit(check_assets(args.profile), args.json)
    elif args.command == "check-partial-assets":
        emit(check_partial_assets(args.profile), args.json)
    elif args.command == "run-profile":
        return run_profile(args.profile, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
