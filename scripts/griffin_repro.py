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
        "expected": profile["expected"],
        "commands": [command],
        "asset_checks": [f"griffin_repro/official/{path}" for path in profile.get("required_paths", [])],
    }


def check_assets(profile_name: str) -> dict[str, Any]:
    payload = profile_payload(profile_name)
    checks = []
    for rel_path in payload["asset_checks"]:
        path = REPO_ROOT / rel_path
        checks.append({"path": rel_path, "exists": path.exists()})
    return {"profile": profile_name, "checks": checks, "ready": all(item["exists"] for item in checks)}


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

    matrix_parser = subparsers.add_parser("matrix")
    matrix_parser.add_argument("--profile", required=True)
    matrix_parser.add_argument("--json", action="store_true")

    assets_parser = subparsers.add_parser("check-assets")
    assets_parser.add_argument("--profile", required=True)
    assets_parser.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser("run-profile")
    run_parser.add_argument("--profile", required=True)
    run_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "verify-layout":
        emit(verify_layout(), args.json)
    elif args.command == "summarize-results":
        emit(result_summary(), args.json)
    elif args.command == "matrix":
        emit(profile_payload(args.profile), args.json)
    elif args.command == "check-assets":
        emit(check_assets(args.profile), args.json)
    elif args.command == "run-profile":
        return run_profile(args.profile, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
