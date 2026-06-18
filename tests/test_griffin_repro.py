import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "griffin_repro.py"
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_griffin_remote.py"
MANIFEST = REPO_ROOT / "griffin_repro" / "manifest.json"
OFFICIAL = REPO_ROOT / "griffin_repro" / "official"


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def test_griffin_official_tree_is_isolated_and_complete():
    assert MANIFEST.exists()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["source"]["repository"] == "https://github.com/wang-jh18-SVM/Griffin"
    assert manifest["source"]["remote_head"] == "9c02ba4a37201edfc2b95ddbcdc2ff9aff47e7f4"

    assert (OFFICIAL / "README.md").exists()
    assert (OFFICIAL / "tools" / "dist_eval.sh").exists()
    assert (OFFICIAL / "tools" / "analysis_tools" / "compute_BPS.py").exists()
    assert (OFFICIAL / "projects" / "mmdet3d_plugin" / "datasets" / "griffin_dataset.py").exists()
    assert not (REPO_ROOT / "projects" / "mmdet3d_plugin").exists()


def test_verify_layout_reports_expected_matrix_counts():
    result = run_cli("verify-layout", "--json")
    payload = json.loads(result.stdout)
    assert payload["official_exists"] is True
    assert payload["config_files"] == 97
    assert payload["detailed_result_rows"] == 142
    assert payload["baseline_rows"] == 28
    assert payload["experiment_profiles"] >= 3


def test_summarize_results_matches_paper_table3_values():
    result = run_cli("summarize-results", "--json")
    summary = json.loads(result.stdout)
    by_key = {(row["dataset"], row["method"]): row for row in summary["baseline"]}

    assert by_key[("50scenes_25m", "1-early fusion")]["AP"] == 0.607
    assert by_key[("50scenes_25m", "1-early fusion")]["AMOTA"] == 0.67
    assert by_key[("50scenes_25m", "2b1-cooptrack")]["AP"] == 0.479
    assert by_key[("100scenes_random", "0-no fusion")]["AMOTA"] == 0.481


def test_matrix_smoke_profile_generates_real_eval_command():
    result = run_cli("matrix", "--profile", "smoke_25m_instance", "--json")
    payload = json.loads(result.stdout)
    assert payload["profile"] == "smoke_25m_instance"
    assert payload["expected"]["AP"] == 0.479
    assert payload["expected"]["AMOTA"] == 0.488
    assert "tools/dist_eval.sh" in payload["commands"][0]
    assert "projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py" in payload["commands"][0]
    assert "ckpts/griffin_50scenes_25m/cooperative/instance_fusion/iter_33024.pth" in payload["commands"][0]
    assert "griffin_repro/official/datasets/griffin_50scenes_25m/griffin-nuscenes/cooperative" in payload["asset_checks"]
    assert "griffin_repro/official/data/infos/griffin_50scenes_25m/drone-side/track_query" in payload["asset_checks"]


def test_smoke_profile_asset_check_includes_data_dependencies():
    result = run_cli("check-assets", "--profile", "smoke_25m_instance", "--json")
    payload = json.loads(result.stdout)
    paths = {item["path"] for item in payload["checks"]}
    assert "griffin_repro/official/projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py" in paths
    assert "griffin_repro/official/ckpts/griffin_50scenes_25m/cooperative/instance_fusion/iter_33024.pth" in paths
    assert "griffin_repro/official/datasets/griffin_50scenes_25m/griffin-nuscenes/cooperative" in paths
    assert "griffin_repro/official/data/infos/griffin_50scenes_25m/cooperative/griffin_infos_val.pkl" in paths
    assert "griffin_repro/official/data/infos/griffin_50scenes_25m/drone-side/track_query" in paths


def test_sync_remote_dry_run_limits_upload_to_repro_files():
    result = subprocess.run(
        [
            sys.executable,
            str(SYNC_SCRIPT),
            "--host",
            "10.2.14.120",
            "--user",
            "fp",
            "--remote-dir",
            "/home/fp/CARLA/CarlaAir-v0.1.7/code",
            "--dry-run",
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert "griffin_repro/manifest.json" in payload["files"]
    assert "griffin_repro/official/projects/mmdet3d_plugin/datasets/griffin_dataset.py" in payload["files"]
    assert "scripts/griffin_repro.py" in payload["files"]
    assert "scripts/sync_griffin_remote.py" in payload["files"]
    assert ".git/" not in "\n".join(payload["files"])
    assert all(not path.startswith("carlaair_active_world/") for path in payload["files"])
