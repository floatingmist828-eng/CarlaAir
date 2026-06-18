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


def test_paper_matrix_cli_covers_paper_scenes_metrics_fusions_and_robustness():
    result = run_cli("paper-matrix", "--json")
    payload = json.loads(result.stdout)

    assert payload["source"]["paper"].startswith("Griffin:")
    assert payload["result_rows"] == 142
    assert payload["baseline_rows"] == 28
    assert payload["baseline_complete"] is True

    assert payload["datasets"]["50scenes_25m"]["scene_count"] == 47
    assert payload["datasets"]["50scenes_40m"]["scene_count"] == 54
    assert payload["datasets"]["50scenes_55m"]["scene_count"] == 50
    assert payload["datasets"]["100scenes_random"]["scene_count"] == 104

    assert set(payload["fusion_methods"]) == {
        "0-no fusion",
        "1-early fusion",
        "2a1-v2x-vit",
        "2a2-where2comm",
        "2b1-cooptrack",
        "2b2-univ2x",
        "3-late fusion",
    }
    assert set(payload["metrics"]) >= {"AP", "AMOTA", "BPS", "FPS"}
    assert payload["robustness"]["communication_latency_ms"] == [100, 200, 300, 400]
    assert payload["robustness"]["packet_loss"] == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert payload["robustness"]["translation_error_m"] == [0.5, 1.0, 1.5, 2.0, 2.5]
    assert payload["robustness"]["rotation_error_deg"] == [1, 2, 3, 4, 5]


def test_list_profiles_marks_runnable_configs_and_expected_metrics():
    result = run_cli("list-profiles", "--json")
    payload = json.loads(result.stdout)
    profiles = payload["profiles"]

    assert {"smoke_25m_instance", "smoke_25m_vehicle", "smoke_25m_early"} <= set(profiles)
    assert profiles["smoke_25m_instance"]["config_exists"] is True
    assert profiles["smoke_25m_instance"]["expected"] == {"AMOTA": 0.488, "AP": 0.479}
    assert profiles["smoke_25m_instance"]["method"] == "2b1-cooptrack"


def test_partial_run_plan_includes_conversion_query_extraction_eval_and_asset_gates():
    result = run_cli("plan-partial-run", "--profile", "smoke_25m_instance", "--json")
    payload = json.loads(result.stdout)
    commands = payload["commands"]
    assets = payload["required_assets"]

    assert payload["profile"] == "smoke_25m_instance"
    assert payload["dataset_prefix"] == "griffin_50scenes_25m"
    assert payload["expected"] == {"AMOTA": 0.488, "AP": 0.479}
    assert commands[0] == "cd griffin_repro/official"
    assert "bash tools/griffin_converter.sh griffin_50scenes_25m" in commands
    assert any("drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval_train.py" in command for command in commands)
    assert any("drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval.py" in command for command in commands)
    assert commands[-1].startswith("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} ./tools/dist_eval.sh")
    assert "projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py" in commands[-1]

    assert "griffin_repro/official/datasets/griffin_50scenes_25m/griffin-release/vehicle-side" in assets
    assert "griffin_repro/official/datasets/griffin_50scenes_25m/griffin-release/drone-side" in assets
    assert "griffin_repro/official/data/split_datas/griffin_50scenes_25m.json" in assets
    assert "griffin_repro/official/ckpts/griffin_50scenes_25m/drone-side/iter_33024.pth" in assets
    assert "griffin_repro/official/data/infos/griffin_50scenes_25m/drone-side/track_query" in assets


def test_check_partial_assets_reports_preprocess_and_evaluation_stages():
    result = run_cli("check-partial-assets", "--profile", "smoke_25m_instance", "--json")
    payload = json.loads(result.stdout)
    stages = {stage["stage"]: stage for stage in payload["stages"]}

    assert payload["profile"] == "smoke_25m_instance"
    assert {"preprocess", "evaluation"} == set(stages)
    assert any(
        item["path"] == "griffin_repro/official/ckpts/griffin_50scenes_25m/drone-side/iter_33024.pth"
        for item in stages["preprocess"]["checks"]
    )
    assert any(
        item["path"] == "griffin_repro/official/data/infos/griffin_50scenes_25m/drone-side/track_query"
        for item in stages["evaluation"]["checks"]
    )
    assert payload["ready"] == all(stage["ready"] for stage in stages.values())


def test_environment_check_reports_runtime_and_required_modules():
    result = run_cli("env-check", "--json")
    payload = json.loads(result.stdout)

    assert payload["python"]["executable"]
    assert payload["python"]["version"]
    assert {"torch", "mmcv", "mmdet", "mmseg", "mmdet3d"} <= set(payload["python_modules"])
    assert payload["python_modules"]["torch"]["expected_version"] == "1.9.1"
    assert payload["python_modules"]["mmseg"]["package"] == "mmsegmentation"
    assert payload["python_modules"]["mmseg"]["expected_version"] == "0.14.1"
    assert "ok" in payload["python_modules"]["torch"]
    assert "nvidia_smi" in payload
    assert payload["ready"] == all(item["ok"] for item in payload["python_modules"].values())


def test_data_packages_lists_griffin_25m_archives_and_download_size():
    result = run_cli("data-packages", "--dataset", "50scenes_25m", "--json")
    payload = json.loads(result.stdout)
    packages = {item["path"]: item for item in payload["packages"]}

    assert payload["dataset_prefix"] == "griffin_50scenes_25m"
    assert payload["package_count"] == 15
    assert packages["datasets/griffin_50scenes_25m/vehicle_metadata.zip"]["size_bytes"] == 10876283
    assert packages["datasets/griffin_50scenes_25m/drone_metadata.zip"]["size_bytes"] == 14384997
    assert packages["datasets/griffin_50scenes_25m/vehicle_lidar.zip"]["size_bytes"] == 214487013
    assert payload["total_size_bytes"] == 167190016122


def test_validate_run_accepts_log_metrics_near_paper_reference(tmp_path):
    log_path = tmp_path / "eval.log"
    log_path.write_text(
        "Evaluation summary\n"
        "pts_bbox_NuScenes/mAP: 0.481\n"
        "pts_bbox_NuScenes/amota: 0.491\n",
        encoding="utf-8",
    )

    result = run_cli("validate-run", "--profile", "smoke_25m_instance", "--log", str(log_path), "--json")
    payload = json.loads(result.stdout)

    assert payload["profile"] == "smoke_25m_instance"
    assert payload["passed"] is True
    assert payload["metrics"]["AP"] == 0.481
    assert payload["metrics"]["AMOTA"] == 0.491
    assert payload["checks"]["AP"]["expected"] == 0.479
    assert payload["checks"]["AMOTA"]["expected"] == 0.488


def test_validate_run_rejects_missing_metrics(tmp_path):
    log_path = tmp_path / "eval.log"
    log_path.write_text("Testing done without metric summary\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "validate-run",
            "--profile",
            "smoke_25m_instance",
            "--log",
            str(log_path),
            "--json",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.returncode == 3
    payload = json.loads(result.stdout)
    assert payload["passed"] is False
    assert payload["missing_metrics"] == ["AP", "AMOTA"]


def test_write_mobaxterm_script_emits_asset_gate_and_isolated_eval(tmp_path):
    out_path = tmp_path / "run_smoke.sh"
    run_cli("write-mobaxterm-script", "--profile", "smoke_25m_instance", "--out", str(out_path), "--json")
    script = out_path.read_text(encoding="utf-8")

    assert "cd griffin_repro/official" in script
    assert "bash tools/griffin_converter.sh griffin_50scenes_25m" in script
    assert "preprocess_assets=(" in script
    assert "evaluation_assets=(" in script
    assert script.index("bash tools/griffin_converter.sh griffin_50scenes_25m") < script.index("evaluation_assets=(")
    assert "missing_assets=0" in script
    assert "projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py" in script
    assert "python3 scripts/griffin_repro.py env-check --strict --json" in script
    assert "python3 scripts/griffin_repro.py validate-run --profile smoke_25m_instance" in script
    assert "-printf '%T@ %p\\n'" in script
    assert "carlaair_active_world" not in script


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
