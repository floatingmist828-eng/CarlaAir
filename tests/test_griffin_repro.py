import json
import importlib.util
import pickle
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
    assert "bash tools/dist_eval.sh" in payload["commands"][0]
    assert "./tools/dist_eval.sh" not in payload["commands"][0]
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


def test_paper_run_matrix_maps_baselines_to_runnable_official_configs():
    result = run_cli("paper-run-matrix", "--json")
    payload = json.loads(result.stdout)
    rows = payload["rows"]
    by_key = {(row["dataset"], row["method"], row["condition_id"]): row for row in rows}

    assert payload["summary"]["paper_result_rows"] == 142
    assert payload["summary"]["baseline_rows"] == 28
    assert payload["summary"]["baseline_complete"] is True

    coop = by_key[("50scenes_25m", "2b1-cooptrack", "baseline")]
    assert coop["AP"] == 0.479
    assert coop["AMOTA"] == 0.488
    assert coop["config"] == "projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py"
    assert coop["checkpoint"] == "ckpts/griffin_50scenes_25m/cooperative/instance_fusion/iter_33024.pth"
    assert coop["command_kind"] == "dist_eval"
    assert coop["config_exists"] is True
    assert coop["checkpoint_exists"] is True
    assert coop["status"] == "runnable_config"

    where2comm = by_key[("50scenes_25m", "2a2-where2comm", "baseline")]
    assert where2comm["status"] == "paper_result_only"
    assert where2comm["config"] is None
    assert where2comm["command"] is None


def test_paper_run_matrix_can_emit_25m_robustness_commands():
    result = run_cli(
        "paper-run-matrix",
        "--dataset",
        "50scenes_25m",
        "--include-robustness",
        "--json",
    )
    payload = json.loads(result.stdout)
    rows = payload["rows"]
    by_key = {(row["method"], row["condition_id"]): row for row in rows}

    assert payload["summary"]["result_rows"] == 142
    assert payload["summary"]["emitted_rows"] > 28

    packet = by_key[("2b1-cooptrack", "packet_loss_0.2")]
    assert packet["condition"] == {"packet_loss": 0.2}
    assert packet["config"] == "projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/drop_noised/tiny_track_r50_stream_bs8_48epoch_3cls_drop20.py"
    assert packet["checkpoint"] == "ckpts/griffin_50scenes_25m/cooperative/instance_fusion/iter_33024.pth"
    assert packet["command_kind"] == "dist_eval"
    assert packet["config_exists"] is True

    early_loc = by_key[("1-early fusion", "translation_error_m_1.5")]
    assert early_loc["config"] == "projects/configs_griffin_50scenes_25m/early-fusion/loc_noised/tiny_track_r50_stream_bs8_48epoch_3cls_loc15.py"
    assert early_loc["command_kind"] == "dist_eval"

    late_latency = by_key[("3-late fusion", "communication_latency_ms_200")]
    assert late_latency["config"] == "projects/configs_griffin_50scenes_25m/cooperative/late_fusion/latency/tiny_track_r50_stream_bs1_3cls_late_fusion_200latency.py"
    assert late_latency["command_kind"] == "late_fusion_pipeline"
    assert "tools/eval_late_fusion.sh" in late_latency["command"]
    assert "tiny_track_r50_stream_bs1_3cls_late_fusion_200latency_ab3dmot.py" in late_latency["command"]


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
    assert commands[-1].startswith("CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} bash tools/dist_eval.sh")
    assert "./tools/dist_eval.sh" not in commands[-1]
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


def test_prepare_partial_eval_writes_scene_subset_and_config(tmp_path):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    official = tmp_path / "official"
    source_ann = official / "data" / "infos" / "griffin_50scenes_25m" / "cooperative" / "griffin_infos_val.pkl"
    out_ann = source_ann.with_name("griffin_infos_val_partial_1scene.pkl")
    out_config = (
        official
        / "projects"
        / "configs_griffin_50scenes_25m"
        / "cooperative"
        / "instance_fusion"
        / "tiny_track_r50_stream_bs8_48epoch_3cls_partial_1scene.py"
    )
    base_config = official / module.profile_payload("smoke_25m_instance")["config"]
    source_ann.parent.mkdir(parents=True)
    base_config.parent.mkdir(parents=True)
    base_config.write_text("# base config\n", encoding="utf-8")
    source_ann.write_bytes(
        pickle.dumps(
            {
                "metadata": {"version": "v1.0-trainval"},
                "infos": [
                    {"token": "b0", "scene_token": "scene-b", "timestamp": 30},
                    {"token": "a0", "scene_token": "scene-a", "timestamp": 10},
                    {"token": "a1", "scene_token": "scene-a", "timestamp": 20},
                ],
            }
        )
    )

    module.OFFICIAL_ROOT = official
    payload = module.prepare_partial_eval(
        "smoke_25m_instance",
        scene_limit=1,
        max_samples=10,
        source_ann=str(source_ann),
        out_ann=str(out_ann),
        out_config=str(out_config),
    )

    written = pickle.loads(out_ann.read_bytes())
    assert [item["token"] for item in written["infos"]] == ["a0", "a1"]
    assert written["metadata"] == {"version": "v1.0-trainval"}
    assert payload["selected_scene_count"] == 1
    assert payload["selected_sample_count"] == 2
    assert payload["ann_file"] == "data/infos/griffin_50scenes_25m/cooperative/griffin_infos_val_partial_1scene.pkl"
    assert payload["config"] == (
        "projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/"
        "tiny_track_r50_stream_bs8_48epoch_3cls_partial_1scene.py"
    )
    assert "bash tools/dist_eval.sh" in payload["command"]
    assert "./tools/dist_eval.sh" not in payload["command"]
    assert payload["expected"] == {"AMOTA": 0.488, "AP": 0.479}

    config = out_config.read_text(encoding="utf-8")
    assert "_base_ = './tiny_track_r50_stream_bs8_48epoch_3cls.py'" in config
    assert "ann_file_val = './data/infos/griffin_50scenes_25m/cooperative/griffin_infos_val_partial_1scene.pkl'" in config
    assert "val=dict(ann_file=ann_file_val)" in config
    assert "test=dict(ann_file=ann_file_val)" in config


def test_prepare_drone_query_partial_eval_matches_cooperative_air_tokens(tmp_path):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    official = tmp_path / "official"
    prefix = "griffin_50scenes_25m"
    coop_ann = official / "data" / "infos" / prefix / "cooperative" / "griffin_infos_val.pkl"
    drone_ann = official / "data" / "infos" / prefix / "drone-side" / "griffin_infos_val.pkl"
    base_config = official / "projects" / f"configs_{prefix}" / "drone-side" / "tiny_track_r50_stream_bs8_24epoch_3cls_eval.py"
    coop_ann.parent.mkdir(parents=True)
    drone_ann.parent.mkdir(parents=True)
    base_config.parent.mkdir(parents=True)
    base_config.write_text("# base drone eval\n", encoding="utf-8")

    coop_infos = [
        {"token": "veh_a", "air_sample_token": "air_b", "scene_token": "scene_0", "timestamp": 20, "cams": {}},
        {"token": "veh_b", "air_sample_token": "air_a", "scene_token": "scene_0", "timestamp": 10, "cams": {}},
    ]
    drone_infos = [
        {"token": "air_a", "scene_token": "scene_0", "timestamp": 10, "cams": {}},
        {"token": "air_b", "scene_token": "scene_0", "timestamp": 20, "cams": {}},
    ]
    with coop_ann.open("wb") as handle:
        pickle.dump({"infos": coop_infos, "metadata": {"version": "v1.0-trainval"}}, handle)
    with drone_ann.open("wb") as handle:
        pickle.dump({"infos": drone_infos, "metadata": {"version": "v1.0-trainval"}}, handle)

    module.OFFICIAL_ROOT = official
    payload = module.prepare_drone_query_partial_eval(
        "smoke_25m_instance",
        scene_limit=1,
        max_samples=2,
        out_tag="partial_1scene_2samples",
    )

    assert payload["selected_sample_count"] == 2
    assert payload["ann_file"] == "data/infos/griffin_50scenes_25m/drone-side/griffin_infos_val_partial_1scene_2samples.pkl"
    assert payload["config"] == (
        "projects/configs_griffin_50scenes_25m/drone-side/"
        "tiny_track_r50_stream_bs8_24epoch_3cls_eval_partial_1scene_2samples.py"
    )
    assert "bash tools/dist_eval.sh" in payload["command"]
    with (official / payload["ann_file"]).open("rb") as handle:
        written = pickle.load(handle)
    assert [info["token"] for info in written["infos"]] == ["air_a", "air_b"]
    config = (official / payload["config"]).read_text(encoding="utf-8")
    assert "_base_ = './tiny_track_r50_stream_bs8_24epoch_3cls_eval.py'" in config
    assert "ann_file_val = './data/infos/griffin_50scenes_25m/drone-side/griffin_infos_val_partial_1scene_2samples.pkl'" in config


def test_partial_image_materialization_plan_maps_drone_archive_members(tmp_path):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    official = tmp_path / "official"
    prefix = "griffin_50scenes_25m"
    coop_ann = official / "data" / "infos" / prefix / "cooperative" / "griffin_infos_val.pkl"
    drone_ann = official / "data" / "infos" / prefix / "drone-side" / "griffin_infos_val.pkl"
    coop_ann.parent.mkdir(parents=True)
    drone_ann.parent.mkdir(parents=True)
    with coop_ann.open("wb") as handle:
        pickle.dump(
            {
                "infos": [
                    {
                        "token": "veh_001",
                        "air_sample_token": "air_001",
                        "scene_token": "scene_0",
                        "timestamp": 1,
                        "cams": {"CAM_FRONT": {"data_path": "samples/CAM_FRONT/000001.png"}},
                    }
                ],
                "metadata": {"version": "v1.0-trainval"},
            },
            handle,
        )
    with drone_ann.open("wb") as handle:
        pickle.dump(
            {
                "infos": [
                    {
                        "token": "air_001",
                        "scene_token": "scene_0",
                        "timestamp": 1,
                        "cams": {"CAM_BOTTOM": {"data_path": "samples/CAM_BOTTOM/000001.png"}},
                    }
                ],
                "metadata": {"version": "v1.0-trainval"},
            },
            handle,
        )

    module.OFFICIAL_ROOT = official
    payload = module.partial_image_materialization_plan(
        "smoke_25m_instance",
        image_side="drone-side",
        scene_limit=1,
        max_samples=1,
    )

    assert payload["image_side"] == "drone-side"
    assert payload["selected_sample_count"] == 1
    assert payload["frames"] == ["000001"]
    assert payload["directions"] == ["bottom"]
    assert payload["items"][0]["archive"] == "drone_camera_bottom.zip"
    assert payload["items"][0]["member"] == (
        "griffin_50scenes_25m/griffin-release/drone-side/camera/bottom/000001.png"
    )
    assert payload["items"][0]["dest"].endswith("griffin-release/drone-side/camera/bottom/000001.png")


def test_materialize_partial_images_dry_run_returns_plan_without_writes(tmp_path):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    official = tmp_path / "official"
    prefix = "griffin_50scenes_25m"
    ann = official / "data" / "infos" / prefix / "vehicle-side" / "griffin_infos_val.pkl"
    ann.parent.mkdir(parents=True)
    with ann.open("wb") as handle:
        pickle.dump(
            {
                "infos": [
                    {
                        "token": "veh_001",
                        "scene_token": "scene_0",
                        "timestamp": 1,
                        "cams": {"CAM_FRONT": {"data_path": "samples/CAM_FRONT/000001.png"}},
                    }
                ],
                "metadata": {"version": "v1.0-trainval"},
            },
            handle,
        )

    module.OFFICIAL_ROOT = official
    payload = module.materialize_partial_images(
        "smoke_25m_vehicle",
        image_side="vehicle-side",
        scene_limit=1,
        max_samples=1,
        dry_run=True,
    )

    assert payload["dry_run"] is True
    assert payload["planned_items"] == 1
    assert payload["written"] == 0
    assert not (official / "datasets").exists()


def test_materialize_partial_images_cli_accepts_shared_out_tag(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    official = tmp_path / "official"
    prefix = "griffin_50scenes_25m"
    ann = official / "data" / "infos" / prefix / "vehicle-side" / "griffin_infos_val.pkl"
    ann.parent.mkdir(parents=True)
    with ann.open("wb") as handle:
        pickle.dump(
            {
                "infos": [
                    {
                        "token": "veh_001",
                        "scene_token": "scene_0",
                        "timestamp": 1,
                        "cams": {"CAM_FRONT": {"data_path": "samples/CAM_FRONT/000001.png"}},
                    }
                ],
                "metadata": {"version": "v1.0-trainval"},
            },
            handle,
        )

    module.OFFICIAL_ROOT = official
    code = module.main(
        [
            "materialize-partial-images",
            "--profile",
            "smoke_25m_vehicle",
            "--image-side",
            "vehicle-side",
            "--scene-limit",
            "1",
            "--max-samples",
            "1",
            "--out-tag",
            "partial_1scene_1samples",
            "--dry-run",
            "--json",
        ]
    )

    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["planned_items"] == 1


def test_script_writers_are_compatible_with_python38_pathlib(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    original_write_text = Path.write_text

    def python38_write_text(self, data, encoding=None, errors=None):
        return original_write_text(self, data, encoding=encoding, errors=errors)

    monkeypatch.setattr(Path, "write_text", python38_write_text)

    payload = module.write_data_script("50scenes_25m", str(tmp_path / "download.sh"))

    assert payload["path"] == str(tmp_path / "download.sh")


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


def test_write_env_script_bootstraps_isolated_paper_environment(tmp_path):
    out_path = tmp_path / "setup_env.sh"
    result = run_cli("write-env-script", "--out", str(out_path), "--json")
    payload = json.loads(result.stdout)
    script = out_path.read_text(encoding="utf-8")

    assert payload["path"] == str(out_path)
    assert "Miniconda3-latest-Linux-x86_64.sh" in script
    assert "conda create -n \"$ENV_NAME\" python=3.8 pip -y" in script
    assert "torch==1.9.1+cu111" in script
    assert "torchvision==0.10.1+cu111" in script
    assert "mmcv-full==1.4.0" in script
    assert "mmdet==2.14.0 mmsegmentation==0.14.1" in script
    assert "timeout 180 git -c http.version=HTTP/1.1 clone --depth 1 --branch v0.17.1" in script
    assert "mmdetection3d/archive/refs/tags/v0.17.1.tar.gz" in script
    assert "python -m pip download mmdet3d==0.17.1" in script
    assert "Skipping mmdet3d spconv extension" in script
    assert "python -m pip install -r \"$MMDET3D_SRC/requirements/runtime.txt\"" in script
    assert "python -m pip install -v -e \"$MMDET3D_SRC\" --no-deps" in script
    assert "CUDA_HOME=\"${CUDA_HOME:-/usr/local/cuda}\"" in script
    assert "python scripts/griffin_repro.py env-check --strict --json" in script
    assert "carlaair_active_world" not in script


def test_data_packages_lists_griffin_25m_archives_and_download_size():
    result = run_cli("data-packages", "--dataset", "50scenes_25m", "--json")
    payload = json.loads(result.stdout)
    packages = {item["path"]: item for item in payload["packages"]}

    assert payload["dataset_prefix"] == "griffin_50scenes_25m"
    assert payload["package_profile"] == "full"
    assert payload["package_count"] == 15
    assert payload["full_package_count"] == 15
    assert packages["datasets/griffin_50scenes_25m/vehicle_metadata.zip"]["size_bytes"] == 10876283
    assert packages["datasets/griffin_50scenes_25m/drone_metadata.zip"]["size_bytes"] == 14384997
    assert packages["datasets/griffin_50scenes_25m/vehicle_lidar.zip"]["size_bytes"] == 214487013
    assert payload["total_size_bytes"] == 167190016122
    assert payload["full_total_size_bytes"] == 167190016122


def test_data_packages_can_select_smoke_eval_archives_only():
    result = run_cli(
        "data-packages",
        "--dataset",
        "50scenes_25m",
        "--package-profile",
        "smoke_25m_instance",
        "--json",
    )
    payload = json.loads(result.stdout)
    package_paths = {item["path"] for item in payload["packages"]}

    assert payload["package_profile"] == "smoke_25m_instance"
    assert payload["package_count"] == 12
    assert payload["full_package_count"] == 15
    assert payload["total_size_bytes"] == 162300524941
    assert payload["full_total_size_bytes"] == 167190016122
    assert "datasets/griffin_50scenes_25m/drone_camera_bottom.zip" in package_paths
    assert "datasets/griffin_50scenes_25m/vehicle_camera_right.zip" in package_paths
    assert "datasets/griffin_50scenes_25m/drone_camera_instance_segmentation.zip" not in package_paths
    assert "datasets/griffin_50scenes_25m/vehicle_lidar.zip" not in package_paths


def test_data_packages_can_select_vehicle_smoke_archives_only():
    result = run_cli(
        "data-packages",
        "--dataset",
        "50scenes_25m",
        "--package-profile",
        "smoke_25m_vehicle",
        "--json",
    )
    payload = json.loads(result.stdout)
    package_paths = {item["path"] for item in payload["packages"]}

    assert payload["package_profile"] == "smoke_25m_vehicle"
    assert payload["package_count"] == 6
    assert "datasets/griffin_50scenes_25m/md5.txt" in package_paths
    assert "datasets/griffin_50scenes_25m/vehicle_metadata.zip" in package_paths
    assert "datasets/griffin_50scenes_25m/vehicle_camera_front.zip" in package_paths
    assert "datasets/griffin_50scenes_25m/vehicle_camera_back.zip" in package_paths
    assert "datasets/griffin_50scenes_25m/vehicle_camera_left.zip" in package_paths
    assert "datasets/griffin_50scenes_25m/vehicle_camera_right.zip" in package_paths
    assert "datasets/griffin_50scenes_25m/drone_camera_front.zip" not in package_paths
    assert "datasets/griffin_50scenes_25m/vehicle_lidar.zip" not in package_paths


def test_write_data_script_downloads_from_mirror_with_checksums(tmp_path):
    out_path = tmp_path / "download_data.sh"
    result = run_cli("write-data-script", "--dataset", "50scenes_25m", "--out", str(out_path), "--json")
    payload = json.loads(result.stdout)
    script = out_path.read_text(encoding="utf-8")

    assert payload["path"] == str(out_path)
    assert payload["dataset"] == "50scenes_25m"
    assert payload["package_profile"] == "smoke_25m_instance"
    assert "https://hf-mirror.com/datasets/wjh-svm/Griffin/resolve/main" in script
    assert "TOTAL_SIZE_BYTES=162300524941" in script
    assert "FULL_TOTAL_SIZE_BYTES=167190016122" in script
    assert "drone_camera_back.zip|19492671867" in script
    assert "vehicle_metadata.zip|10876283" in script
    assert "vehicle_lidar.zip|214487013" not in script
    assert "DOWNLOAD_JOBS=\"${GRIFFIN_DOWNLOAD_JOBS:-3}\"" in script
    assert "DOWNLOAD_MAX_PASSES=\"${GRIFFIN_DOWNLOAD_MAX_PASSES:-12}\"" in script
    assert "LOCK_FILE=\"$ARCHIVE_DIR/.download.lock\"" in script
    assert "flock -n 9" in script
    assert "Another Griffin data download is already active" in script
    assert "curl --retry 5 --connect-timeout 30 -L -C -" in script
    assert "is larger than expected; deleting corrupt partial archive before retry" in script
    assert "rm -f \"$output\"" in script
    assert "wait -n" in script
    assert "md5.selected.txt" in script
    assert "md5sum -c md5.selected.txt" in script
    assert "DATA_PARENT=\"$ROOT/griffin_repro/official/datasets\"" in script
    assert "unzip -oq \"$archive\" -d \"$DATA_PARENT\"" in script
    assert "extracted.to-data-parent" in script
    assert "check-partial-assets --profile smoke_25m_instance" in script
    assert "carlaair_active_world" not in script


def test_check_data_packages_reports_missing_local_archives():
    result = run_cli(
        "check-data-packages",
        "--dataset",
        "50scenes_25m",
        "--package-profile",
        "smoke_25m_instance",
        "--json",
    )
    payload = json.loads(result.stdout)

    assert payload["package_profile"] == "smoke_25m_instance"
    assert payload["package_count"] == 12
    assert payload["total_size_bytes"] == 162300524941
    assert payload["ready"] is False
    assert any(item["path"].endswith("drone_camera_back.zip") for item in payload["checks"])


def test_check_data_packages_reports_oversized_corrupt_archives(tmp_path):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkey_root = tmp_path / "repo"
    archive_dir = monkey_root / "griffin_repro" / "official" / "datasets" / "tiny_prefix" / "archives"
    archive_dir.mkdir(parents=True)
    corrupt = archive_dir / "oversized.zip"
    corrupt.write_bytes(b"0123456789AB")

    module.REPO_ROOT = monkey_root
    module.OFFICIAL_ROOT = monkey_root / "griffin_repro" / "official"
    module.DATASETS["tiny"] = {
        "dataset_prefix": "tiny_prefix",
        "scene_count": 1,
        "altitude": "unit",
    }
    module.DATA_PACKAGES["tiny"] = [("datasets/tiny_prefix/oversized.zip", 10)]
    module.DATA_PACKAGE_PROFILES["oversize_test"] = {"datasets/tiny_prefix/oversized.zip"}

    payload = module.check_data_packages("tiny", "oversize_test")

    assert payload["ready"] is False
    assert payload["checks"][0]["actual_size_bytes"] == 12
    assert payload["checks"][0]["expected_size_bytes"] == 10
    assert payload["checks"][0]["missing_size_bytes"] == 0
    assert payload["checks"][0]["oversize_size_bytes"] == 2
    assert payload["checks"][0]["size_delta_bytes"] == 2


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


def test_validate_run_accepts_official_tracking_table_metrics(tmp_path):
    log_path = tmp_path / "eval.log"
    log_path.write_text(
        "mAP: 0.1625\n"
        "Aggregated results:\n"
        "AMOTA\t0.138\n"
        "AMOTP\t1.741\n",
        encoding="utf-8",
    )

    result = run_cli(
        "validate-run",
        "--profile",
        "smoke_25m_vehicle",
        "--log",
        str(log_path),
        "--tolerance",
        "1.0",
        "--json",
    )
    payload = json.loads(result.stdout)

    assert payload["passed"] is True
    assert payload["metrics"]["AP"] == 0.1625
    assert payload["metrics"]["AMOTA"] == 0.138


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
    assert "CONDA_HOME=\"${GRIFFIN_CONDA_HOME:-$HOME/miniconda3}\"" in script
    assert "conda activate \"$GRIFFIN_ENV_NAME\"" in script
    assert "python scripts/griffin_repro.py env-check --strict --json" in script
    assert "import mmdet3d.ops.spconv" in script
    assert "bash griffin_repro/build_mmdet3d_spconv_ext_mobaxterm.sh" in script
    assert "python scripts/griffin_repro.py validate-run --profile smoke_25m_instance" in script
    assert "-printf '%T@ %p\\n'" in script
    assert "carlaair_active_world" not in script


def test_write_mobaxterm_script_can_run_partial_final_eval(tmp_path):
    out_path = tmp_path / "run_smoke.sh"
    run_cli("write-mobaxterm-script", "--profile", "smoke_25m_instance", "--out", str(out_path), "--json")
    script = out_path.read_text(encoding="utf-8")

    assert "GRIFFIN_PARTIAL_SCENE_LIMIT" in script
    assert "GRIFFIN_PARTIAL_MAX_SAMPLES" in script
    assert 'partial_scene_limit="${GRIFFIN_PARTIAL_SCENE_LIMIT:-1}"' in script
    assert 'partial_max_samples="${GRIFFIN_PARTIAL_MAX_SAMPLES:-20}"' in script
    assert "prepare-partial-eval --profile smoke_25m_instance" in script
    assert "smoke_25m_instance_partial_eval.json" in script
    assert "partial_eval_command=" in script
    assert "final_eval_to_run=\"$partial_eval_command\"" in script
    assert "eval \"$final_eval_to_run\"" in script
    assert "GRIFFIN_PARTIAL_METRIC_TOLERANCE" in script


def test_write_mobaxterm_script_prepares_partial_images_and_drone_query(tmp_path):
    out_path = tmp_path / "run_smoke.sh"
    run_cli("write-mobaxterm-script", "--profile", "smoke_25m_instance", "--out", str(out_path), "--json")
    script = out_path.read_text(encoding="utf-8")

    assert "materialize-partial-images --profile smoke_25m_instance --image-side vehicle-side" in script
    assert "materialize-partial-images --profile smoke_25m_instance --image-side drone-side" in script
    assert "prepare-drone-query-partial-eval --profile smoke_25m_instance" in script
    assert script.index("prepare-drone-query-partial-eval --profile smoke_25m_instance") < script.index(
        "prepare-partial-eval --profile smoke_25m_instance"
    )


def test_write_vehicle_mobaxterm_script_uses_profile_partial_eval_file(tmp_path):
    out_path = tmp_path / "run_smoke_vehicle.sh"
    run_cli("write-mobaxterm-script", "--profile", "smoke_25m_vehicle", "--out", str(out_path), "--json")
    script = out_path.read_text(encoding="utf-8")

    assert "prepare-partial-eval --profile smoke_25m_vehicle" in script
    assert "smoke_25m_vehicle_partial_eval.json" in script
    assert "smoke_25m_instance_partial_eval.json" not in script
    assert "else\n\nfi" not in script


def test_vehicle_partial_run_plan_uses_vehicle_only_preprocess():
    result = run_cli("plan-partial-run", "--profile", "smoke_25m_vehicle", "--json")
    payload = json.loads(result.stdout)
    commands = payload["commands"]
    assets = payload["required_assets"]

    assert payload["method"] == "0-no fusion"
    assert "GriffinKittiToNuScenesConverter" in "\n".join(commands)
    assert "sys.path.insert" in "\n".join(commands)
    assert "tools/griffin_data_converter" in "\n".join(commands)
    assert "converter.convert({})" in "\n".join(commands)
    assert "converter.convert([])" not in "\n".join(commands)
    assert not any("drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval" in command for command in commands)
    assert "griffin_repro/official/ckpts/griffin_50scenes_25m/vehicle-side/iter_33024.pth" in assets
    assert "griffin_repro/official/ckpts/griffin_50scenes_25m/drone-side/iter_33024.pth" not in assets
    assert "griffin_repro/official/data/infos/griffin_50scenes_25m/drone-side/track_query" not in assets


def test_write_supervisor_script_retries_data_download_until_smoke_ready(tmp_path):
    out_path = tmp_path / "supervise_smoke.sh"
    result = run_cli(
        "write-supervisor-script",
        "--profile",
        "smoke_25m_instance",
        "--dataset",
        "50scenes_25m",
        "--package-profile",
        "smoke_25m_instance",
        "--out",
        str(out_path),
        "--json",
    )
    payload = json.loads(result.stdout)
    script = out_path.read_text(encoding="utf-8")

    assert payload["path"] == str(out_path)
    assert payload["profile"] == "smoke_25m_instance"
    assert payload["dataset"] == "50scenes_25m"
    assert payload["package_profile"] == "smoke_25m_instance"
    assert "GRIFFIN_SUPERVISOR_MAX_ATTEMPTS" in script
    assert "check-data-packages --dataset 50scenes_25m --package-profile smoke_25m_instance --json" in script
    assert "cleanup_stale_downloads()" in script
    assert "pkill -f \"curl .*griffin_50scenes_25m/archives\"" in script
    assert "pkill -f \"bash griffin_repro/download_50scenes_25m_mobaxterm.sh\"" in script
    assert "bash griffin_repro/download_50scenes_25m_mobaxterm.sh" in script
    assert "download_status=$?" in script
    assert "sleep \"$SUPERVISOR_SLEEP_SEC\"" in script
    assert "bash griffin_repro/run_smoke_25m_instance_mobaxterm.sh" in script
    assert "smoke_25m_instance_supervisor.latest" in script
    assert "carlaair_active_world" not in script


def test_write_vehicle_supervisor_uses_vehicle_download_and_run_scripts(tmp_path):
    out_path = tmp_path / "supervise_vehicle.sh"
    result = run_cli(
        "write-supervisor-script",
        "--profile",
        "smoke_25m_vehicle",
        "--dataset",
        "50scenes_25m",
        "--package-profile",
        "smoke_25m_vehicle",
        "--out",
        str(out_path),
        "--json",
    )
    payload = json.loads(result.stdout)
    script = out_path.read_text(encoding="utf-8")

    assert payload["profile"] == "smoke_25m_vehicle"
    assert payload["package_profile"] == "smoke_25m_vehicle"
    assert "check-data-packages --dataset 50scenes_25m --package-profile smoke_25m_vehicle --json" in script
    assert "bash griffin_repro/download_50scenes_25m_vehicle_mobaxterm.sh" in script
    assert "bash griffin_repro/run_smoke_25m_vehicle_mobaxterm.sh" in script
    assert "smoke_25m_vehicle_supervisor.latest" in script
    assert "run_smoke_25m_instance_mobaxterm.sh" not in script


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
