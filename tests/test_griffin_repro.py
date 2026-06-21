import json
import hashlib
import importlib.util
import io
import pickle
import struct
import subprocess
import sys
import threading
import time
import zipfile
import zlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "griffin_repro.py"
SYNC_SCRIPT = REPO_ROOT / "scripts" / "sync_griffin_remote.py"
MANIFEST = REPO_ROOT / "griffin_repro" / "manifest.json"
OFFICIAL = REPO_ROOT / "griffin_repro" / "official"
FINALIZER_SCRIPT = REPO_ROOT / "griffin_repro" / "finalize_25m_full_validation_mobaxterm.sh"


class _StrictQueryPayload:
    def __init__(self, fields):
        self._fields = fields

    def get(self, name):
        if name not in self._fields:
            raise KeyError(name)
        return self._fields[name]


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
    assert payload["config_files"] >= 97
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


def test_late_fusion_converter_defaults_optional_noise_fields():
    converter = (
        REPO_ROOT
        / "griffin_repro"
        / "official"
        / "tools"
        / "result_converter"
        / "det_result_late_fusion.py"
    ).read_text(encoding="utf-8")

    assert "drop_prob = cfg.get('drop_prob', 0.0)" in converter
    assert "loc_noise_std = cfg.get('loc_noise_std', 0.0)" in converter
    assert "orien_noise_std = cfg.get('orien_noise_std', 0.0)" in converter


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


def test_prepare_partial_eval_can_sample_each_selected_scene(tmp_path):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    official = tmp_path / "official"
    prefix = "griffin_50scenes_25m"
    source_ann = official / "data" / "infos" / prefix / "cooperative" / "griffin_infos_val.pkl"
    base_config = official / module.profile_payload("smoke_25m_instance")["config"]
    out_config = (
        official
        / "projects"
        / f"configs_{prefix}"
        / "cooperative"
        / "instance_fusion"
        / "tiny_track_r50_stream_bs8_48epoch_3cls_partial_3scene_2per_scene.py"
    )
    source_ann.parent.mkdir(parents=True)
    base_config.parent.mkdir(parents=True)
    base_config.write_text("# base config\n", encoding="utf-8")

    infos = []
    for scene_idx in range(3):
        for sample_idx in range(4):
            infos.append(
                {
                    "token": f"scene_{scene_idx}_{sample_idx}",
                    "scene_token": f"scene_{scene_idx}",
                    "timestamp": scene_idx * 100 + sample_idx,
                    "cams": {},
                }
            )
    with source_ann.open("wb") as handle:
        pickle.dump({"infos": infos, "metadata": {"version": "v1.0-trainval"}}, handle)

    module.OFFICIAL_ROOT = official
    payload = module.prepare_partial_eval(
        "smoke_25m_instance",
        scene_limit=3,
        samples_per_scene=2,
        out_tag="partial_3scene_2per_scene",
    )

    assert payload["selected_scene_count"] == 3
    assert payload["selected_sample_count"] == 6
    assert payload["selected_scenes"] == ["scene_0", "scene_1", "scene_2"]
    assert payload["config"] == module._relative_posix(out_config, official)
    with (official / payload["ann_file"]).open("rb") as handle:
        written = pickle.load(handle)
    assert [info["token"] for info in written["infos"]] == [
        "scene_0_0",
        "scene_0_1",
        "scene_1_0",
        "scene_1_1",
        "scene_2_0",
        "scene_2_1",
    ]


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


def test_describe_partial_subset_reports_scene_and_class_coverage(tmp_path):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    official = tmp_path / "official"
    prefix = "griffin_50scenes_25m"

    def write_infos(side, names_by_frame):
        ann = official / "data" / "infos" / prefix / side / "griffin_infos_val.pkl"
        ann.parent.mkdir(parents=True)
        infos = []
        for index, names in enumerate(names_by_frame):
            scene = "scene_0" if index < 3 else "scene_1"
            infos.append(
                {
                    "token": f"{side}_{index}",
                    "scene_token": scene,
                    "timestamp": index,
                    "gt_names": names,
                }
            )
        with ann.open("wb") as handle:
            pickle.dump({"infos": infos, "metadata": {"version": "v1.0-trainval"}}, handle)

    write_infos("cooperative", [["car"], ["car", "pedestrian"], ["bicycle"], ["car"], ["pedestrian"]])
    write_infos("vehicle-side", [["car"], ["car"], [], ["car"], []])
    write_infos("early-fusion", [["car"], ["pedestrian"], ["bicycle"], ["car"], ["pedestrian"]])
    write_infos("drone-side", [["pedestrian"], ["car"], ["bicycle"], ["car"], ["pedestrian"]])
    module.OFFICIAL_ROOT = official

    payload = module.describe_partial_subset(
        "smoke_25m_instance",
        scene_limit=2,
        samples_per_scene=2,
    )

    assert payload["dataset"] == "50scenes_25m"
    assert payload["paper_scene_count"] == 47
    assert payload["scene_limit"] == 2
    assert payload["samples_per_scene"] == 2
    assert payload["sides"]["cooperative"]["total_samples"] == 5
    assert payload["sides"]["cooperative"]["scene_count"] == 2
    assert payload["sides"]["cooperative"]["selected_sample_count"] == 4
    assert payload["sides"]["cooperative"]["selected_scene_frame_counts"] == {"scene_0": 2, "scene_1": 2}
    assert payload["sides"]["cooperative"]["class_annotation_counts"] == {
        "car": 3,
        "pedestrian": 2,
    }
    assert payload["sides"]["vehicle-side"]["frames_with_any_gt_names"] == 3
    assert payload["sides"]["vehicle-side"]["class_frame_presence"] == {"car": 3}


def test_materialize_partial_images_reports_progress_to_stderr(tmp_path, monkeypatch, capsys):
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
    monkeypatch.setattr(module, "_zip_entry_from_url", lambda *_args: b"png-bytes")

    payload = module.materialize_partial_images(
        "smoke_25m_vehicle",
        image_side="vehicle-side",
        scene_limit=1,
        max_samples=1,
    )

    captured = capsys.readouterr()
    assert payload["written"] == 1
    assert "Materializing vehicle-side images: 1/1" in captured.err
    assert "000001.png" in captured.err


def test_materialize_partial_images_can_fetch_missing_url_images_in_parallel(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.OFFICIAL_ROOT = tmp_path / "official"
    first_dest = module.OFFICIAL_ROOT / "datasets" / "prefix" / "first.png"
    second_dest = module.OFFICIAL_ROOT / "datasets" / "prefix" / "second.png"
    plan = {
        "profile": "smoke_25m_vehicle",
        "dataset": "50scenes_25m",
        "dataset_prefix": "prefix",
        "image_side": "vehicle-side",
        "source_ann": "ann.pkl",
        "selected_scene_count": 1,
        "selected_sample_count": 2,
        "selected_scenes": ["scene_0"],
        "frames": ["first", "second"],
        "directions": ["front"],
        "items": [
            {
                "archive": "vehicle_camera_front.zip",
                "archive_size_bytes": 123,
                "dest": str(first_dest),
                "member": "prefix/first.png",
                "url": "https://example.invalid/archive.zip",
            },
            {
                "archive": "vehicle_camera_front.zip",
                "archive_size_bytes": 123,
                "dest": str(second_dest),
                "member": "prefix/second.png",
                "url": "https://example.invalid/archive.zip",
            },
        ],
    }
    monkeypatch.setattr(module, "partial_image_materialization_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setenv("GRIFFIN_MATERIALIZE_JOBS", "2")

    lock = threading.Lock()
    active_fetches = 0
    max_active_fetches = 0

    def fake_zip_entry_from_url(_url, _archive_size, member):
        nonlocal active_fetches, max_active_fetches
        with lock:
            active_fetches += 1
            max_active_fetches = max(max_active_fetches, active_fetches)
        time.sleep(0.05)
        with lock:
            active_fetches -= 1
        return member.encode("utf-8")

    monkeypatch.setattr(module, "_zip_entry_from_url", fake_zip_entry_from_url)
    monkeypatch.setattr(module, "_central_directory_from_url", lambda *_args: (0, b""))

    payload = module.materialize_partial_images(
        "smoke_25m_vehicle",
        image_side="vehicle-side",
        scene_limit=1,
        max_samples=2,
    )

    assert payload["materialize_jobs"] == 2
    assert payload["written"] == 2
    assert payload["skipped_existing"] == 0
    assert max_active_fetches == 2
    assert first_dest.read_bytes() == b"prefix/first.png"
    assert second_dest.read_bytes() == b"prefix/second.png"
    captured = capsys.readouterr()
    assert "Materializing vehicle-side images: submitted 2/2" in captured.err
    assert "Materializing vehicle-side images: completed 2/2" in captured.err


def test_zip_entry_from_url_reads_zip64_central_directory(monkeypatch):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    member = "griffin_50scenes_25m/griffin-release/vehicle-side/camera/back/012901.png"
    member_bytes = member.encode("utf-8")
    payload = b"png-bytes"
    crc = zlib.crc32(payload) & 0xFFFFFFFF

    local_offset = 0
    local_header = (
        struct.pack(
            "<4s5H3I2H",
            b"PK\x03\x04",
            45,
            0,
            0,
            0,
            0,
            crc,
            len(payload),
            len(payload),
            len(member_bytes),
            0,
        )
        + member_bytes
    )
    central_offset = len(local_header) + len(payload)
    zip64_extra = struct.pack("<HHQ", 0x0001, 8, local_offset)
    central = (
        struct.pack(
            "<4s6H3I5H2I",
            b"PK\x01\x02",
            45,
            45,
            0,
            0,
            0,
            0,
            crc,
            len(payload),
            len(payload),
            len(member_bytes),
            len(zip64_extra),
            0,
            0,
            0,
            0,
            0xFFFFFFFF,
        )
        + member_bytes
        + zip64_extra
    )
    zip64_eocd_offset = central_offset + len(central)
    zip64_eocd = struct.pack(
        "<4sQ2H2I4Q",
        b"PK\x06\x06",
        44,
        45,
        45,
        0,
        0,
        1,
        1,
        len(central),
        central_offset,
    )
    zip64_locator = struct.pack("<4sIQI", b"PK\x06\x07", 0, zip64_eocd_offset, 1)
    eocd = struct.pack(
        "<4s4H2IH",
        b"PK\x05\x06",
        0,
        0,
        0xFFFF,
        0xFFFF,
        0xFFFFFFFF,
        0xFFFFFFFF,
        0,
    )
    archive = local_header + payload + central + zip64_eocd + zip64_locator + eocd

    def fake_curl_range(_url, start, end):
        return archive[start : end + 1]

    monkeypatch.setattr(module, "_curl_range", fake_curl_range)

    assert module._zip_entry_from_url("https://example.invalid/archive.zip", len(archive), member) == payload


def test_url_zip_entry_reader_reuses_central_directory_for_same_archive(monkeypatch):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive_file:
        archive_file.writestr("root/first.txt", b"first")
        archive_file.writestr("root/second.txt", b"second")
    archive = archive_buffer.getvalue()

    eocd_at = archive.rfind(b"PK\x05\x06")
    assert eocd_at >= 0
    _, _, _, _, _, central_size, central_offset, _ = struct.unpack(
        "<4s4H2IH",
        archive[eocd_at : eocd_at + 22],
    )

    calls = []

    def fake_curl_range(_url, start, end):
        calls.append((start, end))
        return archive[start : end + 1]

    monkeypatch.setattr(module, "_curl_range", fake_curl_range)

    assert module._zip_entry_from_url("https://example.invalid/archive.zip", len(archive), "root/first.txt") == b"first"
    assert module._zip_entry_from_url("https://example.invalid/archive.zip", len(archive), "root/second.txt") == b"second"

    assert calls.count((central_offset, central_offset + central_size - 1)) == 1


def test_curl_range_uses_retries_and_timeout(monkeypatch):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0, stdout=b"ok", stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module._curl_range("https://example.invalid/archive.zip", 10, 20) == b"ok"

    assert "--retry" in captured["args"]
    assert "--retry-all-errors" in captured["args"]
    assert "--http1.1" in captured["args"]
    assert "--connect-timeout" in captured["args"]
    assert "--max-time" in captured["args"]


def test_ab3dmot_xinshuo_compat_helpers(tmp_path):
    sys.path.insert(0, str(OFFICIAL / "projects" / "ab3dmot_plugin"))
    try:
        import easydict
        import xinshuo_io
        import xinshuo_miscellaneous
        import xinshuo_visualization
    finally:
        sys.path.pop(0)

    output_file = tmp_path / "nested" / "values.txt"
    xinshuo_io.save_txt_file(["2", "1"], output_file)

    lines, count = xinshuo_io.load_txt_file(output_file)
    folder_items, folder_count = xinshuo_io.load_list_from_folder(output_file.parent)

    assert lines == ["2", "1"]
    assert count == 2
    assert folder_items == [str(output_file)]
    assert folder_count == 1
    assert xinshuo_io.fileparts(output_file) == (str(output_file.parent), "values", ".txt")
    assert xinshuo_io.is_path_exists(output_file)
    assert xinshuo_miscellaneous.merge_listoflist([["b", "a"], ["a"]], unique=True) == ["b", "a"]
    log_buffer = io.StringIO()
    xinshuo_miscellaneous.print_log("tracking started", log=log_buffer, display=False)
    assert log_buffer.getvalue() == "tracking started\n"
    assert len(xinshuo_miscellaneous.get_timestring()) == len("20260620_022401")
    assert xinshuo_visualization.random_colors(2) == [(1.0, 0.0, 0.0), (0.0, 1.0, 1.0)]
    assert easydict.EasyDict({"tracker": {"name": "ab3dmot"}}).tracker.name == "ab3dmot"


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
    assert "python -m pip install \"filterpy==1.4.5\"" in script
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


def test_data_packages_lists_released_non_25m_subsets():
    result_55m = run_cli("data-packages", "--dataset", "50scenes_55m", "--json")
    payload_55m = json.loads(result_55m.stdout)
    packages_55m = {item["path"]: item for item in payload_55m["packages"]}

    assert payload_55m["dataset_prefix"] == "griffin_50scenes_55m"
    assert payload_55m["package_count"] == 15
    assert payload_55m["total_size_bytes"] == 192907983888
    assert packages_55m["datasets/griffin_50scenes_55m/vehicle_lidar.zip"]["size_bytes"] == 1811791394

    result_random = run_cli("data-packages", "--dataset", "100scenes_random", "--json")
    payload_random = json.loads(result_random.stdout)
    package_paths = {item["path"] for item in payload_random["packages"]}

    assert payload_random["dataset_prefix"] == "griffin_100scenes_random"
    assert payload_random["package_count"] == 24
    assert payload_random["total_size_bytes"] == 403922969356
    assert "datasets/griffin_100scenes_random/drone_camera_front.z01" in package_paths


def test_checkpoint_packages_lists_released_model_weights():
    result = run_cli("checkpoint-packages", "--dataset", "50scenes_40m", "--json")
    payload = json.loads(result.stdout)
    packages = {item["path"]: item for item in payload["packages"]}

    assert payload["dataset_prefix"] == "griffin_50scenes_40m"
    assert payload["package_count"] == 7
    assert payload["total_size_bytes"] == 1544556037
    assert packages["ckpts/griffin_50scenes_40m/cooperative/instance_fusion/iter_38784.pth"]["size_bytes"] == 222769163

    result_55m = run_cli("checkpoint-packages", "--dataset", "50scenes_55m", "--json")
    payload_55m = json.loads(result_55m.stdout)
    assert payload_55m["package_count"] == 4
    assert payload_55m["total_size_bytes"] == 882185460


def test_write_checkpoint_script_downloads_released_model_weights(tmp_path):
    out_path = tmp_path / "download_ckpts.sh"
    result = run_cli("write-checkpoint-script", "--dataset", "50scenes_25m", "--out", str(out_path), "--json")
    payload = json.loads(result.stdout)
    script = out_path.read_text(encoding="utf-8")

    assert payload["path"] == str(out_path)
    assert payload["dataset"] == "50scenes_25m"
    assert payload["dataset_prefix"] == "griffin_50scenes_25m"
    assert payload["total_size_bytes"] == 1104957567
    assert "https://hf-mirror.com/datasets/wjh-svm/Griffin/resolve/main" in script
    assert "TOTAL_SIZE_BYTES=1104957567" in script
    assert "ckpts/griffin_50scenes_25m/drone-side/iter_33024.pth|219802531" in script
    assert "CKPT_ROOT=\"$ROOT/griffin_repro/official/ckpts/griffin_50scenes_25m\"" in script
    assert "curl --retry 5 --connect-timeout 30 -L -C -" in script
    assert "is larger than expected; deleting corrupt partial checkpoint before retry" in script
    assert "check-checkpoint-packages --dataset 50scenes_25m --json" in script
    assert "unzip" not in script
    assert "carlaair_active_world" not in script


def test_check_checkpoint_packages_reports_local_file_completeness(tmp_path):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkey_root = tmp_path / "repo"
    checkpoint_root = monkey_root / "griffin_repro" / "official" / "ckpts" / "tiny_prefix"
    checkpoint_root.mkdir(parents=True)
    complete = checkpoint_root / "complete.pth"
    complete.write_bytes(b"1234")

    module.REPO_ROOT = monkey_root
    module.OFFICIAL_ROOT = monkey_root / "griffin_repro" / "official"
    module.DATASETS["tiny"] = {
        "dataset_prefix": "tiny_prefix",
        "scene_count": 1,
        "altitude": "unit",
    }
    module.CHECKPOINT_PACKAGES["tiny"] = [
        ("ckpts/tiny_prefix/complete.pth", 4),
        ("ckpts/tiny_prefix/missing.pth", 6),
    ]

    summary = module.check_checkpoint_packages("tiny")

    assert summary["dataset"] == "tiny"
    assert summary["checkpoint_dir"] == "griffin_repro/official/ckpts/tiny_prefix"
    assert summary["package_count"] == 2
    assert summary["complete_count"] == 1
    assert summary["complete_size_bytes"] == 4
    assert summary["ready"] is False
    missing = next(item for item in summary["checks"] if item["path"].endswith("missing.pth"))
    assert missing["actual_size_bytes"] == 0
    assert missing["missing_size_bytes"] == 6
    assert missing["complete"] is False


def test_verify_data_package_md5_reports_matches_mismatches_and_missing_files(tmp_path):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkey_root = tmp_path / "repo"
    module.REPO_ROOT = monkey_root
    module.OFFICIAL_ROOT = monkey_root / "griffin_repro" / "official"
    archive_dir = module.OFFICIAL_ROOT / "datasets" / "tiny_prefix" / "archives"
    archive_dir.mkdir(parents=True)
    good_payload = b"good"
    bad_payload = b"bad"
    (archive_dir / "good.zip").write_bytes(good_payload)
    (archive_dir / "bad.zip").write_bytes(bad_payload)
    (archive_dir / "md5.txt").write_text(
        "\n".join(
            [
                f"{hashlib.md5(good_payload).hexdigest()}  ./good.zip",
                f"{hashlib.md5(b'expected').hexdigest()}  ./bad.zip",
                f"{hashlib.md5(b'missing').hexdigest()}  ./missing.zip",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    module.DATASETS["tiny"] = {
        "dataset_prefix": "tiny_prefix",
        "scene_count": 1,
        "altitude": "unit",
    }
    module.DATA_PACKAGES["tiny"] = [
        ("datasets/tiny_prefix/md5.txt", len((archive_dir / "md5.txt").read_bytes())),
        ("datasets/tiny_prefix/good.zip", len(good_payload)),
        ("datasets/tiny_prefix/bad.zip", len(bad_payload)),
        ("datasets/tiny_prefix/partial.zip", 7),
        ("datasets/tiny_prefix/missing.zip", 7),
    ]
    partial_payload = b"part"
    (archive_dir / "partial.zip").write_bytes(partial_payload)
    module.DATA_PACKAGE_PROFILES["tiny_profile"] = {
        "datasets/tiny_prefix/md5.txt",
        "datasets/tiny_prefix/good.zip",
        "datasets/tiny_prefix/bad.zip",
        "datasets/tiny_prefix/partial.zip",
        "datasets/tiny_prefix/missing.zip",
    }

    summary = module.verify_data_package_md5("tiny", "tiny_profile")

    assert summary["dataset"] == "tiny"
    assert summary["package_profile"] == "tiny_profile"
    assert summary["ready"] is False
    assert summary["checked_count"] == 4
    assert summary["matched_count"] == 1
    checks = {item["archive"]: item for item in summary["checks"]}
    assert checks["good.zip"]["status"] == "matched"
    assert checks["bad.zip"]["status"] == "mismatch"
    assert checks["partial.zip"]["status"] == "size-mismatch"
    assert checks["partial.zip"]["actual_md5"] is None
    assert checks["missing.zip"]["status"] == "missing"


def test_audit_25m_assets_summarizes_fixed_height_reproduction_inputs(tmp_path):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkey_root = tmp_path / "repo"
    module.REPO_ROOT = monkey_root
    module.REPRO_ROOT = monkey_root / "griffin_repro"
    module.OFFICIAL_ROOT = monkey_root / "griffin_repro" / "official"
    module.MANIFEST_PATH = monkey_root / "griffin_repro" / "manifest.json"
    module.RESULTS_CSV = module.OFFICIAL_ROOT / "docs" / "detailed_results.csv"

    split_dir = module.OFFICIAL_ROOT / "data" / "split_datas"
    split_dir.mkdir(parents=True)
    (split_dir / "griffin_50scenes_25m.json").write_text(
        json.dumps({"batch_split": {"train": ["scene-a"], "val": ["scene-b", "scene-c"]}}),
        encoding="utf-8",
    )

    archive_dir = module.OFFICIAL_ROOT / "datasets" / "griffin_50scenes_25m" / "archives"
    archive_dir.mkdir(parents=True)
    module.DATA_PACKAGES["50scenes_25m"] = [
        ("datasets/griffin_50scenes_25m/md5.txt", 3),
        ("datasets/griffin_50scenes_25m/vehicle_metadata.zip", 4),
    ]
    (archive_dir / "md5.txt").write_bytes(b"123")
    (archive_dir / "vehicle_metadata.zip").write_bytes(b"1234")

    ckpt_root = module.OFFICIAL_ROOT / "ckpts" / "griffin_50scenes_25m"
    (ckpt_root / "drone-side").mkdir(parents=True)
    module.CHECKPOINT_PACKAGES["50scenes_25m"] = [
        ("ckpts/griffin_50scenes_25m/drone-side/iter_33024.pth", 5),
    ]
    (ckpt_root / "drone-side" / "iter_33024.pth").write_bytes(b"12345")

    for rel_path in [
        "datasets/griffin_50scenes_25m/griffin-release/vehicle-side",
        "datasets/griffin_50scenes_25m/griffin-release/drone-side",
        "datasets/griffin_50scenes_25m/griffin-nuscenes/cooperative",
        "data/infos/griffin_50scenes_25m/vehicle-side",
        "data/infos/griffin_50scenes_25m/drone-side",
        "data/infos/griffin_50scenes_25m/cooperative",
        "projects/configs_griffin_50scenes_25m/vehicle-side",
        "projects/configs_griffin_50scenes_25m/early-fusion",
        "projects/configs_griffin_50scenes_25m/cooperative/instance_fusion",
        "projects/configs_griffin_50scenes_25m/cooperative/late_fusion",
        "tools/analysis_tools",
        "tools",
    ]:
        (module.OFFICIAL_ROOT / rel_path).mkdir(parents=True, exist_ok=True)

    for side in ["vehicle-side", "drone-side", "early-fusion", "cooperative"]:
        metadata_dir = (
            module.OFFICIAL_ROOT
            / "datasets"
            / "griffin_50scenes_25m"
            / "griffin-nuscenes"
            / side
            / "v1.0-trainval"
        )
        metadata_dir.mkdir(parents=True, exist_ok=True)
        for name in [
            "scene.json",
            "sample.json",
            "sample_data.json",
            "sample_annotation.json",
            "instance.json",
            "calibrated_sensor.json",
            "ego_pose.json",
        ]:
            (metadata_dir / name).write_text("[]", encoding="utf-8")
    for rel_path in [
        "data/infos/griffin_50scenes_25m/vehicle-side/griffin_infos_val.pkl",
        "data/infos/griffin_50scenes_25m/drone-side/griffin_infos_val.pkl",
        "data/infos/griffin_50scenes_25m/cooperative/griffin_infos_val.pkl",
        "projects/configs_griffin_50scenes_25m/vehicle-side/tiny_track_r50_stream_bs8_48epoch_3cls.py",
        "projects/configs_griffin_50scenes_25m/early-fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py",
        "projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py",
        "projects/configs_griffin_50scenes_25m/cooperative/late_fusion/tiny_track_r50_stream_bs1_3cls_late_fusion.py",
        "tools/dist_eval.sh",
        "tools/analysis_tools/compute_BPS.py",
    ]:
        (module.OFFICIAL_ROOT / rel_path).write_bytes(b"0")

    summary = module.audit_25m_assets()

    assert summary["dataset"] == "50scenes_25m"
    assert summary["fixed_height"] == "25 +/- 2 m"
    assert summary["data_packages"]["ready"] is True
    assert summary["checkpoint_packages"]["ready"] is True
    assert summary["split"]["exists"] is True
    assert summary["split"]["counts"] == {"train": 1, "val": 2}
    assert summary["directories"]["griffin_release_vehicle_side"]["exists"] is True
    assert summary["nuscenes_metadata"]["vehicle-side"]["ego_pose.json"]["exists"] is True
    assert summary["nuscenes_metadata"]["drone-side"]["sample_data.json"]["exists"] is True
    assert summary["nuscenes_metadata"]["early-fusion"]["scene.json"]["exists"] is True
    assert summary["nuscenes_metadata"]["cooperative"]["calibrated_sensor.json"]["exists"] is True
    assert summary["configs"]["2b1-cooptrack"]["exists"] is True
    assert summary["evaluator"]["dist_eval.sh"]["exists"] is True


def test_official_source_diff_tracks_config_and_tool_changes(tmp_path):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    repo_root = tmp_path / "repo"
    official_root = repo_root / "griffin_repro" / "official"
    reference_root = tmp_path / "upstream"
    module.REPO_ROOT = repo_root
    module.OFFICIAL_ROOT = official_root

    rel_config = "projects/configs_griffin_50scenes_25m/vehicle-side/base.py"
    rel_tool = "tools/dist_eval.sh"
    rel_plugin = "projects/mmdet3d_plugin/datasets/griffin_dataset.py"
    rel_generated = "projects/work_dirs_griffin_50scenes_25m/result.pkl"
    for root in (official_root, reference_root):
        (root / rel_config).parent.mkdir(parents=True, exist_ok=True)
        (root / rel_tool).parent.mkdir(parents=True, exist_ok=True)
        (root / rel_plugin).parent.mkdir(parents=True, exist_ok=True)
    (reference_root / rel_config).write_text("model='official'\n", encoding="utf-8")
    (official_root / rel_config).write_text("model='patched'\n", encoding="utf-8")
    (reference_root / rel_tool).write_text("bash official\n", encoding="utf-8")
    (official_root / rel_tool).write_text("bash official\n", encoding="utf-8")
    (reference_root / rel_plugin).write_text("dataset = 'official'\n", encoding="utf-8")
    (official_root / rel_plugin).write_text("dataset = 'official'\n", encoding="utf-8")
    (official_root / "tools" / "extra_wrapper.sh").write_text("echo helper\n", encoding="utf-8")
    (official_root / rel_generated).parent.mkdir(parents=True, exist_ok=True)
    (official_root / rel_generated).write_bytes(b"ignored")

    summary = module.official_source_diff(str(reference_root))

    assert summary["reference_root"].endswith("upstream")
    assert summary["modified_count"] == 1
    assert summary["missing_count"] == 0
    assert summary["extra_count"] == 1
    assert summary["ignored_prefixes"] == ["ckpts/", "data/", "datasets/", "projects/work_dirs"]
    by_path = {item["path"]: item for item in summary["differences"]}
    assert by_path[rel_config]["status"] == "modified"
    assert by_path["tools/extra_wrapper.sh"]["status"] == "extra"
    assert rel_generated not in by_path


def test_official_source_diff_ignores_line_endings_and_pycache(tmp_path):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    repo_root = tmp_path / "repo"
    official_root = repo_root / "griffin_repro" / "official"
    reference_root = tmp_path / "upstream"
    module.REPO_ROOT = repo_root
    module.OFFICIAL_ROOT = official_root

    rel_config = "projects/configs_griffin_50scenes_25m/vehicle-side/base.py"
    for root in (official_root, reference_root):
        (root / rel_config).parent.mkdir(parents=True, exist_ok=True)
    (reference_root / rel_config).write_bytes(b"model='official'\r\nvalue=1\r\n")
    (official_root / rel_config).write_bytes(b"model='official'\nvalue=1\n")
    pycache_file = official_root / "projects" / "ab3dmot_plugin" / "__pycache__" / "helper.cpython-312.pyc"
    pycache_file.parent.mkdir(parents=True)
    pycache_file.write_bytes(b"ignored")
    for rel_generated_config in (
        "projects/configs_griffin_50scenes_25m/vehicle-side/codex_score07_eval.py",
        "projects/configs_griffin_50scenes_25m/vehicle-side/tiny_track_r50_stream_bs8_48epoch_3cls_partial_1scene.py",
        "projects/configs_griffin_50scenes_25m/vehicle-side/tiny_track_r50_stream_bs8_48epoch_3cls_stable_query_topk.py",
    ):
        generated_config = official_root / rel_generated_config
        generated_config.parent.mkdir(parents=True, exist_ok=True)
        generated_config.write_text("# generated experiment wrapper\n", encoding="utf-8")

    summary = module.official_source_diff(str(reference_root))

    assert summary["modified_count"] == 0
    assert summary["extra_count"] == 0
    assert summary["differences"] == []


def test_efficiency_audit_computes_25m_bps_and_log_fps(tmp_path):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    late_result = tmp_path / "late.pkl"
    with late_result.open("wb") as handle:
        pickle.dump(
            {
                "bbox_results": [
                    {
                        "token": "sample-a",
                        "labels_3d": [0, 1],
                        "scores_3d": [0.9, 0.8],
                        "boxes_3d": {"center": [[0.0] * 7, [1.0] * 7]},
                    },
                    {
                        "token": "sample-b",
                        "labels_3d": [0],
                        "scores_3d": [0.7],
                        "boxes_3d": {"center": [[2.0] * 7]},
                    },
                ]
            },
            handle,
        )

    query_dir = tmp_path / "track_query"
    query_dir.mkdir()
    with (query_dir / "sample-a.pkl").open("wb") as handle:
        pickle.dump(
            {
                "query_feats": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                "cache_motion_feats": [[1.0, 2.0], [3.0, 4.0]],
                "ref_pts": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            },
            handle,
        )
    with (query_dir / "sample-b.pkl").open("wb") as handle:
        pickle.dump(
            {
                "query_feats": [[1.0, 2.0, 3.0]],
                "cache_motion_feats": [[1.0, 2.0]],
                "ref_pts": [[0.1, 0.2, 0.3]],
            },
            handle,
        )

    log_path = tmp_path / "eval.log"
    log_path.write_text("FPS: 12.5\nEval time: 0.45\n", encoding="utf-8")

    summary = module.efficiency_audit(
        dataset="50scenes_25m",
        late_result_pkl=str(late_result),
        cooptrack_query_dir=str(query_dir),
        logs=[str(log_path)],
    )

    assert summary["dataset"] == "50scenes_25m"
    assert summary["hardware_note"].startswith("Paper FPS was reported")
    assert summary["methods"]["3-late fusion"]["BPS"] == 495.0
    assert summary["methods"]["3-late fusion"]["result_per_frame"] == 1.5
    assert summary["methods"]["2b1-cooptrack"]["BPS"] == 480.0
    assert summary["methods"]["2b1-cooptrack"]["result_per_frame"] == 1.5
    assert summary["logs"][0]["fps"] == 12.5
    assert summary["logs"][0]["eval_time_seconds"] == 0.45


def test_efficiency_audit_pickle_loader_can_resolve_official_project_modules(tmp_path):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    official_root = tmp_path / "official"
    package_dir = official_root / "projects"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "fake_payload.py").write_text(
        "class Payload:\n"
        "    def __init__(self, value):\n"
        "        self.value = value\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(official_root))
    try:
        fake_payload = importlib.import_module("projects.fake_payload")
        payload_path = tmp_path / "payload.pkl"
        with payload_path.open("wb") as handle:
            pickle.dump(fake_payload.Payload(7), handle)
    finally:
        sys.path.remove(str(official_root))
        sys.modules.pop("projects.fake_payload", None)
        sys.modules.pop("projects", None)

    module.OFFICIAL_ROOT = official_root

    payload = module._read_pickle_file(payload_path)

    assert payload.value == 7


def test_efficiency_audit_tolerates_missing_optional_query_fields(tmp_path):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    query_dir = tmp_path / "track_query"
    query_dir.mkdir()
    with (query_dir / "sample-a.pkl").open("wb") as handle:
        pickle.dump(
            _StrictQueryPayload(
                {
                    "query_feats": [[1.0, 2.0, 3.0]],
                    "ref_pts": [[0.1, 0.2, 0.3]],
                }
            ),
            handle,
        )

    summary = module.efficiency_audit(
        dataset="50scenes_25m",
        cooptrack_query_dir=str(query_dir),
    )

    assert summary["methods"]["2b1-cooptrack"]["samples"] == 1
    assert summary["methods"]["2b1-cooptrack"]["BPS"] == 240.0
    assert summary["methods"]["2b1-cooptrack"]["result_per_frame"] == 1.0


def test_efficiency_log_parser_records_completed_progress_throughput(tmp_path):
    spec = importlib.util.spec_from_file_location("griffin_repro_module", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    log_path = tmp_path / "progress.log"
    log_path.write_text(
        "[>>>>>>>>] 1490/1490, 7.8 task/s, elapsed: 191s, ETA: 0s\n"
        "[>>>>>>>>] 1490/1490, 1421.0 task/s, elapsed: 1s, ETA: 0s\n"
        "Eval time: 16.5s\n",
        encoding="utf-8",
    )

    summary = module._parse_efficiency_log(log_path)

    assert summary["fps"] is None
    assert summary["estimated_fps_from_progress"] == 7.8
    assert summary["completed_progress_task_s"] == [7.8, 1421.0]
    assert summary["eval_time_seconds"] == 16.5


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


def test_write_data_script_full_profile_verifies_full_archive_set(tmp_path):
    out_path = tmp_path / "download_full_data.sh"
    result = run_cli(
        "write-data-script",
        "--dataset",
        "50scenes_25m",
        "--package-profile",
        "full",
        "--out",
        str(out_path),
        "--json",
    )
    payload = json.loads(result.stdout)
    script = out_path.read_text(encoding="utf-8")

    assert payload["package_profile"] == "full"
    assert payload["total_size_bytes"] == 167190016122
    assert "vehicle_lidar.zip|214487013" in script
    assert "drone_camera_instance_segmentation.zip|3558624019" in script
    assert "check-data-packages --dataset 50scenes_25m --package-profile full --json" in script
    assert "check-partial-assets --profile smoke_25m_instance" in script


def test_full_validation_finalizer_waits_for_download_and_records_audits():
    script = FINALIZER_SCRIPT.read_text(encoding="utf-8")

    assert "download_50scenes_25m_full" in script
    assert "ps -eo pid=,comm=,args=" in script
    assert "comm == \"curl\"" in script
    assert "griffin_repro/download_50scenes_25m_full_mobaxterm.sh" in script
    assert "pgrep -af 'download_50scenes_25m_full|curl" not in script
    assert "check-data-packages --dataset 50scenes_25m --package-profile full --json" in script
    assert "verify-data-md5 --dataset 50scenes_25m --package-profile full --json" in script
    assert "check-checkpoint-packages --dataset 50scenes_25m --json" in script
    assert "audit-25m-assets --json" in script
    assert "official-source-diff --reference-root" in script
    assert "-m pytest tests/test_griffin_repro.py -q" in script
    assert "official_25m_full_finalize_" in script
    assert "carlaair_active_world" not in script


def test_full_md5_repair_script_redownloads_mismatches_safely():
    script = (REPO_ROOT / "griffin_repro/repair_50scenes_25m_full_md5_mobaxterm.sh").read_text(
        encoding="utf-8"
    )

    assert "verify-data-md5" in script
    assert 'REPAIR_JOBS="${GRIFFIN_REPAIR_JOBS:-3}"' in script
    assert 'REPAIR_PARTS="${GRIFFIN_REPAIR_PARTS:-1}"' in script
    assert "GRIFFIN_REPAIR_JOBS must be a positive integer" in script
    assert "GRIFFIN_REPAIR_PARTS must be a positive integer" in script
    assert "REPAIR_PARTS=1" in script
    assert "wait -n" in script
    assert "status\") == \"mismatch\"" in script
    assert (
        'curl --silent --show-error --fail --retry 5 --retry-all-errors --connect-timeout 30 --speed-limit 1024 '
        '--speed-time 120 -L -o "$tmp"'
    ) in script
    assert '-r "$start-$end" -o "$part_path"' in script
    assert "Range part size mismatch" in script
    assert " -C - " not in script
    assert "$target.corrupt.$STAMP" in script
    assert "MD5 repair did not converge" in script
    assert 'unzip -oq "$ARCHIVE_DIR/$name" -d "$DATA_PARENT"' in script


def test_check_data_packages_reports_local_archive_state():
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
    assert payload["ready"] is all(item["complete"] for item in payload["checks"])
    assert any(item["path"].endswith("drone_camera_back.zip") for item in payload["checks"])
    assert any(item["path"].endswith("md5.txt") for item in payload["checks"])


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


def test_validate_run_can_use_paper_car_class_metrics(tmp_path):
    log_path = tmp_path / "eval.log"
    log_path.write_text(
        "Per-class results:\n"
        "Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE\n"
        "car\t0.607\t0.380\t0.142\t0.333\t3.404\t1.000\n"
        "bicycle\t0.092\t0.681\t0.286\t0.032\t0.913\t1.000\n"
        "pedestrian\t0.000\t0.818\t0.577\t1.088\t1.938\t1.000\n"
        "======\n"
        "Aggregated results:\n"
        "AMOTA\t0.270\n"
        "{'pts_bbox/mAP': 0.2332, 'pts_bbox/amota': 0.2699}\n"
        "Per-class results:\n"
        "\t\tAMOTA\tAMOTP\tRECALL\tMOTAR\tGT\tMOTA\tMOTP\tMT\tML\tFAF\tTP\tFP\tFN\tIDS\tFRAG\tTID\tLGD\n"
        "car     \t0.670\t0.820\t0.711\t0.933\t8320\t0.662\t0.468\t33\t18\t26.7\t5908\t398\t2403\t9\t27\t5.25\t11.21\n"
        "bicycle \t0.140\t1.804\t0.195\t1.000\t783\t0.195\t0.694\t1\t8\t0.0\t153\t0\t630\t0\t3\t0.50\t30.25\n",
        encoding="utf-8",
    )

    result = run_cli(
        "validate-run",
        "--profile",
        "smoke_25m_early",
        "--log",
        str(log_path),
        "--tolerance",
        "0.001",
        "--metric-scope",
        "paper",
        "--json",
    )
    payload = json.loads(result.stdout)

    assert payload["metric_scope"] == "paper"
    assert payload["passed"] is True
    assert payload["metrics"] == {"AP": 0.607, "AMOTA": 0.67}


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


def test_analyze_result_pkl_reports_prediction_class_coverage(tmp_path):
    result_path = tmp_path / "results.pkl"
    payload = {
        "bbox_results": [
            {
                "labels_3d": [0, 0, 1],
                "scores_3d": [0.95, 0.45, 0.2],
                "track_ids": [4, 4, -1],
                "labels_3d_det": [0, 1, 2],
                "scores_3d_det": [0.8, 0.12, 0.04],
            },
            {
                "labels_3d": [2],
                "scores_3d": [0.55],
                "track_ids": [7],
                "labels_3d_det": [1, 2],
                "scores_3d_det": [0.31, 0.51],
            },
        ]
    }
    result_path.write_bytes(pickle.dumps(payload))

    result = run_cli("analyze-result-pkl", "--path", str(result_path), "--json")
    summary = json.loads(result.stdout)

    assert summary["samples"] == 2
    assert summary["prediction_sets"]["tracking"]["total_predictions"] == 4
    assert summary["prediction_sets"]["tracking"]["classes"]["car"]["count"] == 2
    assert summary["prediction_sets"]["tracking"]["classes"]["car"]["frames"] == 1
    assert summary["prediction_sets"]["tracking"]["classes"]["car"]["score_bins"][">=0.4"] == 2
    assert summary["prediction_sets"]["tracking"]["classes"]["pedestrian"]["score_bins"][">=0.5"] == 1
    assert summary["prediction_sets"]["tracking"]["track_ids"]["duplicate_id_frames"] == 1
    assert summary["prediction_sets"]["tracking"]["track_ids"]["negative_ids"] == 1
    assert summary["prediction_sets"]["tracking"]["track_ids"]["unique_id_count"] == 2
    assert summary["prediction_sets"]["tracking"]["track_ids"]["ids_per_frame"]["max"] == 2
    assert summary["prediction_sets"]["detection"]["total_predictions"] == 5
    assert summary["prediction_sets"]["detection"]["classes"]["bicycle"]["frames"] == 2
    assert summary["prediction_sets"]["detection"]["classes"]["pedestrian"]["score_bins"][">=0.5"] == 1


def test_analyze_track_query_cache_summarizes_coverage_and_fields(tmp_path):
    query_dir = tmp_path / "track_query"
    query_dir.mkdir()
    ann_path = tmp_path / "griffin_infos_val.pkl"
    ann_path.write_bytes(
        pickle.dumps(
            {
                "infos": [
                    {"air_sample_token": "air_a"},
                    {"air_sample_token": "air_b"},
                    {"air_sample_token": "air_missing"},
                ]
            }
        )
    )
    (query_dir / "air_a.pkl").write_bytes(
        pickle.dumps(
            {
                "query_feats": [[1.0, 2.0], [3.0, 4.0]],
                "query_embeds": [[0.0, 0.0], [1.0, 1.0]],
                "ref_pts": [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]],
                "obj_idxes": [-1, 7],
                "scores": [0.1, 0.9],
            }
        )
    )
    (query_dir / "air_b.pkl").write_bytes(
        pickle.dumps(
            {
                "query_feats": [[5.0, 6.0]],
                "query_embeds": [[2.0, 2.0]],
                "ref_pts": [[0.1, 0.1, 0.1]],
                "obj_idxes": [8],
                "scores": [0.8],
            }
        )
    )
    (query_dir / "extra.pkl").write_bytes(
        pickle.dumps(
            {
                "query_feats": [],
                "query_embeds": [],
                "ref_pts": [],
                "obj_idxes": [],
                "scores": [],
            }
        )
    )

    result = run_cli(
        "analyze-track-query-cache",
        "--query-dir",
        str(query_dir),
        "--ann-file",
        str(ann_path),
        "--json",
    )
    summary = json.loads(result.stdout)

    assert summary["track_query_files"] == 3
    assert summary["ann_samples"] == 3
    assert summary["expected_coverage"] == 2
    assert summary["missing_expected"] == ["air_missing"]
    assert summary["extra_files"] == ["extra"]
    assert summary["rows"] == {"count": 3, "max": 2, "mean": 1.0, "min": 0, "sum": 3}
    assert summary["valid_obj_idx_ge0"] == {"count": 3, "max": 1, "mean": 0.6667, "min": 0, "sum": 2}
    assert summary["keys"]["query_feats"]["present_files"] == 3
    assert summary["keys"]["query_feats"]["shapes"] == {"[0]": 1, "[1, 2]": 1, "[2, 2]": 1}
    assert summary["active_score_bins"] == {"all": 2, ">=0.05": 2, ">=0.1": 2, ">=0.3": 2, ">=0.35": 2, ">=0.4": 2, ">=0.5": 2}
    assert summary["query_timing"]["first_missing_index"] == 2


def test_audit_cooptrack_gap_combines_eval_result_query_and_config(tmp_path):
    result_path = tmp_path / "results.pkl"
    result_path.write_bytes(
        pickle.dumps(
            {
                "bbox_results": [
                    {
                        "labels_3d": [0, 0],
                        "scores_3d": [0.9, 0.2],
                        "track_ids": [3, 3],
                        "labels_3d_det": [0],
                        "scores_3d_det": [0.8],
                    }
                ]
            }
        )
    )
    query_dir = tmp_path / "track_query"
    query_dir.mkdir()
    (query_dir / "air_a.pkl").write_bytes(
        pickle.dumps(
            {
                "query_feats": [[1.0, 2.0]],
                "query_embeds": [[0.0, 0.0]],
                "ref_pts": [[0.2, 0.3, 0.4]],
                "obj_idxes": [9],
                "scores": [0.6],
            }
        )
    )
    ann_file = tmp_path / "griffin_infos_val.pkl"
    ann_file.write_bytes(pickle.dumps({"infos": [{"air_sample_token": "air_a"}]}))
    eval_dir = tmp_path / "json_output"
    (eval_dir / "det").mkdir(parents=True)
    (eval_dir / "track").mkdir()
    (eval_dir / "det" / "metrics_summary.json").write_text(
        json.dumps({"mean_dist_aps": {"car": 0.42}, "label_aps": {"car": {"1.0": 0.3}}}),
        encoding="utf-8",
    )
    (eval_dir / "track" / "metrics_summary.json").write_text(
        json.dumps({"label_metrics": {"amota": {"car": 0.453}, "tp": {"car": 10.0}, "fp": {"car": 4.0}, "fn": {"car": 5.0}, "ids": {"car": 1.0}}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.py"
    config_path.write_text(
        "score_thresh=0.4\nfilter_score_thresh=0.35\nbbox_coder=dict(type=\"NMSFreeCoder\", max_num=300)\n",
        encoding="utf-8",
    )

    result = run_cli(
        "audit-cooptrack-gap",
        "--result-pkl",
        str(result_path),
        "--query-dir",
        str(query_dir),
        "--ann-file",
        str(ann_file),
        "--eval-dir",
        str(eval_dir),
        "--config",
        str(config_path),
        "--json",
    )
    audit = json.loads(result.stdout)

    assert audit["method"] == "2b1-cooptrack"
    assert audit["summary"]["metrics"] == {"AMOTA": 0.453, "AP": 0.42, "FN": 5.0, "FP": 4.0, "IDS": 1.0, "TP": 10.0}
    assert audit["result_pkl"]["prediction_sets"]["tracking"]["track_ids"]["duplicate_id_frames"] == 1
    assert audit["track_query"]["expected_coverage"] == 1
    assert audit["config_thresholds"]["score_thresh"] == 0.4
    assert audit["config_thresholds"]["filter_score_thresh"] == 0.35
    assert audit["config_thresholds"]["bbox_coder_type"] == "NMSFreeCoder"


def test_summarize_run_log_collects_method_validation_entries(tmp_path):
    log_path = tmp_path / "combined.log"
    vehicle = {
        "profile": "smoke_25m_vehicle",
        "dataset": "50scenes_25m",
        "method": "0-no fusion",
        "metrics": {"AP": 0.18, "AMOTA": 0.15},
        "checks": {
            "AP": {"actual": 0.18, "expected": 0.375, "delta": -0.195, "abs_delta": 0.195, "passed": False},
            "AMOTA": {"actual": 0.15, "expected": 0.365, "delta": -0.215, "abs_delta": 0.215, "passed": False},
        },
        "missing_metrics": [],
        "passed": False,
    }
    early = {
        "profile": "smoke_25m_early",
        "dataset": "50scenes_25m",
        "method": "1-early fusion",
        "metrics": {"AP": 0.23, "AMOTA": 0.27},
        "checks": {
            "AP": {"actual": 0.23, "expected": 0.607, "delta": -0.377, "abs_delta": 0.377, "passed": False},
            "AMOTA": {"actual": 0.27, "expected": 0.67, "delta": -0.4, "abs_delta": 0.4, "passed": False},
        },
        "missing_metrics": [],
        "passed": False,
    }
    log_path.write_text(
        "run vehicle\n"
        f"{json.dumps(vehicle)}\n"
        "run early\n"
        f"validation: {json.dumps(early)}\n"
        "run late\n"
        '{"expected": {"AMOTA": 0.488, "AP": 0.479}, "method": "2b1-cooptrack"}\n'
        "mAP: 0.1336\n"
        "Aggregated results:\n"
        "AMOTA\t0.125\n",
        encoding="utf-8",
    )

    result = run_cli("summarize-run-log", "--log", str(log_path), "--json")
    summary = json.loads(result.stdout)

    assert summary["method_count"] == 3
    assert summary["methods"] == ["0-no fusion", "1-early fusion", "3-late fusion"]
    assert summary["all_passed"] is False
    assert summary["paper_tolerance"] == 0.02
    assert summary["all_within_paper_tolerance"] is False
    assert summary["missing_runnable_methods"] == ["2b1-cooptrack"]
    assert summary["paper_mismatches"][0] == {
        "method": "0-no fusion",
        "metric": "AP",
        "actual": 0.18,
        "expected": 0.375,
        "abs_delta": 0.195,
    }
    assert summary["entries"][0]["checks"]["AP"]["expected"] == 0.375
    assert summary["entries"][1]["checks"]["AMOTA"]["delta"] == -0.4
    assert summary["entries"][2]["checks"]["AP"]["expected"] == 0.378
    assert summary["entries"][2]["checks"]["AMOTA"]["actual"] == 0.125


def test_summarize_run_log_can_reparse_paper_car_class_metrics(tmp_path):
    official_log = tmp_path / "official_early.log"
    official_log.write_text(
        "Per-class results:\n"
        "Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE\n"
        "car\t0.607\t0.380\t0.142\t0.333\t3.404\t1.000\n"
        "======\n"
        "Per-class results:\n"
        "\t\tAMOTA\tAMOTP\tRECALL\tMOTAR\tGT\tMOTA\tMOTP\tMT\tML\tFAF\tTP\tFP\tFN\tIDS\tFRAG\tTID\tLGD\n"
        "car     \t0.670\t0.820\t0.711\t0.933\t8320\t0.662\t0.468\t33\t18\t26.7\t5908\t398\t2403\t9\t27\t5.25\t11.21\n",
        encoding="utf-8",
    )
    combined_log = tmp_path / "combined.log"
    combined_log.write_text(
        json.dumps(
            {
                "profile": "smoke_25m_early",
                "dataset": "50scenes_25m",
                "method": "1-early fusion",
                "log": str(official_log),
                "metric_scope": "aggregate",
                "metrics": {"AP": 0.2332, "AMOTA": 0.27},
                "checks": {
                    "AP": {"actual": 0.2332, "expected": 0.607, "delta": -0.3738, "abs_delta": 0.3738, "passed": False},
                    "AMOTA": {"actual": 0.27, "expected": 0.67, "delta": -0.4, "abs_delta": 0.4, "passed": False},
                },
                "missing_metrics": [],
                "passed": False,
            }
        ),
        encoding="utf-8",
    )

    result = run_cli(
        "summarize-run-log",
        "--log",
        str(combined_log),
        "--paper-tolerance",
        "0.001",
        "--metric-scope",
        "paper",
        "--json",
    )
    summary = json.loads(result.stdout)

    assert summary["metric_scope"] == "paper"
    assert summary["entries"][0]["metric_scope"] == "paper"
    assert summary["entries"][0]["metrics"] == {"AP": 0.607, "AMOTA": 0.67}
    assert summary["paper_mismatches"] == []
    assert summary["all_within_paper_tolerance"] is True


def test_summarize_eval_json_extracts_paper_scope_metrics(tmp_path):
    eval_dir = tmp_path / "json_output"
    det_dir = eval_dir / "det"
    track_dir = eval_dir / "track"
    det_dir.mkdir(parents=True)
    track_dir.mkdir(parents=True)
    (det_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "mean_dist_aps": {"car": 0.3747686248091491},
                "label_aps": {"car": {"0.5": 0.1978, "1.0": 0.3054, "2.0": 0.4263, "4.0": 0.5696}},
            }
        ),
        encoding="utf-8",
    )
    (track_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "label_metrics": {
                    "amota": {"car": 0.3651430631449719},
                    "gt": {"car": 8320.0},
                    "tp": {"car": 3141.0},
                    "fp": {"car": 418.0},
                    "fn": {"car": 5177.0},
                    "ids": {"car": 2.0},
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_cli(
        "summarize-eval-json",
        "--eval-dir",
        str(eval_dir),
        "--dataset",
        "50scenes_25m",
        "--method",
        "0-no fusion",
        "--json",
    )
    summary = json.loads(result.stdout)

    assert summary["metric_scope"] == "paper"
    assert summary["metrics"]["AP"] == 0.3747686248091491
    assert summary["metrics"]["AMOTA"] == 0.3651430631449719
    assert summary["metrics"]["GT"] == 8320.0
    assert summary["checks"]["AP"]["passed"] is True
    assert summary["checks"]["AMOTA"]["passed"] is True
    assert summary["passed"] is True


def test_summarize_eval_json_can_compare_condition_specific_paper_rows(tmp_path):
    eval_dir = tmp_path / "json_output"
    det_dir = eval_dir / "det"
    track_dir = eval_dir / "track"
    det_dir.mkdir(parents=True)
    track_dir.mkdir(parents=True)
    (det_dir / "metrics_summary.json").write_text(
        json.dumps({"mean_dist_aps": {"car": 0.564}, "label_aps": {"car": {"2.0": 0.61}}}),
        encoding="utf-8",
    )
    (track_dir / "metrics_summary.json").write_text(
        json.dumps(
            {
                "label_metrics": {
                    "amota": {"car": 0.64},
                    "gt": {"car": 8265.0},
                    "tp": {"car": 5791.0},
                    "fp": {"car": 588.0},
                    "fn": {"car": 2462.0},
                    "ids": {"car": 12.0},
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_cli(
        "summarize-eval-json",
        "--eval-dir",
        str(eval_dir),
        "--dataset",
        "50scenes_25m",
        "--method",
        "1-early fusion",
        "--condition-id",
        "communication_latency_ms_100",
        "--paper-tolerance",
        "0.001",
        "--json",
    )
    summary = json.loads(result.stdout)

    assert summary["condition_id"] == "communication_latency_ms_100"
    assert summary["checks"]["AP"]["expected"] == 0.564
    assert summary["checks"]["AMOTA"]["expected"] == 0.64
    assert summary["paper_metrics"]["GT"] == 8265.0
    assert summary["passed"] is True


def test_summarize_official_log_compares_direct_paper_class_metrics(tmp_path):
    official_log = tmp_path / "official_late.log"
    official_log.write_text(
        "Per-class results:\n"
        "Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE\n"
        "car\t0.378\t0.330\t0.698\t0.144\t2.945\t1.000\n"
        "======\n"
        "Per-class results:\n"
        "\t\tAMOTA\tAMOTP\tRECALL\tMOTAR\tGT\tMOTA\tMOTP\tMT\tML\tFAF\tTP\tFP\tFN\tIDS\tFRAG\tTID\tLGD\n"
        "car     \t0.377\t1.047\t0.365\t0.739\t8320\t0.269\t0.355\t17\t57\t53.2\t3033\t792\t5282\t5\t12\t9.89\t13.45\n",
        encoding="utf-8",
    )

    result = run_cli(
        "summarize-official-log",
        "--log",
        str(official_log),
        "--dataset",
        "50scenes_25m",
        "--method",
        "3-late fusion",
        "--paper-tolerance",
        "0.001",
        "--json",
    )
    summary = json.loads(result.stdout)

    assert summary["profile"] == "official_log"
    assert summary["metric_scope"] == "paper"
    assert summary["metrics"] == {"AP": 0.378, "AMOTA": 0.377}
    assert summary["checks"]["AP"]["expected"] == 0.378
    assert summary["checks"]["AMOTA"]["expected"] == 0.377
    assert summary["passed"] is True


def test_summarize_run_logs_merges_parallel_method_logs(tmp_path):
    main_log = tmp_path / "main.log"
    instance_log = tmp_path / "instance.log"
    late_log = tmp_path / "late.log"
    vehicle = {
        "profile": "smoke_25m_vehicle",
        "dataset": "50scenes_25m",
        "method": "0-no fusion",
        "metrics": {"AP": 0.1986, "AMOTA": 0.16},
        "checks": {
            "AP": {"actual": 0.1986, "expected": 0.375, "delta": -0.1764, "abs_delta": 0.1764, "passed": False},
            "AMOTA": {"actual": 0.16, "expected": 0.365, "delta": -0.205, "abs_delta": 0.205, "passed": False},
        },
        "missing_metrics": [],
        "passed": False,
    }
    early = {
        "profile": "smoke_25m_early",
        "dataset": "50scenes_25m",
        "method": "1-early fusion",
        "metrics": {"AP": 0.2332, "AMOTA": 0.27},
        "checks": {
            "AP": {"actual": 0.2332, "expected": 0.607, "delta": -0.3738, "abs_delta": 0.3738, "passed": False},
            "AMOTA": {"actual": 0.27, "expected": 0.67, "delta": -0.4, "abs_delta": 0.4, "passed": False},
        },
        "missing_metrics": [],
        "passed": False,
    }
    instance = {
        "profile": "smoke_25m_instance",
        "dataset": "50scenes_25m",
        "method": "2b1-cooptrack",
        "metrics": {"AP": 0.145, "AMOTA": 0.151},
        "checks": {
            "AP": {"actual": 0.145, "expected": 0.479, "delta": -0.334, "abs_delta": 0.334, "passed": False},
            "AMOTA": {"actual": 0.151, "expected": 0.488, "delta": -0.337, "abs_delta": 0.337, "passed": False},
        },
        "missing_metrics": [],
        "passed": False,
    }
    main_log.write_text(
        f"{json.dumps(vehicle)}\n"
        f"{json.dumps(early)}\n",
        encoding="utf-8",
    )
    instance_log.write_text(f"validation: {json.dumps(instance)}\n", encoding="utf-8")
    late_log.write_text(
        "Starting late fusion\n"
        "mAP: 0.1341\n"
        "{'pts_bbox/amota': 0.12645220901640866}\n",
        encoding="utf-8",
    )

    result = run_cli(
        "summarize-run-logs",
        "--log",
        str(main_log),
        "--log",
        str(instance_log),
        "--log",
        str(late_log),
        "--json",
    )
    summary = json.loads(result.stdout)

    assert summary["method_count"] == 4
    assert summary["methods"] == ["0-no fusion", "1-early fusion", "2b1-cooptrack", "3-late fusion"]
    assert summary["missing_runnable_methods"] == []
    assert summary["all_within_paper_tolerance"] is False
    assert summary["entries"][2]["metrics"] == {"AP": 0.145, "AMOTA": 0.151}
    assert summary["entries"][3]["metrics"]["AP"] == 0.1341
    assert summary["entries"][3]["metrics"]["AMOTA"] == 0.12645220901640866
    assert summary["logs"] == [str(main_log), str(instance_log), str(late_log)]


def test_write_mobaxterm_script_emits_asset_gate_and_isolated_eval(tmp_path):
    out_path = tmp_path / "run_smoke.sh"
    run_cli("write-mobaxterm-script", "--profile", "smoke_25m_instance", "--out", str(out_path), "--json")
    script = out_path.read_text(encoding="utf-8")

    assert "cd griffin_repro/official" in script
    assert "bash tools/griffin_converter.sh griffin_50scenes_25m" in script
    assert "preprocess_assets=(" in script
    assert "evaluation_assets=(" in script
    assert script.index("bash tools/griffin_converter.sh griffin_50scenes_25m") < script.index(
        'check_assets "evaluation" "${evaluation_assets[@]}"'
    )
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


def test_write_mobaxterm_script_can_sample_each_scene(tmp_path):
    out_path = tmp_path / "run_smoke.sh"
    run_cli("write-mobaxterm-script", "--profile", "smoke_25m_instance", "--out", str(out_path), "--json")
    script = out_path.read_text(encoding="utf-8")

    assert "GRIFFIN_PARTIAL_SAMPLES_PER_SCENE" in script
    assert '--samples-per-scene "$partial_samples_per_scene"' in script
    assert 'partial_${partial_scene_limit}scene_${partial_samples_per_scene}per_scene' in script


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


def test_write_mobaxterm_script_can_skip_converter_after_asset_check(tmp_path):
    out_path = tmp_path / "run_smoke.sh"
    run_cli("write-mobaxterm-script", "--profile", "smoke_25m_instance", "--out", str(out_path), "--json")
    script = out_path.read_text(encoding="utf-8")

    assert 'skip_converter="${GRIFFIN_SKIP_CONVERTER:-0}"' in script
    assert 'if [ "$skip_converter" = "1" ]; then' in script
    assert "Skipping Griffin converter because GRIFFIN_SKIP_CONVERTER=1" in script
    assert 'check_assets "converted data" "${evaluation_assets[@]}"' in script
    assert "bash tools/griffin_converter.sh griffin_50scenes_25m" in script
    assert script.index('if [ "$skip_converter" = "1" ]; then') < script.index(
        "bash tools/griffin_converter.sh griffin_50scenes_25m"
    )


def test_write_vehicle_mobaxterm_script_uses_profile_partial_eval_file(tmp_path):
    out_path = tmp_path / "run_smoke_vehicle.sh"
    run_cli("write-mobaxterm-script", "--profile", "smoke_25m_vehicle", "--out", str(out_path), "--json")
    script = out_path.read_text(encoding="utf-8")

    assert "prepare-partial-eval --profile smoke_25m_vehicle" in script
    assert "smoke_25m_vehicle_partial_eval.json" in script
    assert "smoke_25m_instance_partial_eval.json" not in script
    assert "else\n\nfi" not in script


def test_late_fusion_mobaxterm_script_reuses_vehicle_and_drone_outputs():
    script_path = REPO_ROOT / "griffin_repro" / "run_smoke_25m_late_mobaxterm.sh"
    script = script_path.read_text(encoding="utf-8")

    assert "prepare-partial-eval --profile smoke_25m_instance" in script
    assert '"$ROOT/$partial_json"' not in script
    assert '"$partial_json" "$partial_tag"' in script
    assert "partial_${partial_scene_limit}scene_${partial_samples_per_scene}per_scene" in script
    assert "tiny_track_r50_stream_bs8_48epoch_3cls_${partial_tag}/results-*.pkl" in script
    assert "tiny_track_r50_stream_bs8_24epoch_3cls_eval_${partial_tag}/results-*.pkl" in script
    assert "tools/eval_late_fusion.sh" in script
    assert "tiny_track_r50_stream_bs1_3cls_late_fusion_${partial_tag}.py" in script
    assert "tiny_track_r50_stream_bs1_3cls_late_fusion_ab3dmot_${partial_tag}.py" in script


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
