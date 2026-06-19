# Griffin Reproduction Package

This directory isolates the Griffin paper reproduction from the legacy CarlaAir autonomous-driving scaffold.

## Scope

The immediate goal is to reproduce the paper structure and run real partial closures, not to download and rerun the full 973 GB-scale dataset in one pass. The smoke profiles cover Griffin-25m vehicle-only (`smoke_25m_vehicle`), early-fusion (`smoke_25m_early`), and cooperative instance-fusion (`smoke_25m_instance`) baselines when the selected dataset and checkpoints are present.

The full paper matrix is represented by `manifest.json` and by `python scripts/griffin_repro.py paper-matrix`: four Griffin scene groups, seven fusion methods, the AP/AMOTA/BPS/FPS metric set, and robustness conditions for latency, packet loss, translation error, and rotation error. Use `paper-run-matrix` when you need the same paper rows expanded into concrete official config/checkpoint commands and runnable-status labels.

## Layout

- `official/`: upstream Griffin source ingested from `E:/a2/Griffin-main.zip`.
- `manifest.json`: source provenance, selected profiles, expected paper metrics, and robustness matrix.
- `setup_griffin_env_mobaxterm.sh`: Linux environment bootstrap for the paper dependency stack on the remote host.
- `build_mmdet3d_spconv_ext_mobaxterm.sh`: builds the missing MMDetection3D `sparse_conv_ext` extension when the editable v0.17.1 install lacks it.
- `download_50scenes_25m_mobaxterm.sh`: resumable Griffin-25m raw data download, checksum, and extraction script.
- `download_50scenes_25m_vehicle_mobaxterm.sh`: smaller vehicle-side metadata/camera download script for the no-fusion baseline.
- `run_smoke_25m_instance_mobaxterm.sh`: ready-to-run Linux shell script for the partial smoke closure from MobaXterm.
- `run_smoke_25m_vehicle_mobaxterm.sh`: vehicle-side partial smoke closure, defaulting to one scene and 20 samples.
- `run_smoke_25m_early_mobaxterm.sh`: early-fusion partial smoke closure script.
- `run_smoke_25m_late_mobaxterm.sh`: late-fusion partial closure that reuses the vehicle/drone pkl outputs and runs AB3DMOT tracking.
- `supervise_smoke_25m_instance_mobaxterm.sh`: end-to-end MobaXterm supervisor that retries data download until the smoke package set is complete, then launches the smoke evaluation.
- `../scripts/griffin_repro.py`: local verification, result summary, asset checks, and eval command generation.
- `../scripts/sync_griffin_remote.py`: password-capable SFTP sync for this reproduction package only.

Large datasets, checkpoints, work directories, and visualization outputs are intentionally ignored by git.

## Local Checks

Run from the CarlaAir repository root:

```bash
python -m pytest tests/test_griffin_repro.py -q
python scripts/griffin_repro.py verify-layout
python scripts/griffin_repro.py summarize-results
python scripts/griffin_repro.py paper-matrix
python scripts/griffin_repro.py paper-run-matrix --dataset 50scenes_25m --include-robustness
python scripts/griffin_repro.py data-packages --dataset 50scenes_25m
python scripts/griffin_repro.py data-packages --dataset 50scenes_25m --package-profile smoke_25m_instance
python scripts/griffin_repro.py write-data-script --dataset 50scenes_25m --package-profile smoke_25m_instance --out griffin_repro/download_50scenes_25m_mobaxterm.sh
python scripts/griffin_repro.py env-check
python scripts/griffin_repro.py write-env-script --out griffin_repro/setup_griffin_env_mobaxterm.sh
python scripts/griffin_repro.py list-profiles
python scripts/griffin_repro.py matrix --profile smoke_25m_vehicle
python scripts/griffin_repro.py matrix --profile smoke_25m_instance
python scripts/griffin_repro.py plan-partial-run --profile smoke_25m_vehicle
python scripts/griffin_repro.py plan-partial-run --profile smoke_25m_instance
python scripts/griffin_repro.py check-partial-assets --profile smoke_25m_vehicle
python scripts/griffin_repro.py check-partial-assets --profile smoke_25m_instance
```

## Real Smoke Evaluation

Prepare the official Griffin environment on Linux. On the remote host used for this reproduction, Conda/Mamba is not preinstalled, so use the project bootstrap script:

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
bash griffin_repro/setup_griffin_env_mobaxterm.sh
```

The script installs Miniconda under `${HOME}/miniconda3` when Conda is missing, creates/updates the `griffin` environment, sets `CUDA_HOME=/usr/local/cuda`, installs PyTorch `1.9.1+cu111`, MMCV `1.4.0`, MMDetection `2.14.0`, MMSegmentation `0.14.1`, MMDetection3D `v0.17.1`, then runs `env-check --strict`. Logs are written under `griffin_repro/artifacts/logs/`.

The equivalent manual environment steps are:

```bash
conda create -n griffin python=3.8 -y
conda activate griffin
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0 mmsegmentation==0.14.1
git clone https://github.com/open-mmlab/mmdetection3d.git /tmp/mmdetection3d-v0.17.1
cd /tmp/mmdetection3d-v0.17.1 && git checkout v0.17.1 && pip install -v -e .
cd -
pip install -r griffin_repro/official/requirements.txt
```

Place the Griffin-25m dataset and checkpoint under:

```text
griffin_repro/official/datasets/griffin_50scenes_25m/
griffin_repro/official/ckpts/griffin_50scenes_25m/drone-side/iter_33024.pth
griffin_repro/official/ckpts/griffin_50scenes_25m/cooperative/instance_fusion/iter_33024.pth
```

The `smoke_25m_instance` profile checks these concrete runtime assets:

```text
griffin_repro/official/projects/configs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls.py
griffin_repro/official/ckpts/griffin_50scenes_25m/cooperative/instance_fusion/iter_33024.pth
griffin_repro/official/datasets/griffin_50scenes_25m/griffin-nuscenes/cooperative
griffin_repro/official/data/infos/griffin_50scenes_25m/cooperative/griffin_infos_val.pkl
griffin_repro/official/data/infos/griffin_50scenes_25m/drone-side/track_query
```

If the raw KITTI-style Griffin-25m data is present but the NuScenes data or info files are missing, run the official conversion flow first:

```bash
cd griffin_repro/official
bash tools/griffin_converter.sh griffin_50scenes_25m
CUDA_VISIBLE_DEVICES=0 ./tools/dist_eval.sh \
  projects/configs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval.py \
  ckpts/griffin_50scenes_25m/drone-side/iter_33024.pth \
  1
cd -
```

The drone-side eval step is needed to populate the `data/infos/griffin_50scenes_25m/drone-side/track_query` features consumed by instance fusion.

The Griffin-25m raw data is packaged as 15 Hugging Face archives totaling `167190016122` bytes. The smoke reproduction only needs the 12 metadata/camera archives used by the official drone query extraction and cooperative instance-fusion evaluation, totaling `162300524941` bytes. Inspect either package list before downloading:

```bash
python scripts/griffin_repro.py data-packages --dataset 50scenes_25m
python scripts/griffin_repro.py data-packages --dataset 50scenes_25m --package-profile smoke_25m_instance
```

On the remote host, Hugging Face mainline timed out while `hf-mirror.com` returned the official `md5.txt`. Use the generated resumable download script:

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
bash griffin_repro/download_50scenes_25m_mobaxterm.sh
```

The script downloads the selected Griffin-25m archives to `griffin_repro/official/datasets/griffin_50scenes_25m/archives/`, verifies a selected-entry MD5 file with `md5sum -c md5.selected.txt`, extracts zip files into the official dataset directory, and ends with `check-partial-assets --profile smoke_25m_instance`. It defaults to three concurrent resumable downloads and up to 12 resume passes; override `GRIFFIN_DOWNLOAD_JOBS` or `GRIFFIN_DOWNLOAD_MAX_PASSES` to tune bandwidth and retry depth. If an interrupted transfer leaves an archive larger than the official expected size, the script deletes that corrupt partial file before retrying. To use Hugging Face mainline instead of the mirror, override `GRIFFIN_DATA_BASE_URL`. To generate the full 15-archive downloader instead, pass `--package-profile full` to `write-data-script`.

From MobaXterm on the remote host, run the staged smoke script:

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
bash griffin_repro/run_smoke_25m_instance_mobaxterm.sh
```

The smoke script auto-activates `${HOME}/miniconda3` environment `griffin` when present. It then reports and enforces the Griffin Python package environment, checks raw Griffin-25m data and both drone/cooperative checkpoints, then runs official conversion, partial image materialization, drone query extraction, and final cooperative instance-fusion evaluation. With the default partial setting (`GRIFFIN_PARTIAL_SCENE_LIMIT=1`, `GRIFFIN_PARTIAL_MAX_SAMPLES=20`), it avoids the full drone val split by generating a drone-side partial eval config that matches the cooperative samples through `air_sample_token`.

For unattended real runs, prefer the supervisor:

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
bash griffin_repro/supervise_smoke_25m_instance_mobaxterm.sh
```

The supervisor repeatedly runs `check-data-packages`, retries `download_50scenes_25m_mobaxterm.sh` after transient failures, and only calls `run_smoke_25m_instance_mobaxterm.sh` after all selected smoke archives match their expected sizes. Use `GRIFFIN_SUPERVISOR_SLEEP_SEC` to tune retry delay and `GRIFFIN_SUPERVISOR_MAX_ATTEMPTS` to bound retries; the default is unlimited attempts.

After evaluation, the script finds the latest official Griffin eval log and runs:

```bash
python3 scripts/griffin_repro.py validate-run --profile smoke_25m_instance --log <latest-log>
```

That validator parses AP and AMOTA from the log and compares them with the paper reference using a default tolerance of `0.02`.

You can also inspect and run the same pieces through the Python helper:

```bash
python scripts/griffin_repro.py check-partial-assets --profile smoke_25m_instance
python scripts/griffin_repro.py materialize-partial-images --profile smoke_25m_instance --image-side drone-side --scene-limit 1 --max-samples 20 --dry-run
python scripts/griffin_repro.py prepare-drone-query-partial-eval --profile smoke_25m_instance --scene-limit 1 --max-samples 20
python scripts/griffin_repro.py validate-run --profile smoke_25m_instance --log griffin_repro/official/<path-to-eval-log>
python scripts/griffin_repro.py run-profile --profile smoke_25m_instance
```

Expected paper reference for this smoke profile is AP `0.479` and AMOTA `0.488`.

### Verified Vehicle-Side Partial Run

The current remote closure was run on `10.2.14.120` under `/home/fp/CARLA/CarlaAir-v0.1.7/code` with the synced `codex/griffin-repro-closure` branch. The default vehicle smoke script now runs one scene and 20 samples unless `GRIFFIN_PARTIAL_SCENE_LIMIT=0` is set for a full validation split run.

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
bash griffin_repro/run_smoke_25m_vehicle_mobaxterm.sh
```

The run used the official vehicle-side config generated by `prepare-partial-eval`:

```text
projects/configs_griffin_50scenes_25m/vehicle-side/tiny_track_r50_stream_bs8_48epoch_3cls_partial_1scene_20samples.py
```

Remote evidence from the default-script run on 2026-06-19:

```text
profile: smoke_25m_vehicle
method: 0-no fusion
subset: scene_0, 20 samples
log: griffin_repro/official/projects/work_dirs_griffin_50scenes_25m/vehicle-side/tiny_track_r50_stream_bs8_48epoch_3cls_partial_1scene_20samples/logs/test_06191602_iter_33024_tiny_track_r50_stream_bs8_48epoch_3cls_partial_1scene_20samples.log
AP: 0.1625
AMOTA: 0.138
missing_metrics: []
validation: passed with partial tolerance 1.0
```

This partial run is not expected to match the full paper AP/AMOTA values exactly because the validator intentionally restricts the official val split to 20 samples for closure speed. It verifies the end-to-end path: official conversion, vehicle-side partial config generation, checkpoint loading, model inference, detection metric export, tracking metric export, AP/AMOTA parsing, and scripted validation.

### Verified CoopTrack Partial Run

The cooperative instance-fusion closure uses the paper's CoopTrack profile (`2b1-cooptrack`) with official vehicle/drone metadata, official conversion, drone-side query extraction, and the cooperative instance-fusion checkpoint. The default script now materializes only the selected frames needed by the one-scene partial subset before running the drone query extraction and cooperative eval.

Remote evidence from the partial CoopTrack run on 2026-06-19:

```text
profile: smoke_25m_instance
method: 2b1-cooptrack
subset: scene_0, 20 samples
versioned script log: griffin_repro/artifacts/logs/smoke_25m_instance_versioned_20260619_164927.log
drone-query log: griffin_repro/official/projects/work_dirs_griffin_50scenes_25m/drone-side/tiny_track_r50_stream_bs8_24epoch_3cls_eval_partial_1scene_20samples/logs/test_06191701_iter_33024_tiny_track_r50_stream_bs8_24epoch_3cls_eval_partial_1scene_20samples.log
cooperative log: griffin_repro/official/projects/work_dirs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls_partial_1scene_20samples/logs/test_06191702_iter_33024_tiny_track_r50_stream_bs8_48epoch_3cls_partial_1scene_20samples.log
drone-side track_query files: 20
AP: 0.1126
AMOTA: 0.116
missing_metrics: []
validation: passed with partial tolerance 1.0
```

As with the vehicle-side smoke run, this is a partial closure rather than the full paper validation split. It verifies the paper-aligned scenario/method path for CoopTrack: cooperative sample selection, vehicle/drone image availability, drone query generation, instance-fusion config generation, checkpoint loading, AP parsing, AMOTA parsing, and validator comparison against the paper reference profile.

### Expanded 10-Scene Partial Verification

After the one-scene closures, the remote validation subset was expanded to all 10 scenes present in the official Griffin-25m val annotation files under `data/infos/griffin_50scenes_25m/*/griffin_infos_val.pkl`. Each val scene exposes 149 frames in this release, so increasing `GRIFFIN_PARTIAL_SCENE_LIMIT` above `10` does not add more scenes; use `GRIFFIN_PARTIAL_SAMPLES_PER_SCENE` to increase the sampled frames per scene.

The following 100-frame runs were verified on `10.2.14.120` under `/home/fp/CARLA/CarlaAir-v0.1.7/code`:

```text
subset: 10 scenes, 10 samples per scene, 100 samples total

method            paper AP   paper AMOTA   partial AP   partial AMOTA   script log
0-no fusion       0.375      0.365         0.2265       0.205           griffin_repro/artifacts/logs/smoke_25m_vehicle_10scene_10per_scene_20260619_225455.log
1-early fusion    0.607      0.670         0.2789       0.308           griffin_repro/artifacts/logs/smoke_25m_early_10scene_10per_scene_20260619_225928.log
2b1-cooptrack     0.479      0.488         0.1654       0.184           griffin_repro/artifacts/logs/smoke_25m_instance_10scene_10per_scene_hardened_20260619_215318.log
```

The corresponding official eval logs are:

```text
vehicle-side: projects/work_dirs_griffin_50scenes_25m/vehicle-side/tiny_track_r50_stream_bs8_48epoch_3cls_partial_10scene_10per_scene/logs/test_06192257_iter_33024_tiny_track_r50_stream_bs8_48epoch_3cls_partial_10scene_10per_scene.log
early-fusion: projects/work_dirs_griffin_50scenes_25m/early-fusion/tiny_track_r50_stream_bs8_48epoch_3cls_partial_10scene_10per_scene/logs/test_06192313_iter_33024_tiny_track_r50_stream_bs8_48epoch_3cls_partial_10scene_10per_scene.log
cooptrack: projects/work_dirs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls_partial_10scene_10per_scene/logs/test_06192245_iter_33024_tiny_track_r50_stream_bs8_48epoch_3cls_partial_10scene_10per_scene.log
```

These values are real official-evaluator outputs, but they are still partial-subset results. They prove the paper-aligned scenarios, metric path, and three runnable fusion baselines can be executed, while the AP/AMOTA values remain below the full paper references because only 100 of 1490 val frames were evaluated.

The next expansion doubled the sampled frames to 20 per scene and added the official late-fusion pipeline. Late-fusion uses the already generated vehicle-side and drone-side pkl outputs, runs `tools/eval_late_fusion.sh`, then runs AB3DMOT tracking through `tools/eval_track_ab3dmot.sh`. The detection stage and tracking stage were both verified on the same remote host:

```text
subset: 10 scenes, 20 samples per scene, 200 samples total

method            paper AP   paper AMOTA   partial AP        partial AMOTA   evidence log
0-no fusion       0.375      0.365         0.2042            0.184           griffin_repro/artifacts/logs/smoke_25m_vehicle_10scene_20per_scene_20260619_232124.log
1-early fusion    0.607      0.670         0.2523            0.290           griffin_repro/artifacts/logs/smoke_25m_early_10scene_20per_scene_20260620_000839.log
2b1-cooptrack     0.479      0.488         0.1384            0.153           griffin_repro/artifacts/logs/smoke_25m_instance_10scene_20per_scene_20260620_012758.log
3-late fusion     0.378      0.377         0.1344 det        0.112           griffin_repro/artifacts/logs/smoke_25m_late_tracking_only_10scene_20per_scene_20260620_022545.log
```

For late-fusion, the tracking-only evidence log reuses the `results.pkl` generated by the preceding late-fusion detection run at `griffin_repro/artifacts/logs/smoke_25m_late_10scene_20per_scene_20260620_022003.log`. That run produced detection mAP `0.1344`; the successful AB3DMOT conversion and tracking evaluation produced tracking AMOTA `0.112` and tracking-result mAP `0.1098`.

The 200-frame results are still below the paper's full-validation numbers. They are useful as a real closure check for method wiring, data conversion, metrics, and relative trends, not as a claim that the paper table has been fully reproduced.

Paper-fit assessment for the 200-frame subset:

- Matches the paper only at the coarse method-ranking level that early fusion is the strongest runnable baseline in this subset.
- Does not yet match the full paper's CoopTrack behavior: the paper reports CoopTrack above no-fusion and late-fusion on Griffin-25m, while this 200-frame subset has CoopTrack below no-fusion.
- Does not yet match paper-level absolute AP/AMOTA. The subset is 200 frames out of the 1490-frame validation split, with weak bicycle and pedestrian coverage in the sampled frames.
- `validate-run` `passed=true` in these logs means the official evaluator ran and AP/AMOTA were parsed inside the configured tolerance. It is not a claim that the partial result equals the paper table.

An attempted 10-scene, 30-samples-per-scene expansion on 2026-06-20 was intentionally stopped during data materialization before evaluation. The local vehicle-side archives on the remote host are incomplete for several camera zips, so the materializer fell back to HTTP Range extraction from the Hugging Face mirror. At stop time, the 10x30 vehicle-side plan still missed 224 image files, 10x22 still missed 40, and 10x21 still missed 20. Treat that run only as a data-prefetch attempt, not as an experimental result.

To rerun the same 100-frame checks from MobaXterm:

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
export GRIFFIN_PARTIAL_SCENE_LIMIT=10
export GRIFFIN_PARTIAL_SAMPLES_PER_SCENE=10

bash griffin_repro/run_smoke_25m_vehicle_mobaxterm.sh 2>&1 | tee griffin_repro/artifacts/logs/manual_vehicle_10scene_10per_scene_$(date +%Y%m%d_%H%M%S).log
bash griffin_repro/run_smoke_25m_early_mobaxterm.sh 2>&1 | tee griffin_repro/artifacts/logs/manual_early_10scene_10per_scene_$(date +%Y%m%d_%H%M%S).log
bash griffin_repro/run_smoke_25m_instance_mobaxterm.sh 2>&1 | tee griffin_repro/artifacts/logs/manual_instance_10scene_10per_scene_$(date +%Y%m%d_%H%M%S).log
```

To rerun the 200-frame checks, set `GRIFFIN_PARTIAL_SAMPLES_PER_SCENE=20` and use the same three smoke scripts. For late-fusion, reuse the latest 200-frame vehicle and drone pkl outputs by running the dedicated late-fusion smoke script:

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
export GRIFFIN_PARTIAL_SCENE_LIMIT=10
export GRIFFIN_PARTIAL_SAMPLES_PER_SCENE=20
bash griffin_repro/run_smoke_25m_late_mobaxterm.sh 2>&1 | tee griffin_repro/artifacts/logs/manual_late_10scene_20per_scene_$(date +%Y%m%d_%H%M%S).log
```

When increasing `GRIFFIN_PARTIAL_SAMPLES_PER_SCENE`, the materialization step now prints per-image progress to stderr while keeping stdout JSON-compatible. If the terminal is still printing `Materializing ... images: N/M`, the job is still filling image files, not yet running model evaluation.

## Remote Sync

Set the password in the shell instead of committing it:

```powershell
$env:GRIFFIN_REMOTE_PASSWORD = "<password>"
python scripts/sync_griffin_remote.py --host 10.2.14.120 --user fp --remote-dir /home/fp/CARLA/CarlaAir-v0.1.7/code
```

The sync script uploads only `griffin_repro`, `scripts/griffin_repro.py`, `scripts/sync_griffin_remote.py`, selected tests, and `.gitignore`. It does not upload or delete legacy CarlaAir driving modules.

## Matrix Notes

`paper-matrix` confirms the parsed official result table contains 142 result rows and all 28 zero-noise dataset/method baselines. It also records the paper scene counts used by the official docs: 47 Griffin-25m scenes, 54 Griffin-40m scenes, 50 Griffin-55m scenes, and 104 random-altitude scenes.

`paper-run-matrix` converts the same rows into executable status records. It maps vehicle-side, early-fusion, CoopTrack instance-fusion, and late-fusion entries to the official config paths, expected checkpoint paths, and MobaXterm-ready commands. With `--include-robustness`, Griffin-25m latency, packet-loss, translation-error, and rotation-error configs are included when they exist in the upstream zip. Rows without runnable configs in this release are retained as `paper_result_only`, so the paper table is still represented without pretending those methods can be launched from missing assets.

Some paper methods are present in `docs/detailed_results.csv` even when this zip does not include runnable config files. In particular, `Where2Comm` is retained in the paper matrix, while runnable status must be checked from the actual config paths exposed by `list-profiles` or direct file checks.
