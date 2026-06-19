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

The smoke script auto-activates `${HOME}/miniconda3` environment `griffin` when present. It then reports and enforces the Griffin Python package environment, checks raw Griffin-25m data and both drone/cooperative checkpoints, then runs official conversion, drone query extraction, and final cooperative instance-fusion evaluation.

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
