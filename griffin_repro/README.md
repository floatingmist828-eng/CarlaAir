# Griffin Reproduction Package

This directory isolates the Griffin paper reproduction from the legacy CarlaAir autonomous-driving scaffold.

## Scope

The immediate goal is to reproduce the paper structure and run a real partial closure, not to download and rerun the full 973 GB-scale dataset in one pass. The selected smoke profile is `smoke_25m_instance`, which runs the Griffin-25m cooperative instance-fusion baseline when the selected dataset and checkpoints are present.

The full paper matrix is represented by `manifest.json` and by `python scripts/griffin_repro.py paper-matrix`: four Griffin scene groups, seven fusion methods, the AP/AMOTA/BPS/FPS metric set, and robustness conditions for latency, packet loss, translation error, and rotation error. Use `paper-run-matrix` when you need the same paper rows expanded into concrete official config/checkpoint commands and runnable-status labels.

## Layout

- `official/`: upstream Griffin source ingested from `E:/a2/Griffin-main.zip`.
- `manifest.json`: source provenance, selected profiles, expected paper metrics, and robustness matrix.
- `setup_griffin_env_mobaxterm.sh`: Linux environment bootstrap for the paper dependency stack on the remote host.
- `download_50scenes_25m_mobaxterm.sh`: resumable Griffin-25m raw data download, checksum, and extraction script.
- `run_smoke_25m_instance_mobaxterm.sh`: ready-to-run Linux shell script for the partial smoke closure from MobaXterm.
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
python scripts/griffin_repro.py matrix --profile smoke_25m_instance
python scripts/griffin_repro.py plan-partial-run --profile smoke_25m_instance
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
