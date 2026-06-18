# Griffin Reproduction Package

This directory isolates the Griffin paper reproduction from the legacy CarlaAir autonomous-driving scaffold.

## Scope

The immediate goal is to reproduce the paper structure and run a real partial closure, not to download and rerun the full 973 GB-scale dataset in one pass. The selected smoke profile is `smoke_25m_instance`, which runs the Griffin-25m cooperative instance-fusion baseline when the selected dataset and checkpoints are present.

The full paper matrix is represented by `manifest.json` and by `python scripts/griffin_repro.py paper-matrix`: four Griffin scene groups, seven fusion methods, the AP/AMOTA/BPS/FPS metric set, and robustness conditions for latency, packet loss, translation error, and rotation error.

## Layout

- `official/`: upstream Griffin source ingested from `E:/a2/Griffin-main.zip`.
- `manifest.json`: source provenance, selected profiles, expected paper metrics, and robustness matrix.
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
python scripts/griffin_repro.py list-profiles
python scripts/griffin_repro.py matrix --profile smoke_25m_instance
python scripts/griffin_repro.py plan-partial-run --profile smoke_25m_instance
python scripts/griffin_repro.py check-partial-assets --profile smoke_25m_instance
```

## Real Smoke Evaluation

Prepare the official Griffin environment on Linux:

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

From MobaXterm on the remote host, run the staged smoke script:

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
bash griffin_repro/run_smoke_25m_instance_mobaxterm.sh
```

The script first checks raw Griffin-25m data and both drone/cooperative checkpoints, then runs official conversion, drone query extraction, and final cooperative instance-fusion evaluation.

You can also inspect and run the same pieces through the Python helper:

```bash
python scripts/griffin_repro.py check-partial-assets --profile smoke_25m_instance
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

Some paper methods are present in `docs/detailed_results.csv` even when this zip does not include runnable config files. In particular, `Where2Comm` is retained in the paper matrix, while runnable status must be checked from the actual config paths exposed by `list-profiles` or direct file checks.
