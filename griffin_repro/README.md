# Griffin Reproduction Package

This directory isolates the Griffin paper reproduction from the legacy CarlaAir autonomous-driving scaffold.

## Scope

The immediate goal is an engineering closure, not a full rerun of every paper table. The selected smoke profile is `smoke_25m_instance`, which runs the Griffin-25m cooperative instance-fusion baseline when the dataset and checkpoint are present.

The full paper matrix remains represented in `manifest.json` so it can be expanded later without touching CarlaAir control code.

## Layout

- `official/`: upstream Griffin source ingested from `E:/a2/Griffin-main.zip`.
- `manifest.json`: source provenance, selected profiles, expected paper metrics, and robustness matrix.
- `../scripts/griffin_repro.py`: local verification, result summary, asset checks, and eval command generation.
- `../scripts/sync_griffin_remote.py`: password-capable SFTP sync for this reproduction package only.

Large datasets, checkpoints, work directories, and visualization outputs are intentionally ignored by git.

## Local Checks

Run from the CarlaAir repository root:

```bash
python -m pytest tests/test_griffin_repro.py -q
python scripts/griffin_repro.py verify-layout
python scripts/griffin_repro.py summarize-results
python scripts/griffin_repro.py matrix --profile smoke_25m_instance
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

Then run:

```bash
python scripts/griffin_repro.py check-assets --profile smoke_25m_instance
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
