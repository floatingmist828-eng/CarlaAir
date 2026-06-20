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
python scripts/griffin_repro.py summarize-official-log --log griffin_repro/artifacts/logs/<official-log>.log --dataset 50scenes_25m --method "3-late fusion" --json
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

The next completed expansions increased the same selected scenes from 21 to 40 samples per scene. These runs keep the paper-aligned Griffin-25m scenario and the same four runnable baselines, then compare the official AP/AMOTA outputs against the paper's full-validation references:

```text
subset: 10 scenes, 21 samples per scene, 210 samples total

method            paper AP   paper AMOTA   partial AP        partial AMOTA   evidence log
0-no fusion       0.375      0.365         0.2030            0.184           griffin_repro/artifacts/logs/expanded_10scene_21per_scene_all_20260620_automated.log
1-early fusion    0.607      0.670         0.2513            0.289           griffin_repro/artifacts/logs/expanded_10scene_21per_scene_all_20260620_automated.log
2b1-cooptrack     0.479      0.488         0.1375            0.153           griffin_repro/artifacts/logs/expanded_10scene_21per_scene_all_20260620_automated.log
3-late fusion     0.378      0.377         0.1346 det        0.115           griffin_repro/artifacts/logs/expanded_10scene_21per_scene_all_20260620_automated.log
```

```text
subset: 10 scenes, 22 samples per scene, 220 samples total

method            paper AP   paper AMOTA   partial AP        partial AMOTA   evidence log
0-no fusion       0.375      0.365         0.2039            0.187           griffin_repro/artifacts/logs/expanded_10scene_22per_scene_all_20260620_automated.log
1-early fusion    0.607      0.670         0.2506            0.289           griffin_repro/artifacts/logs/expanded_10scene_22per_scene_all_20260620_automated.log
2b1-cooptrack     0.479      0.488         0.1364            0.151           griffin_repro/artifacts/logs/expanded_10scene_22per_scene_all_20260620_automated.log
3-late fusion     0.378      0.377         0.1349 det        0.116           griffin_repro/artifacts/logs/expanded_10scene_22per_scene_all_20260620_automated.log
```

```text
subset: 10 scenes, 25 samples per scene, 250 samples total

method            paper AP   paper AMOTA   partial AP        partial AMOTA   evidence log
0-no fusion       0.375      0.365         0.2022            0.185           griffin_repro/artifacts/logs/expanded_10scene_25per_scene_all_20260620_automated.log
1-early fusion    0.607      0.670         0.2565            0.290           griffin_repro/artifacts/logs/expanded_10scene_25per_scene_all_20260620_automated.log
2b1-cooptrack     0.479      0.488         0.1365            0.150           griffin_repro/artifacts/logs/expanded_10scene_25per_scene_all_20260620_automated.log
3-late fusion     0.378      0.377         0.1367 det        0.118           griffin_repro/artifacts/logs/expanded_10scene_25per_scene_all_20260620_automated.log
```

```text
subset: 10 scenes, 30 samples per scene, 300 samples total

method            paper AP   paper AMOTA   partial AP        partial AMOTA   evidence log
0-no fusion       0.375      0.365         0.1998            0.185           griffin_repro/artifacts/logs/expanded_10scene_30per_scene_all_20260620_automated.log
1-early fusion    0.607      0.670         0.2518            0.285           griffin_repro/artifacts/logs/expanded_10scene_30per_scene_all_20260620_automated.log
2b1-cooptrack     0.479      0.488         0.1370            0.150           griffin_repro/artifacts/logs/expanded_10scene_30per_scene_all_20260620_automated.log
3-late fusion     0.378      0.377         0.1381 det        0.120           griffin_repro/artifacts/logs/expanded_10scene_30per_scene_all_20260620_automated.log
```

```text
subset: 10 scenes, 40 samples per scene, 400 samples total

method            paper AP   paper AMOTA   partial AP        partial AMOTA   evidence log
0-no fusion       0.375      0.365         0.2042            0.192           griffin_repro/artifacts/logs/expanded_10scene_40per_scene_all_20260620_061121_automated.log
1-early fusion    0.607      0.670         0.2436            0.280           griffin_repro/artifacts/logs/expanded_10scene_40per_scene_all_20260620_061121_automated.log
2b1-cooptrack     0.479      0.488         0.1357            0.147           griffin_repro/artifacts/logs/expanded_10scene_40per_scene_all_20260620_061121_automated.log
3-late fusion     0.378      0.377         0.1404 det        0.119           griffin_repro/artifacts/logs/expanded_10scene_40per_scene_all_20260620_061121_automated.log
```

```text
subset: 10 scenes, 60 samples per scene, 600 samples total

method            paper AP   paper AMOTA   partial AP        partial AMOTA   evidence log
0-no fusion       0.375      0.365         0.1932            0.167           griffin_repro/artifacts/logs/expanded_10scene_60per_scene_all_20260620_082047_automated.log
1-early fusion    0.607      0.670         0.2375            0.270           griffin_repro/artifacts/logs/expanded_10scene_60per_scene_all_20260620_082047_automated.log
2b1-cooptrack     0.479      0.488         0.1362            0.142           griffin_repro/artifacts/logs/expanded_10scene_60per_scene_all_20260620_082047_automated.log
3-late fusion     0.378      0.377         0.1316 det        0.119           griffin_repro/artifacts/logs/expanded_10scene_60per_scene_all_20260620_082047_automated.log
```

```text
subset: 10 scenes, 80 samples per scene, 800 samples total

method            paper AP   paper AMOTA   partial AP        partial AMOTA   evidence log
0-no fusion       0.375      0.365         0.1895            0.152           griffin_repro/artifacts/logs/expanded_10scene_80per_scene_all_20260620_101439_automated.log
1-early fusion    0.607      0.670         0.2321            0.276           griffin_repro/artifacts/logs/expanded_10scene_80per_scene_all_20260620_101439_automated.log
2b1-cooptrack     0.479      0.488         0.1359            0.142           griffin_repro/artifacts/logs/expanded_10scene_80per_scene_all_20260620_101439_automated.log
3-late fusion     0.378      0.377         0.1303 det        0.121           griffin_repro/artifacts/logs/expanded_10scene_80per_scene_all_20260620_101439_automated.log
```

```text
subset: 10 scenes, 100 samples per scene, 1000 samples total

method            paper AP   paper AMOTA   partial AP        partial AMOTA   evidence log
0-no fusion       0.375      0.365         0.1883            0.151           griffin_repro/artifacts/logs/expanded_10scene_100per_scene_all_20260620_121812_automated.log
1-early fusion    0.607      0.670         0.2314            0.268           griffin_repro/artifacts/logs/expanded_10scene_100per_scene_all_20260620_121812_automated.log
2b1-cooptrack     0.479      0.488         0.1376            0.141           griffin_repro/artifacts/logs/expanded_10scene_100per_scene_all_20260620_121812_automated.log
3-late fusion     0.378      0.377         0.1336 det        0.125           griffin_repro/artifacts/logs/expanded_10scene_100per_scene_all_20260620_121812_automated.log
```

For late-fusion in the 1000-frame run, the detection stage produced mAP `0.1336`; the AB3DMOT tracking stage produced tracking AMOTA `0.125` and tracking-result mAP `0.1211`.

The 250-frame run first filled 64 missing vehicle-side images and 150 missing drone-side images, then completed all four methods. The later 300-frame, 400-frame, 600-frame, 800-frame, and 1000-frame runs completed the same four-method sequence and stayed in the same metric range. The 600-frame run used `GRIFFIN_MATERIALIZE_JOBS=8` and completed the full four-method validation sequence in about 92 minutes, including 800 newly materialized vehicle-side images and 1000 newly materialized drone-side images. The 800-frame run used the same materialization setting and completed on `10.2.14.120` from 2026-06-20 10:14:39 CST to 12:03:20 CST. The 1000-frame run completed on `10.2.14.120` from 2026-06-20 12:18:12 CST to 14:24:45 CST. Early fusion remains the strongest runnable baseline in these subsets, which agrees with the paper's broad ordering. CoopTrack remains below no-fusion in these subsets, which still disagrees with the paper's Griffin-25m result and shows the current subset is not representative enough for a paper-level effect claim.

The added subset coverage diagnostic records the exact annotation coverage used by the partial runs. For the 600-frame subset, cooperative annotations contain 7165 car, 851 bicycle, and 2359 pedestrian labels, with class frame-presence of 600/528/360 frames respectively. For the 800-frame subset, cooperative annotations contain 9548 car, 1102 bicycle, and 3117 pedestrian labels, with class frame-presence of 800/684/488 frames; vehicle-side annotations contain 5742 car, 449 bicycle, and 1687 pedestrian labels, with frame-presence of 800/407/303 frames. For the 1000-frame subset, cooperative annotations contain 12003 car, 1411 bicycle, and 3809 pedestrian labels, with class frame-presence of 1000/855/628 frames; vehicle-side annotations contain 6993 car, 549 bicycle, and 2045 pedestrian labels, with frame-presence of 1000/506/395 frames. For the complete local Griffin-25m val annotation exposed by the official pkl files, the diagnostic reports 10 scenes and 1490 samples, not the 47-scene / 7000-sample paper-level raw scope. Its cooperative annotations contain 17926 car, 2096 bicycle, and 5123 pedestrian labels. The 1000-frame metric blocks show the current paper-fit problem is not empty data but weak non-car performance: CoopTrack reaches car AP@2m `0.4840`, bicycle AP@2m `0.0067`, and pedestrian AP@2m `0.0`; early fusion reaches car AP@2m `0.7797`, bicycle AP@2m `0.1174`, and pedestrian AP@2m `0.0`.

The result-pkl diagnostic adds a second check on the same 1000-frame outputs. Official detection mAP is computed from `boxes_3d_det/scores_3d_det/labels_3d_det`, while official tracking AMOTA is computed from `boxes_3d/scores_3d/labels_3d`. For the 1000-frame CoopTrack pkl, the tracking output contains 4652 car predictions but only 47 bicycle and 87 pedestrian predictions; its detection output has only 55 bicycle and 100 pedestrian boxes at score `>=0.3`. For early fusion, the tracking output contains 4745 car predictions, 213 bicycle predictions, and only 30 pedestrian predictions. This confirms the paper-fit gap is caused by weak bicycle/pedestrian usable predictions in the current subset/checkpoint path, not by missing annotations or a class-name ordering mismatch.

The 1490-frame validation split expansion covers the full local Griffin-25m validation subset exposed by the available official pkl files: 10 scenes, 149 samples per scene. The run first completed the sequential no-fusion and early-fusion baselines in `griffin_repro/artifacts/logs/expanded_10scene_149per_scene_all_20260620_143510_automated.log`, then used additional A100 GPUs to run CoopTrack and late-fusion in isolated output tags:

```text
subset: 10 scenes, 149 samples per scene, 1490 samples total
method            paper AP   paper AMOTA   partial AP        partial AMOTA   evidence log
0-no fusion       0.375      0.365         0.1986            0.160           griffin_repro/artifacts/logs/expanded_10scene_149per_scene_all_20260620_143510_automated.log
1-early fusion    0.607      0.670         0.2332            0.270           griffin_repro/artifacts/logs/expanded_10scene_149per_scene_all_20260620_143510_automated.log
2b1-cooptrack     0.479      0.488         0.1450            0.151           griffin_repro/artifacts/logs/parallel_instance_10scene_149per_scene_20260620_172629.log
3-late fusion     0.378      0.377         0.1341 det        0.126           griffin_repro/artifacts/logs/parallel_late_10scene_149per_scene_20260620_172540.log
```

For late-fusion in the 1490-frame run, the detection stage produced mAP `0.1341`; the AB3DMOT tracking stage produced tracking AMOTA `0.126` and tracking-result mAP `0.1258`.

The table above is the official aggregate output over all three classes. A later audit found that `docs/detailed_results.csv` aligns with the official log's `car` class rows: detection `car` AP from the first per-class table and tracking `car` AMOTA from the second per-class table. Rechecking the same 1490-frame logs with `--metric-scope paper` gives:

```text
method            paper AP   paper AMOTA   car AP   car AMOTA   paper-scope status
0-no fusion       0.375      0.365         0.477    0.456       outside tolerance, higher than paper
1-early fusion    0.607      0.670         0.607    0.670       matches
2b1-cooptrack     0.479      0.488         0.420    0.453       outside tolerance, below paper
3-late fusion     0.378      0.377         0.377    0.379       matches
```

The corrected paper-scope check changes the interpretation: early-fusion and late-fusion now match the paper table within tolerance; CoopTrack is close but still below the paper by AP `0.059` and AMOTA `0.035`; no-fusion is above the paper reference by AP `0.102` and AMOTA `0.091`, so it is not a table match either.

The no-fusion mismatch above is now explained by the GT side used for evaluation, not by model quality. The vehicle-side no-fusion log evaluates vehicle predictions against vehicle-side GT (`car GT=6932`), while the paper table uses the cooperative/benchmark GT (`car GT=8320`). Re-evaluating the saved vehicle-side predictions from `results-06201519.pkl` against the cooperative 1490-frame GT with the official nuScenes evaluator gives:

```text
method            paper AP   paper AMOTA   re-eval car AP   re-eval car AMOTA   GT     TP     FP    FN     IDS   status
0-no fusion       0.375      0.365         0.3748           0.3651              8320   3141   418   5177   2     matches
2b1-cooptrack     0.479      0.488         0.4203           0.4532              8320   4685   1293  3611   24    below paper
```

The command used to summarize those evaluator JSON outputs is:

```bash
python scripts/griffin_repro.py summarize-eval-json \
  --eval-dir griffin_repro/official/projects/work_dirs_griffin_50scenes_25m/vehicle-side/reeval_vehicle_as_coop_1490/json_output \
  --dataset 50scenes_25m \
  --method "0-no fusion" \
  --json
```

For CoopTrack, the remaining mismatch is now isolated to the instance-fusion output quality. The 1490-frame track-query cache has complete coverage (`1490/1490` files, no missing air-token files), the successful CoopTrack run loaded `ckpts/griffin_50scenes_25m/cooperative/instance_fusion/iter_33024.pth`, and the saved tracking output has no negative track ids or duplicate ids within a frame. The difference from the paper is that this run produces more car matches but many more high-confidence false positives and identity switches: paper `TP/FP/FN/IDS = 3755/599/4563/2`, current `4685/1293/3611/24`.

The remote checkpoint audit on 2026-06-20 matched the upstream Griffin md5 list exactly: cooperative instance-fusion `7e1448188b6e99ca6303575c3466b97f`, drone-side `41734b8d764d1213935a231a3419f655`, early-fusion `ccf39651d3e4381ec06e4d0709821949`, and vehicle-side `1201f5692e390a75e5b5ab0efa0cae19`. The local upstream source manifest also matches `wang-jh18-SVM/Griffin` HEAD `9c02ba4a37201edfc2b95ddbcdc2ff9aff47e7f4`, so the CoopTrack gap is not currently explained by a wrong released checkpoint or stale official source snapshot.

The track-query cache audit uses the new `analyze-track-query-cache` command. On the 1490-frame CoopTrack run it reported `1490/1490` coverage, no extra files, no NaN/inf tensors, no `ref_pts` values outside `[0, 1]`, `query_feats/query_embeds/ref_pts/obj_idxes/scores` present in every file, and an average of `905.37` queries per frame with `5.37` active `obj_idxes >= 0` per frame. The cache does not contain `cache_motion_feats` or other `cache_*` fields, but the active instance-fusion path uses `CrossAgentSparseInteraction`, which consumes the query/ref/score/id fields and does not require those cache-motion fields.

The remote asset audit currently shows only `datasets/griffin_50scenes_25m`, the Griffin-25m val info files, and the four Griffin-25m checkpoints are present. The 40m, 55m, and 100-scenes-random paper scene groups are represented in the matrix and runnable-command metadata, but they have not been real-run on the remote host because their datasets/checkpoints are not present there yet.

### Official 25m Paper-Scope Verification

The 2026-06-20 official-config batch moved beyond smoke/partial tags and ran the upstream Griffin-25m configs directly on the remote multi-A100 host. The batch manifest is `griffin_repro/artifacts/logs/official_25m_batch_20260620_194535.json`; each row below was summarized from official `metrics_summary.json` or official log output with paper-scope car-class AP/AMOTA, then compared with `docs/detailed_results.csv` using `paper_tolerance=0.02`.

```text
method            condition                 actual AP   paper AP   actual AMOTA   paper AMOTA   status      evidence
1-early fusion    baseline                  0.607394    0.607      0.669747       0.670         matches     official_25m_early_baseline_20260620_194535.log
1-early fusion    packet_loss_0.3           0.564822    0.565      0.639051       0.639         matches     official_25m_early_drop30_20260620_194535.log
1-early fusion    translation_error_m_1.5   0.495353    0.495      0.576453       0.576         matches     official_25m_early_loc15_20260620_194535.log
1-early fusion    rotation_error_deg_3      0.522681    0.523      0.583572       0.584         matches     official_25m_early_orien3_20260620_194535.log
2b1-cooptrack     baseline                  0.420317    0.479      0.453223       0.488         below paper official_25m_cooptrack_baseline_20260620_194535.log
2b1-cooptrack     communication_latency_100 0.412454    0.463      0.452798       0.467         AP below    official_25m_cooptrack_latency100_20260620_194535.log
3-late fusion     baseline                  0.377       0.378      0.379          0.377         matches     official_25m_late_baseline_20260620_200431.log
```

The late-fusion baseline in this table used freshly generated official vehicle/drone pkl inputs from `official_25m_vehicle_baseline_20260620_195720.log` and `official_25m_drone_query_eval_20260620_195720.log`, then ran `tools/eval_late_fusion.sh` with the upstream late-fusion and AB3DMOT configs. The new helper command can summarize such bare official logs directly:

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
/home/fp/miniconda3/envs/griffin/bin/python scripts/griffin_repro.py summarize-official-log \
  --log griffin_repro/artifacts/logs/official_25m_late_baseline_20260620_200431.log \
  --dataset 50scenes_25m \
  --method "3-late fusion" \
  --paper-tolerance 0.02 \
  --json
```

This official-config batch strengthens the current paper-fit conclusion: early-fusion matches the Griffin-25m baseline and representative robustness axes (communication latency was separately verified at 100ms with AP `0.564283` vs paper `0.564`, AMOTA `0.639659` vs paper `0.640`), and late-fusion baseline matches after regenerating its official inputs. CoopTrack remains the outstanding mismatch: the official baseline config and 100ms latency config both run successfully with complete track-query cache coverage, but car AP remains about `0.05-0.06` below the released paper table.

The class-level 1490-frame metrics show the same paper-fit problem as the smaller subsets. Detection AP@2m is: no-fusion car `0.5401`, bicycle `0.1560`, pedestrian `0.0004`; early-fusion car `0.7570`, bicycle `0.1344`, pedestrian `0.0`; CoopTrack car `0.5061`, bicycle `0.0233`, pedestrian `0.0`; late-fusion car `0.4658`, bicycle `0.0124`, pedestrian `0.0`. Result-pkl diagnostics confirm the usable tracking predictions are heavily car-dominated: early-fusion tracking contains 6911 car, 289 bicycle, and 40 pedestrian predictions; CoopTrack tracking contains 6776 car, 81 bicycle, and 100 pedestrian predictions.

Paper-fit assessment for the 200/210/220/250/300/400/600/800/1000/1490-frame subsets:

- Use `--metric-scope paper` for paper-table comparison; it reads the car-class AP/AMOTA rows that match the CSV metric convention.
- Under the corrected paper-table scope, no-fusion, early-fusion, and late-fusion match the paper on the completed 1490-frame validation split when no-fusion is evaluated against the cooperative GT used by the paper table.
- CoopTrack still does not fully match the paper table: it remains below the paper reference and below early-fusion in this run.
- The aggregate three-class outputs remain useful diagnostics. They expose weak bicycle/pedestrian predictions, but they should not be used as the paper-table AP/AMOTA comparison.
- `validate-run` `passed=true` in old aggregate logs means the official evaluator ran and AP/AMOTA were parsed inside the configured tolerance. It is not by itself a claim that the result equals the paper table.

The cooperative validation info currently exposes 10 val `scene_token` groups under this partial selector, even though the official raw Griffin-25m nuScenes metadata records 47 scenes and 7000 samples. Increasing `GRIFFIN_PARTIAL_SCENE_LIMIT` above 10 therefore does not expand the selected val subset yet; the current expansion axis is `GRIFFIN_PARTIAL_SAMPLES_PER_SCENE`, up to the full 149 frames per val scene.

To rerun the same 100-frame checks from MobaXterm:

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
export GRIFFIN_PARTIAL_SCENE_LIMIT=10
export GRIFFIN_PARTIAL_SAMPLES_PER_SCENE=10

bash griffin_repro/run_smoke_25m_vehicle_mobaxterm.sh 2>&1 | tee griffin_repro/artifacts/logs/manual_vehicle_10scene_10per_scene_$(date +%Y%m%d_%H%M%S).log
bash griffin_repro/run_smoke_25m_early_mobaxterm.sh 2>&1 | tee griffin_repro/artifacts/logs/manual_early_10scene_10per_scene_$(date +%Y%m%d_%H%M%S).log
bash griffin_repro/run_smoke_25m_instance_mobaxterm.sh 2>&1 | tee griffin_repro/artifacts/logs/manual_instance_10scene_10per_scene_$(date +%Y%m%d_%H%M%S).log
```

To rerun a larger partial check, set `GRIFFIN_PARTIAL_SAMPLES_PER_SCENE` to the desired samples per val scene (`20` for 200 frames, `40` for 400 frames, `60` for 600 frames, `80` for 800 frames, `100` for 1000 frames, up to `149` for the 1490-frame val split) and use the same smoke scripts. Set `GRIFFIN_MATERIALIZE_JOBS=8` on the remote host to fetch missing image files concurrently. For late-fusion, reuse the latest matching vehicle and drone pkl outputs by running the dedicated late-fusion smoke script:

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
export GRIFFIN_PARTIAL_SCENE_LIMIT=10
export GRIFFIN_PARTIAL_SAMPLES_PER_SCENE=20
export GRIFFIN_MATERIALIZE_JOBS=8
bash griffin_repro/run_smoke_25m_late_mobaxterm.sh 2>&1 | tee griffin_repro/artifacts/logs/manual_late_10scene_20per_scene_$(date +%Y%m%d_%H%M%S).log
```

On the multi-A100 remote host, safe parallelism is possible after the shared drone-side query outputs have been generated. Do not run multiple drone-query producers against the same tag at once because the official configs share `data/infos/griffin_50scenes_25m/drone-side/track_query/`. The safe pattern used for the 1490-frame closure was:

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
export GRIFFIN_PARTIAL_SCENE_LIMIT=10
export GRIFFIN_PARTIAL_SAMPLES_PER_SCENE=149
export GRIFFIN_SKIP_CONVERTER=1
export GRIFFIN_MATERIALIZE_JOBS=8

# First produce shared vehicle/drone-side pkl and track_query outputs.
CUDA_VISIBLE_DEVICES=0 bash griffin_repro/run_smoke_25m_early_mobaxterm.sh

# Then run independent final evaluators with isolated out-tags and unique MASTER_PORT values.
# Use scripts/griffin_repro.py prepare-partial-eval --out-tag <unique_tag>
# and assign CUDA_VISIBLE_DEVICES / MASTER_PORT per process.
```

When increasing `GRIFFIN_PARTIAL_SAMPLES_PER_SCENE`, the materialization step now prints per-image progress to stderr while keeping stdout JSON-compatible. If the terminal is still printing `Materializing ... images: N/M`, the job is still filling image files, not yet running model evaluation. With `GRIFFIN_MATERIALIZE_JOBS>1`, the `N/M` lines represent plan traversal, not completed downloads; the script now also prints `submitted ... missing downloads` and `completed ...` lines so a terminal can distinguish queued work from finished fetches.

To audit class-level prediction coverage from a saved official result pkl:

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
/home/fp/miniconda3/envs/griffin/bin/python scripts/griffin_repro.py analyze-result-pkl \
  --path griffin_repro/official/projects/work_dirs_griffin_50scenes_25m/cooperative/instance_fusion/tiny_track_r50_stream_bs8_48epoch_3cls_partial_10scene_100per_scene/results-06201404.pkl \
  --json
```

To audit the drone-side track-query cache consumed by CoopTrack:

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
/home/fp/miniconda3/envs/griffin/bin/python scripts/griffin_repro.py analyze-track-query-cache \
  --query-dir griffin_repro/official/data/infos/griffin_50scenes_25m/drone-side/track_query \
  --ann-file griffin_repro/official/data/infos/griffin_50scenes_25m/cooperative/griffin_infos_val_partial_10scene_149per_scene_parallel_instance_20260620_172629.pkl \
  --key query_feats \
  --key query_embeds \
  --key obj_idxes \
  --key ref_pts \
  --key scores \
  --key cache_motion_feats \
  --json
```

To summarize paper-reference deltas from a combined run log after each smoke script writes its `validate-run` JSON:

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
/home/fp/miniconda3/envs/griffin/bin/python scripts/griffin_repro.py summarize-run-log \
  --log griffin_repro/artifacts/logs/expanded_10scene_149per_scene_all_20260620_143510_automated.log \
  --paper-tolerance 0.02 \
  --metric-scope paper \
  --json
```

When the 1490-frame checks are run in parallel on separate A100 GPUs, combine the main, CoopTrack, and late-fusion logs before judging paper fit:

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
/home/fp/miniconda3/envs/griffin/bin/python scripts/griffin_repro.py summarize-run-logs \
  --log griffin_repro/artifacts/logs/expanded_10scene_149per_scene_all_20260620_143510_automated.log \
  --log griffin_repro/artifacts/logs/parallel_instance_10scene_149per_scene_20260620_172629.log \
  --log griffin_repro/artifacts/logs/parallel_late_10scene_149per_scene_20260620_172540.log \
  --paper-tolerance 0.02 \
  --metric-scope paper \
  --json
```

In that summary, `metric_scope=paper` means the parser reparses each official eval log and compares the car-class rows against `docs/detailed_results.csv`. `all_within_paper_tolerance` applies the explicit paper-level tolerance above; use `paper_mismatches` when deciding whether a partial or expanded run actually matches the paper table.

To inspect coverage before a longer run:

```bash
cd /home/fp/CARLA/CarlaAir-v0.1.7/code
/home/fp/miniconda3/envs/griffin/bin/python scripts/griffin_repro.py describe-partial-subset --profile smoke_25m_instance --scene-limit 10 --samples-per-scene 60 --json
/home/fp/miniconda3/envs/griffin/bin/python scripts/griffin_repro.py describe-partial-subset --profile smoke_25m_instance --scene-limit 10 --samples-per-scene 149 --json
```

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
