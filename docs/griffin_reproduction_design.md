# Griffin Reproduction Design

## Assumptions

- The user selected a partial-data closure: the full paper matrix must be represented, but the dataset does not need to be downloaded and evaluated end-to-end in this pass.
- The current CarlaAir driving code is out of scope and must remain isolated.
- Large Griffin datasets and checkpoints are external assets and must not be committed.

## Design

The official Griffin source is vendored under `griffin_repro/official`. A lightweight manifest records source provenance, selected smoke profiles, expected paper metrics, the full matrix boundaries from the paper, and the concrete assets required for the selected partial run.

Repository-level helpers live in `scripts/` so they can be run from the CarlaAir root without importing the legacy CarlaAir package. The helpers only use the Python standard library, except for the remote sync script, which optionally uses Paramiko for password-based SFTP.

`scripts/griffin_repro.py paper-matrix` is the reproducibility contract for the paper-level coverage:

- Scene groups: Griffin-25m, Griffin-40m, Griffin-55m, and Griffin-Random.
- Fusion methods: no fusion, early fusion, V2X-ViT, Where2Comm, CoopTrack, UniV2X, and late fusion.
- Metrics: AP, AMOTA, BPS, and FPS.
- Robustness axes: 100-400 ms latency, 10-50% packet loss, 0.5-2.5 m translation error, and 1-5 degree rotation error.

The generated `griffin_repro/run_smoke_25m_instance_mobaxterm.sh` is the operational closure for the user-run MobaXterm path. It stages raw-data checks, official conversion, drone track-query extraction, final evaluation asset checks, and the cooperative instance-fusion eval command.

## Verification

The local test suite verifies that the official tree is isolated, the paper result CSV can be parsed, all 28 zero-noise baselines are represented, the selected smoke profile resolves to real Griffin eval commands, and remote sync dry-run scope excludes legacy driving code.

Real experimental verification requires the Griffin Linux GPU environment plus the selected Griffin-25m raw data, drone checkpoint, cooperative instance-fusion checkpoint, and generated info/query assets. The smoke profile target is the paper's `50scenes_25m / 2b1-cooptrack` result: AP `0.479`, AMOTA `0.488`.
