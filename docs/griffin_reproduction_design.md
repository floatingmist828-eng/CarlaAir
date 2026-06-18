# Griffin Reproduction Design

## Assumptions

- The user selected an engineering closure: one real Griffin evaluation loop is enough for now, and the full paper matrix does not need to run immediately.
- The current CarlaAir driving code is out of scope and must remain isolated.
- Large Griffin datasets and checkpoints are external assets and must not be committed.

## Design

The official Griffin source is vendored under `griffin_repro/official`. A lightweight manifest records source provenance, selected smoke profiles, expected paper metrics, and the full matrix boundaries from the paper.

Repository-level helpers live in `scripts/` so they can be run from the CarlaAir root without importing the legacy CarlaAir package. The helpers only use the Python standard library, except for the remote sync script, which optionally uses Paramiko for password-based SFTP.

## Verification

The local test suite verifies that the official tree is isolated, the paper result CSV can be parsed, the selected smoke profile resolves to a real Griffin eval command, and remote sync dry-run scope excludes legacy driving code.

Real experimental verification requires the Griffin Linux GPU environment plus the selected Griffin-25m dataset and checkpoint. The smoke profile target is the paper's `50scenes_25m / 2b1-cooptrack` result: AP `0.479`, AMOTA `0.488`.
