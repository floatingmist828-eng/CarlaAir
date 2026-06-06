# UAV Fusion Benchmark Design

## Assumptions

- Keep TCP-Lite as the primary driving policy. UAV information may only produce a bounded residual correction.
- Support three experiment groups from config: no UAV, rule UAV-BEV correction, and learned fusion planner correction.
- The learned planner starts as a small offline-loadable residual model. If no checkpoint is configured or loadable, it must report diagnostics and apply zero correction.
- Scene complexity should be represented by explicit scenario configs and metadata first. CARLA-only actors such as pedestrians or occluders can be enabled later without changing the experiment group schema.
- Evaluation must run on recorded `EpisodeRecorder` JSON without requiring CARLA/AirSim.

## Architecture

`ScenarioConfig` gets explicit `uav_fusion_mode` values:

- `none`: no UAV fusion correction.
- `rule`: current UAV-BEV steer residual.
- `learned`: learned residual correction from TCP-Lite local state plus UAV-BEV feature.

The old `uav_bev_fusion_enabled` boolean remains supported for existing configs. When `uav_fusion_mode` is missing, `true` maps to `rule` and `false` maps to `none`.

`TcpLiteVisionPolicy` keeps the current control flow: TCP-Lite predicts local trajectory/control, stabilization adds lane centering and optional UAV residual, then normal steer limits and rate limits apply. The learned planner is not allowed to output throttle/brake or absolute steering.

## Learned Planner

Add `carlaair_active_world/vision_models/fusion_planner.py` with:

- `FusionPlannerConfig` for mode, checkpoint path, gain, max correction, and min confidence.
- `LinearFusionPlanner` as a small deterministic residual model over numeric features.
- `FusionPlannerAdapter` that returns `(correction, diagnostics)` and gracefully disables itself when unavailable.

The initial checkpoint format is JSON so tests and remote deployment do not depend on PyTorch. A future torch planner can be added behind the same adapter.

## Scenarios

Create a scenario ladder under `configs/scenarios/experiments/`:

- `clean`: single-lane stable driving.
- `meeting`: adds oncoming traffic metadata.
- `slow_lead`: adds slow lead vehicle metadata.
- `pedestrian_crossing`: adds walker metadata.
- `junction`: junction command/route emphasis.
- `occlusion`: marks occluder factor.
- `rain_fog`: uses existing hard rain/fog weather.
- `texture_attack`: uses existing texture attack.

Each stage only adds one complexity factor. Each stage can be run with the three experiment group configs.

## Metrics And Visualization

Add a standalone recorded-episode evaluator that emits:

- Per-run and grouped metrics: collision rate, lane departure rate, off-road rate, average/max lane offset, path completion proxy, net displacement, average speed, hard brake count, steering oscillation count, safety gate count, scene success rate.
- Segment metrics for path distance every 30 seconds.
- CSV summary plus optional PNG plots for collision rate, lane offset, steer, speed, segmented path distance, and scene success.

The evaluator must tolerate partial historical recordings by using available fields and reporting missing metrics as conservative zeros or `None` where appropriate.

## Documentation And Sync

Update the current progress document with the new experiment design, commands, and rollout order. Add a remote sync helper script that stages deployment to `/home/fp/CARLA/CarlaAir-v0.1.7/code` via `rsync` when run from the repository root on a machine with SSH access.

## Testing

Unit tests should cover:

- Scenario config round-trip for fusion mode and planner settings.
- Rule vs learned residual gating in `TcpLiteVisionPolicy`.
- Learned planner checkpoint loading and bounded correction.
- Evaluation metrics and CSV/plot generation from synthetic recordings.
- Experiment scenario configs load and preserve the group/stage metadata.
