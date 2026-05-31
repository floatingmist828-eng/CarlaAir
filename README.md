# CarlaAir Active Air-Ground World V0

This folder contains an initial research scaffold for an active UAV + ground-traffic world-model task on top of CarlaAir.

## What is included

- A scenario config format based on JSON
- An environment wrapper that connects CARLA and AirSim
- Candidate UAV viewpoint generation
- Baseline policies:
  - fixed
  - random
  - ego-follow
  - intersection-center
  - occlusion heuristic
- Ground-truth labels based on constant-velocity trajectory projection
- Episode recording and simple evaluation helpers

## Entry points

### Run one episode

```bash
python3 code/scripts/run_active_world.py --scenario code/configs/scenarios/town10hd_v0.json --policy heuristic --record code/recordings/demo.json
```

### Collect a small dataset

```bash
python3 code/scripts/collect_dataset.py --scenario code/configs/scenarios/town10hd_v0.json --policy fixed --episodes 3 --output-dir code/recordings/active_world
```

### Evaluate recorded episodes

```bash
python3 code/scripts/evaluate_policy.py --input-dir code/recordings/active_world
```

### Interactive UAV task

```bash
python3 code/scripts/run_uav_task.py
```

Interactive commands:

- `takeoff`
- `hover`
- `up 5`
- `down 5`
- `forward 5`
- `back 5`
- `left 5`
- `right 5`
- `goto x y z`
- `yaw 30`
- `status`
- `sample`
- `patrol on`
- `patrol off`
- `quit`

Vehicle view windows:

- On servers, live OpenCV windows are disabled by default to avoid X11/Qt crashes.
- Set `CARLAAIR_ENABLE_VIEWER=1` before launching if you are running in a desktop session and want live windows for each tracked ground vehicle.
- Window names use the pattern `VehicleView-<actor_id>`.
- The default scenario now tracks 2 ground vehicles, so you should see up to 2 vehicle windows in addition to the UAV view.
- The default scenario keeps the UAV hovering at the intersection start pose.
- You can still type `patrol on` later to try the experimental automatic viewpoint cycling.

## Data layout

Each run writes one `episode.json` plus a `samples/` directory under the output folder.

Example:

```text
recordings/active_uav_task/
  episode.json
  samples/
    step_000001/
      uav/
        rgb.png
        depth.png
      vehicle_123/
        rgb.png
        depth.png
      vehicle_456/
        rgb.png
        depth.png
```

`episode.json` contains per-step records:

- `ego`: ego vehicle pose/state
- `traffic`: other vehicle states
- `uav`: UAV image shapes
- `vehicle_sensors`: per-vehicle image shapes
- `data_files`: file paths for the saved PNGs
- `labels`: simple future-trajectory proxy labels
- `captured`: whether the step was actually saved
- `reason`: why the step was skipped when `captured` is false

Sampling is gated:

- Data is only captured when the ego or one of the tracked traffic vehicles is within the configured hotspot radius.
- `sample_min_interval_sec` controls the minimum time between saved samples.
- If a sample is skipped, the interactive mode prints the reason.

How to use it later:

- Read `episode.json`
- For each step, load the PNGs from `data_files`
- Join `ego` / `traffic` / `uav` / `vehicle_sensors` into your model input
- Use `labels` as a starting target before you replace them with a stronger world-model label

## Notes

- This is V0 scaffolding, not a full benchmark.
- The label generation currently uses constant-velocity rollout.
- The UAV policies are heuristic baselines that are easy to extend later.
- This scaffold preserves existing traffic in CarlaAir, but it does not yet create a new multi-vehicle traffic stream by itself.
- Ground vehicles are spawned with CARLA autopilot / Traffic Manager control, not manual driving.
- The UAV starts in a hover state over the intersection, and manual commands can still move it if needed.
- The bundled AirSim settings use `SimpleFlight` as the multirotor vehicle name. If you change `settings.json`, update the scenario JSON to match.
- Task scripts use Traffic Manager port `8001` by default so they do not collide with CarlaAir's auto-spawned city traffic on port `8000`. Override with `CARLAAIR_TASK_TRAFFIC_MANAGER_PORT` only when you know the target port is free.
