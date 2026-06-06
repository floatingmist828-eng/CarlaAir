# UAV Fusion Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible three-group UAV fusion benchmark path that keeps TCP-Lite as the primary driver while adding rule and learned residual fusion comparison.

**Architecture:** Extend `ScenarioConfig` with explicit fusion mode and planner settings, then route those settings into `TcpLiteVisionPolicy`. Add a JSON-backed learned residual adapter, experiment scenario configs, recorded-run metrics/plots, and sync/docs support.

**Tech Stack:** Python dataclasses, pytest, JSON scenario configs, matplotlib optional plotting, existing CarlaAir recorder JSON.

---

## File Structure

- Modify `carlaair_active_world/scenario.py` to parse and serialize fusion mode, planner checkpoint, planner gain, planner cap, and experiment metadata.
- Create `carlaair_active_world/vision_models/fusion_planner.py` for bounded learned residual corrections.
- Modify `carlaair_active_world/vision_models/tcp_lite_policy.py` to choose `none`, `rule`, or `learned` UAV residuals.
- Modify `carlaair_active_world/env.py` and `carlaair_active_world/task_app.py` to pass new settings into TCP-Lite policy.
- Create `scripts/evaluate_fusion_benchmark.py` for metrics, CSV, and optional PNG plots from recordings.
- Create `scripts/sync_remote_code.sh` for explicit remote copy to `/home/fp/CARLA/CarlaAir-v0.1.7/code`.
- Create `configs/scenarios/experiments/*.json` for the scenario ladder and comparison groups.
- Modify tests under `tests/` with red-green coverage for each new behavior.

### Task 1: Fusion Config Schema

**Files:**
- Modify: `carlaair_active_world/scenario.py`
- Test: `tests/test_scenario_config.py`

- [ ] **Step 1: Write the failing test**

```python
def test_scenario_config_round_trips_uav_fusion_mode_and_planner():
    config = ScenarioConfig.from_dict({
        "name": "fusion",
        "uav_bev_fusion_enabled": True,
        "uav_fusion_mode": "learned",
        "uav_fusion_planner_path": "models/fusion.json",
        "uav_fusion_planner_gain": 0.5,
        "uav_fusion_max_steer_correction": 0.04,
        "uav_fusion_min_confidence": 0.3,
        "experiment_group": "learned_fusion",
        "scenario_stage": "clean",
        "scenario_complexity": ["clean"],
    })
    assert config.uav_fusion_mode == "learned"
    assert config.to_dict()["uav_fusion_planner_path"] == "models/fusion.json"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scenario_config.py::test_scenario_config_round_trips_uav_fusion_mode_and_planner -q`

Expected: fail because `ScenarioConfig` has no new attributes.

- [ ] **Step 3: Implement minimal schema fields**

Add dataclass fields, parse them in `from_dict`, and serialize them in `to_dict`. Normalize missing mode from `uav_bev_fusion_enabled`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_scenario_config.py::test_scenario_config_round_trips_uav_fusion_mode_and_planner -q`

Expected: pass.

### Task 2: Learned Planner Adapter

**Files:**
- Create: `carlaair_active_world/vision_models/fusion_planner.py`
- Test: `tests/test_fusion_planner.py`

- [ ] **Step 1: Write failing tests**

```python
def test_linear_fusion_planner_applies_bounded_json_weights(tmp_path):
    path = tmp_path / "fusion.json"
    path.write_text(json.dumps({"weights": {"center_bias": 1.0}, "bias": 0.0}), encoding="utf-8")
    adapter = FusionPlannerAdapter(FusionPlannerConfig(mode="learned", checkpoint_path=str(path), gain=0.5, max_correction=0.2))
    correction, diagnostics = adapter.predict({"uav_bev": {"available": True, "road_confidence": 0.8, "center_bias": 1.0}})
    assert correction == 0.2
    assert diagnostics["applied"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fusion_planner.py -q`

Expected: import failure because module does not exist.

- [ ] **Step 3: Implement adapter**

Implement config dataclass, JSON checkpoint loading, numeric feature extraction, min confidence gate, gain, and clamp.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fusion_planner.py -q`

Expected: pass.

### Task 3: TCP-Lite Fusion Modes

**Files:**
- Modify: `carlaair_active_world/vision_models/tcp_lite_policy.py`
- Test: `tests/test_tcp_lite.py`

- [ ] **Step 1: Write failing tests**

Add tests proving `uav_fusion_mode="none"` ignores UAV features, `rule` keeps current center-bias correction, and `learned` uses the planner while still passing through steer stabilization and caps.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_tcp_lite.py::test_tcp_lite_policy_uses_learned_uav_fusion_residual -q`

Expected: fail because constructor has no learned planner settings.

- [ ] **Step 3: Implement minimal mode switch**

Add constructor parameters, instantiate `FusionPlannerAdapter`, and update `_uav_bev_correction`.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_tcp_lite.py -q`

Expected: pass.

### Task 4: Environment Wiring And Scenario Ladder

**Files:**
- Modify: `carlaair_active_world/env.py`
- Modify: `carlaair_active_world/task_app.py`
- Create: `configs/scenarios/experiments/*.json`
- Test: `tests/test_scenario_config.py`
- Test: `tests/test_tcp_lite.py`

- [ ] **Step 1: Write failing tests**

Add tests that env/task app pass fusion kwargs to `TcpLiteVisionPolicy`, and scenario ladder configs load.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_scenario_config.py tests/test_tcp_lite.py -q`

Expected: fail on missing kwargs/configs.

- [ ] **Step 3: Implement wiring and configs**

Pass the fields into policy constructors and add JSON configs for clean, meeting, slow lead, pedestrian, junction, occlusion, rain/fog, and texture attack across no-UAV/rule/learned groups.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_scenario_config.py tests/test_tcp_lite.py -q`

Expected: pass.

### Task 5: Metrics And Visualization

**Files:**
- Create: `scripts/evaluate_fusion_benchmark.py`
- Test: `tests/test_fusion_benchmark_eval.py`

- [ ] **Step 1: Write failing tests**

Create synthetic recorder JSON and assert collision rate, lane offsets, net displacement, hard brake count, steer oscillations, safety gate count, 30-second segments, and CSV output.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_fusion_benchmark_eval.py -q`

Expected: import failure because evaluator does not exist.

- [ ] **Step 3: Implement evaluator**

Load files, compute metrics from available fields, write `summary.csv`, `segments.csv`, and PNG plots when matplotlib is available.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_fusion_benchmark_eval.py -q`

Expected: pass.

### Task 6: Docs, Sync, And Verification

**Files:**
- Modify: `docs/CarlaAir 当前工作进展-20260606.md`
- Create: `scripts/sync_remote_code.sh`

- [ ] **Step 1: Update docs**

Add the three-group experiment plan, staged scene ladder, metrics, visualization commands, and recommended rollout order.

- [ ] **Step 2: Add sync helper**

Add an `rsync` helper that excludes recordings, model binaries, caches, and git internals.

- [ ] **Step 3: Run full tests**

Run: `python -m pytest tests -q`

Expected: all existing and new tests pass or skip optional dependency tests.

- [ ] **Step 4: Commit and push**

Stage only this task's files, commit with `build uav fusion benchmark tooling`, push to `origin main` unless remote authentication blocks it, and report the exact result.
