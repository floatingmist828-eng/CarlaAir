# TCP-Lite 与 YOLO 本地闭环设计

## 背景

CarlaAir 当前已经完成规则视觉自动驾驶 baseline：车载 RGB/depth/semantic 输入经过 `VisionEgoDriver` 和 `SimpleLaneVisionPolicy` 输出 `carla.VehicleControl`。正常场景、纹理攻击、雨雾攻击已各有 scenario 配置，并已有 60 秒远程实验结果。

本阶段只推进第二、第三阶段的本地代码闭环：

1. 第二阶段：车载 RGB + speed + navigation command 输入到 TCP-Lite，输出 trajectory 和 steer/throttle/brake。
2. 第三阶段：车载 RGB 输入到 YOLOv8/YOLO11 或轻量启发式检测，输出 obstacle / attack-pattern diagnostics，并进入 safety gate。

无人机 BEV 融合暂不纳入本阶段。

## 范围

本阶段交付边界是方案 A：在 `E:\a2\CarlaAir` 完成可训练、可推理、可测试的 TCP-Lite + YOLO safety gate 代码闭环。真实 CARLA 60 秒效果实验、远程服务器同步、GitHub push、权重下载和论文级 TCP 复现不在本阶段范围内。

成功标准：

- 保留现有 `vision_simple` 和 `vision_rgb_only` 行为。
- 新增 `vision_tcp_lite` 控制模式。
- 新增轻量 TCP-Lite 模型、dataset、训练脚本、推理 policy。
- 没有 checkpoint 时 TCP-Lite policy 安全刹车并写 diagnostics。
- 有 mock 或小 checkpoint 时 TCP-Lite policy 能从 RGB/speed/command 输出 trajectory 和 control。
- YOLO obstacle detector 继续可选，detector 判定前方障碍物时 safety gate 触发刹车。
- attack-pattern detection 先做轻量 RGB 纹理启发式 diagnostics，不依赖自定义 YOLO 权重。
- 单元测试和小样本训练 smoke test 通过。

## 非目标

- 不训练可上路的最终模型。
- 不承诺 TCP-Lite 在 Town10HD 的实际 60 秒效果优于规则 baseline。
- 不引入无人机 RGB、BEVFormer-Lite 或融合 planner。
- 不把模型权重、recordings 或大数据集纳入 Git。
- 不在本地阶段强依赖 `ultralytics`、CUDA 或 CARLA 运行环境。

## 架构

现有规则链路保持不变：

```text
VehicleSensorRig RGB/depth/semantic
  -> VisionEgoDriver
  -> SimpleLaneVisionPolicy
  -> VehicleControl
```

新增学习式链路：

```text
VehicleSensorRig RGB
  + ego speed
  + navigation command
  + optional detector diagnostics
  -> VisionEgoDriver
  -> TcpLiteVisionPolicy
  -> TcpLiteModel trajectory/control heads
  -> safety gate
  -> VehicleControl
```

`VisionEgoDriver` 继续做传感器快照、速度读取、detector 调用和 diagnostics 聚合。它根据 scenario 中的 `ego_control_mode` 或显式 policy 选择规则 policy 或 TCP-Lite policy。

## 组件

### Scenario 配置

`ScenarioConfig` 新增 TCP-Lite 相关字段：

- `vision_model_path`: TCP-Lite checkpoint 路径，默认为空。
- `vision_model_device`: `cpu` 或 `cuda`，默认 `cpu`。
- `vision_navigation_command`: 当前阶段固定命令，默认 `lane_follow`。
- `vision_safety_gate_enabled`: 是否启用 detector/attack gate，默认 `true`。
- `vision_attack_pattern_gate`: 是否把启发式纹理攻击 diagnostics 转成降速或刹车，默认 `false`。

新增 scenario：

```text
configs/scenarios/town10hd_vision_tcp_lite.json
```

它使用 `ego_control_mode: "vision_tcp_lite"`，默认不填 checkpoint，因此可安全启动并刹车，不会伪装成有效学习模型。

### TCP-Lite 模型

新增模块负责纯 PyTorch 轻量模型：

```text
carlaair_active_world/vision_models/tcp_lite.py
```

最小接口：

```python
class TcpLiteModel(torch.nn.Module):
    def forward(self, rgb, speed, command):
        return {
            "trajectory": trajectory,  # [B, K, 2]
            "control": control,        # [B, 3], steer/throttle/brake logits or values
        }
```

本阶段模型是 TCP-Lite 风格，而不是完整 TCP 论文复现。它包含：

- 小型 CNN encoder。
- speed scalar embedding。
- navigation command embedding。
- trajectory head。
- control head。

### TCP-Lite Policy

新增 policy：

```text
carlaair_active_world/vision_models/tcp_lite_policy.py
```

职责：

- 预处理 RGB 到固定尺寸张量。
- 编码 speed 和 command。
- 加载 checkpoint。
- 调用 `TcpLiteModel`。
- clamp `steer/throttle/brake` 到 CARLA 控制范围。
- 如果缺 RGB、缺 checkpoint、模型异常或 safety gate 触发，则安全刹车。
- 记录 `last_diagnostics`，包括 `model_ready`、`command`、`trajectory`、`raw_control`、`safety_gate`、`reason`。

### YOLO 与 Safety Gate

现有 `UltralyticsObstacleDetector` 保留，并继续通过 `vision_detector_model_path` 和 `vision_detector_confidence` 配置。新增轻量 safety gate 辅助逻辑：

```text
carlaair_active_world/vision_models/safety_gate.py
```

职责：

- 将 detector 输出的 `obstacle=True` 解释为前方障碍物。
- 从 RGB 计算高对比重复纹理启发式指标，记录 `attack_pattern_score`。
- 在 `vision_attack_pattern_gate=true` 且 score 超阈值时触发保守动作。

默认只让 obstacle gate 触发刹车；attack-pattern gate 默认只记录 diagnostics，避免误杀正常道路纹理。

### 数据集与训练

新增离线 imitation 数据格式：

```text
dataset_root/
  samples.jsonl
  images/
    000001.png
```

每行 JSON：

```json
{
  "rgb": "images/000001.png",
  "speed_mps": 2.1,
  "command": "lane_follow",
  "trajectory": [[2.0, 0.0], [4.0, 0.1], [6.0, 0.2]],
  "control": {"steer": 0.02, "throttle": 0.3, "brake": 0.0}
}
```

新增：

```text
carlaair_active_world/vision_models/tcp_lite_dataset.py
scripts/train_tcp_lite.py
```

训练脚本读取 JSONL，最小化 trajectory MSE 和 control MSE，保存 checkpoint。测试只使用临时小数据集跑 smoke，不要求真实收敛效果。

## 数据流

1. `ActiveAirGroundEnv` 或 `ActiveUAVTaskApp` 读取 scenario。
2. `ego_control_mode == "vision_tcp_lite"` 时创建 `VisionEgoDriver`，并传入 `TcpLiteVisionPolicy`。
3. `VisionEgoDriver.predict()` 读取 RGB/depth/semantic，但 TCP-Lite 只消费 RGB、speed、command、detector diagnostics。
4. Policy 推理产生 trajectory/control。
5. Safety gate 在输出前检查 obstacle 和可选 attack-pattern。
6. Diagnostics 写入 `ego_control`，供 recorder 和后续实验分析。

## 错误处理

错误处理只覆盖本阶段真实会遇到的情况：

- RGB 未就绪：刹车，`reason="missing_rgb"`。
- checkpoint 为空或不存在：刹车，`reason="missing_model_path"`。
- checkpoint 加载失败：刹车，记录异常类型。
- 推理异常：刹车，记录异常类型。
- detector 缺失或 `ultralytics` 不可用：不阻断 TCP-Lite，只记录 detector unavailable。

不增加复杂重试、下载权重、远程同步或自动 fallback 到规则 policy。

## 测试策略

使用 TDD。先添加失败测试，再实现最小代码。

测试覆盖：

- `ScenarioConfig` 能 round-trip 新字段。
- `vision_tcp_lite` scenario 能加载。
- `VisionEgoDriver` 在 `vision_tcp_lite` 模式下构造 TCP-Lite policy。
- TCP-Lite policy 缺 checkpoint 时安全刹车。
- TCP-Lite policy 使用 mock model 时输出 clamped control 和 trajectory diagnostics。
- Safety gate 收到 obstacle diagnostics 时触发刹车。
- Attack-pattern heuristic 对高对比重复纹理给出高于干净图的 score。
- `scripts/train_tcp_lite.py` 能在临时小数据集上跑一个 epoch 并保存 checkpoint。

## 验证

本地阶段验证命令：

```bash
pytest
```

如果环境安装了 PyTorch 和 Pillow，再跑：

```bash
python scripts/train_tcp_lite.py --dataset tests/fixtures/tcp_lite_tiny --output checkpoints/tcp_lite_smoke.pt --epochs 1 --batch-size 2 --device cpu
```

真实 CARLA 效果验证留到下一阶段，在远程 `/home/fp/CARLA/CarlaAir-v0.1.7/code` 同步后运行 60 秒 normal/attack scenario 对比。

## Git 与同步

本阶段只在本地 Git 仓库提交代码和测试。提交后不自动 push，不自动同步远程服务器 code 目录。远程同步应在本地测试通过并经确认后执行。

## 风险

- 本地没有 CARLA 时只能验证接口和离线训练 smoke，不能证明驾驶效果。
- 小型 TCP-Lite 从零训练需要足够专家数据，否则真实性能可能弱于规则 baseline。
- YOLO 权重下载和自定义 attack 类别训练不在本阶段，attack-pattern detection 只能作为轻量 diagnostics。
- `torch` 如果在本地环境不可用，训练 smoke 会被测试跳过或以清晰错误提示结束；远程训练环境需单独确认。
