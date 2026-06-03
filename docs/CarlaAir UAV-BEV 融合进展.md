# CarlaAir UAV-BEV 融合进展

更新时间：2026-06-04

## 目标

第四阶段目标是把无人机视角引入车端闭环控制：

```text
UAV RGB
  -> BEVFormer-Lite
  -> global BEV feature

车载 TCP/Local-BEV feature + UAV global BEV feature
  -> fusion planner
  -> vehicle control
```

当前实现采用轻量 BEVFormer-Lite 风格原型，不直接训练完整 BEVFormer 大模型。原因是当前项目前三阶段已经有可运行的 TCP-Lite-YOLO11n 车端闭环，第四阶段优先要证明 UAV RGB 能进入车端控制链路，并且真实 CARLA 闭环运行不退化。

当前服务器上的 AirSim 运行环境返回的是 `PhysXCar`，不支持 `getMultirotorState(vehicle_name=...)` 多旋翼状态 API。因此本阶段新增了 `uav_control_enabled` 开关：完整 UAV 控制场景仍可保留旧逻辑，当前 `town10hd_vision_tcp_lite_yolo_uav_bev.json` 使用 `uav_control_enabled=false`，通过 AirSim scene camera 获取 UAV-like 全局 RGB 做 BEV 融合验证。

## 当前实现

新增配置：

```text
configs/scenarios/town10hd_vision_tcp_lite_yolo_uav_bev.json
```

新增代码：

```text
carlaair_active_world/vision_models/uav_bev.py
```

核心改动：

```text
UAVSensorRig.snapshot()
  -> CachedUAVBEVProvider
  -> extract_uav_bev_feature()
  -> VisionEgoDriver observation["uav_bev"]
  -> TcpLiteVisionPolicy._uav_bev_correction()
  -> fused steer/throttle/brake diagnostics
```

`extract_uav_bev_feature()` 会从 UAV RGB 中提取紧凑全局 BEV-like 特征，包括：

- `road_confidence`
- `center_bias`
- `forward_density`
- `left_right_balance`
- `feature`

这些特征不是直接替代 TCP-Lite，而是作为全局上下文进入 `TcpLiteVisionPolicy` 的稳定控制阶段。

## 融合策略

车端 TCP-Lite 仍然负责主驾驶轨迹预测：

```text
车载 RGB + speed + navigation_command
  -> TCP-Lite
  -> local trajectory/control
```

UAV BEV feature 只做受限修正：

```text
UAV global BEV center_bias
  -> confidence gate
  -> max steer correction gate
  -> steering stabilization
```

这样可以避免 UAV 图像异常、AirSim 采样延迟或 BEV 特征置信度不足时破坏 clean 场景已有稳定性。

## 关键配置

```json
{
  "uav_enabled": true,
  "uav_control_enabled": false,
  "uav_bev_fusion_enabled": true,
  "uav_bev_camera_name": "front_center",
  "uav_bev_refresh_hz": 2.0,
  "uav_bev_min_confidence": 0.20,
  "uav_bev_steer_gain": 0.06,
  "uav_bev_max_steer_correction": 0.06
}
```

`uav_bev_refresh_hz` 限制 UAV 图像采样频率，避免车端 8Hz 控制循环每帧都阻塞 AirSim RPC。

`uav_bev_max_steer_correction` 当前设置很保守，目标是先证明融合链路真实有效且不破坏 clean 驾驶。

`uav_control_enabled=false` 表示本场景只使用 AirSim camera RGB，不调用 takeoff、hover、move 或 multirotor state API。

## 运行命令

终端 1 启动 CARLA：

```bash
cd ~/CARLA/CarlaAir-v0.1.7
source ~/miniconda3/etc/profile.d/conda.sh
conda activate carlaAir

export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CARLAAIR_ENABLE_VIEWER=1

./CarlaAir.sh Town10HD --quality Low --res 800x600 --windowed
```

终端 2 运行 UAV-BEV 融合 clean 场景：

```bash
cd ~/CARLA/CarlaAir-v0.1.7/code
source ~/miniconda3/etc/profile.d/conda.sh
conda activate carlaAir

export DISPLAY=localhost:10.0
export XAUTHORITY=$HOME/.Xauthority
export QT_X11_NO_MITSHM=1

python3 scripts/run_active_world.py \
  --scenario configs/scenarios/town10hd_vision_tcp_lite_yolo_uav_bev.json \
  --policy fixed \
  --record recordings/manual_uav_bev_clean_$(date +%Y%m%d_%H%M%S).json \
  --viewer
```

## 如何看效果

记录文件中每一步的 `observation.ego_control` 应包含：

```text
uav_bev
uav_bev_fusion
stabilization.uav_bev_fusion
```

如果 UAV RGB 正常，`uav_bev.available` 应为 `true`。

如果融合实际参与修正，`uav_bev_fusion.applied` 应为 `true`，并有非零 `steer_correction`。

clean 场景评价仍以闭环驾驶为准，重点看：

- `path_distance_m`
- `net_displacement_m`
- `mean_speed_mps`
- `brake_steps`
- `reason` 分布
- `safety` 分布
- lane offset
- viewer 中是否稳定行驶

## 当前边界

当前版本不是完整 BEVFormer 训练模型，也没有把高维 BEV tensor 输入到 TCP-Lite checkpoint 中，因为现有 checkpoint 没有这一路输入。

当前版本的意义是先完成真实数据链路：

```text
AirSim UAV/camera RGB -> global BEV-like feature -> ego policy observation -> fusion planner -> vehicle control
```

后续如果要继续增强，可以再把 UAV BEV feature 记录成训练数据，训练一个显式的 fusion planner 或扩展 TCP-Lite 输入结构。
