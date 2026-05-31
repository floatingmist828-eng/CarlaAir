# CarlaAir 项目解析文档

> 生成日期: 2026-05-16 | 版本: CarlaAir v0.1.7

---

## 一、项目总览

### 1.1 项目是什么

**CarlaAir** 是一个 **空-地联合仿真平台（Air-Ground Co-Simulation Platform）**，将两个仿真器集成到同一个 Unreal Engine 世界中：

| 仿真器 | 角色 | 用途 |
|--------|------|------|
| **CARLA** | 地面交通仿真 | 模拟地面车辆（自车 + 环境交通车）的行驶、物理、传感器 |
| **AirSim** | 空中无人机仿真 | 模拟多旋翼无人机的飞行控制、机载摄像头 |

核心理念：**用无人机（UAV）在空中观察地面交通，构建一个 "主动空-地世界模型"（Active Air-Ground World Model）研究框架。**

### 1.2 整体架构

```
┌───────────────────────────────────────────────────────────┐
│                    CarlaAir 平台                           │
│                                                           │
│  ┌──────────────────────┐   ┌──────────────────────────┐  │
│  │    CARLA 仿真引擎      │   │    AirSim 仿真引擎         │  │
│  │  (CarlaUE4-Linux-     │   │  (Multirotor无人机)       │  │
│  │   Shipping, 端口2000) │   │  端口 41451)              │  │
│  │                       │   │                          │  │
│  │  • 地面车辆物理        │   │  • 无人机飞行动力学        │  │
│  │  • Traffic Manager    │   │  • 机载RGB/深度相机        │  │
│  │  • 地图/路网           │   │  • SimpleFlight 模型      │  │
│  │  • 车载RGB/深度相机    │   │                          │  │
│  └──────────┬───────────┘   └────────────┬─────────────┘  │
│             │                            │                 │
│             │    坐标校准 (CARLA z-up     │                 │
│             │    ↔ AirSim NED z-down)     │                 │
│             │                            │                 │
│  ┌──────────┴────────────────────────────┴─────────────┐  │
│  │              Python 控制层 (Python 3.10)              │  │
│  │                                                      │  │
│  │  • carla Python API  • airsim Python API             │  │
│  │  • carlaair_active_world 包 (本项目核心代码)          │  │
│  └─────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
```

### 1.3 技术栈

- **引擎层**: Unreal Engine 4 (CarlaUE4 定制版，内嵌 AirSim 插件)
- **仿真层**: CARLA 0.9.x + AirSim (Multirotor)
- **控制层**: Python 3.10, `carla` Python API, `airsim` Python API
- **地图**: Town10HD (城市十字路口场景)
- **坐标转换**: CARLA 左手系 (z轴向上) ↔ AirSim NED 坐标系 (z轴向下)

---

## 二、code 文件夹代码模块详解

### 2.1 目录总览

```
code/
├── carlaair_active_world/   ← 核心 Python 包（框架主体）
│   ├── __init__.py           - 包初始化
│   ├── core.py               - 底层连接、生成、坐标转换
│   ├── env.py                - RL风格的仿真环境封装
│   ├── scenario.py           - 场景配置（JSON驱动）
│   ├── geometry.py           - 几何数据结构
│   ├── ego_driver.py         - 自车控制器（非视觉）
│   ├── policies.py           - 无人机视角选择策略
│   ├── labels.py             - 真值标签生成
│   ├── sensors.py            - 传感器管理（相机）
│   ├── traffic.py            - 交通车辆生成
│   ├── control.py            - 无人机手动控制接口
│   ├── recorder.py           - 数据记录器
│   └── task_app.py           - 交互式任务应用（总集成）
├── scripts/                  ← 入口脚本
│   ├── run_active_world.py   - 策略驱动的自动化运行
│   ├── run_uav_task.py       - 交互式手动控制
│   ├── collect_dataset.py    - 批量数据集采集
│   └── evaluate_policy.py    - 策略评估
├── follow_1uav-1car.py       ← 独立脚本：1架无人机跟随1辆车
├── follow-4uav-1car.py       ← 独立脚本：4架无人机绕1辆车编队
├── configs/scenarios/        ← 场景JSON配置
│   └── town10hd_v0.json
├── models/                   ← 模型存放（当前为空）
├── tests/                    ← 单元测试
└── docs/                     ← 文档
```

---

### 2.2 核心模块详细说明

#### `core.py` — 底层基础设施

项目的"操作系统层"，提供最基础的 CARLA/AirSim 连接、actor 生成、坐标系统转换。

| 功能 | 关键函数 | 说明 |
|------|---------|------|
| CARLA连接 | `connect_carla()` | 连接 localhost:2000 |
| AirSim连接 | `connect_airsim()` | 连接端口 41451，启用 API 控制 |
| 自车生成 | `spawn_ego_vehicle()` | 用 role_name="ego" 标记，自动选 spawn point |
| 自动导航 | `configure_autopilot()` | 配置 CARLA Traffic Manager (端口8001) |
| 坐标校准 | `calibrate_offset()` | 计算 CARLA↔AirSim 坐标偏移量 |
| 坐标转换 | `carla_to_airsim_ned()` | CARLA (z-up) → AirSim NED (z-down) |
| 候选点计算 | `local_candidate_to_world()` | 将无人机相对自车的局部坐标转为世界坐标 |
| 无人机移动 | `move_uav_to()` | 通过 AirSim API 设置无人机位姿 |
| 状态获取 | `get_actor_state()`, `collect_vehicle_states()` | 获取车辆位置/速度/朝向 |
| 无人机查找 | `find_drone_actor()` | 在 CARLA 中按类型名匹配无人机 actor |

**关键设计:**
- 自车使用独立 Traffic Manager 端口 8001（不与 CarlaAir 自带城市交通的端口 8000 冲突）
- 坐标校准通过找到 CARLA 世界中的无人机 actor，对比其 AirSim 坐标来计算偏移

---

#### `env.py` — 仿真环境封装 (`ActiveAirGroundEnv`)

**RL 风格的仿真环境**，提供标准的 `reset()` / `step(action)` / `observe()` / `close()` 接口。

```
reset() → observation
step(action) → ScenarioResult(observation, label, info, done)
```

**工作流程：**
1. `reset()`: 连接 CARLA/AirSim → 生成自车 → 启动自车控制器 → 初始化无人机位置 → 返回观测
2. `step(action)`: 将无人机移动到选定候选视角 → 推进仿真 → 收集观测 → 生成标签 → 返回结果
3. `observe()`: 收集 ego 状态、其他车辆状态、无人机状态、路网信息、候选视角

**自车控制支持两种模式：**
- `autopilot`: 使用 CARLA Traffic Manager（基于规则，无视觉）
- `route_follow`: 使用自定义 `RouteFollowingDriver`（基于规则，无视觉）

---

#### `ego_driver.py` — 自车控制器 (`RouteFollowingDriver`)

**这是项目中自车自动驾驶的核心模块。关键结论：非视觉输入。**

```
输入: CARLA actor + world (不是图像！)
输出: carla.VehicleControl (steer, throttle, brake)
```

**算法原理（纯规则驱动）：**

```
┌─────────────────────────────────────────────────┐
│            RouteFollowingDriver 控制流程          │
│                                                  │
│  1. 路由目标计算 (_route_target_point)            │
│     ├── 从 CARLA 路网获取当前车道 waypoint         │
│     ├── 沿路网向前 lookahead_m (默认10m) 取候选点  │
│     └── 选航向偏差最小 + 横向偏移最小的点          │
│                                                  │
│  2. 前方车辆检测 (_vehicle_ahead_clearance)       │
│     ├── 遍历场景中所有车辆                        │
│     ├── 筛选同车道前方 35m 内、横向偏差 < 4.5m    │
│     └── 返回最近车辆的净空距离 (clearance)         │
│                                                  │
│  3. 转向控制 (steer)                             │
│     └── 基于路由目标的角度偏差 → 比例控制 + 平滑   │
│                                                  │
│  4. 速度控制 (throttle/brake)                    │
│     ├── 目标速度 = 基础速度 × 障碍因子            │
│     │              × 弯道因子 × 车道因子           │
│     ├── 路口限速: junction_speed_mps (3.0 m/s)   │
│     ├── 跟车减速: 净空 < follow_distance_m 时减速 │
│     ├── 紧急制动: 净空 < emergency_clearance_m    │
│     ├── 红灯停车: is_at_traffic_light() 检测      │
│     └── PID-like 油门 + 条件刹车                 │
└─────────────────────────────────────────────────┘
```

**关键参数（`EgoDriveConfig`）：**
| 参数 | 默认值 | 说明 |
|------|--------|------|
| target_speed_mps | 8.0 | 目标速度 (m/s) ≈ 28.8 km/h |
| lookahead_m | 10.0 | 前方预瞄距离 |
| min_clearance_m | 7.0 | 最小安全净空 |
| brake_distance_m | 12.0 | 开始减速距离 |
| junction_speed_mps | 3.0 | 路口限速 |
| follow_distance_m | 16.0 | 跟车距离 |
| steer_smoothing | 0.70 | 转向平滑系数 |

**结论：完全不依赖摄像头/视觉输入。** 所有决策基于：
- CARLA 路网 API (waypoint 查询)
- Actor 位置/速度 (直接 API 获取，非视觉)
- 红绿灯状态 API
- 数值计算（距离、角度）

---

#### `scenario.py` — 场景配置

`ScenarioConfig` 数据类，所有场景参数通过 JSON 文件定义。

当前默认场景 [town10hd_v0.json](../configs/scenarios/town10hd_v0.json) 的核心参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| 自车 | Tesla Model 3 | 自己生成的车辆 |
| 自车控制模式 | route_follow | 使用 RouteFollowingDriver |
| 环境交通车 | 2 辆 | 由 CARLA Traffic Manager 控制 |
| 无人机 | SimpleFlight ×1 | 带 RGB+深度相机 |
| 运行时 | 30 秒 | 每步 0.5 秒 |
| 数据采集 | 热点半径 70m | 最小间隔 0.5 秒 |

**7 个无人机候选视角（相对自车坐标，x前/y左/z上）：**

| 名称 | 偏移量 (x, y, z) | 权重 |
|------|------------------|------|
| front_high | (18, 0, 10) | 1.0 |
| front_left | (15, 8, 10) | 1.0 |
| front_right | (15, -8, 10) | 1.0 |
| top | (0, 0, 22) | 0.8 |
| rear_high | (-10, 0, 12) | 0.7 |
| left_high | (0, 14, 12) | 0.8 |
| right_high | (0, -14, 12) | 0.8 |

---

#### `policies.py` — 无人机视角选择策略

决定无人机在每个时刻飞到哪个候选视角。**全部是启发式策略，不使用学习模型。**

| 策略 | 名称 | 逻辑 |
|------|------|------|
| FixedPolicy | `fixed` | 始终选择固定索引的候选点 |
| RandomPolicy | `random` | 随机选择 |
| EgoFollowPolicy | `follow` | 偏好前方、高处的候选点 |
| IntersectionCenterPolicy | `intersection` | 在路口时偏好中心候选点 |
| OcclusionHeuristicPolicy | `heuristic` | 综合高度、交通密度、路口等因素 |

---

#### `labels.py` — 真值标签生成

为训练世界模型提供标签数据。

- **车辆轨迹预测**: 使用**常速度模型 (Constant Velocity)** 向前投影 `future_horizon_sec` 秒
- **风险代理指标**: 自车周围 25m 内其他车辆的距离加权和

**注意: 这是 V0 脚手架，标签使用的是最简单的常速度假设，未来需替换为更复杂的世界模型标签。**

---

#### `sensors.py` — 传感器管理

| 传感器组件 | 模拟器 | 输出 |
|-----------|--------|------|
| `VehicleSensorRig` | CARLA | RGB 图像 (640×360) + 深度图 |
| `UAVSensorRig` | AirSim | RGB 图像 (1280×960) + 深度图 |

- 车载相机安装在车辆上方 (x=1.5m, z=1.8m)，FOV=90°
- 无人机使用 `front_center` 相机（AirSim 默认）
- 深度图被归一化到 0-80m 范围并可视化为灰度图

---

#### `traffic.py` — 环境交通车生成

- 从 CARLA spawn points 生成指定数量的交通车辆（role_name="task_traffic"）
- 全部由 CARLA Traffic Manager 控制（基于规则导航，无视觉）
- 不使用自定义控制器

---

#### `control.py` — 无人机手动控制 (`UAVCommandController`)

提供对 AirSim 无人机的完整手动控制接口：
- `takeoff()` / `hover()` — 起降
- `move_relative()` — 相对移动（上下）
- `move_body_relative()` — 机体坐标系移动（前后左右）
- `goto()` — 飞到绝对坐标
- `rotate_yaw()` — 旋转
- `move_velocity()` / `move_body_velocity()` — 速度控制

---

#### `task_app.py` — 交互式任务应用 (`ActiveUAVTaskApp`)

**最完整的应用入口**，集成了所有模块：

- 连接 CARLA + AirSim → 生成自车 + 环境交通车 → 安装传感器 → 启动采样器
- 支持交互式命令（键盘输入控制无人机）
- 支持自动 UAV 巡逻模式
- 支持车载/机载传感器数据自动采集和保存
- 支持实时 OpenCV 可视化窗口（需设置 `CARLAAIR_ENABLE_VIEWER=1`）

---

#### `recorder.py` — 数据记录

`EpisodeRecorder` 将每步数据保存为 JSON，格式如下：

```json
{
  "meta": { "scenario": {...}, "policy": "heuristic" },
  "steps": [
    {
      "step": 1,
      "action": 2,
      "observation": { "ego": {...}, "vehicles": [...], "drone": {...} },
      "label": { "risk_proxy": 0.05, "vehicles": [...] },
      "info": { "candidate_count": 7 }
    }
  ]
}
```

---

### 2.3 入口脚本说明

| 脚本 | 用途 | 使用场景 |
|------|------|---------|
| `run_active_world.py` | 用指定策略自动运行一个 episode | 自动化实验、批量测试 |
| `run_uav_task.py` | 交互式手动控制无人机 | 手动调试、数据采集 |
| `collect_dataset.py` | 批量采集多个 episode | 构建数据集 |
| `evaluate_policy.py` | 统计已录制 episode 的风险指标 | 离线评估 |
| `follow_1uav-1car.py` | 1架无人机平滑跟随1辆车 | 无人机编队演示 |
| `follow-4uav-1car.py` | 4架无人机绕车圆形编队 | 多无人机编队演示 |

---

### 2.4 两个独立跟随脚本

#### `follow_1uav-1car.py` — 1跟1编队

- 1 架无人机（SimpleFlight）平滑跟随 1 辆地面车辆后方上方
- 无人机位置: 车后 7m + 上方 9m（参数可调）
- 使用速度估计器 (`VehicleMotionEstimator`) 做前向预测减少滞后
- 速度低通滤波平滑 (α=0.65)

#### `follow-4uav-1car.py` — 4机绕车编队

- 4 架无人机（Drone1~Drone4）围绕 1 辆地面车辆做圆形轨道飞行
- 轨道半径 13m，高度 12m，角速度 14°/s
- 4 机相位差 90°，均匀分布在圆周上
- 无人机机头朝向车辆中心（可切换为切线方向）
- 包含前馈速度 + 位置误差反馈

---

## 三、地面车辆自动驾驶算法总结

### 3.1 车辆分类

| 类别 | role_name | 数量(默认) | 控制器 | 视觉输入 |
|------|-----------|-----------|--------|---------|
| 自车 (ego) | `"ego"` | 1 | RouteFollowingDriver 或 Traffic Manager | **无** |
| 环境交通车 | `"task_traffic"` | 2 | CARLA Traffic Manager | **无** |

### 3.2 自车控制模式

自车支持两种控制模式（在场景 JSON 的 `ego_control_mode` 中配置）：

#### 模式一：`autopilot` (CARLA Traffic Manager)
- CARLA 内置的交通管理系统
- 基于路网拓扑的路径规划
- 规则驱动的加减速和转向
- 不支持视觉输入

#### 模式二：`route_follow` (RouteFollowingDriver) — **当前默认**
- 自定义 Python 控制器
- 基于 CARLA 路网 API 获取前方 waypoint 做路径跟踪
- 基于距离检测前方车辆做跟车/制动
- 基于 API 检测红绿灯状态
- 弯道自动减速
- **完全不使用摄像头图像**

### 3.3 关键结论

> **目前地面车辆的自动驾驶模型不是基于视觉输入的。**
>
> 所有控制决策都来自：
> 1. CARLA 路网 API（获取道路几何信息）
> 2. Actor 位置/速度 API（获取其他车辆状态，非视觉感知）
> 3. 红绿灯 API
> 4. 纯数值计算（距离、角度、PID控制）

这是一个**经典的规则驱动控制器**，相当于拥有"上帝视角"（直接获取仿真器的真值数据），而非从摄像头图像中感知环境。

---

## 四、无人机情况

当前默认配置包含 **1 架无人机**（AirSim SimpleFlight 模型），有两个独立脚本分别演示了 1 架和 4 架无人机的编队场景。

| 脚本 | 无人机数 | 车辆数 | 编队方式 |
|------|---------|--------|---------|
| `run_active_world.py` / `run_uav_task.py` | 1 | 1 自车 + 2 交通车 | 策略选择候选视角 |
| `follow_1uav-1car.py` | 1 | 1 | 车后上方跟随 |
| `follow-4uav-1car.py` | 4 | 1 | 圆形轨道编队 |

---

## 五、数据流与坐标系

### 5.1 坐标转换链路

```
场景 JSON (候选视角偏移量, 相对自车)
    │
    ▼ local_candidate_to_world()
CARLA 世界坐标 (z-up, 左手系)
    │
    ▼ carla_to_airsim_ned()
AirSim NED 坐标 (z-down)
    │
    ▼ move_uav_to() → simSetVehiclePose()
无人机移动到目标位姿
```

### 5.2 数据采集流程

```
仿真运行（每步 0.5s）
    │
    ├─→ 热点检测 (ego/交通车是否在 70m 半径内)
    ├─→ 采样间隔门控 (0.5s 最小间隔)
    │
    ├─→ UAV 传感器快照 → samples/step_N/uav/rgb.png + depth.png
    ├─→ 车载传感器快照 → samples/step_N/vehicle_X/rgb.png + depth.png
    │
    ├─→ 车辆状态 (位置/速度/朝向)
    ├─→ 标签生成 (常速度轨迹预测 + 风险代理)
    │
    └─→ 写入 episode.json
```

---

## 六、V0 脚手架说明

根据 README 和代码注释，当前是 **V0 研究脚手架（scaffold）**，存在以下限制：

1. **标签简单**: 使用常速度模型做轨迹预测，未使用学习模型
2. **策略启发式**: 所有 UAV 策略都是手工规则，无学习
3. **不生成新交通流**: 依赖 CARLA 已有的 Traffic Manager
4. **地面车辆控制规则驱动**: 没有端到端视觉模型
5. **models/ 文件夹为空**: 尚未集成任何神经网络模型

这些是下一步可以扩展的方向。
