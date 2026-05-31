import carla
import airsim
import time
import math
import traceback
import os


# ============================================================
# 基础配置
# ============================================================

CARLA_HOST = "localhost"
CARLA_PORT = 2000

# CarlaAir Windows 版本通常是 41451
AIRSIM_PORT = 41451

TRAFFIC_MANAGER_PORT = int(os.environ.get("CARLAAIR_TASK_TRAFFIC_MANAGER_PORT", "8001"))

VEHICLE_BLUEPRINT = "vehicle.tesla.model3"

# 是否清理旧车辆
DESTROY_OLD_VEHICLES = True

# 车辆自动驾驶速度控制
# 正数：比限速慢；负数：比限速快
VEHICLE_SPEED_PERCENTAGE = 40.0

# 无人机跟随参数
DRONE_HEIGHT = 9.0
DRONE_BACK = 7.0

# 预测补偿，车辆前进速度越快，无人机目标点越提前一点
DRONE_LOOKAHEAD_SEC = 0.35

# 无人机速度控制参数
UPDATE_INTERVAL = 0.05
MAX_DRONE_SPEED = 28.0
KP_DRONE_POSITION = 1.15

# 无人机速度低通滤波，越大越平滑，但响应越慢
DRONE_VELOCITY_SMOOTH = 0.65


# ============================================================
# 工具函数
# ============================================================

def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def connect_carla():
    print("[1] Connecting to CARLA...")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(15.0)

    world = client.get_world()
    settings = world.get_settings()

    print(f"    Connected to world: {world.get_map().name}")
    print(f"    synchronous_mode = {settings.synchronous_mode}")
    print(f"    fixed_delta_seconds = {settings.fixed_delta_seconds}")

    return client, world


def connect_airsim():
    print("[2] Connecting to AirSim...")
    air_client = airsim.MultirotorClient(port=AIRSIM_PORT)
    air_client.confirmConnection()
    air_client.enableApiControl(True)
    air_client.armDisarm(True)
    print("    AirSim drone connected")
    return air_client


def cleanup_old_vehicles(world):
    if not DESTROY_OLD_VEHICLES:
        return

    count = 0
    for actor in world.get_actors().filter("vehicle.*"):
        try:
            actor.destroy()
            count += 1
        except Exception:
            pass

    if count > 0:
        print(f"[Cleanup] Destroyed {count} old vehicles")
        time.sleep(0.5)


# ============================================================
# 车辆速度估计器：不用 get_velocity，而是用位置差
# ============================================================

class VehicleMotionEstimator:
    def __init__(self, alpha=0.35):
        self.last_loc = None
        self.last_time = None

        self.speed_kmh = 0.0

        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        self.alpha = alpha

    def update(self, vehicle):
        now = time.time()
        loc = vehicle.get_location()

        if self.last_loc is None:
            self.last_loc = carla.Location(loc.x, loc.y, loc.z)
            self.last_time = now
            return {
                "speed_kmh": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0
            }

        dt = now - self.last_time

        if dt <= 1e-6:
            return {
                "speed_kmh": self.speed_kmh,
                "vx": self.vx,
                "vy": self.vy,
                "vz": self.vz
            }

        dx = loc.x - self.last_loc.x
        dy = loc.y - self.last_loc.y
        dz = loc.z - self.last_loc.z

        raw_vx = dx / dt
        raw_vy = dy / dt
        raw_vz = dz / dt

        raw_speed_mps = math.sqrt(raw_vx ** 2 + raw_vy ** 2 + raw_vz ** 2)
        raw_speed_kmh = raw_speed_mps * 3.6

        # 低通滤波，避免输出抖动
        self.vx = self.alpha * raw_vx + (1.0 - self.alpha) * self.vx
        self.vy = self.alpha * raw_vy + (1.0 - self.alpha) * self.vy
        self.vz = self.alpha * raw_vz + (1.0 - self.alpha) * self.vz
        self.speed_kmh = self.alpha * raw_speed_kmh + (1.0 - self.alpha) * self.speed_kmh

        self.last_loc = carla.Location(loc.x, loc.y, loc.z)
        self.last_time = now

        return {
            "speed_kmh": self.speed_kmh,
            "vx": self.vx,
            "vy": self.vy,
            "vz": self.vz
        }


# ============================================================
# 坐标校准
# ============================================================

def find_drone_actor_in_carla(world):
    keywords = [
        "drone",
        "uav",
        "quad",
        "airsim",
        "multirotor",
        "quadcopter"
    ]

    for actor in world.get_actors():
        type_id = actor.type_id.lower()
        for key in keywords:
            if key in type_id:
                print(f"[Calibration] Found drone actor in CARLA: {actor.type_id}, id={actor.id}")
                return actor

    return None


def calibrate_offset(world, air_client):
    """
    CARLA 坐标：x, y, z，其中 z 向上。
    AirSim NED 坐标：x, y, z，其中 z 向下。

    AirSim z = -CARLA z + offset_z
    """
    print("[3] Calibrating CARLA <-> AirSim coordinate offset...")

    time.sleep(0.5)

    drone_actor = find_drone_actor_in_carla(world)

    if drone_actor is None:
        print("    Warning: cannot find drone actor in CARLA.")
        print("    Use zero offset: AirSim(x,y,z) = CARLA(x,y,-z)")
        return 0.0, 0.0, 0.0

    carla_loc = drone_actor.get_location()
    air_pos = air_client.getMultirotorState().kinematics_estimated.position

    ox = air_pos.x_val - carla_loc.x
    oy = air_pos.y_val - carla_loc.y
    oz = air_pos.z_val - (-carla_loc.z)

    print(f"    CARLA drone location: ({carla_loc.x:.3f}, {carla_loc.y:.3f}, {carla_loc.z:.3f})")
    print(f"    AirSim drone position: ({air_pos.x_val:.3f}, {air_pos.y_val:.3f}, {air_pos.z_val:.3f})")
    print(f"    Offset: ox={ox:.3f}, oy={oy:.3f}, oz={oz:.3f}")

    return ox, oy, oz


def carla_to_airsim_ned(carla_x, carla_y, carla_z, ox, oy, oz):
    ned_x = carla_x + ox
    ned_y = carla_y + oy
    ned_z = -carla_z + oz
    return ned_x, ned_y, ned_z


# ============================================================
# 车辆生成与自动驾驶
# ============================================================

def spawn_vehicle(world):
    print("[4] Spawning vehicle...")

    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find(VEHICLE_BLUEPRINT)

    if vehicle_bp.has_attribute("color"):
        vehicle_bp.set_attribute("color", "255,0,0")

    spawn_points = world.get_map().get_spawn_points()

    vehicle = None
    used_spawn_point = None

    for sp in spawn_points:
        try:
            vehicle = world.spawn_actor(vehicle_bp, sp)
            used_spawn_point = sp
            break
        except RuntimeError:
            continue

    if vehicle is None:
        raise RuntimeError("车辆生成失败，可能所有 spawn point 都被占用了。")

    vehicle.set_simulate_physics(True)

    print(f"    Vehicle spawned: {vehicle.type_id}, id={vehicle.id}")
    print(
        f"    Spawn location: "
        f"({used_spawn_point.location.x:.2f}, "
        f"{used_spawn_point.location.y:.2f}, "
        f"{used_spawn_point.location.z:.2f})"
    )

    return vehicle


def start_vehicle_autopilot(client, vehicle):
    print("[6] Starting vehicle autopilot...")

    tm = client.get_trafficmanager(TRAFFIC_MANAGER_PORT)

    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.global_percentage_speed_difference(VEHICLE_SPEED_PERCENTAGE)

    try:
        tm.set_hybrid_physics_mode(True)
        tm.set_hybrid_physics_radius(70.0)
    except Exception:
        pass

    vehicle.set_autopilot(True, TRAFFIC_MANAGER_PORT)

    try:
        tm.ignore_lights_percentage(vehicle, 100)
        tm.auto_lane_change(vehicle, False)
        tm.distance_to_leading_vehicle(vehicle, 5.0)
    except Exception:
        pass

    print("    Vehicle autopilot enabled")


# ============================================================
# 无人机控制
# ============================================================

def set_drone_pose_teleport(air_client, x, y, z, yaw_deg):
    pose = airsim.Pose(
        airsim.Vector3r(x, y, z),
        airsim.to_quaternion(
            math.radians(-20.0),
            0.0,
            math.radians(yaw_deg)
        )
    )

    air_client.simSetVehiclePose(pose, True)


def get_follow_target(vehicle, motion_info, ox, oy, oz):
    """
    根据车辆位置、朝向和估计速度，计算无人机目标点。

    目标：
    - 车辆后方 DRONE_BACK 米
    - 车辆上方 DRONE_HEIGHT 米
    - 根据车辆估计速度做一点前向预测
    """
    tf = vehicle.get_transform()

    yaw_deg = tf.rotation.yaw
    yaw_rad = math.radians(yaw_deg)

    loc = tf.location

    # 用位置差估计出的速度做预测，不使用 vehicle.get_velocity()
    predicted_x = loc.x + DRONE_LOOKAHEAD_SEC * motion_info["vx"]
    predicted_y = loc.y + DRONE_LOOKAHEAD_SEC * motion_info["vy"]
    predicted_z = loc.z + DRONE_LOOKAHEAD_SEC * motion_info["vz"]

    target_carla_x = predicted_x - DRONE_BACK * math.cos(yaw_rad)
    target_carla_y = predicted_y - DRONE_BACK * math.sin(yaw_rad)
    target_carla_z = predicted_z + DRONE_HEIGHT

    ned_x, ned_y, ned_z = carla_to_airsim_ned(
        target_carla_x,
        target_carla_y,
        target_carla_z,
        ox,
        oy,
        oz
    )

    return ned_x, ned_y, ned_z, yaw_deg


def initial_place_drone(air_client, vehicle, motion_info, ox, oy, oz):
    """
    初始化阶段用 teleport 把无人机放到车后上方。
    后续主循环用速度控制，不再瞬移。
    """
    print("[5] Taking off and placing drone near vehicle...")

    try:
        air_client.takeoffAsync(timeout_sec=8).join()
    except TypeError:
        air_client.takeoffAsync().join()
    except Exception:
        air_client.takeoffAsync()
        time.sleep(2.0)

    time.sleep(1.0)

    x, y, z, yaw = get_follow_target(vehicle, motion_info, ox, oy, oz)

    set_drone_pose_teleport(air_client, x, y, z, yaw)

    time.sleep(1.0)

    print("    Drone initialized behind and above vehicle")


def follow_by_velocity(air_client, target_x, target_y, target_z, yaw_deg, drone_state):
    """
    无人机速度控制跟随。
    """
    state = air_client.getMultirotorState()
    pos = state.kinematics_estimated.position

    dx = target_x - pos.x_val
    dy = target_y - pos.y_val
    dz = target_z - pos.z_val

    raw_vx = clamp(KP_DRONE_POSITION * dx, -MAX_DRONE_SPEED, MAX_DRONE_SPEED)
    raw_vy = clamp(KP_DRONE_POSITION * dy, -MAX_DRONE_SPEED, MAX_DRONE_SPEED)
    raw_vz = clamp(KP_DRONE_POSITION * dz, -MAX_DRONE_SPEED, MAX_DRONE_SPEED)

    prev_vx = drone_state.get("vx", 0.0)
    prev_vy = drone_state.get("vy", 0.0)
    prev_vz = drone_state.get("vz", 0.0)

    vx = DRONE_VELOCITY_SMOOTH * prev_vx + (1.0 - DRONE_VELOCITY_SMOOTH) * raw_vx
    vy = DRONE_VELOCITY_SMOOTH * prev_vy + (1.0 - DRONE_VELOCITY_SMOOTH) * raw_vy
    vz = DRONE_VELOCITY_SMOOTH * prev_vz + (1.0 - DRONE_VELOCITY_SMOOTH) * raw_vz

    drone_state["vx"] = vx
    drone_state["vy"] = vy
    drone_state["vz"] = vz

    duration = UPDATE_INTERVAL * 2.0

    air_client.moveByVelocityAsync(
        float(vx),
        float(vy),
        float(vz),
        duration=float(duration),
        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
        yaw_mode=airsim.YawMode(False, float(yaw_deg))
    )


def print_status(vehicle, motion_info, air_client, last_print_time):
    now = time.time()

    if now - last_print_time < 1.0:
        return last_print_time

    loc = vehicle.get_location()
    drone_pos = air_client.getMultirotorState().kinematics_estimated.position

    print(
        f"Vehicle: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f}), "
        f"speed_est={motion_info['speed_kmh']:.1f} km/h, "
        f"v_est=({motion_info['vx']:.2f}, {motion_info['vy']:.2f}, {motion_info['vz']:.2f}) m/s | "
        f"Drone NED: ({drone_pos.x_val:.1f}, {drone_pos.y_val:.1f}, {drone_pos.z_val:.1f})"
    )

    return now


# ============================================================
# 主程序
# ============================================================

def main():
    vehicle = None
    air_client = None

    try:
        client, world = connect_carla()
        settings = world.get_settings()
        sync_mode = settings.synchronous_mode

        cleanup_old_vehicles(world)

        air_client = connect_airsim()

        ox, oy, oz = calibrate_offset(world, air_client)

        vehicle = spawn_vehicle(world)

        if sync_mode:
            world.tick()

        motion_estimator = VehicleMotionEstimator(alpha=0.35)

        # 初始化估计器
        motion_info = motion_estimator.update(vehicle)

        # 先把无人机放到车辆后上方
        initial_place_drone(air_client, vehicle, motion_info, ox, oy, oz)

        # 然后再启动车辆自动驾驶，避免车先跑掉
        start_vehicle_autopilot(client, vehicle)

        print("")
        print("============================================================")
        print("Autopilot vehicle + smooth drone following started.")
        print("This version estimates vehicle speed from position difference.")
        print(f"DRONE_HEIGHT = {DRONE_HEIGHT}")
        print(f"DRONE_BACK = {DRONE_BACK}")
        print(f"DRONE_LOOKAHEAD_SEC = {DRONE_LOOKAHEAD_SEC}")
        print(f"UPDATE_INTERVAL = {UPDATE_INTERVAL}")
        print(f"MAX_DRONE_SPEED = {MAX_DRONE_SPEED}")
        print(f"KP_DRONE_POSITION = {KP_DRONE_POSITION}")
        print("Press Ctrl+C to stop.")
        print("============================================================")
        print("")

        drone_state = {}
        last_print_time = 0.0

        while True:
            # 同步模式下需要 tick
            if sync_mode:
                world.tick()

            # 1. 更新车辆运动估计
            motion_info = motion_estimator.update(vehicle)

            # 2. 计算无人机跟随目标
            target_x, target_y, target_z, yaw_deg = get_follow_target(
                vehicle,
                motion_info,
                ox,
                oy,
                oz
            )

            # 3. 平滑速度控制无人机
            follow_by_velocity(
                air_client,
                target_x,
                target_y,
                target_z,
                yaw_deg,
                drone_state
            )

            # 4. 打印状态
            last_print_time = print_status(
                vehicle,
                motion_info,
                air_client,
                last_print_time
            )

            time.sleep(UPDATE_INTERVAL)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    except Exception as e:
        print("\nError occurred:")
        print(e)
        traceback.print_exc()

    finally:
        print("\nCleaning up...")

        if vehicle is not None:
            try:
                vehicle.set_autopilot(False)
            except Exception:
                pass

            try:
                vehicle.destroy()
                print("Vehicle destroyed.")
            except Exception:
                pass

        if air_client is not None:
            try:
                air_client.hoverAsync()
                time.sleep(0.5)
            except Exception:
                pass

            try:
                air_client.armDisarm(False)
                air_client.enableApiControl(False)
                print("Drone API control released.")
            except Exception:
                pass

        print("Done.")


if __name__ == "__main__":
    main()
