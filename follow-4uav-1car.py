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

AIRSIM_PORT = 41451
TRAFFIC_MANAGER_PORT = int(os.environ.get("CARLAAIR_TASK_TRAFFIC_MANAGER_PORT", "8001"))

VEHICLE_BLUEPRINT = "vehicle.tesla.model3"

# 这里假设 AirSim / CarlaAir 中已经存在这 4 架无人机
# 如果你的 settings.json 里不是这个名字，需要改成对应名字
DRONE_NAMES = ["Drone1", "Drone2", "Drone3", "Drone4"]

DESTROY_OLD_VEHICLES = True

# 车辆自动驾驶速度控制
# 正数：比限速慢；负数：比限速快
VEHICLE_SPEED_PERCENTAGE = 45.0

# ============================================================
# 4 架无人机绕车伴飞参数
# ============================================================

ORBIT_RADIUS = 13.0          # 无人机绕车半径，单位米
ORBIT_HEIGHT = 12.0          # 无人机相对车辆高度，单位米
ORBIT_ANGULAR_SPEED_DEG = 14.0   # 绕车角速度，单位 度/秒

# 无人机跟随控制参数
UPDATE_INTERVAL = 0.05
MAX_DRONE_SPEED = 38.0
KP_DRONE_POSITION = 1.2

# 越大越平滑，但响应越慢
DRONE_VELOCITY_SMOOTH = 0.45

# 是否让无人机镜头/机头朝向车辆中心
FACE_VEHICLE_CENTER = True

# 用车辆估计速度做一点预测，减少无人机滞后
CENTER_LOOKAHEAD_SEC = 0.25


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
    print("    AirSim connected")

    return air_client


def setup_drones(air_client):
    """
    启用 4 架无人机 API 控制。
    如果这里报错，基本说明 CarlaAir 当前只启动了 1 架无人机，
    或者 settings.json 里的无人机名字不是 Drone1~Drone4。
    """
    print("[3] Setting up drones...")

    available = []

    for name in DRONE_NAMES:
        try:
            air_client.enableApiControl(True, vehicle_name=name)
            air_client.armDisarm(True, vehicle_name=name)
            state = air_client.getMultirotorState(vehicle_name=name)
            pos = state.kinematics_estimated.position
            print(
                f"    {name}: OK, "
                f"pos=({pos.x_val:.2f}, {pos.y_val:.2f}, {pos.z_val:.2f})"
            )
            available.append(name)
        except Exception as e:
            print(f"    {name}: FAILED -> {e}")

    if len(available) != len(DRONE_NAMES):
        raise RuntimeError(
            "没有成功连接到 4 架无人机。\n"
            "请确认 AirSim / CarlaAir 的 settings.json 中已经配置 Drone1、Drone2、Drone3、Drone4。\n"
            "如果当前 CarlaAir 只启动了 1 架无人机，这个脚本不能凭空生成另外 3 架。"
        )

    print("    All drones are ready.")


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
# 车辆运动估计器：用位置差估计速度
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
    print("[4] Calibrating CARLA <-> AirSim coordinate offset...")

    time.sleep(0.5)

    drone_actor = find_drone_actor_in_carla(world)

    if drone_actor is None:
        print("    Warning: cannot find drone actor in CARLA.")
        print("    Use zero offset: AirSim(x,y,z) = CARLA(x,y,-z)")
        return 0.0, 0.0, 0.0

    carla_loc = drone_actor.get_location()

    # 用 Drone1 的 AirSim 位置和 CARLA 中找到的 drone actor 做坐标偏移估计
    air_pos = air_client.getMultirotorState(
        vehicle_name=DRONE_NAMES[0]
    ).kinematics_estimated.position

    ox = air_pos.x_val - carla_loc.x
    oy = air_pos.y_val - carla_loc.y
    oz = air_pos.z_val - (-carla_loc.z)

    print(f"    CARLA drone location: ({carla_loc.x:.3f}, {carla_loc.y:.3f}, {carla_loc.z:.3f})")
    print(f"    AirSim {DRONE_NAMES[0]} position: ({air_pos.x_val:.3f}, {air_pos.y_val:.3f}, {air_pos.z_val:.3f})")
    print(f"    Offset: ox={ox:.3f}, oy={oy:.3f}, oz={oz:.3f}")

    return ox, oy, oz


def carla_to_airsim_ned(carla_x, carla_y, carla_z, ox, oy, oz):
    ned_x = carla_x + ox
    ned_y = carla_y + oy
    ned_z = -carla_z + oz
    return ned_x, ned_y, ned_z


def carla_vel_to_airsim_ned_vel(vx, vy, vz):
    """
    CARLA z 向上，AirSim NED z 向下。
    因此速度 z 方向也要取反。
    """
    return vx, vy, -vz


# ============================================================
# 车辆生成与自动驾驶
# ============================================================

def spawn_vehicle(world):
    print("[5] Spawning vehicle...")

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
    print("[7] Starting vehicle autopilot...")

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
# 四无人机绕车目标计算
# ============================================================

def get_orbit_target_for_drone(
    vehicle,
    motion_info,
    drone_index,
    start_time,
    ox,
    oy,
    oz
):
    """
    计算第 drone_index 架无人机的绕车目标点。

    4 架无人机相位差 90 度：
        Drone1: 0 度
        Drone2: 90 度
        Drone3: 180 度
        Drone4: 270 度

    所有无人机围绕车辆中心旋转。
    """
    now = time.time()
    elapsed = now - start_time

    tf = vehicle.get_transform()
    loc = tf.location

    # 车辆中心预测
    center_x = loc.x + CENTER_LOOKAHEAD_SEC * motion_info["vx"]
    center_y = loc.y + CENTER_LOOKAHEAD_SEC * motion_info["vy"]
    center_z = loc.z + CENTER_LOOKAHEAD_SEC * motion_info["vz"]

    phase_deg = drone_index * 90.0
    orbit_angle_deg = phase_deg + ORBIT_ANGULAR_SPEED_DEG * elapsed
    orbit_angle_rad = math.radians(orbit_angle_deg)

    # CARLA 平面中的绕车圆轨迹
    target_carla_x = center_x + ORBIT_RADIUS * math.cos(orbit_angle_rad)
    target_carla_y = center_y + ORBIT_RADIUS * math.sin(orbit_angle_rad)
    target_carla_z = center_z + ORBIT_HEIGHT

    # 绕圈轨迹的切向速度，加上车辆中心速度作为前馈
    omega = math.radians(ORBIT_ANGULAR_SPEED_DEG)

    target_vel_carla_x = motion_info["vx"] - ORBIT_RADIUS * math.sin(orbit_angle_rad) * omega
    target_vel_carla_y = motion_info["vy"] + ORBIT_RADIUS * math.cos(orbit_angle_rad) * omega
    target_vel_carla_z = motion_info["vz"]

    # 转为 AirSim NED 坐标
    ned_x, ned_y, ned_z = carla_to_airsim_ned(
        target_carla_x,
        target_carla_y,
        target_carla_z,
        ox,
        oy,
        oz
    )

    ned_vx, ned_vy, ned_vz = carla_vel_to_airsim_ned_vel(
        target_vel_carla_x,
        target_vel_carla_y,
        target_vel_carla_z
    )

    if FACE_VEHICLE_CENTER:
        # 让无人机朝向车辆中心
        yaw_deg = math.degrees(
            math.atan2(
                center_y - target_carla_y,
                center_x - target_carla_x
            )
        )
    else:
        # 让无人机朝向切向方向
        yaw_deg = orbit_angle_deg + 90.0

    return {
        "x": ned_x,
        "y": ned_y,
        "z": ned_z,
        "vx": ned_vx,
        "vy": ned_vy,
        "vz": ned_vz,
        "yaw_deg": yaw_deg,
        "target_carla_x": target_carla_x,
        "target_carla_y": target_carla_y,
        "target_carla_z": target_carla_z
    }


# ============================================================
# 无人机控制
# ============================================================

def set_drone_pose_teleport(air_client, drone_name, x, y, z, yaw_deg):
    """
    初始化时使用 teleport，把无人机放到初始绕车队形中。
    后续主循环使用速度控制。
    """
    pose = airsim.Pose(
        airsim.Vector3r(x, y, z),
        airsim.to_quaternion(
            math.radians(-20.0),
            0.0,
            math.radians(yaw_deg)
        )
    )

    air_client.simSetVehiclePose(
        pose,
        True,
        vehicle_name=drone_name
    )


def takeoff_all_drones(air_client):
    print("[6] Taking off all drones...")

    futures = []

    for name in DRONE_NAMES:
        try:
            future = air_client.takeoffAsync(
                timeout_sec=8,
                vehicle_name=name
            )
        except TypeError:
            future = air_client.takeoffAsync(vehicle_name=name)

        futures.append((name, future))

    for name, future in futures:
        try:
            future.join()
            print(f"    {name}: takeoff OK")
        except Exception as e:
            print(f"    {name}: takeoff join failed -> {e}")

    time.sleep(1.0)


def initial_place_drones(air_client, vehicle, motion_info, start_time, ox, oy, oz):
    """
    起飞后，把 4 架无人机摆成绕车圆形队形。
    """
    print("[6.5] Placing drones around vehicle...")

    for i, name in enumerate(DRONE_NAMES):
        target = get_orbit_target_for_drone(
            vehicle=vehicle,
            motion_info=motion_info,
            drone_index=i,
            start_time=start_time,
            ox=ox,
            oy=oy,
            oz=oz
        )

        set_drone_pose_teleport(
            air_client=air_client,
            drone_name=name,
            x=target["x"],
            y=target["y"],
            z=target["z"],
            yaw_deg=target["yaw_deg"]
        )

        print(
            f"    {name}: placed at NED="
            f"({target['x']:.1f}, {target['y']:.1f}, {target['z']:.1f})"
        )

    time.sleep(1.0)


def follow_drone_by_velocity(air_client, drone_name, target, drone_state):
    """
    对单架无人机进行速度控制。
    target 包含目标位置和目标速度前馈。
    """
    state = air_client.getMultirotorState(vehicle_name=drone_name)
    pos = state.kinematics_estimated.position

    dx = target["x"] - pos.x_val
    dy = target["y"] - pos.y_val
    dz = target["z"] - pos.z_val

    # 前馈速度 + 位置误差反馈
    raw_vx = target["vx"] + KP_DRONE_POSITION * dx
    raw_vy = target["vy"] + KP_DRONE_POSITION * dy
    raw_vz = target["vz"] + KP_DRONE_POSITION * dz

    raw_vx = clamp(raw_vx, -MAX_DRONE_SPEED, MAX_DRONE_SPEED)
    raw_vy = clamp(raw_vy, -MAX_DRONE_SPEED, MAX_DRONE_SPEED)
    raw_vz = clamp(raw_vz, -MAX_DRONE_SPEED, MAX_DRONE_SPEED)

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
        yaw_mode=airsim.YawMode(False, float(target["yaw_deg"])),
        vehicle_name=drone_name
    )


def print_status(vehicle, motion_info, air_client, last_print_time):
    now = time.time()

    if now - last_print_time < 1.0:
        return last_print_time

    loc = vehicle.get_location()

    status = [
        f"Vehicle=({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})",
        f"speed_est={motion_info['speed_kmh']:.1f} km/h"
    ]

    for name in DRONE_NAMES:
        try:
            pos = air_client.getMultirotorState(
                vehicle_name=name
            ).kinematics_estimated.position

            status.append(
                f"{name}=({pos.x_val:.1f}, {pos.y_val:.1f}, {pos.z_val:.1f})"
            )
        except Exception:
            status.append(f"{name}=ERR")

    print(" | ".join(status))

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
        setup_drones(air_client)

        ox, oy, oz = calibrate_offset(world, air_client)

        vehicle = spawn_vehicle(world)

        if sync_mode:
            world.tick()

        motion_estimator = VehicleMotionEstimator(alpha=0.35)
        motion_info = motion_estimator.update(vehicle)

        # 起飞并摆成四机绕车队形
        takeoff_all_drones(air_client)

        start_time = time.time()

        initial_place_drones(
            air_client=air_client,
            vehicle=vehicle,
            motion_info=motion_info,
            start_time=start_time,
            ox=ox,
            oy=oy,
            oz=oz
        )

        # 车辆启动自动驾驶
        start_vehicle_autopilot(client, vehicle)

        print("")
        print("============================================================")
        print("4 drones orbiting around autopilot vehicle started.")
        print(f"DRONE_NAMES = {DRONE_NAMES}")
        print(f"ORBIT_RADIUS = {ORBIT_RADIUS}")
        print(f"ORBIT_HEIGHT = {ORBIT_HEIGHT}")
        print(f"ORBIT_ANGULAR_SPEED_DEG = {ORBIT_ANGULAR_SPEED_DEG}")
        print(f"UPDATE_INTERVAL = {UPDATE_INTERVAL}")
        print(f"MAX_DRONE_SPEED = {MAX_DRONE_SPEED}")
        print("Press Ctrl+C to stop.")
        print("============================================================")
        print("")

        drone_states = {
            name: {}
            for name in DRONE_NAMES
        }

        last_print_time = 0.0

        while True:
            if sync_mode:
                world.tick()

            # 更新车辆运动估计
            motion_info = motion_estimator.update(vehicle)

            # 控制 4 架无人机绕车
            for i, name in enumerate(DRONE_NAMES):
                target = get_orbit_target_for_drone(
                    vehicle=vehicle,
                    motion_info=motion_info,
                    drone_index=i,
                    start_time=start_time,
                    ox=ox,
                    oy=oy,
                    oz=oz
                )

                follow_drone_by_velocity(
                    air_client=air_client,
                    drone_name=name,
                    target=target,
                    drone_state=drone_states[name]
                )

            last_print_time = print_status(
                vehicle=vehicle,
                motion_info=motion_info,
                air_client=air_client,
                last_print_time=last_print_time
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
            for name in DRONE_NAMES:
                try:
                    air_client.hoverAsync(vehicle_name=name)
                except Exception:
                    pass

            time.sleep(0.5)

            for name in DRONE_NAMES:
                try:
                    air_client.armDisarm(False, vehicle_name=name)
                    air_client.enableApiControl(False, vehicle_name=name)
                    print(f"{name}: API control released.")
                except Exception:
                    pass

        print("Done.")


if __name__ == "__main__":
    main()
