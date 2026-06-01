#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_ROOT="$(cd "${CODE_ROOT}/.." && pwd)"
CARLA_BIN="${CARLAAIR_BIN:-${DIST_ROOT}/CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping}"
CONDA_PYTHON="${CARLAAIR_PYTHON:-${HOME}/miniconda3/envs/carlaAir/bin/python3}"

MAP_NAME="Town10HD"
RESOLUTION="800x600"
QUALITY="Low"
CARLA_PORT="2000"
LOG_FILE="${DIST_ROOT}/CarlaAir_vehicle_eval_$(date +%Y%m%d_%H%M%S).log"
NO_CITY_TRAFFIC=1
DISPLAY_NAME="${CARLAAIR_DISPLAY:-localhost:10.0}"
XAUTHORITY_PATH="${CARLAAIR_XAUTHORITY:-${HOME}/.Xauthority}"

usage() {
    cat <<EOF
Usage: $0 [--map Town10HD] [--res 800x600] [--quality Low] [--port 2000] [--log PATH] [--display DISPLAY] [--xauthority PATH] [--no-city-traffic]

Launch CarlaAir for vehicle-only clean evaluation. This writes AirSim SimMode=Car
and uses the MobaX X11 display by default to avoid SimpleFlight/ComputerVision
and offscreen startup crashes seen during vehicle-only TCP/YOLO runs. Run the
evaluation command soon after CARLA_READY=1.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --map)
            shift; MAP_NAME="$1" ;;
        --res)
            shift; RESOLUTION="$1" ;;
        --quality)
            shift; QUALITY="$1" ;;
        --port)
            shift; CARLA_PORT="$1" ;;
        --log)
            shift; LOG_FILE="$1" ;;
        --display)
            shift; DISPLAY_NAME="$1" ;;
        --xauthority)
            shift; XAUTHORITY_PATH="$1" ;;
        --no-city-traffic)
            NO_CITY_TRAFFIC=1 ;;
        --help|-h)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1 ;;
    esac
    shift
done

RES_X="${RESOLUTION%%x*}"
RES_Y="${RESOLUTION##*x}"

kill_matching() {
    local pattern="$1"
    local signal="${2:-TERM}"
    while read -r pid; do
        [ -n "${pid}" ] && kill "-${signal}" "${pid}" 2>/dev/null || true
    done < <(ps -eo pid,args | awk -v pat="${pattern}" '$0 ~ pat && $0 !~ /awk/ {print $1}')
}

write_vehicle_airsim_settings() {
    mkdir -p "${HOME}/Documents/AirSim"
    if [ -f "${HOME}/Documents/AirSim/settings.json" ]; then
        cp -a "${HOME}/Documents/AirSim/settings.json" \
            "${HOME}/Documents/AirSim/settings.before_vehicle_eval_$(date +%Y%m%d_%H%M%S).json"
    fi
    cat > "${HOME}/Documents/AirSim/settings.json" <<'JSON'
{
  "SettingsVersion": 1.2,
  "SimMode": "Car",
  "Vehicles": {
    "PhysXCar": {
      "VehicleType": "PhysXCar",
      "AutoCreate": true
    }
  }
}
JSON
}

wait_for_carla() {
    for _ in $(seq 1 60); do
        if "${CONDA_PYTHON}" - <<PY >/tmp/carla_vehicle_eval_probe.out 2>/tmp/carla_vehicle_eval_probe.err
import carla
client = carla.Client("localhost", ${CARLA_PORT})
client.set_timeout(5.0)
world = client.get_world()
print(world.get_map().name)
PY
        then
            cat /tmp/carla_vehicle_eval_probe.out
            return 0
        fi
        if [ -n "${CARLA_PID:-}" ] && ! kill -0 "${CARLA_PID}" 2>/dev/null; then
            echo "CARLA process exited before RPC became ready." >&2
            return 1
        fi
        sleep 2
    done
    cat /tmp/carla_vehicle_eval_probe.err >&2 || true
    return 1
}

if [ "${NO_CITY_TRAFFIC}" -eq 1 ]; then
    kill_matching "examples/auto_traffic.py" TERM
fi
kill_matching "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" TERM
sleep 2
if [ "${NO_CITY_TRAFFIC}" -eq 1 ]; then
    kill_matching "examples/auto_traffic.py" KILL
fi
kill_matching "CarlaUE4/Binaries/Linux/CarlaUE4-Linux-Shipping" KILL

write_vehicle_airsim_settings

export DISPLAY="${DISPLAY_NAME}"
export XAUTHORITY="${XAUTHORITY_PATH}"
unset WAYLAND_DISPLAY
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json}"
export __GLX_VENDOR_LIBRARY_NAME="${__GLX_VENDOR_LIBRARY_NAME:-nvidia}"

"${CARLA_BIN}" CarlaUE4 "${MAP_NAME}" \
    -windowed \
    -ResX="${RES_X}" -ResY="${RES_Y}" \
    -carla-rpc-port="${CARLA_PORT}" \
    -quality-level="${QUALITY}" \
    -TexturePoolSize=2048 \
    -unattended -nosound -UseVSync \
    -stdout -FullStdOutLogOutput \
    > "${LOG_FILE}" 2>&1 &

CARLA_PID=$!
echo "CARLA_PID=${CARLA_PID}"
echo "CARLA_LOG=${LOG_FILE}"

if ! wait_for_carla; then
    tail -120 "${LOG_FILE}" >&2 || true
    exit 1
fi

echo "CARLA_READY=1"
echo "CITY_TRAFFIC=disabled"
echo "DISPLAY=${DISPLAY}"
echo "XAUTHORITY=${XAUTHORITY}"
