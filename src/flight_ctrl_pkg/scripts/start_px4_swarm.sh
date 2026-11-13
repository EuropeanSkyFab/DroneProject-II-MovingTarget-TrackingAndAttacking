#!/usr/bin/env bash
# 一键启动 PX4 SITL + Gazebo Harmonic + MicroXRCEAgent（支持单机与三机集群）
# 用法:
#   ./start_px4_swarm.sh [1|3] [headless=1|0] [with_camera=1|0]
# 示例:
#   ./start_px4_swarm.sh 1 1 1   # 无头单机 + 下视相机
#   ./start_px4_swarm.sh 3 1 1   # 无头三机协同 + 三路下视相机

set -euo pipefail

NUM_DRONES="${1:-1}"
HEADLESS_MODE="${2:-1}"
WITH_CAMERA="${3:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

PX4_BIN="${PX4_AUTOPILOT_DIR:-}/build/px4_sitl_default/bin/px4"
if [ -z "${PX4_AUTOPILOT_DIR:-}" ] || [ ! -x "$PX4_BIN" ]; then
    echo "ERROR: 找不到编译好的 PX4 SITL 可执行文件: $PX4_BIN" >&2
    echo "请先设置 PX4_AUTOPILOT_DIR 或执行: make px4_sitl_default" >&2
    exit 1
fi

UXRCE_BIN="$(command -v MicroXRCEAgent 2>/dev/null || true)"
if [ -z "$UXRCE_BIN" ]; then
    echo "ERROR: 找不到 MicroXRCEAgent，请先安装 Micro-XRCE-DDS-Agent 并确保其位于 PATH 中。" >&2
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/px4-rc.params" ]; then
    echo "WARN: 未在 $SCRIPT_DIR 找到 px4-rc.params，PX4 将使用默认参数。" >&2
fi

WORLD_FILE="$PKG_ROOT/sim/worlds/moving_target_arena.sdf"
MODEL_DIR="$PKG_ROOT/sim/models"
PX4_GZ_MODELS="$PX4_AUTOPILOT_DIR/Tools/simulation/gz/models"
PX4_GZ_WORLDS="$PX4_AUTOPILOT_DIR/Tools/simulation/gz/worlds"

export GZ_SIM_RESOURCE_PATH="${MODEL_DIR}:${PX4_GZ_MODELS}:${PX4_GZ_WORLDS}:${GZ_SIM_RESOURCE_PATH:-}"
export PATH="${SCRIPT_DIR}:${PATH}"

PIDS=()
cleanup() {
    echo ""
    echo "正在停止仿真进程..."
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1. 启动 Micro-XRCE-DDS Agent
echo "[1/4] 启动 MicroXRCEAgent (UDP 8888): $UXRCE_BIN"
"$UXRCE_BIN" udp4 -p 8888 &
PIDS+=($!)
sleep 1

# 2. 启动 Gazebo Harmonic 世界（无头模式下显式解除 DISPLAY；若存在 NVIDIA EGL 则绑定硬件加速）
echo "[2/4] 启动 Gazebo moving_target_arena (HEADLESS=${HEADLESS_MODE})..."
if [ "$HEADLESS_MODE" = "1" ]; then
    if [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
        env -u DISPLAY __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json \
            gz sim -r -s "$WORLD_FILE" &
    else
        env -u DISPLAY gz sim -r -s "$WORLD_FILE" &
    fi
else
    gz sim -r "$WORLD_FILE" &
fi
PIDS+=($!)
sleep 3

# 3. 启动 PX4 SITL 实例（自动注入 px4-rc.params 中的安全参数）
SIM_MODEL="gz_x500"
if [ "$WITH_CAMERA" = "1" ]; then
    SIM_MODEL="gz_x500_down_cam"
fi

POSES=("0,0,0.5,0,0,0" "0,2,0.5,0,0,0" "0,4,0.5,0,0,0")

echo "[3/4] 启动 ${NUM_DRONES} 架 PX4 SITL (${SIM_MODEL})..."
mkdir -p /tmp/px4_sitl_instances
for ((i=0; i<NUM_DRONES; i++)); do
    INST_DIR="/tmp/px4_sitl_instances/drone_$((i+1))"
    mkdir -p "$INST_DIR"
    DRONE_NAME="drone_$((i+1))"
    GZ_MODEL_INST="${SIM_MODEL#gz_}_${i}"

    PX4_GZ_STANDALONE=1 \
    PX4_SYS_AUTOSTART=4001 \
    PX4_SIM_MODEL="$SIM_MODEL" \
    PX4_GZ_WORLD="moving_target_arena" \
    PX4_GZ_MODEL_POSE="${POSES[$i]}" \
        "$PX4_BIN" -i "$i" -d "$PX4_AUTOPILOT_DIR/build/px4_sitl_default/etc" -w "$INST_DIR" \
        > "/tmp/px4_${DRONE_NAME}.log" 2>&1 &
    PIDS+=($!)
    sleep 2
done

# 4. 提示：相机传感器由 flight_ctrl_pkg 的 camera_relay_node 直接通过 gz.transport13 桥接
if [ "$NUM_DRONES" -eq 1 ]; then
    LAUNCH_FILE="flight_ctrl_single.launch.py"
else
    LAUNCH_FILE="flight_ctrl_swarm.launch.py"
fi
echo "仿真集群已全部拉起（MicroXRCEAgent + Gazebo + ${NUM_DRONES}x PX4）。按 Ctrl+C 退出。"
echo "请另开终端执行: ros2 launch flight_ctrl_pkg ${LAUNCH_FILE}"
wait
