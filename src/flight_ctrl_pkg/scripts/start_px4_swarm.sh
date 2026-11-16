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
else
    # PX4 v1.17+ rcS 移除 . px4-rc.params，改由 PX4_PARAM_* 环境变量注入；同时保留 PATH 兼容 v1.15.4
    while IFS= read -r _pline; do
        _pline="${_pline%%#*}"
        # shellcheck disable=SC2086
        set -- $_pline
        if [ "${1:-}" = "param" ] && [ "${2:-}" = "set" ] && [ -n "${3:-}" ] && [ -n "${4:-}" ]; then
            export "PX4_PARAM_${3}=${4}"
        fi
    done < "$SCRIPT_DIR/px4-rc.params"
fi

WORLD_FILE="$PKG_ROOT/sim/worlds/moving_target_arena.sdf"
MODEL_DIR="$PKG_ROOT/sim/models"
export PX4_GZ_MODELS="${PX4_GZ_MODELS:-$PX4_AUTOPILOT_DIR/Tools/simulation/gz/models}"
export PX4_GZ_WORLDS="${PX4_GZ_WORLDS:-$PX4_AUTOPILOT_DIR/Tools/simulation/gz/worlds}"

export GZ_SIM_RESOURCE_PATH="${MODEL_DIR}:${PX4_GZ_MODELS}:${PX4_GZ_WORLDS}:${GZ_SIM_RESOURCE_PATH:-}"
export PATH="${SCRIPT_DIR}:${PATH}"
if [ -f "$SCRIPT_DIR/px4-rc.params" ]; then
    export PX4_RC_PARAMS_POST="$SCRIPT_DIR/px4-rc.params"
fi

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
# 等待 Gazebo 世界就绪再生成无人机模型
for _w in $(seq 1 30); do
    if gz topic -l 2>/dev/null | grep -q "/world/moving_target_arena/clock"; then
        break
    fi
    sleep 0.5
done

# 3. 通过 GZ_SIM_RESOURCE_PATH 在 Gazebo 中生成机型，并启动 PX4 SITL 挂载实例（兼容 v1.15/v1.17/main 且不修改 PX4 源码树）
SIM_MODEL="gz_x500"
MODEL_RES_NAME="x500"
if [ "$WITH_CAMERA" = "1" ]; then
    SIM_MODEL="gz_x500_down_cam"
    MODEL_RES_NAME="x500_down_cam"
fi

POSES=("0,0,0.5,0,0,0" "0,2,0.5,0,0,0" "0,4,0.5,0,0,0")

echo "[3/4] 启动 ${NUM_DRONES} 架 PX4 SITL (${SIM_MODEL})..."
mkdir -p /tmp/px4_sitl_instances
for ((i=0; i<NUM_DRONES; i++)); do
    INST_DIR="/tmp/px4_sitl_instances/drone_$((i+1))"
    mkdir -p "$INST_DIR"
    DRONE_NAME="drone_$((i+1))"
    GZ_MODEL_INST="${MODEL_RES_NAME}_${i}"
    UXRCE_NS=""
    if [ "$i" -gt 0 ]; then
        UXRCE_NS="px4_${i}"
    fi

    IFS=',' read -r px py pz pr pp pyaw <<< "${POSES[$i]}"
    SDF_WRAP="<sdf version='1.9'><include><uri>model://${MODEL_RES_NAME}</uri><pose>${px:-0} ${py:-0} ${pz:-0.5} ${pr:-0} ${pp:-0} ${pyaw:-0}</pose></include></sdf>"
    gz service -s "/world/moving_target_arena/create" \
        --reqtype gz.msgs.EntityFactory \
        --reptype gz.msgs.Boolean \
        --timeout 5000 \
        --req "name: \"${GZ_MODEL_INST}\", allow_renaming: false, sdf: \"${SDF_WRAP}\"" >/dev/null 2>&1 || true

    PX4_GZ_STANDALONE=1 \
    PX4_SYS_AUTOSTART=4001 \
    PX4_SIM_MODEL="gz_x500" \
    PX4_GZ_WORLD="moving_target_arena" \
    PX4_GZ_MODEL_NAME="${GZ_MODEL_INST}" \
    PX4_UXRCE_DDS_NS="${UXRCE_NS}" \
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
