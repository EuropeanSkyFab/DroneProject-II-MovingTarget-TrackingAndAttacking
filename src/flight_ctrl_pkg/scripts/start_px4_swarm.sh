#!/usr/bin/env bash
# 一键启动 PX4 SITL + Gazebo Harmonic + MicroXRCEAgent（支持 N=1..8+ 无人机与 GPU-3 硬件隔离）
# 用法:
#   ./start_px4_swarm.sh [NUM_DRONES=1] [HEADLESS=1] [WITH_CAMERA=1] [WORLD=moving_target_arena] [NAVCAM_MODE=auto]
#   可选标志: [--world=<name>] [--navcam=auto|full|hybrid|lite] [--obstacles=<yaml>]
# 示例:
#   ./start_px4_swarm.sh 3 1 1
#   ./start_px4_swarm.sh 6 1 1 urban_counter_swarm_arena hybrid

set -euo pipefail

POSITIONAL_ARGS=()
OPT_WORLD=""
OPT_NAVCAM=""
OPT_OBSTACLES=""

_require_opt_val() {
    local flag="$1"
    if [ "$#" -lt 2 ] || [ -z "${2:-}" ] || [[ "${2:-}" == --* ]]; then
        echo "ERROR: ${flag} requires a non-empty value." >&2
        exit 1
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --world=*)
            OPT_WORLD="${1#--world=}"
            [ -n "$OPT_WORLD" ] || { echo "ERROR: --world requires a non-empty value." >&2; exit 1; }
            shift
            ;;
        --world)
            _require_opt_val "$@"
            OPT_WORLD="$2"
            shift 2
            ;;
        --navcam=*)
            OPT_NAVCAM="${1#--navcam=}"
            [ -n "$OPT_NAVCAM" ] || { echo "ERROR: --navcam requires a non-empty value." >&2; exit 1; }
            shift
            ;;
        --navcam)
            _require_opt_val "$@"
            OPT_NAVCAM="$2"
            shift 2
            ;;
        --obstacles=*)
            OPT_OBSTACLES="${1#--obstacles=}"
            [ -n "$OPT_OBSTACLES" ] || { echo "ERROR: --obstacles requires a non-empty value." >&2; exit 1; }
            shift
            ;;
        --obstacles)
            _require_opt_val "$@"
            OPT_OBSTACLES="$2"
            shift 2
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

NUM_DRONES="${POSITIONAL_ARGS[0]:-1}"
if ! [[ "$NUM_DRONES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: NUM_DRONES must be a positive integer (>= 1), got: ${NUM_DRONES}" >&2
    exit 1
fi
HEADLESS_MODE="${POSITIONAL_ARGS[1]:-1}"
WITH_CAMERA="${POSITIONAL_ARGS[2]:-1}"
WORLD_NAME="${OPT_WORLD:-${POSITIONAL_ARGS[3]:-moving_target_arena}}"
WORLD_NAME="${WORLD_NAME%.sdf}"
NAVCAM_MODE="${OPT_NAVCAM:-${POSITIONAL_ARGS[4]:-auto}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 0. 加载工作空间环境并强制激活 GPU-3 隔离（EGL + CUDA）
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
if [ ! -f "$SCRIPT_DIR/gpu3_guard.sh" ]; then
    echo "ERROR: 找不到 $SCRIPT_DIR/gpu3_guard.sh，拒绝无 GPU-3 隔离启动 Gazebo。" >&2
    exit 1
fi
# shellcheck disable=SC1091
GPU3_GUARD_HEADLESS="$HEADLESS_MODE" source "$SCRIPT_DIR/gpu3_guard.sh"

# 1. 验证 PX4 SITL 与 MicroXRCEAgent 可用性
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

# 2. 注入 px4-rc.params 环境变量（避免使用 set -- 覆盖位置参数）
if [ ! -f "$SCRIPT_DIR/px4-rc.params" ]; then
    echo "WARN: 未在 $SCRIPT_DIR 找到 px4-rc.params，PX4 将使用默认参数。" >&2
else
    while IFS= read -r _pline; do
        _pline="${_pline%%#*}"
        read -r _cmd _sub _pkey _pval _rest <<< "$_pline" || true
        if [ "${_cmd:-}" = "param" ] && [ "${_sub:-}" = "set" ] && [ -n "${_pkey:-}" ] && [ -n "${_pval:-}" ]; then
            export "PX4_PARAM_${_pkey}=${_pval}"
        fi
    done < "$SCRIPT_DIR/px4-rc.params"
fi

# 3. 定位世界 SDF、模型资源与共享 obstacles.yaml
WORLD_FILE="$PKG_ROOT/sim/worlds/${WORLD_NAME}.sdf"
MODEL_DIR="$PKG_ROOT/sim/models"
if [ ! -f "$WORLD_FILE" ]; then
    # 回退到 share/flight_ctrl_pkg/sim/worlds（若脚本从 install 目录运行）
    _ALT_WORLD="$(cd "$PKG_ROOT" && pwd)/sim/worlds/${WORLD_NAME}.sdf"
    if [ -f "$_ALT_WORLD" ]; then
        WORLD_FILE="$_ALT_WORLD"
    else
        echo "ERROR: 找不到 Gazebo 世界文件: $WORLD_FILE" >&2
        exit 1
    fi
fi

OBSTACLES_YAML="${OPT_OBSTACLES:-$PKG_ROOT/config/obstacles.yaml}"
export PX4_GZ_MODELS="${PX4_GZ_MODELS:-$PX4_AUTOPILOT_DIR/Tools/simulation/gz/models}"
export PX4_GZ_WORLDS="${PX4_GZ_WORLDS:-$PX4_AUTOPILOT_DIR/Tools/simulation/gz/worlds}"
export GZ_SIM_RESOURCE_PATH="${MODEL_DIR}:${PX4_GZ_MODELS}:${PX4_GZ_WORLDS}:${GZ_SIM_RESOURCE_PATH:-}"
export PATH="${SCRIPT_DIR}:${PATH}"
if [ -f "$SCRIPT_DIR/px4-rc.params" ]; then
    export PX4_RC_PARAMS_POST="$SCRIPT_DIR/px4-rc.params"
fi

# 解析 N 架无人机的起降坪位姿（优先读取 obstacles.yaml 中的 helipads）
mapfile -t POSES < <(python3 - "$OBSTACLES_YAML" "$NUM_DRONES" <<'PYEOF'
import os, sys
yaml_path = sys.argv[1]
num_drones = int(sys.argv[2])
pads = {}
if os.path.isfile(yaml_path):
    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        for pad in data.get('helipads', []):
            d_id = str(pad.get('drone_id', ''))
            if d_id:
                pads[d_id] = (
                    float(pad.get('x', 0.0)),
                    float(pad.get('y', 0.0)),
                    float(pad.get('z', 0.0)) + 0.5,
                    float(pad.get('yaw', 0.0)),
                )
    except Exception as exc:
        print(f"WARN: failed to parse {yaml_path}: {exc}", file=sys.stderr)

for i in range(num_drones):
    d_id = f"drone_{i + 1}"
    if d_id in pads:
        x, y, z, yaw = pads[d_id]
    elif i < 3:
        x, y, z, yaw = 0.0, 3.0 * i, 0.5, 0.0
    else:
        x, y, z, yaw = -3.5 * (i // 3), 3.0 * (i % 3), 0.5, 0.0
    print(f"{x},{y},{z},0,0,{yaw}")
PYEOF
)

PIDS=()
_CLEANUP_DONE=0
_collect_descendants() {
    local parent="$1"
    local child
    for child in $(pgrep -P "$parent" 2>/dev/null || true); do
        echo "$child"
        _collect_descendants "$child"
    done
}
cleanup() {
    trap - EXIT INT TERM
    if [ "$_CLEANUP_DONE" -eq 1 ]; then
        return 0
    fi
    _CLEANUP_DONE=1
    echo ""
    echo "正在停止仿真进程..."

    local all_targets=()
    local pid desc
    for pid in "${PIDS[@]:-}"; do
        [ -n "$pid" ] || continue
        all_targets+=("$pid")
        while IFS= read -r desc; do
            [ -n "$desc" ] && all_targets+=("$desc")
        done < <(_collect_descendants "$pid")
    done

    # 1. Send SIGTERM to each isolated process group (PGID == $pid) and collected descendants
    for pid in "${PIDS[@]:-}"; do
        [ -n "$pid" ] && kill -TERM -- "-$pid" 2>/dev/null || true
    done
    for pid in "${all_targets[@]:-}"; do
        [ -n "$pid" ] && kill -TERM "$pid" 2>/dev/null || true
    done

    # 2. Bounded wait up to 2.5s (25 x 0.1s) for graceful shutdown
    local _t alive
    for _t in $(seq 1 25); do
        alive=0
        for pid in "${all_targets[@]:-}"; do
            [ -n "$pid" ] || continue
            if kill -0 "$pid" 2>/dev/null; then
                alive=1
                break
            fi
        done
        [ "$alive" -eq 0 ] && break
        sleep 0.1
    done

    # 3. Escalate to SIGKILL for any surviving PGIDs or descendants
    for pid in "${PIDS[@]:-}"; do
        [ -n "$pid" ] && kill -KILL -- "-$pid" 2>/dev/null || true
    done
    for pid in "${all_targets[@]:-}"; do
        [ -n "$pid" ] && kill -KILL "$pid" 2>/dev/null || true
    done

    # 4. Reap tracked PIDs (non-blocking after SIGKILL)
    for pid in "${PIDS[@]:-}"; do
        [ -n "$pid" ] && wait "$pid" 2>/dev/null || true
    done
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap 'cleanup' EXIT

# 4. 启动 Micro-XRCE-DDS Agent
echo "[1/4] 启动 MicroXRCEAgent (UDP 8888): $UXRCE_BIN"
setsid "$UXRCE_BIN" udp4 -p 8888 &
PIDS+=($!)
sleep 1

# 5. 启动 Gazebo Harmonic 世界（GPU-3 隔离由 gpu3_guard.sh 保证）
echo "[2/4] 启动 Gazebo ${WORLD_NAME} (HEADLESS=${HEADLESS_MODE}, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})..."
if [ "$HEADLESS_MODE" = "1" ]; then
    setsid env -u DISPLAY gz sim -r -s --headless-rendering "$WORLD_FILE" &
else
    setsid gz sim -r "$WORLD_FILE" &
fi
PIDS+=($!)

# 等待 Gazebo 世界时钟就绪
_clock_ready=0
for _w in $(seq 1 40); do
    if gz topic -l 2>/dev/null | grep -q "/world/${WORLD_NAME}/clock"; then
        _clock_ready=1
        break
    fi
    sleep 0.5
done
if [ "$_clock_ready" -ne 1 ]; then
    echo "ERROR: 等待 /world/${WORLD_NAME}/clock 超时。" >&2
    exit 1
fi

# 6. 生成 N 架无人机模型并启动 PX4 SITL 实例
if [ "$NAVCAM_MODE" = "auto" ]; then
    if [ "$NUM_DRONES" -le 3 ]; then
        EFFECTIVE_NAVCAM="full"
    else
        EFFECTIVE_NAVCAM="hybrid"
    fi
else
    EFFECTIVE_NAVCAM="$NAVCAM_MODE"
fi

echo "[3/4] 启动 ${NUM_DRONES} 架 PX4 SITL (WITH_CAMERA=${WITH_CAMERA}, NAVCAM_MODE=${EFFECTIVE_NAVCAM})..."
mkdir -p /tmp/px4_sitl_instances
for ((i=0; i<NUM_DRONES; i++)); do
    DRONE_NAME="drone_$((i+1))"
    INST_DIR="/tmp/px4_sitl_instances/${DRONE_NAME}"
    rm -rf "$INST_DIR"
    mkdir -p "$INST_DIR"

    if [ "$WITH_CAMERA" = "1" ]; then
        case "$EFFECTIVE_NAVCAM" in
            full)
                MODEL_RES_NAME="x500_down_cam"
                ;;
            lite)
                MODEL_RES_NAME="x500_down_cam_lite"
                ;;
            hybrid)
                if [ "$i" -eq 0 ]; then
                    MODEL_RES_NAME="x500_down_cam"
                else
                    MODEL_RES_NAME="x500_down_cam_lite"
                fi
                ;;
            *)
                echo "ERROR: 未知 NAVCAM_MODE=${EFFECTIVE_NAVCAM} (支持 auto|full|hybrid|lite)" >&2
                exit 1
                ;;
        esac
        GZ_MODEL_INST="x500_down_cam_${i}"
    else
        MODEL_RES_NAME="x500"
        GZ_MODEL_INST="x500_${i}"
    fi

    UXRCE_NS=""
    if [ "$i" -gt 0 ]; then
        UXRCE_NS="px4_${i}"
    fi

    IFS=',' read -r px py pz pr pp pyaw <<< "${POSES[$i]}"
    SDF_WRAP="<sdf version='1.9'><include><uri>model://${MODEL_RES_NAME}</uri><pose>${px:-0} ${py:-0} ${pz:-0.5} ${pr:-0} ${pp:-0} ${pyaw:-0}</pose></include></sdf>"
    _spawn_out="$(gz service -s "/world/${WORLD_NAME}/create" \
        --reqtype gz.msgs.EntityFactory \
        --reptype gz.msgs.Boolean \
        --timeout 5000 \
        --req "name: \"${GZ_MODEL_INST}\", allow_renaming: false, sdf: \"${SDF_WRAP}\"" 2>&1 || true)"
    if ! echo "$_spawn_out" | grep -q "data: true"; then
        echo "WARN: 生成模型 ${GZ_MODEL_INST} (${MODEL_RES_NAME}) 到 /world/${WORLD_NAME}/create 返回: ${_spawn_out}" >&2
    fi

    PX4_GZ_STANDALONE=1 \
    PX4_SYS_AUTOSTART=4001 \
    PX4_SIM_MODEL="gz_x500" \
    PX4_GZ_WORLD="${WORLD_NAME}" \
    PX4_GZ_MODEL_NAME="${GZ_MODEL_INST}" \
    PX4_UXRCE_DDS_NS="${UXRCE_NS}" \
        setsid "$PX4_BIN" -i "$i" -d "$PX4_AUTOPILOT_DIR/build/px4_sitl_default/etc" -w "$INST_DIR" \
        > "/tmp/px4_${DRONE_NAME}.log" 2>&1 &
    PIDS+=($!)
    sleep 2
done

# 7. 启动提示
if [ "$NUM_DRONES" -eq 1 ] && [ "$WORLD_NAME" = "moving_target_arena" ]; then
    LAUNCH_CMD="ros2 launch flight_ctrl_pkg flight_ctrl_single.launch.py"
elif [ "$WORLD_NAME" = "urban_counter_swarm_arena" ]; then
    LAUNCH_CMD="ros2 launch flight_ctrl_pkg showcase_sitl.launch.py num_drones:=${NUM_DRONES}"
else
    LAUNCH_CMD="ros2 launch flight_ctrl_pkg flight_ctrl_swarm.launch.py num_drones:=${NUM_DRONES} gz_world:=${WORLD_NAME}"
fi
echo "仿真集群已全部拉起（MicroXRCEAgent + Gazebo ${WORLD_NAME} + ${NUM_DRONES}x PX4）。按 Ctrl+C 退出。"
echo "请另开终端执行: ${LAUNCH_CMD}"
wait
