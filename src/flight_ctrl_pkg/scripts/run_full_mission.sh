#!/usr/bin/env bash
# 课题二全系统端到端一键闭环运行脚本（PX4 v1.17.0 x3 + Gazebo Harmonic + 视觉 + 规划 + 中枢 + 自主战术指挥官）
# 用法:
#   ./run_full_mission.sh [headless=1|0] [target_profile=evasive_weave|figure8|circle]

set -euo pipefail

HEADLESS_MODE="${1:-1}"
TARGET_PROFILE="${2:-evasive_weave}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"

SWARM_PID=""
cleanup() {
    echo ""
    echo "正在安全回收课题二全系统仿真进程..."
    if [ -n "$SWARM_PID" ] && kill -0 "$SWARM_PID" 2>/dev/null; then
        kill -TERM "$SWARM_PID" 2>/dev/null || true
    fi
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[Stage 1/2] 启动底层三机 PX4 v1.17.0 SITL + Gazebo 战术试飞场..."
"$SCRIPT_DIR/start_px4_swarm.sh" 3 "$HEADLESS_MODE" 1 > /tmp/project2_px4_swarm.log 2>&1 &
SWARM_PID=$!

echo "等待 3 架 PX4 实例完成 EKF2 与 uXRCE-DDS 实体注册 (24s)..."
sleep 24

echo "[Stage 2/2] 启动课题二四包全系统闭环节点群与自主战术指挥官..."
ros2 launch flight_ctrl_pkg project2_full_swarm.launch.py \
    auto_mission:=true \
    exit_on_complete:=true \
    target_profile:="$TARGET_PROFILE"
