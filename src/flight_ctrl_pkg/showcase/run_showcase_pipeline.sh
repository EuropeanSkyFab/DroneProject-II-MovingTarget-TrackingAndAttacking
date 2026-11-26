#!/usr/bin/env bash
# End-to-end Project 2 multi-UAV counter-swarm showcase runner (GPU-3 isolated).

set -euo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "${SELF_DIR}/gpu3_guard.sh" ]; then
    SCRIPT_DIR="${SELF_DIR}"
elif [ -f "${SELF_DIR}/../scripts/gpu3_guard.sh" ]; then
    SCRIPT_DIR="$(cd "${SELF_DIR}/../scripts" && pwd)"
else
    SCRIPT_DIR="/home/hkw/flying/ros2_ws/src/flight_ctrl_pkg/scripts"
fi

PKG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [ ! -f "${PKG_ROOT}/showcase/cinema_renderer.py" ] && [ -f "/home/hkw/flying/ros2_ws/src/flight_ctrl_pkg/showcase/cinema_renderer.py" ]; then
    PKG_ROOT="/home/hkw/flying/ros2_ws/src/flight_ctrl_pkg"
fi

# 1. Enforce strict GPU-3 exclusion + clean conda/ROS 2 environment
source "${SCRIPT_DIR}/gpu3_guard.sh"
source "${SCRIPT_DIR}/env.sh"

WS_DIR="${ROS2_WS_DIR:-/home/hkw/flying/ros2_ws}"
if [ -f "${WS_DIR}/install/setup.bash" ]; then
    set +u
    source "${WS_DIR}/install/setup.bash"
    set -u
fi

TRACE_PATH="${PROJECT2_TRACE_PATH:-/tmp/project2_mission_trace.jsonl}"
SUMMARY_PATH="${PROJECT2_SUMMARY_PATH:-/tmp/project2_mission_summary.json}"
OUT_DIR="${SHOWCASE_OUTPUT_DIR:-/home/hkw/.gemini/antigravity/brain/d204ee53-0b23-4a81-ac6e-d2bd544d5472}"
OUTPUT_MP4="${OUT_DIR}/swarm_showcase_cinema.mp4"
KEYFRAMES_DIR="${OUT_DIR}"
OBSTACLES_YAML="${PKG_ROOT}/config/obstacles.yaml"
RENDER_ONLY=0
FULL_SITL=0
DRY_RUN=0
DURATION="18.0"

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Options:
  --help, -h            Show this help message and exit
  --dry-run             Print resolved paths and GPU-3 guard configuration without executing
  --render-only         Render cinema MP4 and keyframes from existing trace/summary JSON
  --full-sitl           Execute full closed-loop 6v3 mission before rendering showcase video
  --trace <path>        Override trace JSONL path (implies --render-only unless --full-sitl)
  --summary <path>      Override summary JSON path
  --output <path>       Override output MP4 path
  --keyframes-dir <dir> Override keyframe PNG output directory
  --obstacles <path>    Override obstacles.yaml path
  --duration <sec>      Video output duration in seconds (default: 18.0)
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --render-only)
            RENDER_ONLY=1
            shift
            ;;
        --full-sitl)
            FULL_SITL=1
            RENDER_ONLY=0
            shift
            ;;
        --trace)
            [ $# -ge 2 ] || { echo "ERROR: --trace requires a value" >&2; exit 2; }
            TRACE_PATH="$2"
            if [ "${FULL_SITL}" -eq 0 ]; then
                RENDER_ONLY=1
            fi
            shift 2
            ;;
        --summary)
            [ $# -ge 2 ] || { echo "ERROR: --summary requires a value" >&2; exit 2; }
            SUMMARY_PATH="$2"
            shift 2
            ;;
        --output)
            [ $# -ge 2 ] || { echo "ERROR: --output requires a value" >&2; exit 2; }
            OUTPUT_MP4="$2"
            KEYFRAMES_DIR="$(dirname "${OUTPUT_MP4}")"
            shift 2
            ;;
        --keyframes-dir)
            [ $# -ge 2 ] || { echo "ERROR: --keyframes-dir requires a value" >&2; exit 2; }
            KEYFRAMES_DIR="$2"
            shift 2
            ;;
        --obstacles)
            [ $# -ge 2 ] || { echo "ERROR: --obstacles requires a value" >&2; exit 2; }
            OBSTACLES_YAML="$2"
            shift 2
            ;;
        --duration)
            [ $# -ge 2 ] || { echo "ERROR: --duration requires a value" >&2; exit 2; }
            DURATION="$2"
            shift 2
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

export PROJECT2_TRACE_PATH="${TRACE_PATH}"
export PROJECT2_SUMMARY_PATH="${SUMMARY_PATH}"
export SHOWCASE_OUTPUT_DIR="${KEYFRAMES_DIR}"

if [ "${DRY_RUN}" -eq 1 ]; then
    echo "DRY-RUN Showcase Pipeline Configuration:"
    echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
    echo "  LD_PRELOAD=${LD_PRELOAD:-}"
    echo "  TRACE_PATH=${TRACE_PATH}"
    echo "  SUMMARY_PATH=${SUMMARY_PATH}"
    echo "  OUTPUT_MP4=${OUTPUT_MP4}"
    echo "  KEYFRAMES_DIR=${KEYFRAMES_DIR}"
    echo "  OBSTACLES_YAML=${OBSTACLES_YAML}"
    echo "  FULL_SITL=${FULL_SITL} RENDER_ONLY=${RENDER_ONLY}"
    exit 0
fi

RENDERER_PY="${PKG_ROOT}/showcase/cinema_renderer.py"
if [ ! -f "${RENDERER_PY}" ]; then
    echo "ERROR: Missing cinema renderer at ${RENDERER_PY}" >&2
    exit 1
fi

if [ "${FULL_SITL}" -eq 1 ] || { [ "${RENDER_ONLY}" -eq 0 ] && [ ! -s "${TRACE_PATH}" ]; }; then
    echo "[1/2] Executing 6v3 closed-loop counter-swarm mission -> ${TRACE_PATH}"
    python3 "${SCRIPT_DIR}/run_closed_loop_mission.py" \
        --obstacles "${OBSTACLES_YAML}" \
        --trace "${TRACE_PATH}" \
        --summary "${SUMMARY_PATH}"
fi

echo "[2/2] Rendering 1080p blockbuster showcase MP4 + 5 act keyframes -> ${OUTPUT_MP4}"
exec python3 "${RENDERER_PY}" \
    --trace "${TRACE_PATH}" \
    --summary "${SUMMARY_PATH}" \
    --obstacles "${OBSTACLES_YAML}" \
    --output "${OUTPUT_MP4}" \
    --keyframes-dir "${KEYFRAMES_DIR}" \
    --duration "${DURATION}"
