#!/usr/bin/env bash
# 课题二飞控开发与联调环境激活脚本（隔离主机 ROS 1 Noetic，激活 px4_ros2 Conda 环境）

# 0. 若调用方开启了 set -u，临时关闭以兼容 conda/ROS setup 脚本
_had_nounset=0
case $- in
    *u*) _had_nounset=1; set +u ;;
esac

# 保存调用方显式指定的 DDS 域与回环隔离变量（防止 conda 钩子覆盖）
_CALLER_ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-}"
_CALLER_ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-}"

# 1. 清理可能继承自宿主终端的 ROS 1 环境变量与路径污染
unset ROS_VERSION ROS_PACKAGE_PATH ROS_ROOT ROS_MASTER_URI ROS_IP ROS_HOSTNAME ROSLISP_PACKAGE_DIRECTORIES ROS_ETC_DIR
if [ "${ROS_DISTRO:-}" = "noetic" ]; then
    unset ROS_DISTRO
fi
_strip_ros1_paths() {
    local var_val="${1:-}"
    local out=""
    local p
    local IFS=":"
    for p in $var_val; do
        case "$p" in
            ""|/opt/ros/noetic*|*ApexNav*) ;;
            *) out="${out:+${out}:}${p}" ;;
        esac
    done
    printf "%s" "$out"
}
export PATH="$(_strip_ros1_paths "${PATH:-}")"
export LD_LIBRARY_PATH="$(_strip_ros1_paths "${LD_LIBRARY_PATH:-}")"
export PYTHONPATH="$(_strip_ros1_paths "${PYTHONPATH:-}")"
export CMAKE_PREFIX_PATH="$(_strip_ros1_paths "${CMAKE_PREFIX_PATH:-}")"
export AMENT_PREFIX_PATH="$(_strip_ros1_paths "${AMENT_PREFIX_PATH:-}")"
export COLCON_PREFIX_PATH="$(_strip_ros1_paths "${COLCON_PREFIX_PATH:-}")"
export PKG_CONFIG_PATH="$(_strip_ros1_paths "${PKG_CONFIG_PATH:-}")"
unset -f _strip_ros1_paths

# 2. 激活 Conda px4_ros2（若已处于 px4_ros2 则仅确保 site-packages 与 bin 位于最前）
if [ "${CONDA_DEFAULT_ENV:-}" = "px4_ros2" ] && [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/bin" ]; then
    case ":${PATH}:" in
        "${CONDA_PREFIX}/bin:"*) ;;
        *) export PATH="${CONDA_PREFIX}/bin:${PATH}" ;;
    esac
    if [ -d "${CONDA_PREFIX}/lib/python3.11/site-packages" ]; then
        case ":${PYTHONPATH:-}:" in
            *":${CONDA_PREFIX}/lib/python3.11/site-packages:"*) ;;
            *) export PYTHONPATH="${CONDA_PREFIX}/lib/python3.11/site-packages${PYTHONPATH:+:${PYTHONPATH}}" ;;
        esac
    fi
    export ROS_DISTRO="${ROS_DISTRO:-humble}"
    export ROS_VERSION=2
    export ROS_PYTHON_VERSION=3
    export ROS_OS_OVERRIDE="conda:linux"
    export ROS_ETC_DIR="${CONDA_PREFIX}/etc/ros"
    case ":${AMENT_PREFIX_PATH:-}:" in
        *":${CONDA_PREFIX}:"*) ;;
        *) export AMENT_PREFIX_PATH="${AMENT_PREFIX_PATH:+${AMENT_PREFIX_PATH}:}${CONDA_PREFIX}" ;;
    esac
else
    _CONDA_SH=""
    for _cand in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
        if [ -f "$_cand" ]; then
            _CONDA_SH="$_cand"
            break
        fi
    done
    if [ -n "$_CONDA_SH" ]; then
        # shellcheck disable=SC1090
        source "$_CONDA_SH"
        _CONDA_BASE="$(dirname "$(dirname "$(dirname "$_CONDA_SH")")")"
        if [ -d "$_CONDA_BASE/envs/px4_ros2" ]; then
            conda activate px4_ros2
        elif [ -f "/opt/ros/${ROS_DISTRO:-humble}/setup.bash" ]; then
            # shellcheck disable=SC1090
            source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
        fi
        unset _CONDA_BASE
    elif [ -f "/opt/ros/${ROS_DISTRO:-humble}/setup.bash" ]; then
        # shellcheck disable=SC1090
        source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
    fi
fi

# 3. 自动定位 ros2_ws 与上层根目录，加载 px4_msgs 与课题二主工作空间
_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${ROS2_WS_DIR:-}" ]; then
    _probe="$_ENV_SCRIPT_DIR"
    for _ in 1 2 3 4 5 6; do
        if [ -d "$_probe/src" ] && { [ -f "$_probe/install/setup.bash" ] || [ -d "$_probe/.git" ]; }; then
            ROS2_WS_DIR="$_probe"
            break
        fi
        _probe="$(dirname "$_probe")"
    done
fi
ROS2_WS_DIR="${ROS2_WS_DIR:-/home/hkw/flying/ros2_ws}"
FLYING_ROOT="${FLYING_ROOT:-$(dirname "$ROS2_WS_DIR")}"

if [ -n "${PX4_MSGS_WS:-}" ] && [ -f "${PX4_MSGS_WS}/install/setup.bash" ]; then
    # shellcheck disable=SC1090
    source "${PX4_MSGS_WS}/install/setup.bash"
elif [ -f "$FLYING_ROOT/px4_msgs_ws/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$FLYING_ROOT/px4_msgs_ws/install/setup.bash"
fi
if [ -f "$ROS2_WS_DIR/install/setup.bash" ]; then
    # shellcheck disable=SC1091
    source "$ROS2_WS_DIR/install/setup.bash"
fi

# 4. 配置 uXRCE-DDS Agent、PX4 根目录与团队统一通信域
if [ -d "$FLYING_ROOT/third_party/uxrce_agent_install/bin" ]; then
    case ":${PATH}:" in
        *":$FLYING_ROOT/third_party/uxrce_agent_install/bin:"*) ;;
        *) export PATH="$FLYING_ROOT/third_party/uxrce_agent_install/bin:$PATH" ;;
    esac
fi
if [ -d "$FLYING_ROOT/third_party/uxrce_agent_install/lib" ]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$FLYING_ROOT/third_party/uxrce_agent_install/lib:"*) ;;
        *) export LD_LIBRARY_PATH="$FLYING_ROOT/third_party/uxrce_agent_install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
fi
if [ -z "${PX4_AUTOPILOT_DIR:-}" ] || [ ! -d "${PX4_AUTOPILOT_DIR}" ]; then
    for _px4_dir in "$FLYING_ROOT/third_party/PX4-Autopilot" "$HOME/PX4-Autopilot" "$HOME/third_party/PX4-Autopilot"; do
        if [ -d "$_px4_dir" ]; then
            export PX4_AUTOPILOT_DIR="$_px4_dir"
            break
        fi
    done
fi
export PX4_AUTOPILOT_DIR="${PX4_AUTOPILOT_DIR:-$FLYING_ROOT/third_party/PX4-Autopilot}"
export ROS_DOMAIN_ID="${_CALLER_ROS_DOMAIN_ID:-${ROS_DOMAIN_ID:-0}}"
export ROS_LOCALHOST_ONLY="${_CALLER_ROS_LOCALHOST_ONLY:-${ROS_LOCALHOST_ONLY:-0}}"

if [ -f "$_ENV_SCRIPT_DIR/gpu3_guard.sh" ]; then
    # shellcheck disable=SC1091
    GPU3_GUARD_HEADLESS="${GPU3_GUARD_HEADLESS:-0}" source "$_ENV_SCRIPT_DIR/gpu3_guard.sh"
fi

unset _CONDA_SH _cand _ENV_SCRIPT_DIR _probe _px4_dir _CALLER_ROS_DOMAIN_ID _CALLER_ROS_LOCALHOST_ONLY

if [ "$_had_nounset" -eq 1 ]; then
    set -u
fi
unset _had_nounset
