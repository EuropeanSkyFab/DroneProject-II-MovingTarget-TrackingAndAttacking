#!/usr/bin/env bash
# 课题二飞控开发与联调环境激活脚本（隔离主机 ROS 1 Noetic，激活 px4_ros2 Conda 环境）

# 0. 若调用方开启了 set -u，临时关闭以兼容 conda/ROS setup 脚本
_had_nounset=0
case $- in
    *u*) _had_nounset=1; set +u ;;
esac

# 1. 清理可能继承自宿主终端的 ROS 1 环境变量
unset ROS_DISTRO ROS_VERSION ROS_PACKAGE_PATH ROS_ROOT ROS_MASTER_URI ROS_IP ROS_HOSTNAME ROSLISP_PACKAGE_DIRECTORIES ROS_ETC_DIR

# 2. 激活 Conda px4_ros2（若不存在则回退到系统 ROS 2 Humble）
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
    if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "px4_ros2"; then
        conda activate px4_ros2
    elif [ -f "/opt/ros/${ROS_DISTRO:-humble}/setup.bash" ]; then
        # shellcheck disable=SC1090
        source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
    fi
elif [ -f "/opt/ros/${ROS_DISTRO:-humble}/setup.bash" ]; then
    # shellcheck disable=SC1090
    source "/opt/ros/${ROS_DISTRO:-humble}/setup.bash"
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
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
export ROS_LOCALHOST_ONLY="${ROS_LOCALHOST_ONLY:-0}"
unset _CONDA_SH _cand _ENV_SCRIPT_DIR _probe _px4_dir

if [ "$_had_nounset" -eq 1 ]; then
    set -u
fi
unset _had_nounset
