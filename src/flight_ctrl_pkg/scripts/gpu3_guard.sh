#!/usr/bin/env bash
# Deterministic GPU-3 exclusion guard for Gazebo/Ogre2 EGL, CUDA, and NVENC.
# Safe to source under `set -u`, idempotent, or invoke as a wrapper:
#   source gpu3_guard.sh
#   GPU3_GUARD_HEADLESS=1 gpu3_guard.sh <command...>

_gpu3_is_sourced=0
if (return 0 2>/dev/null); then
    _gpu3_is_sourced=1
fi

_gpu3_had_nounset=0
case $- in
    *u*) _gpu3_had_nounset=1; set +u ;;
esac

export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 1. Enforce CUDA_VISIBLE_DEVICES excluding GPU 3 (0000:D9:00.0)
_GPU3_UUID="GPU-617ae266-eed3-ef41-ae0f-5f25c730f5d5"
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    export CUDA_VISIBLE_DEVICES="2"
elif [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "WARN [gpu3_guard]: CUDA_VISIBLE_DEVICES is empty; defaulting to '2'." >&2
    export CUDA_VISIBLE_DEVICES="2"
else
    _filtered_cvd=""
    _stripped_gpu3=0
    _cvd_tmp="${CUDA_VISIBLE_DEVICES//$'\n'/}"
    _cvd_tmp="${_cvd_tmp//$'\r'/}"
    IFS=',' read -ra _cvd_tokens <<< "${_cvd_tmp}"
    for _tok in "${_cvd_tokens[@]}"; do
        _tok="${_tok#"${_tok%%[![:space:]]*}"}"
        _tok="${_tok%"${_tok##*[![:space:]]}"}"
        if [ -z "$_tok" ]; then
            continue
        fi
        if [ "$_tok" = "3" ] || [ "$_tok" = "$_GPU3_UUID" ]; then
            _stripped_gpu3=1
            continue
        fi
        if [ -n "$_filtered_cvd" ]; then
            _filtered_cvd="${_filtered_cvd},${_tok}"
        else
            _filtered_cvd="${_tok}"
        fi
    done
    if [ -z "$_filtered_cvd" ]; then
        if [ "$_stripped_gpu3" -eq 1 ]; then
            echo "WARN [gpu3_guard]: CUDA_VISIBLE_DEVICES contained only GPU 3; defaulting to '2'." >&2
        else
            echo "WARN [gpu3_guard]: CUDA_VISIBLE_DEVICES contained no valid devices; defaulting to '2'." >&2
        fi
        export CUDA_VISIBLE_DEVICES="2"
    elif [ "$_stripped_gpu3" -eq 1 ]; then
        echo "WARN [gpu3_guard]: Stripped GPU 3 from CUDA_VISIBLE_DEVICES -> '${_filtered_cvd}'." >&2
        export CUDA_VISIBLE_DEVICES="$_filtered_cvd"
    elif [ "$_filtered_cvd" != "${CUDA_VISIBLE_DEVICES}" ]; then
        export CUDA_VISIBLE_DEVICES="$_filtered_cvd"
    fi
    unset _filtered_cvd _stripped_gpu3 _cvd_tmp _cvd_tokens _tok
fi
unset _GPU3_UUID

# 2. Locate libegl_gpu_filter.so in priority order (fail closed if missing)
_GUARD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_EGL_FILTER_SO=""

# (a) Active AMENT_PREFIX_PATH / COLCON_CURRENT_PREFIX
for _prefix_list in "${AMENT_PREFIX_PATH:-}" "${COLCON_CURRENT_PREFIX:-}"; do
    if [ -n "$_prefix_list" ]; then
        IFS=':' read -ra _prefixes <<< "$_prefix_list"
        for _pfx in "${_prefixes[@]}"; do
            if [ -n "$_pfx" ] && [ -r "$_pfx/lib/libegl_gpu_filter.so" ]; then
                _EGL_FILTER_SO="$_pfx/lib/libegl_gpu_filter.so"
                break 2
            fi
        done
    fi
done
unset _prefix_list _prefixes _pfx

# (b) Standard ROS 2 workspace install path
if [ -z "$_EGL_FILTER_SO" ]; then
    _ws_dir="${ROS2_WS_DIR:-/home/hkw/flying/ros2_ws}"
    if [ -r "$_ws_dir/install/flight_ctrl_pkg/lib/libegl_gpu_filter.so" ]; then
        _EGL_FILTER_SO="$_ws_dir/install/flight_ctrl_pkg/lib/libegl_gpu_filter.so"
    fi
    unset _ws_dir
fi

# (c) Relative package install dir (when sourced from install*/flight_ctrl_pkg/share/flight_ctrl_pkg/scripts)
if [ -z "$_EGL_FILTER_SO" ]; then
    _rel_lib="$(cd "$_GUARD_DIR/../../../lib" 2>/dev/null && pwd || true)"
    if [ -n "$_rel_lib" ] && [ -r "$_rel_lib/libegl_gpu_filter.so" ]; then
        _EGL_FILTER_SO="$_rel_lib/libegl_gpu_filter.so"
    fi
    unset _rel_lib
fi

# (d) Prebuilt fallback next to gpu3_guard.sh (src/flight_ctrl_pkg/scripts/libegl_gpu_filter.so)
if [ -z "$_EGL_FILTER_SO" ] && [ -r "$_GUARD_DIR/libegl_gpu_filter.so" ]; then
    _EGL_FILTER_SO="$_GUARD_DIR/libegl_gpu_filter.so"
fi

if [ -z "$_EGL_FILTER_SO" ] || [ ! -r "$_EGL_FILTER_SO" ]; then
    echo "ERROR [gpu3_guard]: libegl_gpu_filter.so not found; refusing unguarded EGL launch." >&2
    _gpu3_ret_sourced="$_gpu3_is_sourced"
    if [ "$_gpu3_had_nounset" -eq 1 ]; then
        set -u
    fi
    unset _gpu3_had_nounset _gpu3_is_sourced _GUARD_DIR _EGL_FILTER_SO
    if [ "$_gpu3_ret_sourced" -eq 1 ]; then
        unset _gpu3_ret_sourced
        return 1
    fi
    unset _gpu3_ret_sourced
    exit 1
fi

# Idempotently prepend libegl_gpu_filter.so to LD_PRELOAD (deduplicating any prior copy by basename)
_new_ld_preload="${_EGL_FILTER_SO}"
if [ -n "${LD_PRELOAD:-}" ]; then
    IFS=':' read -ra _ldp_entries <<< "${LD_PRELOAD}"
    for _ldp in "${_ldp_entries[@]}"; do
        if [ -z "$_ldp" ] || [ "$(basename "$_ldp")" = "libegl_gpu_filter.so" ]; then
            continue
        fi
        _new_ld_preload="${_new_ld_preload}:${_ldp}"
    done
    unset _ldp_entries _ldp
fi
export LD_PRELOAD="${_new_ld_preload}"
unset _new_ld_preload

# 3. Pin NVIDIA GLVND EGL vendor
if [ -r /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then
    export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
fi

# 4. Headless DISPLAY guard (default 1; env.sh sets 0 for interactive shells unless overridden)
GPU3_GUARD_HEADLESS="${GPU3_GUARD_HEADLESS:-1}"
if [ "$GPU3_GUARD_HEADLESS" = "1" ]; then
    unset DISPLAY
fi

unset _GUARD_DIR _EGL_FILTER_SO

if [ "$_gpu3_had_nounset" -eq 1 ]; then
    set -u
fi
unset _gpu3_had_nounset

if [ "$_gpu3_is_sourced" -eq 0 ] && [ "$#" -gt 0 ]; then
    unset _gpu3_is_sourced
    exec "$@"
fi
unset _gpu3_is_sourced
