"""Tier 1 functional tests for F1: Deterministic GPU-3 EGL & CUDA Exclusion."""

from __future__ import annotations

import ctypes
import json
import py_compile
import sys
from pathlib import Path

import pytest

from helpers.hardware_git_guard import (
    check_cuda_env_excludes_gpu3,
    check_gpu3_zero_occupancy,
    check_ogre2_log_gpu3_excluded,
)
from helpers.ros_probe import run_command_with_gpu_guard

pytestmark = [pytest.mark.tier1, pytest.mark.feature(id="F1")]

ROS2_WS = Path("/home/hkw/flying/ros2_ws")
SCRIPTS_DIR = ROS2_WS / "src" / "flight_ctrl_pkg" / "scripts"
GPU3_GUARD_SH = SCRIPTS_DIR / "gpu3_guard.sh"
ENV_SH = SCRIPTS_DIR / "env.sh"
START_SWARM_SH = SCRIPTS_DIR / "start_px4_swarm.sh"
VERIFY_GPU3_PY = SCRIPTS_DIR / "verify_gpu3_isolation.py"
INSTALL_FILTER_SO = ROS2_WS / "install" / "flight_ctrl_pkg" / "lib" / "libegl_gpu_filter.so"
FALLBACK_FILTER_SO = SCRIPTS_DIR / "libegl_gpu_filter.so"
NVIDIA_ICD_JSON = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
OGRE2_LOG_PATH = Path.home() / ".gz" / "rendering" / "ogre2.log"


def _resolve_filter_so() -> Path:
    if INSTALL_FILTER_SO.is_file():
        return INSTALL_FILTER_SO
    if FALLBACK_FILTER_SO.is_file():
        return FALLBACK_FILTER_SO
    pytest.fail(
        f"libegl_gpu_filter.so missing at both {INSTALL_FILTER_SO} and {FALLBACK_FILTER_SO}"
    )


@pytest.mark.requires_m1
def test_f01_gpu3_guard_env_and_ctypes_egl_probe() -> None:
    """Verify gpu3_guard.sh configures CUDA/EGL isolation and ctypes EGL returns only GPU 2."""
    assert GPU3_GUARD_SH.is_file(), f"Missing gpu3_guard.sh: {GPU3_GUARD_SH}"
    probe_cmd = (
        f"unset CUDA_VISIBLE_DEVICES LD_PRELOAD __EGL_VENDOR_LIBRARY_FILENAMES; "
        f"export DISPLAY=:99; "
        f"GPU3_GUARD_HEADLESS=1 source '{GPU3_GUARD_SH}' && "
        f"echo \"CVD=$CUDA_VISIBLE_DEVICES\" && "
        f"echo \"LDP=$LD_PRELOAD\" && "
        f"echo \"ICD=$__EGL_VENDOR_LIBRARY_FILENAMES\" && "
        f"echo \"DISP=${{DISPLAY:-UNSET}}\""
    )
    res = run_command_with_gpu_guard(
        ["bash", "-lc", probe_cmd],
        timeout_s=15.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "2"},
    )
    assert res.returncode == 0, f"gpu3_guard.sh failed: {res.stderr}\n{res.stdout}"
    kv = {}
    for line in res.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

    assert kv.get("CVD") == "2", f"Expected CUDA_VISIBLE_DEVICES=2, got {kv}"
    assert "libegl_gpu_filter.so" in kv.get("LDP", ""), f"LD_PRELOAD missing filter: {kv}"
    assert kv.get("ICD") == str(NVIDIA_ICD_JSON), f"Unexpected EGL ICD: {kv}"
    assert kv.get("DISP") == "UNSET", f"Headless guard did not unset DISPLAY: {kv}"

    filter_so = _resolve_filter_so()
    egl_probe_py = (
        "import ctypes, json\n"
        "ctypes.CDLL('libEGL.so.1', mode=ctypes.RTLD_GLOBAL)\n"
        "lib = ctypes.CDLL(None)\n"
        "lib.eglGetProcAddress.argtypes = [ctypes.c_char_p]\n"
        "lib.eglGetProcAddress.restype = ctypes.c_void_p\n"
        "q_ptr = lib.eglGetProcAddress(b'eglQueryDevicesEXT')\n"
        "a_ptr = lib.eglGetProcAddress(b'eglQueryDeviceAttribEXT')\n"
        "assert q_ptr and a_ptr, 'Missing EGL device query symbols'\n"
        "QueryDevices = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int, "
        "ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))(q_ptr)\n"
        "QueryAttrib = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, "
        "ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t))(a_ptr)\n"
        "n = ctypes.c_int(0)\n"
        "assert QueryDevices(0, None, ctypes.byref(n)) == 1\n"
        "devs = (ctypes.c_void_p * max(1, n.value))()\n"
        "assert QueryDevices(n.value, devs, ctypes.byref(n)) == 1\n"
        "ids = []\n"
        "for i in range(n.value):\n"
        "    val = ctypes.c_ssize_t(-1)\n"
        "    ok = QueryAttrib(devs[i], 0x323A, ctypes.byref(val))\n"
        "    ids.append(val.value if ok else None)\n"
        "print(json.dumps({'count': n.value, 'cuda_ids': ids}))\n"
    )
    egl_res = run_command_with_gpu_guard(
        [sys.executable, "-c", egl_probe_py],
        timeout_s=15.0,
        extra_env={
            "CUDA_VISIBLE_DEVICES": "2",
            "LD_PRELOAD": str(filter_so),
            "__EGL_VENDOR_LIBRARY_FILENAMES": str(NVIDIA_ICD_JSON),
        },
    )
    assert egl_res.returncode == 0, f"EGL ctypes probe failed: {egl_res.stderr}"
    parsed = json.loads(egl_res.stdout.strip().splitlines()[-1])
    assert parsed["count"] == 1, f"Expected exactly 1 filtered EGL device, got {parsed}"
    assert parsed["cuda_ids"] == [0], f"Expected EGL_CUDA_DEVICE_NV=[0], got {parsed}"


@pytest.mark.requires_m1
def test_f01_libegl_gpu_filter_exists_and_exports_symbols() -> None:
    """Verify libegl_gpu_filter.so exists and exports eglQueryDevicesEXT and eglGetProcAddress."""
    filter_so = _resolve_filter_so()
    assert filter_so.stat().st_size > 1024, f"Unexpectedly small .so: {filter_so}"

    nm_res = run_command_with_gpu_guard(["nm", "-D", str(filter_so)], timeout_s=10.0)
    assert nm_res.returncode == 0, f"nm -D failed on {filter_so}: {nm_res.stderr}"
    exported = nm_res.stdout
    assert "eglQueryDevicesEXT" in exported, f"eglQueryDevicesEXT not exported:\n{exported}"
    assert "eglGetProcAddress" in exported, f"eglGetProcAddress not exported:\n{exported}"

    lib = ctypes.CDLL(str(filter_so))
    assert getattr(lib, "eglQueryDevicesEXT", None) is not None
    assert getattr(lib, "eglGetProcAddress", None) is not None


@pytest.mark.requires_m1
def test_f01_scripts_source_gpu3_guard_and_exclude_gpu3(tmp_path: Path) -> None:
    """Verify env.sh, start_px4_swarm.sh, and verify_gpu3_isolation.py enforce GPU-3 exclusion."""
    for path in (ENV_SH, START_SWARM_SH, VERIFY_GPU3_PY):
        assert path.is_file(), f"Missing F1 script: {path}"

    py_compile.compile(
        str(VERIFY_GPU3_PY),
        cfile=str(tmp_path / "verify_gpu3_isolation.pyc"),
        doraise=True,
    )
    help_res = run_command_with_gpu_guard(
        [sys.executable, "-B", str(VERIFY_GPU3_PY), "--help"],
        timeout_s=10.0,
    )
    assert help_res.returncode == 0, f"verify_gpu3_isolation.py --help failed: {help_res.stderr}"
    for cli_flag in ("--world", "--iterations", "--cuda-visible-devices"):
        assert cli_flag in help_res.stdout, f"Missing CLI flag {cli_flag} in --help output"

    env_txt = ENV_SH.read_text(encoding="utf-8")
    swarm_txt = START_SWARM_SH.read_text(encoding="utf-8")
    assert "gpu3_guard.sh" in env_txt, "env.sh must source gpu3_guard.sh"
    assert "gpu3_guard.sh" in swarm_txt, "start_px4_swarm.sh must source gpu3_guard.sh"

    env_probe = (
        f"unset CUDA_VISIBLE_DEVICES LD_PRELOAD __EGL_VENDOR_LIBRARY_FILENAMES; "
        f"source '{ENV_SH}' >/dev/null 2>&1 && "
        f"echo \"CVD=$CUDA_VISIBLE_DEVICES\" && "
        f"echo \"LDP=$LD_PRELOAD\" && "
        f"echo \"ICD=$__EGL_VENDOR_LIBRARY_FILENAMES\""
    )
    res = run_command_with_gpu_guard(
        ["bash", "-lc", env_probe],
        timeout_s=20.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "2"},
    )
    assert res.returncode == 0, f"env.sh failed: {res.stderr}"
    vals = dict(line.split("=", 1) for line in res.stdout.splitlines() if "=" in line)
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": vals.get("CVD", "")}) is True
    assert "libegl_gpu_filter.so" in vals.get("LDP", ""), f"LD_PRELOAD missing filter: {vals}"
    assert vals.get("ICD") == str(NVIDIA_ICD_JSON), f"ICD mismatch: {vals}"


@pytest.mark.requires_m1
def test_f01_ogre2_log_excludes_gpu3_and_software_renderer() -> None:
    """Audit ~/.gz/rendering/ogre2.log via helper to verify GPU 3 & llvmpipe exclusion."""
    assert OGRE2_LOG_PATH.is_file(), (
        f"Ogre2 log not found at {OGRE2_LOG_PATH}; run Gazebo headless smoke/verify_gpu3_isolation.py first"
    )
    audit = check_ogre2_log_gpu3_excluded(OGRE2_LOG_PATH)
    assert audit.ok is True, f"ogre2.log GPU-3 audit failed: {audit.violations}"
    assert audit.num_egl_devices is not None and 1 <= audit.num_egl_devices <= 3
    assert "#3" not in audit.detected_devices
    assert not any("renderd131" in v.lower() or "llvmpipe" in v.lower() for v in audit.violations)


@pytest.mark.pre_integration_safe
def test_f01_live_nvidia_smi_gpu3_zero_local_occupancy() -> None:
    """Verify live nvidia-smi reports zero local workspace occupancy on GPU 3 and valid NVIDIA ICD."""
    assert NVIDIA_ICD_JSON.is_file(), f"Missing NVIDIA GLVND ICD: {NVIDIA_ICD_JSON}"
    icd_data = json.loads(NVIDIA_ICD_JSON.read_text(encoding="utf-8"))
    assert icd_data.get("ICD", {}).get("library_path") == "libEGL_nvidia.so.0"

    status = check_gpu3_zero_occupancy()
    assert status.ok is True, f"GPU-3 local occupancy violation detected: {status.notes}\n{status.raw_output}"
    assert status.gpu3_pids == [], f"Unexpected local PIDs on GPU 3: {status.gpu3_pids}"
    assert status.gpu3_mem_mib == 0, f"Unexpected local VRAM on GPU 3: {status.gpu3_mem_mib} MiB"
