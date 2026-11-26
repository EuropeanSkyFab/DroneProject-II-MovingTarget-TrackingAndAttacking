"""Tier 2 BVA & corner-case tests for F1: Deterministic GPU-3 EGL & CUDA Exclusion."""

from __future__ import annotations

import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from helpers.hardware_git_guard import (
    check_cuda_env_excludes_gpu3,
    check_ogre2_log_gpu3_excluded,
)
from helpers.ros_probe import run_command_with_gpu_guard

pytestmark = [pytest.mark.tier2, pytest.mark.feature(id="F1"), pytest.mark.requires_m1]

ROS2_WS = Path("/home/hkw/flying/ros2_ws")
SCRIPTS_DIR = ROS2_WS / "src" / "flight_ctrl_pkg" / "scripts"
GPU3_GUARD_SH = SCRIPTS_DIR / "gpu3_guard.sh"
INSTALL_FILTER_SO = ROS2_WS / "install" / "flight_ctrl_pkg" / "lib" / "libegl_gpu_filter.so"
FALLBACK_FILTER_SO = SCRIPTS_DIR / "libegl_gpu_filter.so"
NVIDIA_ICD_JSON = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
GPU3_UUID = "GPU-617ae266-eed3-ef41-ae0f-5f25c730f5d5"


def _resolve_filter_so() -> Path:
    if INSTALL_FILTER_SO.is_file():
        return INSTALL_FILTER_SO
    if FALLBACK_FILTER_SO.is_file():
        return FALLBACK_FILTER_SO
    pytest.fail(
        f"libegl_gpu_filter.so missing at both {INSTALL_FILTER_SO} and {FALLBACK_FILTER_SO}"
    )


@pytest.mark.parametrize(
    ("input_cvd", "expected_sanitized"),
    [
        ("3", "2"),
        ("0,1,2,3", "0,1,2"),
        (GPU3_UUID, "2"),
        (f"0,{GPU3_UUID},2", "0,2"),
    ],
)
def test_f01_bva_reject_and_sanitize_gpu3_cuda_visible_devices(
    input_cvd: str, expected_sanitized: str
) -> None:
    """Reject GPU-3 inputs in check_cuda_env_excludes_gpu3() and verify gpu3_guard.sh strips them."""
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": input_cvd}) is False
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": "all"}) is False
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": ""}) is False

    cmd = (
        f"export CUDA_VISIBLE_DEVICES='{input_cvd}'; "
        f"source '{GPU3_GUARD_SH}' && printf '%s' \"$CUDA_VISIBLE_DEVICES\""
    )
    res = run_command_with_gpu_guard(
        ["bash", "-lc", cmd],
        timeout_s=15.0,
        extra_env={"CUDA_VISIBLE_DEVICES": input_cvd},
    )
    assert res.returncode == 0, f"gpu3_guard.sh failed for {input_cvd}: {res.stderr}"
    sanitized = res.stdout.strip()
    assert sanitized == expected_sanitized, (
        f"Expected sanitized CUDA_VISIBLE_DEVICES={expected_sanitized!r}, got {sanitized!r}"
    )
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": sanitized}) is True


def test_f01_bva_accept_multi_gpu_012_without_gpu3() -> None:
    """Verify multi-GPU CUDA_VISIBLE_DEVICES=0,1,2 is preserved and EGL returns 3 CUDA devices."""
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": "0,1,2"}) is True

    cmd = (
        f"export CUDA_VISIBLE_DEVICES='0,1,2'; "
        f"source '{GPU3_GUARD_SH}' && printf '%s' \"$CUDA_VISIBLE_DEVICES\""
    )
    res = run_command_with_gpu_guard(
        ["bash", "-lc", cmd],
        timeout_s=15.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "0,1,2"},
    )
    assert res.returncode == 0, f"gpu3_guard.sh failed for 0,1,2: {res.stderr}"
    assert res.stdout.strip() == "0,1,2"

    filter_so = _resolve_filter_so()
    egl_probe_py = (
        "import ctypes, json\n"
        "ctypes.CDLL('libEGL.so.1', mode=ctypes.RTLD_GLOBAL)\n"
        "lib = ctypes.CDLL(None)\n"
        "lib.eglGetProcAddress.argtypes = [ctypes.c_char_p]\n"
        "lib.eglGetProcAddress.restype = ctypes.c_void_p\n"
        "q_ptr = lib.eglGetProcAddress(b'eglQueryDevicesEXT')\n"
        "a_ptr = lib.eglGetProcAddress(b'eglQueryDeviceAttribEXT')\n"
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
        "print(json.dumps({'count': n.value, 'cuda_ids': sorted(ids)}))\n"
    )
    egl_res = run_command_with_gpu_guard(
        [sys.executable, "-c", egl_probe_py],
        timeout_s=15.0,
        extra_env={
            "CUDA_VISIBLE_DEVICES": "0,1,2",
            "LD_PRELOAD": str(filter_so),
            "__EGL_VENDOR_LIBRARY_FILENAMES": str(NVIDIA_ICD_JSON),
        },
    )
    assert egl_res.returncode == 0, f"EGL multi-GPU probe failed: {egl_res.stderr}"
    parsed = json.loads(egl_res.stdout.strip().splitlines()[-1])
    assert parsed["count"] == 3, f"Expected 3 filtered EGL devices, got {parsed}"
    assert parsed["cuda_ids"] == [0, 1, 2], f"Expected [0, 1, 2], got {parsed}"


def test_f01_bva_fail_closed_missing_filter_so_and_icd_check(tmp_path: Path) -> None:
    """Verify gpu3_guard.sh fails closed when libegl_gpu_filter.so cannot be located."""
    isolated_guard = tmp_path / "gpu3_guard.sh"
    shutil.copy2(GPU3_GUARD_SH, isolated_guard)
    fake_ws = tmp_path / "fake_ws"
    fake_ws.mkdir()

    cmd = (
        f"unset AMENT_PREFIX_PATH COLCON_CURRENT_PREFIX LD_PRELOAD; "
        f"export ROS2_WS_DIR='{fake_ws}'; "
        f"source '{isolated_guard}'"
    )
    res = run_command_with_gpu_guard(["bash", "-lc", cmd], timeout_s=10.0)
    assert res.returncode != 0, "gpu3_guard.sh must fail closed when filter .so is missing"
    assert "libegl_gpu_filter.so not found" in res.stderr

    missing_audit = check_ogre2_log_gpu3_excluded(tmp_path / "nonexistent_ogre2.log")
    assert missing_audit.ok is False

    assert NVIDIA_ICD_JSON.is_file()
    icd_data = json.loads(NVIDIA_ICD_JSON.read_text(encoding="utf-8"))
    assert icd_data.get("ICD", {}).get("library_path") == "libEGL_nvidia.so.0"


@pytest.mark.parametrize(
    ("label", "log_body"),
    [
        (
            "llvmpipe",
            "Found Num EGL Devices: 1\nEGL Device: #0\nGL_RENDERER = llvmpipe (LLVM 15.0.7)\n",
        ),
        (
            "device_3",
            "Found Num EGL Devices: 3\nEGL Device: EGL_NV_device_cuda #0\nCreated GL 4.5 context for device #3\n",
        ),
        (
            "renderD131",
            "Found Num EGL Devices: 1\nEGL Device: /dev/dri/renderD131 #0\nGL_RENDERER = NVIDIA\n",
        ),
        (
            "four_egl_devices",
            "Found Num EGL Devices: 4\nEGL Device: EGL_NV_device_cuda #0\nGL_RENDERER = NVIDIA\n",
        ),
    ],
)
def test_f01_bva_detect_ogre2_log_violations(
    tmp_path: Path, label: str, log_body: str
) -> None:
    """Verify check_ogre2_log_gpu3_excluded() rejects llvmpipe, #3, renderD131, and 4 EGL devices."""
    bad_log = tmp_path / f"ogre2_{label}.log"
    bad_log.write_text(log_body, encoding="utf-8")
    audit = check_ogre2_log_gpu3_excluded(bad_log)
    assert audit.ok is False, f"Expected violation for {label}, got ok=True"
    assert len(audit.violations) >= 1


def test_f01_bva_concurrent_gpu3_guard_idempotent_ld_preload() -> None:
    """Run 4 parallel gpu3_guard.sh sourcing subprocesses and verify idempotent deduplication."""
    cmd = (
        f"export LD_PRELOAD='/tmp/nonexistent_other.so:/tmp/old/libegl_gpu_filter.so'; "
        f"export CUDA_VISIBLE_DEVICES='3'; "
        f"source '{GPU3_GUARD_SH}' && source '{GPU3_GUARD_SH}' && "
        f"printf '%s|%s' \"$CUDA_VISIBLE_DEVICES\" \"$LD_PRELOAD\""
    )

    def _run_once(_: int) -> tuple[int, str, str]:
        res = run_command_with_gpu_guard(
            ["bash", "-lc", cmd],
            timeout_s=15.0,
            extra_env={"CUDA_VISIBLE_DEVICES": "3"},
        )
        return res.returncode, res.stdout.strip(), res.stderr.strip()

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_run_once, range(4)))

    for rc, out, err in results:
        assert rc == 0, f"Concurrent gpu3_guard.sh failed: {err}"
        cvd, ldp = out.split("|", 1)
        assert cvd == "2", f"Expected sanitized CVD='2', got {cvd!r}"
        entries = [e for e in ldp.split(":") if e]
        filter_hits = [e for e in entries if Path(e).name == "libegl_gpu_filter.so"]
        assert len(filter_hits) == 1, f"Expected deduplicated filter entry, got {entries}"
        assert Path(entries[0]).name == "libegl_gpu_filter.so"
        assert "/tmp/nonexistent_other.so" in entries
