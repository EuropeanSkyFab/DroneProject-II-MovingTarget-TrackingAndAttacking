"""Tier 2 BVA & corner-case opaque-box tests for F11: Audit Gates & Verifiers."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict

import pytest

from helpers.arena_parser import load_obstacles_yaml
from helpers.hardware_git_guard import (
    DEFAULT_GPU3_UUID,
    check_cuda_env_excludes_gpu3,
    check_git_anchors,
    check_gpu3_zero_occupancy,
    check_ogre2_log_gpu3_excluded,
)
from helpers.metrics_validator import (
    REQUIRED_SUMMARY_KEYS,
    validate_mission_summary,
    validate_mission_trace,
)
from helpers.ros_probe import run_colcon_test

pytestmark = [
    pytest.mark.tier2,
    pytest.mark.feature(id="F11"),
    pytest.mark.pre_integration_safe,
]


def _valid_summary() -> Dict[str, Any]:
    return {
        "trace_id": "trace_f11_bva",
        "drone_count": 6,
        "target_count": 3,
        "obstacle_count": 6,
        "vision_samples": 120,
        "stereo_3d_samples": 100,
        "vision_3d_rmse_m": 0.42,
        "imm_pred_1s_rmse_m": 0.65,
        "min_inter_drone_dist_m": 1.0,
        "min_obstacle_clearance_m": 0.3,
        "inter_uav_collisions": 0,
        "obstacle_collisions": 0,
        "targets_neutralized": 3,
        "target_kill_details": [
            {"target_id": "t1", "striker_id": "drone_1", "miss_dist_m": 0.35},
            {"target_id": "t2", "striker_id": "drone_2", "miss_dist_m": 0.41},
            {"target_id": "t3", "striker_id": "drone_3", "miss_dist_m": 0.28},
        ],
        "re_tasking_events": 2,
        "return_helipad_error_m": 2.0,
        "process_crashes": 0,
        "gpu3_violation": False,
        "completed": True,
    }


def _write_trace(path: Path, min_inter_dist: float, min_obs_x: float = 20.0, tid: str = "trace_f11_bva") -> None:
    frames = [
        {
            "timestamp": 1.0,
            "trace_id": tid,
            "span_id": "s1",
            "phase": "ATTACK",
            "drones": {
                "drone_1": {"pos": [min_obs_x, 0.0, 8.0], "vel": [1.0, 0.0, 0.0]},
                "drone_2": {"pos": [min_obs_x, min_inter_dist, 8.0], "vel": [1.0, 0.0, 0.0]},
            },
        },
        {
            "timestamp": 2.0,
            "trace_id": tid,
            "span_id": "s2",
            "phase": "RETURN",
            "drones": {
                "drone_1": {"pos": [0.0, 0.0, 0.2], "vel": [0.0, 0.0, 0.0]},
                "drone_2": {"pos": [0.0, 3.0, 0.2], "vel": [0.0, 0.0, 0.0]},
            },
        },
    ]
    path.write_text("\n".join(json.dumps(f) for f in frames) + "\n", encoding="utf-8")


def test_f11_bva_git_anchor_drift_and_missing_repo_detection(tmp_path: Path) -> None:
    """Verify check_git_anchors catches 1-commit drift, deleted branch, missing repo, and non-git dir."""
    non_git = tmp_path / "not_a_repo"
    non_git.mkdir()
    assert check_git_anchors(non_git).ok is False
    assert check_git_anchors(tmp_path / "missing_repo").ok is False

    repo = tmp_path / "repo"
    repo.mkdir()
    git_env = dict(
        os.environ,
        GIT_AUTHOR_NAME="E2E",
        GIT_AUTHOR_EMAIL="e2e@example.com",
        GIT_COMMITTER_NAME="E2E",
        GIT_COMMITTER_EMAIL="e2e@example.com",
    )
    subprocess.run(["git", "-C", str(repo), "init"], check=True, capture_output=True)
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "README.md"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "c1"], check=True, env=git_env, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "branch", "-M", "main"], check=True, capture_output=True)
    c1 = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--short", "main"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    subprocess.run(["git", "-C", str(repo), "branch", "feat/flight-ctrl-pkg", "main"], check=True, capture_output=True)

    expected = {"main": c1, "feat/flight-ctrl-pkg": c1}
    assert check_git_anchors(repo, expected=expected).ok is True

    # Advance feat/flight-ctrl-pkg by 1 commit
    subprocess.run(["git", "-C", str(repo), "checkout", "feat/flight-ctrl-pkg"], check=True, capture_output=True)
    (repo / "README.md").write_text("moved\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "commit", "-am", "c2"], check=True, env=git_env, capture_output=True)
    moved = check_git_anchors(repo, expected=expected)
    assert moved.ok is False and any("feat/flight-ctrl-pkg" in e for e in moved.errors)

    # Delete branch
    subprocess.run(["git", "-C", str(repo), "checkout", "main"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "branch", "-D", "feat/flight-ctrl-pkg"], check=True, capture_output=True)
    assert check_git_anchors(repo, expected=expected).ok is False


def test_f11_bva_gpu3_compute_vram_and_cuda_env_violations() -> None:
    """Verify GPU-3 compute/VRAM, local PID, pmon graphics, and CUDA env boundaries."""
    gpu_map = (
        "0, GPU-b5e964a1-8266-9595-863d-e26136c268ca, 00000000:3B:00.0\n"
        "2, GPU-5643e739-f46d-feb6-e907-3aa2c1ad4963, 00000000:B1:00.0\n"
        "3, GPU-617ae266-eed3-ef41-ae0f-5f25c730f5d5, 00000000:D9:00.0\n"
    )
    # 1 MiB named compute violation on GPU 3 fails
    res_vram = check_gpu3_zero_occupancy(
        nvidia_smi_output="00000000:D9:00.0, 998877, gz, 1\n",
        gpu_map_output=gpu_map,
    )
    assert res_vram.ok is False and res_vram.gpu3_mem_mib == 1

    # Local /proc/<pid> with [Not Found] fails
    res_local = check_gpu3_zero_occupancy(
        nvidia_smi_output=f"{DEFAULT_GPU3_UUID.upper()}, {os.getpid()}, [Not Found], 16\n",
        gpu_map_output=gpu_map,
    )
    assert res_local.ok is False and os.getpid() in res_local.gpu3_pids

    # pmon Type=G ffmpeg on GPU 3 fails
    res_pmon = check_gpu3_zero_occupancy(
        nvidia_smi_output="",
        pmon_output="# gpu pid type sm mem enc dec command\n  3 55555 G 1 2 0 0 ffmpeg\n",
        gpu_map_output=gpu_map,
    )
    assert res_pmon.ok is False and 55555 in res_pmon.gpu3_pids

    # Unowned sibling [Not Found] + Codex GUI allowed when ignore_external_unowned=True
    res_ext = check_gpu3_zero_occupancy(
        nvidia_smi_output="00000000:D9:00.0, 999991, [Not Found], 4096\n",
        pmon_output="# gpu pid type sm mem enc dec command\n  3 3216 G - - - - Codex -\n",
        gpu_map_output=gpu_map,
        ignore_external_unowned=True,
    )
    assert res_ext.ok is True
    assert check_gpu3_zero_occupancy(
        nvidia_smi_output="00000000:D9:00.0, 999991, [Not Found], 4096\n",
        gpu_map_output=gpu_map,
        ignore_external_unowned=False,
    ).ok is False

    # CUDA_VISIBLE_DEVICES boundaries
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": "2"}) is True
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": "0,1,2"}) is True
    for bad_val in ("3", "0,1,2,3", "all", "", DEFAULT_GPU3_UUID):
        assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": bad_val}) is False


def test_f11_bva_non_zero_collision_and_partial_neutralization_rejection(
    tmp_path: Path,
) -> None:
    """Verify exact Contract #7 floor pass vs collision, partial kill, and missing key rejection."""
    base_p = tmp_path / "base_summary.json"
    base_data = _valid_summary()
    base_p.write_text(json.dumps(base_data), encoding="utf-8")
    assert validate_mission_summary(base_p).ok is True

    mutations = [
        {"inter_uav_collisions": 1},
        {"obstacle_collisions": 1},
        {"targets_neutralized": 2},
        {"drone_count": 5},
        {"target_count": 2, "targets_neutralized": 2},
        {"obstacle_count": 5},
        {"process_crashes": 1},
        {"vision_samples": 0},
        {"stereo_3d_samples": 121},
    ]
    for idx, mut in enumerate(mutations):
        d = copy.deepcopy(base_data)
        d.update(mut)
        p = tmp_path / f"bad_mut_{idx}.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        assert validate_mission_summary(p).ok is False, f"Mutation {mut} was not rejected"

    for key in REQUIRED_SUMMARY_KEYS:
        d = copy.deepcopy(base_data)
        d.pop(key)
        p = tmp_path / f"missing_{key}.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        assert validate_mission_summary(p).ok is False


def test_f11_bva_colcon_xml_skipped_error_failure_rejection(tmp_path: Path) -> None:
    """Verify run_colcon_test rejects skipped, errored, failed, and 0-test pytest XML summaries."""
    def _setup_ws(name: str, tests: int, errors: int, failures: int, skipped: int) -> Path:
        ws = tmp_path / name
        xml_dir = ws / "build" / "mock_pkg"
        xml_dir.mkdir(parents=True, exist_ok=True)
        (xml_dir / "pytest.xml").write_text(
            f'<?xml version="1.0" encoding="utf-8"?>'
            f'<testsuites><testsuite name="pytest" tests="{tests}" errors="{errors}" '
            f'failures="{failures}" skipped="{skipped}" time="0.1">'
            f'<testcase classname="t" name="test_a" time="0.01"/></testsuite></testsuites>\n',
            encoding="utf-8",
        )
        return ws

    ws_ok = _setup_ws("ws_ok", tests=22, errors=0, failures=0, skipped=0)
    assert run_colcon_test(ws_ok, result_only=True, timeout_s=20.0).ok is True

    for label, counts in (
        ("ws_fail", (22, 0, 1, 0)),
        ("ws_err", (22, 1, 0, 0)),
        ("ws_skip", (22, 0, 0, 1)),
        ("ws_zero", (0, 0, 0, 0)),
    ):
        ws_bad = _setup_ws(label, *counts)
        res = run_colcon_test(ws_bad, result_only=True, timeout_s=20.0)
        assert res.ok is False, f"colcon gate failed to reject {label} ({counts}): {res}"


def test_f11_bva_helipad_error_completed_false_gpu3_flag_boundaries(
    tmp_path: Path,
) -> None:
    """Verify exact numeric & boolean gate boundaries on helipad error, clearance, and flags."""
    base_data = _valid_summary()
    mutations = [
        {"return_helipad_error_m": 2.01},
        {"return_helipad_error_m": {"drone_1": 0.4, "drone_2": 2.01}},
        {"min_inter_drone_dist_m": 0.999},
        {"min_obstacle_clearance_m": 0.299},
        {"completed": False},
        {"gpu3_violation": True},
    ]
    for idx, mut in enumerate(mutations):
        d = copy.deepcopy(base_data)
        d.update(mut)
        p = tmp_path / f"bva_{idx}.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        assert validate_mission_summary(p).ok is False, f"Boundary {mut} was not rejected"

    ok_dict = copy.deepcopy(base_data)
    ok_dict["return_helipad_error_m"] = {"drone_1": 0.5, "drone_2": 2.0}
    ok_p = tmp_path / "ok_dict.json"
    ok_p.write_text(json.dumps(ok_dict), encoding="utf-8")
    assert validate_mission_summary(ok_p).ok is True


def test_f11_bva_ogre2_log_and_trace_recomputation_boundaries(
    tmp_path: Path, ros2_ws_root: Path
) -> None:
    """Verify ogre2.log device count/token gates and trace recomputation 0.25m tolerance."""
    # Ogre2 log boundaries
    log_1 = tmp_path / "ogre_1.log"
    log_1.write_text(
        "Found Num EGL Devices: 1\nEGL Device: #0 NVIDIA\nGL_RENDERER = NVIDIA GeForce RTX 3090\n",
        encoding="utf-8",
    )
    assert check_ogre2_log_gpu3_excluded(log_1).ok is True

    log_3 = tmp_path / "ogre_3.log"
    log_3.write_text(
        "Found Num EGL Devices: 3\nEGL Device: #0\nEGL Device: #1\nEGL Device: #2\n",
        encoding="utf-8",
    )
    assert check_ogre2_log_gpu3_excluded(log_3).ok is True

    for name, content in (
        ("ogre_0.log", "Found Num EGL Devices: 0\n"),
        ("ogre_4.log", "Found Num EGL Devices: 4\nEGL Device: #0\nEGL Device: #3\n"),
        ("ogre_drm.log", "Found Num EGL Devices: 3\nEGL Device: #0 /dev/dri/renderD131\n"),
        ("ogre_bus.log", "Found Num EGL Devices: 3\nEGL Device: #0 pci 0000:d9:00.0\n"),
        ("ogre_sw.log", "Found Num EGL Devices: 1\nEGL Device: #0\nGL_RENDERER = llvmpipe\n"),
    ):
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        assert check_ogre2_log_gpu3_excluded(p).ok is False, f"{name} should fail ogre2 audit"

    # Trace recomputation tolerance: summary min_inter_drone_dist_m = 2.50m
    cfg = load_obstacles_yaml(
        ros2_ws_root / "src" / "flight_ctrl_pkg" / "config" / "obstacles.yaml"
    )
    summary_data = _valid_summary()
    summary_data["min_inter_drone_dist_m"] = 2.50
    summary_data["min_obstacle_clearance_m"] = 27.6

    t_pass = tmp_path / "trace_tol_ok.jsonl"
    _write_trace(t_pass, min_inter_dist=2.26)  # delta 0.24 <= 0.25
    res_ok = validate_mission_trace(t_pass, summary_data=summary_data, dist_tolerance_m=0.25)
    assert res_ok.ok is True, f"Delta 0.24m should pass 0.25m tolerance: {res_ok.errors}"

    t_fail = tmp_path / "trace_tol_bad.jsonl"
    _write_trace(t_fail, min_inter_dist=2.24)  # delta 0.26 > 0.25
    assert validate_mission_trace(t_fail, summary_data=summary_data, dist_tolerance_m=0.25).ok is False

    # Obstacle penetration (-0.01m inside obs_comm_pylon_center at (49,1.5, r=1.4))
    t_pen = tmp_path / "trace_pen.jsonl"
    _write_trace(t_pen, min_inter_dist=2.50, min_obs_x=49.0)
    assert validate_mission_trace(t_pen, obstacles_cfg=cfg).ok is False

    # Trace ID mismatch
    t_tid = tmp_path / "trace_tid.jsonl"
    _write_trace(t_tid, min_inter_dist=2.50, tid="wrong_trace_id")
    assert validate_mission_trace(t_tid, summary_data=summary_data).ok is False
