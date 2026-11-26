"""Tier 1 functional opaque-box tests for F11: Regression, Hardware Safety & Git Anchors."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from helpers.arena_parser import load_obstacles_yaml
from helpers.hardware_git_guard import (
    Gpu3Status,
    check_cuda_env_excludes_gpu3,
    check_git_anchors,
    check_gpu3_zero_occupancy,
    check_ogre2_log_gpu3_excluded,
)
from helpers.media_validator import check_video_quality
from helpers.metrics_validator import validate_mission_summary, validate_mission_trace
from helpers.ros_probe import run_colcon_test, run_command_with_gpu_guard

pytestmark = [
    pytest.mark.tier1,
    pytest.mark.feature(id="F11"),
]


@pytest.mark.pre_integration_safe
def test_f11_colcon_test_result_baseline_pass(ros2_ws_root: Path) -> None:
    """Verify existing 4-package colcon test-result baseline passes >=22 tests with 0 defects."""
    res = run_colcon_test(ros2_ws_root, result_only=True, timeout_s=30.0)
    assert res.ok is True, f"colcon test-result baseline failed: {res.raw_summary}"
    assert res.total >= 22, f"Expected >= 22 baseline tests, got {res.total}"
    assert res.errors == 0
    assert res.failures == 0
    assert res.skipped == 0
    for pkg in ("vision_pkg", "planning_pkg", "center_node_pkg", "flight_ctrl_pkg"):
        xml_path = ros2_ws_root / "build" / pkg / "pytest.xml"
        assert xml_path.is_file(), f"Missing baseline pytest.xml for package '{pkg}': {xml_path}"


@pytest.mark.requires_m1
@pytest.mark.requires_m2
@pytest.mark.requires_m3
@pytest.mark.requires_m4
def test_f11_full_four_package_colcon_test_expanded(ros2_ws_root: Path) -> None:
    """Execute full 4-package colcon test suite post-integration and verify expanded >22 tests."""
    res = run_colcon_test(
        ros2_ws_root,
        packages=["vision_pkg", "planning_pkg", "center_node_pkg", "flight_ctrl_pkg"],
        result_only=False,
        timeout_s=300.0,
    )
    assert res.ok is True, f"Full 4-package colcon test failed:\n{res.raw_summary}"
    assert res.errors == 0
    assert res.failures == 0
    assert res.skipped == 0
    assert res.total > 22, (
        f"Expected expanded test count > 22 after M1-M4 integration, got {res.total}"
    )


def _check_gpu3_live_with_external_race_reprobe() -> Gpu3Status:
    """Check GPU-3 zero local occupancy, re-probing once on transient external-PID name race."""
    status = check_gpu3_zero_occupancy()
    if (
        not status.ok
        and status.gpu3_pids
        and all(not Path(f"/proc/{pid}").exists() for pid in status.gpu3_pids)
    ):
        time.sleep(0.25)
        reprobe = check_gpu3_zero_occupancy()
        if (
            reprobe.ok
            and set(status.gpu3_pids).issubset(set(reprobe.external_pids))
            and all(not Path(f"/proc/{pid}").exists() for pid in status.gpu3_pids)
        ):
            return reprobe
    return status


@pytest.mark.pre_integration_safe
def test_f11_gpu3_zero_local_occupancy_live() -> None:
    """Verify CUDA env excludes GPU 3 and nvidia-smi reports zero local occupancy on GPU 3."""
    assert check_cuda_env_excludes_gpu3() is True, (
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} does not exclude GPU 3"
    )
    status = _check_gpu3_live_with_external_race_reprobe()
    assert status.ok is True, f"GPU-3 local occupancy violation: {status.notes}\n{status.raw_output}"
    assert status.gpu3_pids == [], f"Unexpected local PIDs on GPU 3: {status.gpu3_pids}"


@pytest.mark.pre_integration_safe
def test_f11_git_rollback_anchors_intact(ros2_ws_root: Path) -> None:
    """Verify rollback branches main (ba401d5) and feat/flight-ctrl-pkg (0157ecc) are untouched."""
    anchors = check_git_anchors(ros2_ws_root)
    assert anchors.ok is True, f"Git rollback anchor violation: {anchors.errors}"
    assert anchors.actual.get("main") == "ba401d5"
    assert anchors.actual.get("feat/flight-ctrl-pkg") == "0157ecc"

    feat_res = run_command_with_gpu_guard(
        ["git", "-C", str(ros2_ws_root), "rev-parse", "--short", "feat/project2-ultimate"],
        timeout_s=10.0,
    )
    assert feat_res.returncode == 0, f"Missing feat/project2-ultimate branch: {feat_res.stderr}"
    assert feat_res.stdout.strip() != "ba401d5"


@pytest.mark.requires_m4
@pytest.mark.requires_sitl
def test_f11_e2e_mission_summary_and_trace_acceptance_gate(
    ros2_ws_root: Path,
) -> None:
    """Verify Contract #7 E2E quantitative metrics & independent trace geometric recomputation."""
    summary_path = Path(
        os.environ.get(
            "E2E_SITL_SUMMARY_PATH",
            os.environ.get("PROJECT2_SUMMARY_PATH", "/tmp/project2_mission_summary.json"),
        )
    )
    trace_path = Path(
        os.environ.get(
            "E2E_SITL_TRACE_PATH",
            os.environ.get("PROJECT2_TRACE_PATH", "/tmp/project2_mission_trace.jsonl"),
        )
    )
    cfg = load_obstacles_yaml(
        ros2_ws_root / "src" / "flight_ctrl_pkg" / "config" / "obstacles.yaml"
    )

    sres = validate_mission_summary(
        summary_path,
        min_drones=6,
        min_targets=3,
        min_obstacles=6,
        min_inter_drone_m=1.0,
        min_obstacle_clearance_m=0.3,
        max_helipad_err_m=2.0,
    )
    assert sres.ok is True, f"E2E mission summary gate failed: {sres.errors}"
    assert sres.data["inter_uav_collisions"] == 0
    assert sres.data["obstacle_collisions"] == 0
    assert sres.data["targets_neutralized"] == sres.data["target_count"] >= 3
    assert sres.data["process_crashes"] == 0
    kill_ids = [item["target_id"] for item in sres.data["target_kill_details"]]
    assert len(set(kill_ids)) == sres.data["targets_neutralized"]

    tres = validate_mission_trace(
        trace_path,
        obstacles_cfg=cfg,
        summary_data=sres.data,
        dist_tolerance_m=0.25,
    )
    assert tres.ok is True, f"E2E mission trace recomputation gate failed: {tres.errors}"


@pytest.mark.requires_m4
@pytest.mark.requires_m5
def test_f11_showcase_ogre2_and_metrics_cross_check(
    ros2_ws_root: Path, showcase_output_dir: Path
) -> None:
    """Cross-check final MP4, Ogre2 GPU-3 log, Contract #7 summary, GPU-3 live & Git anchors."""
    mp4 = showcase_output_dir / "swarm_showcase_cinema.mp4"
    qres = check_video_quality(
        mp4, min_w=1920, min_h=1080, min_fps=30.0, sample_frames=12, min_duration_s=10.0
    )
    assert qres.ok is True, f"Showcase MP4 gate failed: {qres.errors}"

    ogre_log = Path(
        os.environ.get("OGRE2_LOG_PATH", str(Path.home() / ".gz" / "rendering" / "ogre2.log"))
    )
    ores = check_ogre2_log_gpu3_excluded(ogre_log)
    assert ores.ok is True, f"Ogre2 GPU-3 audit failed: {ores.violations}"

    summary_path = Path(
        os.environ.get("E2E_SITL_SUMMARY_PATH")
        or os.environ.get("PROJECT2_SUMMARY_PATH")
        or "/tmp/project2_mission_summary.json"
    )
    sres = validate_mission_summary(summary_path)
    assert sres.ok is True, f"Summary gate failed: {sres.errors}"
    assert sres.data["gpu3_violation"] is False
    assert sres.data["completed"] is True
    kill_ids = [item["target_id"] for item in sres.data["target_kill_details"]]
    assert len(set(kill_ids)) >= sres.data["targets_neutralized"], (
        f"Duplicate or insufficient unique target_id kills: {kill_ids}"
    )

    gpu_status = _check_gpu3_live_with_external_race_reprobe()
    assert gpu_status.ok is True, (
        f"GPU-3 local occupancy violation: {gpu_status.notes}\n{gpu_status.raw_output}"
    )
    assert check_git_anchors(ros2_ws_root).ok is True
