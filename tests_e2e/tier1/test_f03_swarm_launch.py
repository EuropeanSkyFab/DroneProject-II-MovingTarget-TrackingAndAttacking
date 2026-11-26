"""Tier 1 functional tests for F3: Scalable N >= 6 Swarm Launch & Unobstructed Drone SDF."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from helpers.arena_parser import load_obstacles_yaml, parse_sdf_collisions
from helpers.ros_probe import run_command_with_gpu_guard

pytestmark = [pytest.mark.tier1, pytest.mark.feature(id="F3"), pytest.mark.requires_m1]

ROS2_WS = Path("/home/hkw/flying/ros2_ws")
FC_PKG = ROS2_WS / "src" / "flight_ctrl_pkg"
OBSTACLES_YAML = FC_PKG / "config" / "obstacles.yaml"
START_SWARM_SH = FC_PKG / "scripts" / "start_px4_swarm.sh"
LAUNCH_DIR = FC_PKG / "launch"
MODELS_DIR = FC_PKG / "sim" / "models"


def test_f03_obstacles_yaml_has_at_least_eight_unique_helipads() -> None:
    """Verify obstacles.yaml defines >=6 (at least 8) unique non-overlapping helipads."""
    cfg = load_obstacles_yaml(OBSTACLES_YAML)
    helipads = cfg["helipads"]
    bounds = cfg["arena_bounds"]

    assert len(helipads) >= 8, f"Expected >= 8 helipads, got {len(helipads)}"
    pad_ids = [h["id"] for h in helipads]
    drone_ids = [h["drone_id"] for h in helipads]
    assert len(set(pad_ids)) == len(helipads), f"Duplicate helipad IDs: {pad_ids}"
    assert len(set(drone_ids)) == len(helipads), f"Duplicate drone IDs: {drone_ids}"

    for i in range(1, 9):
        assert f"drone_{i}" in drone_ids, f"Missing drone_{i} helipad mapping"

    for hp in helipads:
        assert bounds["x_min"] <= hp["x"] <= bounds["x_max"]
        assert bounds["y_min"] <= hp["y"] <= bounds["y_max"]
        assert bounds["z_min"] <= hp["z"] <= bounds["z_max"]


def test_f03_start_px4_swarm_syntax_and_n6_flags() -> None:
    """Verify start_px4_swarm.sh passes bash -n and supports N>=6, --world, --navcam, --obstacles."""
    assert START_SWARM_SH.is_file(), f"Missing start_px4_swarm.sh: {START_SWARM_SH}"
    syn = run_command_with_gpu_guard(["bash", "-n", str(START_SWARM_SH)], timeout_s=10.0)
    assert syn.returncode == 0, f"bash -n syntax error in {START_SWARM_SH}: {syn.stderr}"

    txt = START_SWARM_SH.read_text(encoding="utf-8")
    for flag in ("--world", "--navcam", "--obstacles"):
        assert flag in txt, f"start_px4_swarm.sh missing CLI flag {flag}"
    assert "NUM_DRONES" in txt
    assert "x500_down_cam_lite" in txt
    assert "hybrid" in txt
    assert "obstacles.yaml" in txt


def test_f03_swarm_launch_files_installed_and_parameterize_n6() -> None:
    """Verify launch files in active install/ and src/ declare N>=6 swarm arguments."""
    expected_files = (
        "flight_ctrl_swarm.launch.py",
        "project2_full_swarm.launch.py",
        "showcase_sitl.launch.py",
    )
    for fname in expected_files:
        src_file = LAUNCH_DIR / fname
        assert src_file.is_file(), f"Missing launch file in src/: {src_file}"
        src_txt = src_file.read_text(encoding="utf-8")
        assert "num_drones" in src_txt
        assert "obstacles_yaml" in src_txt

        show_res = run_command_with_gpu_guard(
            ["ros2", "launch", "flight_ctrl_pkg", fname, "--show-args"],
            timeout_s=15.0,
            source_env=True,
        )
        assert show_res.returncode == 0, (
            f"Active install/ missing or broken launch file '{fname}': "
            f"{show_res.stderr.strip() or show_res.stdout.strip()}"
        )
        assert "'num_drones'" in show_res.stdout, (
            f"Installed {fname} missing 'num_drones' arg:\n{show_res.stdout}"
        )
        assert "'obstacles_yaml'" in show_res.stdout, (
            f"Installed {fname} missing 'obstacles_yaml' arg:\n{show_res.stdout}"
        )


def test_f03_x500_down_cam_and_lite_unobstructed_camera_sdfs() -> None:
    """Verify x500_down_cam and x500_down_cam_lite place down_cam ahead of gimbal without occlusion."""
    full_sdf = MODELS_DIR / "x500_down_cam" / "model.sdf"
    lite_sdf = MODELS_DIR / "x500_down_cam_lite" / "model.sdf"
    assert full_sdf.is_file()
    assert lite_sdf.is_file()

    def _inspect_cam(sdf_path: Path) -> tuple[list[float], int, int, int, float, float]:
        root = ET.parse(sdf_path).getroot()
        down_link = root.find(".//link[@name='down_cam_link']")
        assert down_link is not None, f"Missing down_cam_link in {sdf_path}"
        pose_vals = [float(v) for v in (down_link.findtext("pose") or "").split()]
        assert len(pose_vals) == 6
        cam = down_link.find("./sensor[@name='down_cam']/camera")
        assert cam is not None
        hfov = float(cam.findtext("horizontal_fov"))
        w = int(cam.findtext("./image/width"))
        h = int(cam.findtext("./image/height"))
        rate = float(down_link.findtext("./sensor[@name='down_cam']/update_rate"))
        return pose_vals, w, h, int(rate), hfov, pose_vals[0]

    full_pose, fw, fh, frate, fhfov, fx = _inspect_cam(full_sdf)
    lite_pose, lw, lh, lrate, lhfov, lx = _inspect_cam(lite_sdf)

    assert fx == pytest.approx(0.12, abs=1e-4)
    assert lx == pytest.approx(0.12, abs=1e-4)
    assert full_pose[4] == pytest.approx(0.61086524, abs=1e-4)
    assert lite_pose[4] == pytest.approx(0.61086524, abs=1e-4)
    assert fhfov == pytest.approx(1.35, abs=1e-4)
    assert lhfov == pytest.approx(1.35, abs=1e-4)
    assert (fw, fh, frate) == (640, 480, 25)
    assert (lw, lh, lrate) == (480, 360, 15)

    # Check gimbal in x500_down_cam is centered at x=0.0 behind down_cam (x=+0.12)
    full_root = ET.parse(full_sdf).getroot()
    gimbal_yaw = full_root.find(".//link[@name='gimbal_yaw_link']")
    assert gimbal_yaw is not None
    gy_pose = [float(v) for v in (gimbal_yaw.findtext("pose") or "").split()]
    assert gy_pose[0] <= 0.0, f"Gimbal x={gy_pose[0]} must stay behind down_cam x={fx}"


def test_f03_multi_target_uav_sdfs_complete() -> None:
    """Verify red_target_uav, amber_target_uav, and slate_target_uav SDFs have valid physics & plugins."""
    for model_name in ("red_target_uav", "amber_target_uav", "slate_target_uav"):
        sdf_path = MODELS_DIR / model_name / "model.sdf"
        assert sdf_path.is_file(), f"Missing target model SDF: {sdf_path}"
        audit = parse_sdf_collisions(sdf_path)
        assert len(audit.collisions) >= 1, f"{model_name} has no <collision>"
        assert any(m["visual_count"] >= 3 for m in audit.models), f"{model_name} missing visuals"
        assert any(m["has_particle_emitter"] for m in audit.models), (
            f"{model_name} missing kill_burst particle emitter"
        )
        plugin_names = {p["name"] for p in audit.plugins}
        assert "gz::sim::systems::Thermal" in plugin_names, f"{model_name} missing Thermal"
        assert "gz::sim::systems::VelocityControl" in plugin_names, (
            f"{model_name} missing VelocityControl"
        )
