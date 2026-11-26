"""Comprehensive positive and negative self-checks for tests_e2e verifiers and runner."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from helpers.arena_parser import (
    check_line_of_sight,
    load_obstacles_yaml,
    parse_sdf_collisions,
    point_min_clearance,
)
from helpers.hardware_git_guard import (
    check_cuda_env_excludes_gpu3,
    check_git_anchors,
    check_gpu3_zero_occupancy,
    check_ogre2_log_gpu3_excluded,
)
from helpers.media_validator import check_video_quality, probe_video
from helpers.metrics_validator import (
    validate_kill_event_msg,
    validate_mission_summary,
    validate_mission_trace,
    validate_planning_target_msg,
    validate_vision_targets_msg,
)
from helpers.ros_probe import run_colcon_test, run_command_with_gpu_guard

pytestmark = [pytest.mark.self_check, pytest.mark.pre_integration_safe]

ROS2_WS = Path("/home/hkw/flying/ros2_ws")
MOVING_TARGET_SDF = (
    ROS2_WS / "src" / "flight_ctrl_pkg" / "sim" / "worlds" / "moving_target_arena.sdf"
)


# ---------------------------------------------------------------------------
# 1. hardware_git_guard.py tests
# ---------------------------------------------------------------------------


def test_gpu3_live_occupancy_allows_external_unowned_only() -> None:
    status = check_gpu3_zero_occupancy()
    assert status.ok is True, f"Unexpected GPU-3 violation: {status}"
    assert status.gpu3_pids == []


def test_gpu3_synthetic_positive_and_negative_fixtures() -> None:
    gpu_map = (
        "0, GPU-b5e964a1-8266-9595-863d-e26136c268ca, 00000000:3B:00.0\n"
        "2, GPU-5643e739-f46d-feb6-e907-3aa2c1ad4963, 00000000:B1:00.0\n"
        "3, GPU-617ae266-eed3-ef41-ae0f-5f25c730f5d5, 00000000:D9:00.0\n"
    )
    external_only = (
        "00000000:D9:00.0, 999991, [Not Found], 6654\n"
        "00000000:B1:00.0, 999992, python3, 1200\n"
    )
    pmon_desktop = (
        "# gpu pid type sm mem enc dec command\n"
        "    3   3216   G   -   -   -   -   Codex -\n"
    )

    ok_res = check_gpu3_zero_occupancy(
        nvidia_smi_output=external_only,
        pmon_output=pmon_desktop,
        gpu_map_output=gpu_map,
        ignore_external_unowned=True,
    )
    assert ok_res.ok is True
    assert ok_res.external_pids == [999991, 3216]

    strict_res = check_gpu3_zero_occupancy(
        nvidia_smi_output=external_only,
        pmon_output="",
        gpu_map_output=gpu_map,
        ignore_external_unowned=False,
    )
    assert strict_res.ok is False
    assert strict_res.gpu3_pids == [999991]
    assert strict_res.gpu3_mem_mib == 6654

    named_gpu3 = "00000000:D9:00.0, 999993, gz sim, 2048\n"
    named_res = check_gpu3_zero_occupancy(
        nvidia_smi_output=named_gpu3, gpu_map_output=gpu_map
    )
    assert named_res.ok is False
    assert named_res.gpu3_pids == [999993]

    # Live local PID in /proc/<pid> must fail even if process_name is [Not Found]
    local_pid_csv = f"GPU-617ae266-eed3-ef41-ae0f-5f25c730f5d5, {os.getpid()}, [Not Found], 512\n"
    local_res = check_gpu3_zero_occupancy(
        nvidia_smi_output=local_pid_csv, gpu_map_output=gpu_map
    )
    assert local_res.ok is False
    assert os.getpid() in local_res.gpu3_pids

    pmon_gz = (
        "# gpu pid type sm mem enc dec command\n"
        "    3   424242   G   5   10   0   0   gz sim -s\n"
    )
    pmon_res = check_gpu3_zero_occupancy(
        nvidia_smi_output="", pmon_output=pmon_gz, gpu_map_output=gpu_map
    )
    assert pmon_res.ok is False
    assert 424242 in pmon_res.gpu3_pids


def test_cuda_env_excludes_gpu3_positive_and_negative() -> None:
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": "2"}) is True
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": "0,1,2"}) is True
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": "-1"}) is True
    assert (
        check_cuda_env_excludes_gpu3(
            {"CUDA_VISIBLE_DEVICES": "GPU-5643e739-f46d-feb6-e907-3aa2c1ad4963"}
        )
        is True
    )

    assert check_cuda_env_excludes_gpu3({}) is False
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": ""}) is False
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": "all"}) is False
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": "3"}) is False
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": "0,1,2,3"}) is False
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": "invalid"}) is False
    assert (
        check_cuda_env_excludes_gpu3(
            {"CUDA_VISIBLE_DEVICES": "GPU-617ae266-eed3-ef41-ae0f-5f25c730f5d5"}
        )
        is False
    )


def test_ogre2_log_audit_positive_and_negative(tmp_path: Path) -> None:
    good_log = tmp_path / "ogre2_good.log"
    good_log.write_text(
        "\n".join(
            [
                "19:16:56: Found Num EGL Devices: 1",
                "19:16:56: EGL Device: EGL_NV_device_cuda EGL_EXT_device_drm #0",
                "19:16:56: Trying to init device: EGL_NV_device_cuda #0...",
                "19:16:56: Created GL 4.5 context for device EGL_NV_device_cuda #0",
                "19:16:56: GL_VENDOR = NVIDIA Corporation",
                "19:16:56: GL_RENDERER = NVIDIA GeForce RTX 3090/PCIe/SSE2",
            ]
        ),
        encoding="utf-8",
    )
    good_audit = check_ogre2_log_gpu3_excluded(good_log)
    assert good_audit.ok is True
    assert good_audit.num_egl_devices == 1
    assert "#0" in good_audit.detected_devices

    bad_log = tmp_path / "ogre2_bad.log"
    bad_log.write_text(
        "\n".join(
            [
                "19:16:56: Found Num EGL Devices: 4",
                "19:16:56: EGL Device: EGL_NV_device_cuda #0",
                "19:16:56: EGL Device: EGL_NV_device_cuda #3",
                "19:16:56: Created GL 4.5 context for device EGL_NV_device_cuda #3",
            ]
        ),
        encoding="utf-8",
    )
    bad_audit = check_ogre2_log_gpu3_excluded(bad_log)
    assert bad_audit.ok is False
    assert bad_audit.num_egl_devices == 4
    assert len(bad_audit.violations) >= 2

    sw_log = tmp_path / "ogre2_sw.log"
    sw_log.write_text(
        "Found Num EGL Devices: 1\nEGL Device: #0\nGL_RENDERER = llvmpipe (LLVM 15)\n",
        encoding="utf-8",
    )
    assert check_ogre2_log_gpu3_excluded(sw_log).ok is False
    assert check_ogre2_log_gpu3_excluded(tmp_path / "missing.log").ok is False


def test_git_anchors_live_and_negative() -> None:
    res = check_git_anchors(ROS2_WS)
    assert res.ok is True, f"Git anchor check failed: {res.errors}"
    assert res.actual["main"] == "ba401d5"
    assert res.actual["feat/flight-ctrl-pkg"] == "0157ecc"

    bad_hash = check_git_anchors(ROS2_WS, expected={"main": "deadbeef"})
    assert bad_hash.ok is False

    non_git = check_git_anchors(ROS2_WS.parent)
    assert non_git.ok is False


# ---------------------------------------------------------------------------
# 2. arena_parser.py tests
# ---------------------------------------------------------------------------


def test_load_obstacles_yaml_valid_and_invalid(tmp_path: Path) -> None:
    yaml_file = tmp_path / "obstacles.yaml"
    yaml_file.write_text(
        """
arena_bounds:
  x_min: -50.0
  x_max: 80.0
  y_min: -50.0
  y_max: 50.0
  z_min: 0.0
  z_max: 35.0
helipads:
  - {id: H1, drone_id: drone_1, x: 0.0, y: 0.0, z: 0.1, yaw: 0.0}
  - {id: H2, drone_id: drone_2, x: 0.0, y: 3.0, z: 0.1, yaw: 0.0}
obstacles:
  - {id: tower_1, type: cylinder, cx: 15.0, cy: 0.0, z_min: 0.0, z_max: 12.0, radius: 2.0, safety_margin_m: 0.5}
  - {id: wall_1, type: box, cx: 30.0, cy: 10.0, z_min: 0.0, z_max: 10.0, sx: 6.0, sy: 2.0, yaw: 0.7853981633974483}
no_fly_zones:
  - {id: nfz_1, cx: -10.0, cy: -10.0, z_min: 0.0, z_max: 25.0, radius: 4.0}
""",
        encoding="utf-8",
    )
    cfg = load_obstacles_yaml(yaml_file)
    assert len(cfg["helipads"]) == 2
    assert len(cfg["obstacles"]) == 2
    assert len(cfg["no_fly_zones"]) == 1

    # Exact clearance checks:
    # Cylinder at (15,0) r=2, z in [0,12]: point (18, 0, 5) -> 1.0m
    assert point_min_clearance((18.0, 0.0, 5.0), cfg) == pytest.approx(1.0, abs=1e-6)
    # 3-4-5 diagonal from top rim (r=2, z=12): point (20, 0, 16) -> dx=3, dz=4 -> 5.0m
    assert point_min_clearance((20.0, 0.0, 16.0), cfg) == pytest.approx(5.0, abs=1e-6)
    # Inside cylinder: point (15.5, 0.0, 6.0) -> -1.5m
    assert point_min_clearance((15.5, 0.0, 6.0), cfg) == pytest.approx(-1.5, abs=1e-6)
    # Rotated box at (30,10) yaw=45deg, sx=6 (hx=3), sy=2 (hy=1):
    # Point displaced along local +Y by 2.0m -> world offset (-sin45*2, cos45*2) -> clearance = 1.0m
    s45 = math.sin(math.pi / 4.0)
    c45 = math.cos(math.pi / 4.0)
    pt_box = (30.0 - s45 * 2.0, 10.0 + c45 * 2.0, 5.0)
    assert point_min_clearance(pt_box, cfg) == pytest.approx(1.0, abs=1e-5)
    # Inside NFZ -> negative unless include_nfz=False
    assert point_min_clearance((-10.0, -10.0, 5.0), cfg, include_nfz=True) == pytest.approx(-4.0)
    assert point_min_clearance((-10.0, -10.0, 5.0), cfg, include_nfz=False) > 10.0

    # Line of sight checks:
    assert check_line_of_sight((10.0, 0.0, 5.0), (20.0, 0.0, 5.0), cfg) is False
    assert check_line_of_sight((10.0, 0.0, 13.0), (20.0, 0.0, 13.0), cfg) is True
    assert check_line_of_sight((10.0, 2.5, 5.0), (20.0, 2.5, 5.0), cfg, margin_m=0.0) is True
    assert check_line_of_sight((10.0, 2.5, 5.0), (20.0, 2.5, 5.0), cfg, margin_m=0.6) is False
    # Through rotated box
    assert check_line_of_sight((27.0, 7.0, 5.0), (33.0, 13.0, 5.0), cfg) is False
    # NaN coordinate must raise ValueError
    with pytest.raises(ValueError):
        point_min_clearance((float("nan"), 0.0, 5.0), cfg)

    # Invalid YAML cases
    bad_yaml = tmp_path / "bad_obstacles.yaml"
    bad_yaml.write_text(
        """
arena_bounds: {x_min: 10, x_max: -10, y_min: 0, y_max: 10, z_min: 0, z_max: 10}
helipads: []
obstacles: []
no_fly_zones: []
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_obstacles_yaml(bad_yaml)


def test_parse_sdf_collisions_real_and_composed_poses(tmp_path: Path) -> None:
    assert MOVING_TARGET_SDF.is_file()
    audit = parse_sdf_collisions(MOVING_TARGET_SDF)
    assert audit.world_name == "moving_target_arena"
    assert any(c.is_ground and c.geometry_type == "plane" for c in audit.collisions)
    assert any(light["name"] == "sunUTC" and light["cast_shadows"] is True for light in audit.lights)
    assert any("gz-sim-sensors-system" in p["filename"] for p in audit.plugins)

    composed_sdf = tmp_path / "composed.sdf"
    composed_sdf.write_text(
        """<?xml version="1.0"?>
<sdf version="1.9">
  <world name="test_world">
    <model name="building_a">
      <static>true</static>
      <pose>10 20 0 0 0 1.5707963267948966</pose>
      <link name="body">
        <pose>2 0 5 0 0 0</pose>
        <collision name="box_col">
          <pose>0 1 0 0 0 0</pose>
          <geometry><box><size>4 2 10</size></box></geometry>
        </collision>
      </link>
    </model>
  </world>
</sdf>
""",
        encoding="utf-8",
    )
    comp_audit = parse_sdf_collisions(composed_sdf)
    assert len(comp_audit.collisions) == 1
    col = comp_audit.collisions[0]
    # Model at (10,20,0) rotated +90 deg yaw: local (2, 1, 5) -> world (10 - 1, 20 + 2, 5) = (9, 22, 5)
    assert col.world_pose[0] == pytest.approx(9.0, abs=1e-5)
    assert col.world_pose[1] == pytest.approx(22.0, abs=1e-5)
    assert col.world_pose[2] == pytest.approx(5.0, abs=1e-5)
    assert col.dimensions == {"sx": 4.0, "sy": 2.0, "sz": 10.0}


# ---------------------------------------------------------------------------
# 3. metrics_validator.py tests
# ---------------------------------------------------------------------------


def _make_valid_summary(trace_id: str = "trace_ok_001") -> dict:
    return {
        "trace_id": trace_id,
        "drone_count": 6,
        "target_count": 3,
        "obstacle_count": 6,
        "vision_samples": 240,
        "stereo_3d_samples": 210,
        "vision_3d_rmse_m": 0.45,
        "imm_pred_1s_rmse_m": 0.62,
        "min_inter_drone_dist_m": 2.5,
        "min_obstacle_clearance_m": 1.0,
        "inter_uav_collisions": 0,
        "obstacle_collisions": 0,
        "targets_neutralized": 3,
        "target_kill_details": [
            {"target_id": "tgt_1", "striker_id": "drone_1", "miss_dist_m": 0.31},
            {"target_id": "tgt_2", "striker_id": "drone_3", "miss_dist_m": 0.28},
            {"target_id": "tgt_3", "striker_id": "drone_5", "miss_dist_m": 0.40},
        ],
        "re_tasking_events": 2,
        "return_helipad_error_m": {f"drone_{i}": 0.2 for i in range(1, 7)},
        "process_crashes": 0,
        "gpu3_violation": False,
        "completed": True,
    }


def test_metrics_validator_summary_and_trace(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_data = _make_valid_summary()
    summary_path.write_text(json.dumps(summary_data), encoding="utf-8")

    res = validate_mission_summary(summary_path)
    assert res.ok is True, res.errors

    # Negative summary: collision + gpu3 violation + NaN + incomplete neutralization
    bad_summary = dict(summary_data)
    bad_summary["inter_uav_collisions"] = 1
    bad_summary["gpu3_violation"] = True
    bad_summary["targets_neutralized"] = 2
    bad_summary["vision_3d_rmse_m"] = float("nan")
    bad_path = tmp_path / "bad_summary.json"
    bad_path.write_text(json.dumps(bad_summary), encoding="utf-8")
    bad_res = validate_mission_summary(bad_path)
    assert bad_res.ok is False
    assert len(bad_res.errors) >= 4

    # Trace recomputation against obstacle YAML
    obs_cfg = {
        "arena_bounds": {"x_min": -50, "x_max": 50, "y_min": -50, "y_max": 50, "z_min": 0, "z_max": 30},
        "helipads": [],
        "obstacles": [
            {
                "id": "cyl_1",
                "type": "cylinder",
                "cx": 10.0,
                "cy": 0.0,
                "z_min": 0.0,
                "z_max": 12.0,
                "radius": 2.0,
                "yaw": 0.0,
                "safety_margin_m": 0.5,
            }
        ],
        "no_fly_zones": [],
    }
    # Drone_1 at (13.0, 0.0, 5.0) -> obstacle clearance = 1.0m
    # Drone_2 at (13.0, 2.5, 5.0) -> inter-drone dist = 2.5m, obstacle clearance = hypot(3,2.5)-2 = 1.905m
    trace_path = tmp_path / "trace.jsonl"
    frames = [
        {
            "timestamp": 100.0,
            "trace_id": "trace_ok_001",
            "span_id": "s-1",
            "phase": "INTERCEPT",
            "drones": {
                "drone_1": {"pos": [13.0, 0.0, 5.0], "vel": [1.0, 0.0, 0.0]},
                "drone_2": {"pos": [13.0, 2.5, 5.0], "vel": [1.0, 0.0, 0.0]},
                "drone_3": {"pos": [20.0, 0.0, 5.0], "vel": [0.0, 1.0, 0.0]},
                "drone_4": {"pos": [20.0, 5.0, 5.0], "vel": [0.0, 1.0, 0.0]},
                "drone_5": {"pos": [25.0, 0.0, 5.0], "vel": [0.0, 0.0, 0.0]},
                "drone_6": {"pos": [25.0, 5.0, 5.0], "vel": [0.0, 0.0, 0.0]},
            },
        },
        {
            "timestamp": 100.1,
            "trace_id": "trace_ok_001",
            "span_id": "s-2",
            "phase": "ATTACK",
            "drones": {
                "drone_1": {"pos": [13.5, 0.0, 5.0], "vel": [1.0, 0.0, 0.0]},
                "drone_2": {"pos": [13.5, 3.0, 5.0], "vel": [1.0, 0.0, 0.0]},
            },
        },
    ]
    trace_path.write_text("\n".join(json.dumps(f) for f in frames) + "\n", encoding="utf-8")
    tr_ok = validate_mission_trace(trace_path, obstacles_cfg=obs_cfg, summary_data=summary_data)
    assert tr_ok.ok is True, tr_ok.errors
    assert tr_ok.computed_min_inter_drone_m == pytest.approx(2.5, abs=1e-6)
    assert tr_ok.computed_min_obstacle_m == pytest.approx(1.0, abs=1e-6)

    # Fabricated summary mismatch must fail
    tampered_summary = dict(summary_data)
    tampered_summary["min_inter_drone_dist_m"] = 9.9
    tr_mismatch = validate_mission_trace(trace_path, obstacles_cfg=obs_cfg, summary_data=tampered_summary)
    assert tr_mismatch.ok is False

    # Obstacle penetration in trace must fail
    pen_trace = tmp_path / "pen_trace.jsonl"
    pen_frame = {
        "timestamp": 100.0,
        "trace_id": "trace_ok_001",
        "span_id": "s-1",
        "phase": "ATTACK",
        "drones": {
            "drone_1": {"pos": [10.5, 0.0, 5.0], "vel": [0.0, 0.0, 0.0]},
            "drone_2": {"pos": [13.0, 0.0, 5.0], "vel": [0.0, 0.0, 0.0]},
        },
    }
    pen_trace.write_text(json.dumps(pen_frame) + "\n", encoding="utf-8")
    tr_pen = validate_mission_trace(pen_trace, obstacles_cfg=obs_cfg)
    assert tr_pen.ok is False
    assert tr_pen.computed_min_obstacle_m < 0.0


def test_ros_json_message_validators() -> None:
    good_vision = {
        "stamp": 1700000000.5,
        "frame_id": "world",
        "targets": [
            {
                "det_id": 0,
                "class_id": "crimson_uav",
                "x": 18.4,
                "y": -2.1,
                "z": 7.5,
                "conf": 0.94,
                "stereo": True,
                "rays_used": 3,
                "parallax_deg": 24.5,
                "skew_rms_m": 0.14,
            }
        ],
    }
    assert validate_vision_targets_msg(good_vision).ok is True

    bad_vision = json.dumps(
        {
            "stamp": 1700000000.5,
            "frame_id": "world",
            "targets": [
                {
                    "det_id": 0,
                    "class_id": "crimson_uav",
                    "x": 1.0,
                    "y": 2.0,
                    "z": 3.0,
                    "conf": 1.5,
                    "stereo": True,
                    "rays_used": 1,
                    "parallax_deg": 10.0,
                    "skew_rms_m": 0.1,
                }
            ],
        }
    )
    assert validate_vision_targets_msg(bad_vision).ok is False

    good_plan = {
        "drone_id": "drone_1",
        "stamp": 1700000001.0,
        "x": 10.0,
        "y": 2.0,
        "z": 6.0,
        "action": "INTERCEPT",
        "t_go": 2.1,
        "imm_probs": {"CV": 0.2, "CTRV": 0.5, "JINK": 0.3},
        "pred_path": [[10.0, 2.0, 6.0], [11.0, 2.5, 6.0]],
    }
    assert validate_planning_target_msg(good_plan).ok is True

    bad_plan = dict(good_plan, action="WARP_SPEED", imm_probs={"CV": 0.9, "CTRV": 0.9})
    assert validate_planning_target_msg(bad_plan).ok is False

    good_kill = {
        "stamp": 1700000010.0,
        "target_id": "tgt_1",
        "striker_id": "drone_2",
        "miss_dist_m": 0.42,
        "closing_speed_ms": 13.8,
        "x": 15.0,
        "y": -1.0,
        "z": 6.2,
    }
    assert validate_kill_event_msg(good_kill).ok is True
    assert validate_kill_event_msg(dict(good_kill, miss_dist_m=-0.2)).ok is False


# ---------------------------------------------------------------------------
# 4. media_validator.py tests
# ---------------------------------------------------------------------------


def _run_ffmpeg(args: list[str]) -> None:
    res = run_command_with_gpu_guard(["ffmpeg", "-y", "-v", "error", *args], timeout_s=30.0)
    assert res.returncode == 0, f"ffmpeg failed: {res.stderr}"


def test_media_validator_synthetic_positive_and_negative_mp4s(tmp_path: Path) -> None:
    good_mp4 = tmp_path / "good_1080p30.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1920x1080:rate=30:duration=1.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(good_mp4),
        ]
    )
    probe = probe_video(good_mp4)
    assert probe.exists is True
    assert probe.width == 1920 and probe.height == 1080
    assert probe.fps == pytest.approx(30.0, abs=0.1)
    assert probe.codec == "h264"
    assert probe.pix_fmt == "yuv420p"
    assert probe.keyframe_count >= 1

    qual = check_video_quality(good_mp4, min_duration_s=1.0, sample_frames=6)
    assert qual.ok is True, qual.errors
    assert qual.black_frames == 0
    assert qual.uniform_frames == 0
    assert qual.frozen_pairs == 0

    # Negative 1: all-black frames
    black_mp4 = tmp_path / "black.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=1920x1080:r=30:d=1.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(black_mp4),
        ]
    )
    black_q = check_video_quality(black_mp4, min_duration_s=1.0, sample_frames=5)
    assert black_q.ok is False
    assert black_q.black_frames > 0

    # Negative 2: uniform/solid mid-gray frames
    solid_mp4 = tmp_path / "solid.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=0x808080:s=1920x1080:r=30:d=1.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(solid_mp4),
        ]
    )
    solid_q = check_video_quality(solid_mp4, min_duration_s=1.0, sample_frames=5)
    assert solid_q.ok is False
    assert solid_q.uniform_frames > 0

    # Negative 3: frozen textured stream (non-uniform, non-black, zero motion)
    frozen_mp4 = tmp_path / "frozen.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "smptebars=s=1920x1080:r=30:d=1.2",
            "-vf",
            "trim=end_frame=1,loop=loop=35:size=1:start=0",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(frozen_mp4),
        ]
    )
    frozen_q = check_video_quality(frozen_mp4, min_duration_s=1.0, sample_frames=6)
    assert frozen_q.ok is False
    assert frozen_q.frozen_pairs > 0

    # Negative 4: low resolution / low fps
    low_mp4 = tmp_path / "low.mp4"
    _run_ffmpeg(
        [
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=640x480:rate=15:duration=1.2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(low_mp4),
        ]
    )
    low_q = check_video_quality(low_mp4, min_w=1920, min_h=1080, min_fps=24.0, min_duration_s=1.0, sample_frames=4)
    assert low_q.ok is False

    # Missing file
    assert probe_video(tmp_path / "does_not_exist.mp4").exists is False


# ---------------------------------------------------------------------------
# 5. ros_probe.py & runner.py tests
# ---------------------------------------------------------------------------


def test_run_command_with_gpu_guard_and_timeout() -> None:
    res = run_command_with_gpu_guard(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('CUDA_VISIBLE_DEVICES')); "
            "print(os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD'))",
        ],
        timeout_s=10.0,
    )
    assert res.returncode == 0
    lines = res.stdout.strip().splitlines()
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": lines[0]}) is True
    assert lines[1] == "1"

    # Timeout cleanly kills child process
    to_res = run_command_with_gpu_guard(
        [sys.executable, "-c", "import time; time.sleep(10)"],
        timeout_s=0.3,
    )
    assert to_res.timed_out is True
    assert to_res.returncode != 0


def test_colcon_test_result_baseline_non_destructive() -> None:
    res = run_colcon_test(ROS2_WS, result_only=True, timeout_s=30.0)
    assert res.ok is True, f"colcon test-result baseline failed: {res.raw_summary}"
    assert res.total >= 22
    assert res.errors == 0
    assert res.failures == 0
    assert res.skipped == 0


def test_runner_cli_rejects_invalid_feature_and_filters_correctly(tmp_path: Path) -> None:
    runner_py = ROS2_WS / "tests_e2e" / "runner.py"
    bad_feat = subprocess.run(
        [sys.executable, str(runner_py), "--tier", "self_check", "--mode", "pre_integration", "--feature", "F99"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_feat.returncode != 0

    # Verify conftest --feature filtering directly on a synthetic test file
    synth = tmp_path / "test_feat_filter.py"
    synth.write_text(
        """
import pytest

@pytest.mark.feature(id="F1")
def test_f1():
    assert True

@pytest.mark.feature(id="F2")
def test_f2():
    assert False
""",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["PYTHONPATH"] = str(ROS2_WS / "tests_e2e")
    filt = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(ROS2_WS / "tests_e2e" / "pytest.ini"),
            "-p",
            "conftest",
            "-o",
            "testpaths=.",
            "--feature",
            "F1",
            str(synth),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert filt.returncode == 0, filt.stdout + filt.stderr
    assert "1 passed" in filt.stdout
    assert "1 deselected" in filt.stdout
