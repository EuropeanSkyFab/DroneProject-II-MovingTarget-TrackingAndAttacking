"""Tier 2 BVA & corner-case tests for F5: Multi-View Epipolar Association & Parallel Vision."""

from __future__ import annotations

import array
import json
import math
import os
import shlex
import signal
import subprocess
import sys as _sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

from helpers.arena_parser import check_line_of_sight
from helpers.metrics_validator import validate_vision_targets_msg

_sys.path[:] = [p for p in _sys.path if "/opt/ros/noetic" not in p]
for _ros_pkg in ("std_msgs", "geometry_msgs", "nav_msgs", "sensor_msgs", "rosgraph_msgs"):
    _mod = _sys.modules.get(_ros_pkg)
    if _mod is not None and "/opt/ros/noetic" in getattr(_mod, "__file__", ""):
        for _k in list(_sys.modules):
            if _k == _ros_pkg or _k.startswith(f"{_ros_pkg}."):
                _sys.modules.pop(_k, None)

from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.context import Context  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402

pytestmark = [pytest.mark.tier2, pytest.mark.feature(id="F5"), pytest.mark.requires_m2]

ROS2_WS = Path("/home/hkw/flying/ros2_ws")
ENV_SH = ROS2_WS / "src" / "flight_ctrl_pkg" / "scripts" / "env.sh"
INSTALL_SETUP = ROS2_WS / "install" / "setup.bash"

CAM_MOUNT_RPY = (0.95993109, 0.0, -1.57079633)
CAM_OFFSET_BODY = (0.12, 0.0, -0.04)
DEFAULT_HFOV_RAD = 1.35
IMG_W = 640
IMG_H = 480

BG_BGR = (55, 120, 70)
CRIMSON_BGR = (22, 22, 232)


def _euler_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def _project_world_to_pixel(
    target_xyz: Tuple[float, float, float],
    drone_xyz: Tuple[float, float, float],
) -> Optional[Tuple[int, int]]:
    fx = 0.5 * IMG_W / math.tan(0.5 * DEFAULT_HFOV_RAD)
    cx, cy = 0.5 * IMG_W, 0.5 * IMG_H
    r_wc = _euler_to_rot(*CAM_MOUNT_RPY) @ np.diag([1.0, -1.0, -1.0])
    cam_origin = np.array(drone_xyz, dtype=float) + np.array(CAM_OFFSET_BODY, dtype=float)
    p_cam = r_wc.T @ (np.array(target_xyz, dtype=float) - cam_origin)
    if p_cam[2] <= 0.4:
        return None
    u = int(round(fx * (p_cam[0] / p_cam[2]) + cx))
    v = int(round(fx * (p_cam[1] / p_cam[2]) + cy))
    if not (0 <= u < IMG_W and 0 <= v < IMG_H):
        return None
    return u, v


def _make_camera_info(stamp_sec: float) -> CameraInfo:
    fx = 0.5 * IMG_W / math.tan(0.5 * DEFAULT_HFOV_RAD)
    msg = CameraInfo()
    sec = int(stamp_sec)
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = int((stamp_sec - sec) * 1e9)
    msg.header.frame_id = "down_cam"
    msg.width = IMG_W
    msg.height = IMG_H
    msg.k = [fx, 0.0, 0.5 * IMG_W, 0.0, fx, 0.5 * IMG_H, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, 0.5 * IMG_W, 0.0, 0.0, fx, 0.5 * IMG_H, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def _make_image_msg(bgr: np.ndarray, stamp_sec: float) -> Image:
    msg = Image()
    sec = int(stamp_sec)
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = int((stamp_sec - sec) * 1e9)
    msg.header.frame_id = "down_cam"
    msg.height = int(bgr.shape[0])
    msg.width = int(bgr.shape[1])
    msg.encoding = "bgr8"
    msg.step = int(bgr.shape[1] * 3)
    msg.data = array.array("B", bgr.tobytes())
    return msg


def _make_odom_msg(xyz: Tuple[float, float, float], stamp_sec: float) -> Odometry:
    msg = Odometry()
    sec = int(stamp_sec)
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = int((stamp_sec - sec) * 1e9)
    msg.header.frame_id = "world"
    msg.pose.pose.position.x = float(xyz[0])
    msg.pose.pose.position.y = float(xyz[1])
    msg.pose.pose.position.z = float(xyz[2])
    msg.pose.pose.orientation.w = 1.0
    return msg


def _format_ros_param(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return json.dumps(v)
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return str(v)


def _run_vision_bva(
    domain_id: int,
    params: Dict[str, Any],
    drones: Dict[str, Tuple[float, float, float]],
    frame_fn,
    stop_when=None,
    wait_s: float = 2.5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["ROS_LOCALHOST_ONLY"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    ros_args = ["ros2", "run", "vision_pkg", "vision_node", "--ros-args"]
    for k, v in params.items():
        ros_args.extend(["-p", f"{k}:={_format_ros_param(v)}"])

    bash_cmd = (
        f"source {shlex.quote(str(ENV_SH))} >/dev/null 2>&1 && "
        f"source {shlex.quote(str(INSTALL_SETUP))} >/dev/null 2>&1 && "
        f"exec {shlex.join(ros_args)}"
    )
    proc = subprocess.Popen(
        ["bash", "-lc", bash_cmd],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    os.environ["ROS_DOMAIN_ID"] = str(domain_id)
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    ctx = Context()
    ctx.init(args=None)
    node = Node(f"f5_bva_{domain_id}", context=ctx)
    executor = SingleThreadedExecutor(context=ctx)
    executor.add_node(node)

    targets_msgs: List[Dict[str, Any]] = []
    diag_msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(String, "/vision/targets", lambda m: targets_msgs.append(json.loads(m.data)), 10)
        node.create_subscription(String, "/vision/diagnostics", lambda m: diag_msgs.append(json.loads(m.data)), 10)
        pubs = {
            d: (
                node.create_publisher(Odometry, f"/{d}/state", 10),
                node.create_publisher(CameraInfo, f"/{d}/camera/camera_info", qos_profile_sensor_data),
                node.create_publisher(Image, f"/{d}/camera/image_raw", qos_profile_sensor_data),
            )
            for d in drones
        }

        timeout_s = 8.0
        deadline = time.monotonic() + timeout_s
        ready = False
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                out, err = proc.communicate(timeout=1.0)
                pytest.fail(f"vision_node exited early (rc={proc.returncode}):\n{err}\n{out}")
            executor.spin_once(timeout_sec=0.05)
            if (
                node.count_publishers("/vision/targets") >= 1
                and all(node.count_subscribers(f"/{d}/camera/image_raw") >= 1 for d in drones)
                and all(node.count_subscribers(f"/{d}/state") >= 1 for d in drones)
            ):
                ready = True
                break
        if not ready:
            pytest.fail(
                "Active install/ vision_node did not advertise /vision/targets within "
                f"{timeout_s}s (stale pre-M2 install/?)"
            )

        t_start = time.monotonic()
        t_end = t_start + wait_s
        while time.monotonic() < t_end:
            now = time.time()
            for d, xyz in drones.items():
                pubs[d][0].publish(_make_odom_msg(xyz, now))
                pubs[d][1].publish(_make_camera_info(now))
                pubs[d][2].publish(_make_image_msg(frame_fn(d, xyz), now))
            for _ in range(4):
                executor.spin_once(timeout_sec=0.02)
            if stop_when is not None and stop_when(targets_msgs, diag_msgs):
                break
            time.sleep(0.05)
        return targets_msgs, diag_msgs, time.monotonic() - t_start
    finally:
        for fn in (lambda: executor.shutdown(timeout_sec=0.5), node.destroy_node, ctx.shutdown):
            try:
                fn()
            except Exception:
                pass
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=1.5)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
            try:
                proc.wait(timeout=1.0)
            except Exception:
                pass


def test_f05_bva_near_collinear_small_baseline_rejected_for_stereo() -> None:
    """Verify baseline < min_baseline_m (1.5m < 3.4m) is rejected for stereo triangulation."""
    drones = {
        "drone_1": (0.0, -0.75, 13.0),
        "drone_2": (0.0, 0.75, 13.0),
    }
    tgt = (11.0, 0.0, 6.0)

    def _draw(_d: str, xyz: Tuple[float, float, float]) -> np.ndarray:
        img = np.full((IMG_H, IMG_W, 3), BG_BGR, dtype=np.uint8)
        uv = _project_world_to_pixel(tgt, xyz)
        assert uv is not None
        cv2.circle(img, uv, 15, CRIMSON_BGR, -1)
        return img

    targets_msgs, _, _ = _run_vision_bva(
        domain_id=61,
        params={
            "num_drones": 2,
            "enable_multi_signature": True,
            "enable_multi_target": True,
            "enable_triangulation": True,
            "min_baseline_m": 3.4,
            "min_parallax_deg": 8.0,
        },
        drones=drones,
        frame_fn=_draw,
        stop_when=lambda t, _d: len(t) >= 5,
    )
    assert targets_msgs, "No /vision/targets emitted"
    stereo_hits = [x for m in targets_msgs for x in m.get("targets", []) if x.get("stereo") is True]
    assert stereo_hits == [], f"Small 1.5m baseline should not produce stereo=True: {stereo_hits}"


def test_f05_bva_single_camera_only_observation_monocular_fallback() -> None:
    """Verify single-camera observation has stereo=False and rays_used==1 under monocular fallback."""
    drones = {
        "drone_1": (0.0, -4.0, 12.0),
        "drone_2": (0.0, 4.0, 12.0),
    }
    tgt = (9.2, -4.0, 5.5)

    def _draw(d_id: str, xyz: Tuple[float, float, float]) -> np.ndarray:
        img = np.full((IMG_H, IMG_W, 3), BG_BGR, dtype=np.uint8)
        if d_id == "drone_1":
            uv = _project_world_to_pixel(tgt, xyz)
            assert uv is not None
            cv2.circle(img, uv, 15, CRIMSON_BGR, -1)
        return img

    targets_msgs, _, _ = _run_vision_bva(
        domain_id=62,
        params={
            "num_drones": 2,
            "enable_multi_signature": True,
            "enable_multi_target": True,
            "enable_triangulation": True,
            "allow_monocular_fallback": True,
            "ground_z": 5.5,
        },
        drones=drones,
        frame_fn=_draw,
        stop_when=lambda t, _d: any(m.get("targets") for m in t),
    )
    non_empty = [m for m in targets_msgs if m.get("targets")]
    assert non_empty, f"Expected monocular fallback detection, got {targets_msgs}"
    assert validate_vision_targets_msg(non_empty[-1]).ok is True
    for t in non_empty[-1]["targets"]:
        assert t["stereo"] is False
        assert t["rays_used"] == 1


def test_f05_bva_symmetric_same_class_targets_no_ghost_crossings() -> None:
    """Verify two symmetric same-class crimson_uav targets observed by 3 UAVs yield exactly 2 true 3D targets."""
    drones = {
        "drone_1": (0.0, -5.0, 13.0),
        "drone_2": (0.0, 0.0, 13.0),
        "drone_3": (0.0, 5.0, 13.0),
    }
    gt_a = (10.5, -2.6, 6.2)
    gt_b = (10.5, 2.6, 6.2)

    def _draw(_d: str, xyz: Tuple[float, float, float]) -> np.ndarray:
        img = np.full((IMG_H, IMG_W, 3), BG_BGR, dtype=np.uint8)
        for p in (gt_a, gt_b):
            uv = _project_world_to_pixel(p, xyz)
            assert uv is not None
            cv2.circle(img, uv, 13, CRIMSON_BGR, -1)
        return img

    targets_msgs, _, _ = _run_vision_bva(
        domain_id=63,
        params={
            "num_drones": 3,
            "process_mode": "threaded_batch",
            "enable_multi_signature": True,
            "enable_multi_target": True,
            "enable_triangulation": True,
        },
        drones=drones,
        frame_fn=_draw,
        stop_when=lambda t, _d: any(
            len(m.get("targets", [])) == 2
            and all(x.get("stereo") and x.get("rays_used", 0) >= 3 for x in m.get("targets", []))
            for m in t
        ),
        wait_s=4.5,
    )
    stereo_two = [
        m for m in targets_msgs
        if sum(1 for x in m.get("targets", []) if x.get("stereo") and x.get("rays_used", 0) >= 3) >= 2
    ]
    assert stereo_two, f"Expected 2 symmetric stereo targets, got {targets_msgs}"
    tgts = stereo_two[-1]["targets"]
    assert len(tgts) == 2, f"Symmetric ambiguity produced phantom ghost targets: {tgts}"
    for est in tgts:
        d_a = math.dist((est["x"], est["y"], est["z"]), gt_a)
        d_b = math.dist((est["x"], est["y"], est["z"]), gt_b)
        assert min(d_a, d_b) <= 1.0, f"Estimated point {est} far from both true targets"


def test_f05_bva_full_obstacle_occlusion_blackout_zero_targets() -> None:
    """Verify full obstacle occlusion across all cameras yields zero bogus 3D targets."""
    drones = {
        "drone_1": (0.0, -4.0, 12.0),
        "drone_2": (0.0, 4.0, 12.0),
    }
    tgt = (10.0, 0.0, 5.5)
    obs_cfg = {
        "obstacles": [
            {
                "id": "full_curtain_wall",
                "type": "box",
                "cx": 5.0,
                "cy": 0.0,
                "z_min": 0.0,
                "z_max": 20.0,
                "sx": 2.0,
                "sy": 14.0,
                "yaw": 0.0,
                "safety_margin_m": 0.0,
            }
        ],
        "no_fly_zones": [],
    }
    assert check_line_of_sight((0.12, -4.0, 11.96), tgt, obs_cfg) is False
    assert check_line_of_sight((0.12, 4.0, 11.96), tgt, obs_cfg) is False

    def _draw(_d: str, xyz: Tuple[float, float, float]) -> np.ndarray:
        img = np.full((IMG_H, IMG_W, 3), BG_BGR, dtype=np.uint8)
        uv = _project_world_to_pixel(tgt, xyz)
        assert uv is not None
        cv2.circle(img, uv, 15, CRIMSON_BGR, -1)
        return img

    targets_msgs, diag_msgs, _ = _run_vision_bva(
        domain_id=64,
        params={
            "num_drones": 2,
            "enable_multi_signature": True,
            "enable_multi_target": True,
            "enable_triangulation": True,
            "enable_los_check": True,
            "obstacles_json": json.dumps(obs_cfg),
        },
        drones=drones,
        frame_fn=_draw,
        stop_when=lambda t, d: len(t) >= 4 and any(dm.get("los_rejected_count", 0) > 0 for dm in d),
    )
    assert any(dm.get("los_rejected_count", 0) > 0 for dm in diag_msgs)
    assert all(m["targets"] == [] for m in targets_msgs), (
        f"Occluded cameras must emit zero targets, got {targets_msgs}"
    )


def test_f05_bva_eight_stream_concurrency_latency_boundary() -> None:
    """Verify N=8 camera streams under threaded_batch maintain active_cams==8 and timely output."""
    drones = {f"drone_{i+1}": (0.0, -7.0 + 2.0 * i, 13.0) for i in range(8)}
    tgt = (10.5, 0.0, 6.0)

    def _draw(_d: str, xyz: Tuple[float, float, float]) -> np.ndarray:
        img = np.full((IMG_H, IMG_W, 3), BG_BGR, dtype=np.uint8)
        uv = _project_world_to_pixel(tgt, xyz)
        if uv is not None:
            cv2.circle(img, uv, 15, CRIMSON_BGR, -1)
        return img

    targets_msgs, diag_msgs, elapsed_s = _run_vision_bva(
        domain_id=65,
        params={
            "num_drones": 8,
            "process_mode": "threaded_batch",
            "max_workers": 8,
            "enable_multi_signature": True,
            "enable_multi_target": True,
            "enable_triangulation": True,
        },
        drones=drones,
        frame_fn=_draw,
        stop_when=lambda t, d: (
            any(isinstance(dm.get("active_cams"), list) and len(dm["active_cams"]) == 8 for dm in d)
            and any(x.get("stereo") for m in t for x in m.get("targets", []))
        ),
        wait_s=4.5,
    )
    assert any(
        isinstance(dm.get("active_cams"), list) and len(dm["active_cams"]) == 8 for dm in diag_msgs
    ), f"Expected active_cams len==8 in diagnostics, got {diag_msgs}"
    assert any(x.get("stereo") for m in targets_msgs for x in m.get("targets", []))
    assert elapsed_s < 4.5
