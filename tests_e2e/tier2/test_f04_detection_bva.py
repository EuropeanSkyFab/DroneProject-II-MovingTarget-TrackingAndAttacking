"""Tier 2 BVA & corner-case tests for F4: Multi-Target Multi-Signature Detection."""

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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pytest

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

pytestmark = [pytest.mark.tier2, pytest.mark.feature(id="F4"), pytest.mark.requires_m2]

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
AMBER_BGR = (20, 120, 245)
GLARE_BGR = (228, 232, 252)


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


def _make_odom_msg(xyz: Tuple[float, float, float], stamp_sec: float, bad_nan: bool = False) -> Odometry:
    msg = Odometry()
    sec = int(stamp_sec)
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = int((stamp_sec - sec) * 1e9)
    msg.header.frame_id = "world"
    msg.pose.pose.position.x = float("nan") if bad_nan else float(xyz[0])
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


def _start_vision(domain_id: int, params: Dict[str, Any]):
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
    node = Node(f"f4_bva_{domain_id}", context=ctx)
    executor = SingleThreadedExecutor(context=ctx)
    executor.add_node(node)
    return proc, ctx, node, executor


def _stop_vision(proc: subprocess.Popen[str], ctx: Context, node: Node, executor: SingleThreadedExecutor) -> None:
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


def _wait_ready(
    proc: subprocess.Popen[str],
    node: Node,
    executor: SingleThreadedExecutor,
    drone_ids: Sequence[str],
    timeout_s: float = 8.0,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            out, err = proc.communicate(timeout=1.0)
            pytest.fail(f"vision_node exited early (rc={proc.returncode}):\n{err}\n{out}")
        executor.spin_once(timeout_sec=0.05)
        if (
            node.count_publishers("/vision/targets") >= 1
            and all(node.count_subscribers(f"/{d}/camera/image_raw") >= 1 for d in drone_ids)
            and all(node.count_subscribers(f"/{d}/state") >= 1 for d in drone_ids)
        ):
            return
    pytest.fail(
        f"Active install/ vision_node did not advertise /vision/targets within {timeout_s}s (stale pre-M2 install/?)"
    )


def test_f04_bva_empty_frame_zero_targets_stays_alive() -> None:
    """Verify uniform empty frames produce valid empty /vision/targets and node stays alive."""
    drones = {"drone_1": (0.0, -4.0, 12.0), "drone_2": (0.0, 4.0, 12.0)}
    proc, ctx, node, executor = _start_vision(51, {"num_drones": 2, "enable_multi_signature": True})
    msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(String, "/vision/targets", lambda m: msgs.append(json.loads(m.data)), 10)
        pubs = {
            d: (
                node.create_publisher(Odometry, f"/{d}/state", 10),
                node.create_publisher(CameraInfo, f"/{d}/camera/camera_info", qos_profile_sensor_data),
                node.create_publisher(Image, f"/{d}/camera/image_raw", qos_profile_sensor_data),
            )
            for d in drones
        }
        _wait_ready(proc, node, executor, list(drones.keys()))
        blank = np.full((IMG_H, IMG_W, 3), BG_BGR, dtype=np.uint8)
        t_end = time.monotonic() + 1.5
        while time.monotonic() < t_end and not msgs:
            now = time.time()
            for d, xyz in drones.items():
                pubs[d][0].publish(_make_odom_msg(xyz, now))
                pubs[d][1].publish(_make_camera_info(now))
                pubs[d][2].publish(_make_image_msg(blank, now))
            executor.spin_once(timeout_sec=0.05)
            time.sleep(0.05)
        assert proc.poll() is None, "vision_node crashed on empty frames"
        assert msgs, "Expected /vision/targets heartbeat on empty frames"
        assert validate_vision_targets_msg(msgs[-1]).ok is True
        assert msgs[-1]["targets"] == []
    finally:
        _stop_vision(proc, ctx, node, executor)


def test_f04_bva_saturated_high_glare_frame_rejected() -> None:
    """Verify saturated high-glare frame yields zero false-positive target detections."""
    drones = {"drone_1": (0.0, -4.0, 12.0), "drone_2": (0.0, 4.0, 12.0)}
    proc, ctx, node, executor = _start_vision(
        52, {"num_drones": 2, "enable_multi_signature": True, "enable_glare_suppression": True}
    )
    msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(String, "/vision/targets", lambda m: msgs.append(json.loads(m.data)), 10)
        pubs = {
            d: (
                node.create_publisher(Odometry, f"/{d}/state", 10),
                node.create_publisher(CameraInfo, f"/{d}/camera/camera_info", qos_profile_sensor_data),
                node.create_publisher(Image, f"/{d}/camera/image_raw", qos_profile_sensor_data),
            )
            for d in drones
        }
        _wait_ready(proc, node, executor, list(drones.keys()))
        glare_img = np.full((IMG_H, IMG_W, 3), GLARE_BGR, dtype=np.uint8)
        t_end = time.monotonic() + 1.5
        while time.monotonic() < t_end and len(msgs) < 3:
            now = time.time()
            for d, xyz in drones.items():
                pubs[d][0].publish(_make_odom_msg(xyz, now))
                pubs[d][1].publish(_make_camera_info(now))
                pubs[d][2].publish(_make_image_msg(glare_img, now))
            executor.spin_once(timeout_sec=0.05)
            time.sleep(0.05)
        assert proc.poll() is None
        assert msgs, "No /vision/targets emitted on glare frame"
        assert all(m["targets"] == [] for m in msgs), f"Glare produced false targets: {msgs}"
    finally:
        _stop_vision(proc, ctx, node, executor)


def test_f04_bva_sub_min_area_speck_rejected_vs_above_min_area_detected() -> None:
    """Verify tiny sub-min_area (2x2=4 px < 24) speck is rejected while >=24 px target is detected."""
    drones = {"drone_1": (0.0, -4.0, 12.0), "drone_2": (0.0, 4.0, 12.0)}
    tgt = (9.2, 0.0, 5.5)
    proc, ctx, node, executor = _start_vision(
        53, {"num_drones": 2, "enable_multi_signature": True, "min_area": 24, "ground_z": 5.5}
    )
    msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(String, "/vision/targets", lambda m: msgs.append(json.loads(m.data)), 10)
        pubs = {
            d: (
                node.create_publisher(Odometry, f"/{d}/state", 10),
                node.create_publisher(CameraInfo, f"/{d}/camera/camera_info", qos_profile_sensor_data),
                node.create_publisher(Image, f"/{d}/camera/image_raw", qos_profile_sensor_data),
            )
            for d in drones
        }
        _wait_ready(proc, node, executor, list(drones.keys()))

        # Phase A: tiny 2x2 px speck (4 px << min_area=24)
        for _ in range(10):
            now = time.time()
            for d, xyz in drones.items():
                img = np.full((IMG_H, IMG_W, 3), BG_BGR, dtype=np.uint8)
                uv = _project_world_to_pixel(tgt, xyz)
                assert uv is not None
                img[uv[1]:uv[1] + 2, uv[0]:uv[0] + 2] = CRIMSON_BGR
                pubs[d][0].publish(_make_odom_msg(xyz, now))
                pubs[d][1].publish(_make_camera_info(now))
                pubs[d][2].publish(_make_image_msg(img, now))
            executor.spin_once(timeout_sec=0.05)
            time.sleep(0.05)
        assert msgs and all(m["targets"] == [] for m in msgs), f"Sub-min_area speck detected: {msgs}"

        # Phase B: 8x8 px target (64 px >= 24)
        msgs.clear()
        t_end = time.monotonic() + 2.0
        while time.monotonic() < t_end and not any(m["targets"] for m in msgs):
            now = time.time()
            for d, xyz in drones.items():
                img = np.full((IMG_H, IMG_W, 3), BG_BGR, dtype=np.uint8)
                uv = _project_world_to_pixel(tgt, xyz)
                assert uv is not None
                cv2.circle(img, uv, 6, CRIMSON_BGR, -1)
                pubs[d][0].publish(_make_odom_msg(xyz, now))
                pubs[d][1].publish(_make_camera_info(now))
                pubs[d][2].publish(_make_image_msg(img, now))
            executor.spin_once(timeout_sec=0.05)
            time.sleep(0.05)
        assert any(m["targets"] for m in msgs), "Target above min_area was not detected"
    finally:
        _stop_vision(proc, ctx, node, executor)


def test_f04_bva_adjacent_distinct_class_targets_in_image_plane() -> None:
    """Verify two nearby distinct-class targets (crimson + amber) remain cleanly separated."""
    drones = {"drone_1": (0.0, -4.0, 12.0), "drone_2": (0.0, 4.0, 12.0)}
    tgt_a = (9.2, -0.9, 5.5)
    tgt_b = (9.2, 0.9, 5.5)
    proc, ctx, node, executor = _start_vision(
        54,
        {
            "num_drones": 2,
            "enable_multi_signature": True,
            "enable_multi_target": True,
            "enable_triangulation": True,
            "ground_z": 5.5,
        },
    )
    msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(String, "/vision/targets", lambda m: msgs.append(json.loads(m.data)), 10)
        pubs = {
            d: (
                node.create_publisher(Odometry, f"/{d}/state", 10),
                node.create_publisher(CameraInfo, f"/{d}/camera/camera_info", qos_profile_sensor_data),
                node.create_publisher(Image, f"/{d}/camera/image_raw", qos_profile_sensor_data),
            )
            for d in drones
        }
        _wait_ready(proc, node, executor, list(drones.keys()))
        t_end = time.monotonic() + 3.5
        while time.monotonic() < t_end:
            now = time.time()
            for d, xyz in drones.items():
                img = np.full((IMG_H, IMG_W, 3), BG_BGR, dtype=np.uint8)
                uva = _project_world_to_pixel(tgt_a, xyz)
                uvb = _project_world_to_pixel(tgt_b, xyz)
                assert uva is not None and uvb is not None
                cv2.circle(img, uva, 12, CRIMSON_BGR, -1)
                cv2.circle(img, uvb, 12, AMBER_BGR, -1)
                pubs[d][0].publish(_make_odom_msg(xyz, now))
                pubs[d][1].publish(_make_camera_info(now))
                pubs[d][2].publish(_make_image_msg(img, now))
            executor.spin_once(timeout_sec=0.05)
            if any(len({x["class_id"] for x in m.get("targets", [])}) >= 2 for m in msgs):
                break
            time.sleep(0.05)
        classes = {t["class_id"] for m in msgs for t in m.get("targets", [])}
        assert {"crimson_uav", "amber_uav"}.issubset(classes), f"Expected both classes, got {msgs}"
    finally:
        _stop_vision(proc, ctx, node, executor)


def test_f04_bva_malformed_image_and_nan_pose_resilience() -> None:
    """Verify corrupt empty Image / NaN Odometry does not crash vision_node and valid recovery succeeds."""
    drones = {"drone_1": (0.0, -4.0, 12.0), "drone_2": (0.0, 4.0, 12.0)}
    tgt = (9.2, 0.0, 5.5)
    proc, ctx, node, executor = _start_vision(
        55, {"num_drones": 2, "enable_multi_signature": True, "ground_z": 5.5}
    )
    msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(String, "/vision/targets", lambda m: msgs.append(json.loads(m.data)), 10)
        pubs = {
            d: (
                node.create_publisher(Odometry, f"/{d}/state", 10),
                node.create_publisher(CameraInfo, f"/{d}/camera/camera_info", qos_profile_sensor_data),
                node.create_publisher(Image, f"/{d}/camera/image_raw", qos_profile_sensor_data),
            )
            for d in drones
        }
        _wait_ready(proc, node, executor, list(drones.keys()))

        # Send malformed Image + NaN pose
        bad_img = Image()
        bad_img.header.frame_id = "down_cam"
        bad_img.encoding = "bgr8"
        bad_img.height = 480
        bad_img.width = 640
        bad_img.step = 640 * 3
        bad_img.data = b"\x00\x01"  # truncated buffer

        for _ in range(5):
            now = time.time()
            for d, xyz in drones.items():
                pubs[d][0].publish(_make_odom_msg(xyz, now, bad_nan=True))
                pubs[d][2].publish(bad_img)
            executor.spin_once(timeout_sec=0.05)
            time.sleep(0.04)

        assert proc.poll() is None, "vision_node crashed on malformed image/NaN pose"

        # Recover with valid frames
        msgs.clear()
        t_end = time.monotonic() + 2.5
        while time.monotonic() < t_end and not any(m.get("targets") for m in msgs):
            now = time.time()
            for d, xyz in drones.items():
                img = np.full((IMG_H, IMG_W, 3), BG_BGR, dtype=np.uint8)
                uv = _project_world_to_pixel(tgt, xyz)
                assert uv is not None
                cv2.circle(img, uv, 15, CRIMSON_BGR, -1)
                pubs[d][0].publish(_make_odom_msg(xyz, now))
                pubs[d][1].publish(_make_camera_info(now))
                pubs[d][2].publish(_make_image_msg(img, now))
            executor.spin_once(timeout_sec=0.05)
            time.sleep(0.05)

        assert proc.poll() is None
        assert any(m.get("targets") for m in msgs), "vision_node failed to recover after corrupt frame"
    finally:
        _stop_vision(proc, ctx, node, executor)
