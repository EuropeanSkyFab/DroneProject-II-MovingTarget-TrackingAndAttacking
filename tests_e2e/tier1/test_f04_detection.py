"""Tier 1 functional tests for F4: Multi-Target Multi-Signature Detection."""

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

from geometry_msgs.msg import PoseStamped  # noqa: E402
from nav_msgs.msg import Odometry  # noqa: E402
from rclpy.context import Context  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import CameraInfo, Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402

pytestmark = [pytest.mark.tier1, pytest.mark.feature(id="F4"), pytest.mark.requires_m2]

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
SLATE_BGR = (108, 90, 75)
GLARE_BGR = (220, 225, 250)
OCHRE_BGR = (38, 146, 194)


def _euler_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def _cam_extrinsics_matrix() -> np.ndarray:
    base = np.diag([1.0, -1.0, -1.0])
    return _euler_to_rot(*CAM_MOUNT_RPY) @ base


def _quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return np.array([
        [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
        [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
        [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
    ], dtype=float)


def _project_world_to_pixel(
    target_xyz: Tuple[float, float, float],
    drone_xyz: Tuple[float, float, float],
    drone_quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    width: int = IMG_W,
    height: int = IMG_H,
    hfov_rad: float = DEFAULT_HFOV_RAD,
) -> Optional[Tuple[int, int]]:
    fx = 0.5 * width / math.tan(0.5 * hfov_rad)
    fy = fx
    cx = 0.5 * width
    cy = 0.5 * height
    r_wb = _quat_to_rot(*drone_quat)
    r_wc = r_wb @ _cam_extrinsics_matrix()
    cam_origin = np.array(drone_xyz, dtype=float) + r_wb @ np.array(CAM_OFFSET_BODY, dtype=float)
    p_cam = r_wc.T @ (np.array(target_xyz, dtype=float) - cam_origin)
    if p_cam[2] <= 0.4:
        return None
    u = int(round(fx * (p_cam[0] / p_cam[2]) + cx))
    v = int(round(fy * (p_cam[1] / p_cam[2]) + cy))
    if not (0 <= u < width and 0 <= v < height):
        return None
    return u, v


def _make_camera_info(stamp_sec: float, width: int = IMG_W, height: int = IMG_H) -> CameraInfo:
    fx = 0.5 * width / math.tan(0.5 * DEFAULT_HFOV_RAD)
    cx = 0.5 * width
    cy = 0.5 * height
    msg = CameraInfo()
    sec = int(stamp_sec)
    nsec = int((stamp_sec - sec) * 1e9)
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nsec
    msg.header.frame_id = "down_cam"
    msg.width = width
    msg.height = height
    msg.k = [fx, 0.0, cx, 0.0, fx, cy, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fx, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def _make_image_msg(bgr: np.ndarray, stamp_sec: float) -> Image:
    msg = Image()
    sec = int(stamp_sec)
    nsec = int((stamp_sec - sec) * 1e9)
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nsec
    msg.header.frame_id = "down_cam"
    msg.height = int(bgr.shape[0])
    msg.width = int(bgr.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(bgr.shape[1] * 3)
    msg.data = array.array("B", bgr.tobytes())
    return msg


def _make_odom_msg(
    xyz: Tuple[float, float, float],
    stamp_sec: float,
    quat: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> Odometry:
    msg = Odometry()
    sec = int(stamp_sec)
    nsec = int((stamp_sec - sec) * 1e9)
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nsec
    msg.header.frame_id = "world"
    msg.pose.pose.position.x = float(xyz[0])
    msg.pose.pose.position.y = float(xyz[1])
    msg.pose.pose.position.z = float(xyz[2])
    msg.pose.pose.orientation.x = float(quat[0])
    msg.pose.pose.orientation.y = float(quat[1])
    msg.pose.pose.orientation.z = float(quat[2])
    msg.pose.pose.orientation.w = float(quat[3])
    return msg


def _format_ros_param(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return json.dumps(v)
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return str(v)


def _start_vision_node(
    domain_id: int,
    params: Dict[str, Any],
) -> Tuple[subprocess.Popen[str], Context, Node, SingleThreadedExecutor]:
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["ROS_LOCALHOST_ONLY"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    ros_args: List[str] = ["ros2", "run", "vision_pkg", "vision_node", "--ros-args"]
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
    node = Node(f"f4_probe_{domain_id}", context=ctx)
    executor = SingleThreadedExecutor(context=ctx)
    executor.add_node(node)
    return proc, ctx, node, executor


def _stop_vision_harness(
    proc: subprocess.Popen[str],
    ctx: Context,
    node: Node,
    executor: SingleThreadedExecutor,
) -> None:
    try:
        executor.shutdown(timeout_sec=0.5)
    except Exception:
        pass
    try:
        node.destroy_node()
    except Exception:
        pass
    try:
        ctx.shutdown()
    except Exception:
        pass
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
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


def _wait_for_vision_ready(
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
        has_targets_pub = node.count_publishers("/vision/targets") >= 1
        has_img_sub = all(
            node.count_subscribers(f"/{d}/camera/image_raw") >= 1 for d in drone_ids
        )
        has_state_sub = all(
            node.count_subscribers(f"/{d}/state") >= 1 for d in drone_ids
        )
        if has_targets_pub and has_img_sub and has_state_sub:
            return
    pytest.fail(
        "Active install/ vision_node did not advertise /vision/targets or subscribe to "
        f"/{drone_ids[0]}/camera/image_raw within {timeout_s}s (stale pre-M2 install/?)"
    )


def _run_vision_scenario(
    domain_id: int,
    params: Dict[str, Any],
    drone_poses: Dict[str, Tuple[float, float, float]],
    frame_fn,
    wait_s: float = 3.0,
    stop_when=None,
) -> Tuple[List[Dict[str, Any]], List[PoseStamped], List[Dict[str, Any]]]:
    proc, ctx, node, executor = _start_vision_node(domain_id, params)
    targets_msgs: List[Dict[str, Any]] = []
    pose_msgs: List[PoseStamped] = []
    diag_msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(
            String, "/vision/targets", lambda m: targets_msgs.append(json.loads(m.data)), 10
        )
        node.create_subscription(
            PoseStamped, "/vision/target_pose", lambda m: pose_msgs.append(m), 10
        )
        node.create_subscription(
            String, "/vision/diagnostics", lambda m: diag_msgs.append(json.loads(m.data)), 10
        )

        pubs = {}
        for d_id in drone_poses:
            pubs[d_id] = (
                node.create_publisher(Odometry, f"/{d_id}/state", 10),
                node.create_publisher(CameraInfo, f"/{d_id}/camera/camera_info", qos_profile_sensor_data),
                node.create_publisher(Image, f"/{d_id}/camera/image_raw", qos_profile_sensor_data),
            )

        _wait_for_vision_ready(proc, node, executor, list(drone_poses.keys()), timeout_s=8.0)

        t_end = time.monotonic() + wait_s
        step = 0
        while time.monotonic() < t_end:
            now = time.time()
            for d_id, d_xyz in drone_poses.items():
                odom_pub, info_pub, img_pub = pubs[d_id]
                odom_pub.publish(_make_odom_msg(d_xyz, now))
                info_pub.publish(_make_camera_info(now))
                img_bgr = frame_fn(d_id, d_xyz, step)
                img_pub.publish(_make_image_msg(img_bgr, now))
            for _ in range(4):
                executor.spin_once(timeout_sec=0.02)
            done = (
                stop_when(targets_msgs, pose_msgs, diag_msgs)
                if stop_when is not None
                else (any(t.get("targets") for t in targets_msgs) and bool(pose_msgs))
            )
            if done:
                break
            step += 1
            time.sleep(0.06)
        return targets_msgs, pose_msgs, diag_msgs
    finally:
        _stop_vision_harness(proc, ctx, node, executor)


def test_f04_vision_targets_contract_schema() -> None:
    """Verify /vision/targets publishes valid Contract #3 JSON schema validated by helper."""
    drone_poses = {
        "drone_1": (0.0, -4.0, 12.0),
        "drone_2": (0.0, 4.0, 12.0),
    }
    tgt_xyz = (9.2, 0.0, 5.5)

    def _draw(_d_id: str, d_xyz: Tuple[float, float, float], _step: int) -> np.ndarray:
        img = np.full((IMG_H, IMG_W, 3), BG_BGR, dtype=np.uint8)
        uv = _project_world_to_pixel(tgt_xyz, d_xyz)
        assert uv is not None
        cv2.circle(img, uv, 16, CRIMSON_BGR, -1)
        return img

    targets_msgs, _, _ = _run_vision_scenario(
        domain_id=41,
        params={
            "num_drones": 2,
            "process_mode": "threaded_batch",
            "enable_multi_signature": True,
            "enable_multi_target": True,
            "enable_triangulation": True,
            "ground_z": 5.5,
        },
        drone_poses=drone_poses,
        frame_fn=_draw,
    )
    assert targets_msgs, "No /vision/targets messages received"
    non_empty = [m for m in targets_msgs if m.get("targets")]
    assert non_empty, f"Expected non-empty targets in /vision/targets, got {targets_msgs}"
    for msg in non_empty:
        res = validate_vision_targets_msg(msg)
        assert res.ok is True, f"Contract #3 schema violation: {res.errors} | {msg}"


def test_f04_multi_target_simultaneous_three_classes() -> None:
    """Verify simultaneous multi-signature detection of crimson_uav, amber_uav, and slate_uav."""
    drone_poses = {
        "drone_1": (0.0, -4.0, 12.0),
        "drone_2": (0.0, 4.0, 12.0),
    }
    gt_targets = {
        "crimson_uav": (8.5, -2.5, 5.5),
        "amber_uav": (9.5, 0.5, 5.5),
        "slate_uav": (10.5, 3.0, 5.5),
    }
    colors = {
        "crimson_uav": CRIMSON_BGR,
        "amber_uav": AMBER_BGR,
        "slate_uav": SLATE_BGR,
    }

    def _draw(_d_id: str, d_xyz: Tuple[float, float, float], _step: int) -> np.ndarray:
        img = np.full((IMG_H, IMG_W, 3), BG_BGR, dtype=np.uint8)
        for cls_name, xyz in gt_targets.items():
            uv = _project_world_to_pixel(xyz, d_xyz)
            assert uv is not None, f"{cls_name} out of view from {d_xyz}"
            radius = 8 if cls_name == "slate_uav" else 14
            cv2.circle(img, uv, radius, colors[cls_name], -1)
        return img

    targets_msgs, _, _ = _run_vision_scenario(
        domain_id=42,
        params={
            "num_drones": 2,
            "process_mode": "threaded_batch",
            "enable_multi_signature": True,
            "enable_multi_target": True,
            "enable_triangulation": True,
            "ground_z": 5.5,
        },
        drone_poses=drone_poses,
        frame_fn=_draw,
        wait_s=4.5,
        stop_when=lambda t, _p, _d: any(
            len({x["class_id"] for x in m.get("targets", [])}) >= 3 for m in t
        ),
    )
    best = max(targets_msgs, key=lambda m: len(m.get("targets", [])), default={})
    detected_classes = {t["class_id"] for t in best.get("targets", [])}
    assert {"crimson_uav", "amber_uav", "slate_uav"}.issubset(detected_classes), (
        f"Expected all 3 target classes, got {best}"
    )


def test_f04_glare_and_ochre_clutter_suppression() -> None:
    """Verify glare suppression rejects specular glare and ochre clutter while detecting crimson_uav."""
    drone_poses = {
        "drone_1": (0.0, -4.0, 12.0),
        "drone_2": (0.0, 4.0, 12.0),
    }
    real_tgt = (9.2, 0.0, 5.5)

    def _draw(_d_id: str, d_xyz: Tuple[float, float, float], _step: int) -> np.ndarray:
        img = np.full((IMG_H, IMG_W, 3), BG_BGR, dtype=np.uint8)
        # Specular glare blob and ochre container clutter far from real target
        cv2.circle(img, (120, 120), 22, GLARE_BGR, -1)
        cv2.rectangle(img, (460, 80), (540, 150), OCHRE_BGR, -1)
        uv = _project_world_to_pixel(real_tgt, d_xyz)
        assert uv is not None
        cv2.circle(img, uv, 15, CRIMSON_BGR, -1)
        return img

    targets_msgs, _, _ = _run_vision_scenario(
        domain_id=43,
        params={
            "num_drones": 2,
            "enable_multi_signature": True,
            "enable_multi_target": True,
            "enable_glare_suppression": True,
            "enable_triangulation": True,
            "ground_z": 5.5,
        },
        drone_poses=drone_poses,
        frame_fn=_draw,
    )
    non_empty = [m for m in targets_msgs if m.get("targets")]
    assert non_empty, "Expected crimson_uav detection despite glare/ochre clutter"
    latest = non_empty[-1]["targets"]
    assert len(latest) == 1, f"Glare/ochre produced false positive targets: {latest}"
    assert latest[0]["class_id"] == "crimson_uav"


def test_f04_calibrated_camera_mount_rpy_localization_accuracy() -> None:
    """Verify calibrated camera_mount_rpy + camera_offset_body achieves <= 1.5m 3D RMSE."""
    drone_poses = {
        "drone_1": (0.0, -5.0, 13.0),
        "drone_2": (0.0, 5.0, 13.0),
    }
    gt_xyz = (10.5, 0.5, 6.2)

    def _draw(_d_id: str, d_xyz: Tuple[float, float, float], _step: int) -> np.ndarray:
        img = np.full((IMG_H, IMG_W, 3), BG_BGR, dtype=np.uint8)
        uv = _project_world_to_pixel(gt_xyz, d_xyz)
        assert uv is not None
        cv2.circle(img, uv, 15, CRIMSON_BGR, -1)
        return img

    targets_msgs, _, _ = _run_vision_scenario(
        domain_id=44,
        params={
            "num_drones": 2,
            "enable_multi_signature": True,
            "enable_multi_target": True,
            "enable_triangulation": True,
            "ground_z": 5.5,
        },
        drone_poses=drone_poses,
        frame_fn=_draw,
        wait_s=4.5,
        stop_when=lambda t, _p, _d: any(
            x.get("stereo") is True for m in t for x in m.get("targets", [])
        ),
    )
    stereo_msgs = [
        t for m in targets_msgs for t in m.get("targets", []) if t.get("stereo") is True
    ]
    assert stereo_msgs, f"Expected stereo localization from calibrated extrinsics, got {targets_msgs}"
    est = stereo_msgs[-1]
    err = math.sqrt(
        (est["x"] - gt_xyz[0]) ** 2
        + (est["y"] - gt_xyz[1]) ** 2
        + (est["z"] - gt_xyz[2]) ** 2
    )
    assert err <= 1.5, f"Localized 3D error {err:.3f}m > 1.5m (est={est}, gt={gt_xyz})"


def test_f04_backward_compatible_target_pose_published() -> None:
    """Verify /vision/target_pose (PoseStamped, frame_id='world') publishes alongside /vision/targets."""
    drone_poses = {
        "drone_1": (0.0, -4.0, 12.0),
        "drone_2": (0.0, 4.0, 12.0),
    }
    tgt_xyz = (9.2, 0.0, 5.5)

    def _draw(_d_id: str, d_xyz: Tuple[float, float, float], _step: int) -> np.ndarray:
        img = np.full((IMG_H, IMG_W, 3), BG_BGR, dtype=np.uint8)
        uv = _project_world_to_pixel(tgt_xyz, d_xyz)
        assert uv is not None
        cv2.circle(img, uv, 16, CRIMSON_BGR, -1)
        return img

    targets_msgs, pose_msgs, _ = _run_vision_scenario(
        domain_id=45,
        params={
            "num_drones": 2,
            "enable_multi_signature": True,
            "enable_multi_target": True,
            "enable_triangulation": True,
            "ground_z": 5.5,
        },
        drone_poses=drone_poses,
        frame_fn=_draw,
    )
    assert any(m.get("targets") for m in targets_msgs)
    assert pose_msgs, "Expected backward-compatible /vision/target_pose messages"
    p = pose_msgs[-1]
    assert p.header.frame_id == "world"
    assert math.isfinite(p.pose.position.x)
    assert math.isfinite(p.pose.position.y)
    assert math.isfinite(p.pose.position.z)
