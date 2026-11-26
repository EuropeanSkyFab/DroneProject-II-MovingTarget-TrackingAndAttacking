"""Tier 2 BVA & corner-case tests for F6: Pure-NumPy Hungarian Solver & 3-Model IMM Tracker."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys as _sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pytest

from helpers.metrics_validator import validate_planning_target_msg

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
from std_msgs.msg import String  # noqa: E402

pytestmark = [pytest.mark.tier2, pytest.mark.feature(id="F6"), pytest.mark.requires_m3]

ROS2_WS = Path("/home/hkw/flying/ros2_ws")
ENV_SH = ROS2_WS / "src" / "flight_ctrl_pkg" / "scripts" / "env.sh"
INSTALL_SETUP = ROS2_WS / "install" / "setup.bash"


def _make_odom(xyz: Tuple[float, float, float], stamp_sec: float) -> Odometry:
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


def _make_targets_json(stamp_sec: float, targets: Sequence[Dict[str, Any]]) -> String:
    return String(data=json.dumps({"stamp": float(stamp_sec), "frame_id": "world", "targets": list(targets)}))


def _start_planning(domain_id: int, params: Dict[str, Any]):
    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(domain_id)
    env["ROS_LOCALHOST_ONLY"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    ros_args = ["ros2", "run", "planning_pkg", "planning_node", "--ros-args"]
    for k, v in params.items():
        if isinstance(v, bool):
            val_str = "true" if v else "false"
        elif isinstance(v, list):
            val_str = json.dumps(v)
        else:
            val_str = str(v)
        ros_args.extend(["-p", f"{k}:={val_str}"])

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
    node = Node(f"f6_bva_{domain_id}", context=ctx)
    executor = SingleThreadedExecutor(context=ctx)
    executor.add_node(node)
    return proc, ctx, node, executor


def _stop_planning(proc: subprocess.Popen[str], ctx: Context, node: Node, executor: SingleThreadedExecutor) -> None:
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
            pytest.fail(f"planning_node exited early (rc={proc.returncode}):\n{err}\n{out}")
        executor.spin_once(timeout_sec=0.05)
        if (
            node.count_subscribers("/vision/targets") >= 1
            and node.count_subscribers("/strategy/cmd") >= 1
            and node.count_publishers("/planning/diagnostics") >= 1
            and node.count_publishers("/planning/target_pose") >= 1
            and all(node.count_subscribers(f"/{d}/state") >= 1 for d in drone_ids)
        ):
            return
    pytest.fail(
        f"Active install/ planning_node did not subscribe to /vision/targets within {timeout_s}s "
        "(stale pre-M3 install/?)"
    )


def _arm_track(cmd_pub, state_pubs, drones, executor) -> None:
    now = time.time()
    for d, xyz in drones.items():
        state_pubs[d].publish(_make_odom(xyz, now))
    cmd_pub.publish(String(data=json.dumps({"drone_id": "ALL", "action": "TAKEOFF"})))
    for _ in range(4):
        executor.spin_once(timeout_sec=0.03)
    time.sleep(0.08)
    now = time.time()
    for d, xyz in drones.items():
        state_pubs[d].publish(_make_odom(xyz, now))
    cmd_pub.publish(String(data=json.dumps({"drone_id": "ALL", "action": "TRACK"})))
    for _ in range(4):
        executor.spin_once(timeout_sec=0.03)


def test_f06_bva_empty_detection_list_coasting_without_crash() -> None:
    """Verify K=0 empty /vision/targets list coasts confirmed tracks cleanly without crashing."""
    drones = {"drone_1": (0.0, 0.0, 9.5)}
    proc, ctx, node, executor = _start_planning(
        81, {"drone_ids": ["drone_1"], "confirm_m": 3, "confirm_n": 5, "max_coast_sec": 3.0}
    )
    plan_msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(String, "/planning/target_pose", lambda m: plan_msgs.append(json.loads(m.data)), 20)
        vis_pub = node.create_publisher(String, "/vision/targets", 10)
        cmd_pub = node.create_publisher(String, "/strategy/cmd", 10)
        state_pubs = {"drone_1": node.create_publisher(Odometry, "/drone_1/state", 10)}
        _wait_ready(proc, node, executor, ["drone_1"])
        _arm_track(cmd_pub, state_pubs, drones, executor)

        for step in range(5):
            ts = time.time()
            state_pubs["drone_1"].publish(_make_odom(drones["drone_1"], ts))
            vis_pub.publish(_make_targets_json(ts, [
                {"det_id": 0, "class_id": "crimson_uav", "x": 18.0 + 0.3 * step, "y": 0.0, "z": 6.0,
                 "conf": 0.9, "stereo": True, "rays_used": 2, "parallax_deg": 18.0, "skew_rms_m": 0.1},
            ]))
            for _ in range(4):
                executor.spin_once(timeout_sec=0.03)
            time.sleep(0.10)

        plan_msgs.clear()
        for _ in range(5):
            ts = time.time()
            state_pubs["drone_1"].publish(_make_odom(drones["drone_1"], ts))
            vis_pub.publish(_make_targets_json(ts, []))
            for _ in range(4):
                executor.spin_once(timeout_sec=0.03)
            time.sleep(0.08)

        assert proc.poll() is None, "planning_node crashed on K=0 empty detections"
        assert plan_msgs, "Expected continued coasting guidance on /planning/target_pose"
        assert validate_planning_target_msg(plan_msgs[-1]).ok is True
    finally:
        _stop_planning(proc, ctx, node, executor)


def test_f06_bva_unbalanced_k_greater_and_less_than_m() -> None:
    """Verify unbalanced K > M (4 targets, 3 drones) and K < M (1 target, 3 drones) assignments."""
    drones = {"drone_1": (0.0, -4.0, 9.5), "drone_2": (0.0, 0.0, 9.5), "drone_3": (0.0, 4.0, 9.5)}

    # Case 1: K=4 > M=3
    proc, ctx, node, executor = _start_planning(
        82, {"drone_ids": list(drones.keys()), "confirm_m": 3, "confirm_n": 5}
    )
    diag_msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(String, "/planning/diagnostics", lambda m: diag_msgs.append(json.loads(m.data)), 10)
        vis_pub = node.create_publisher(String, "/vision/targets", 10)
        cmd_pub = node.create_publisher(String, "/strategy/cmd", 10)
        state_pubs = {d: node.create_publisher(Odometry, f"/{d}/state", 10) for d in drones}
        _wait_ready(proc, node, executor, list(drones.keys()))
        _arm_track(cmd_pub, state_pubs, drones, executor)

        t_end = time.monotonic() + 2.5
        step = 0
        while time.monotonic() < t_end:
            ts = time.time()
            for d, xyz in drones.items():
                state_pubs[d].publish(_make_odom(xyz, ts))
            vis_pub.publish(_make_targets_json(ts, [
                {"det_id": i, "class_id": "crimson_uav", "x": 18.0 + 0.2 * step, "y": -6.0 + 4.0 * i, "z": 6.0,
                 "conf": 0.9, "stereo": True, "rays_used": 3, "parallax_deg": 18.0, "skew_rms_m": 0.1}
                for i in range(4)
            ]))
            for _ in range(5):
                executor.spin_once(timeout_sec=0.02)
            if step >= 4 and any(
                sum(1 for t in dm.get("tracks", []) if t.get("status") == "CONFIRMED") >= 4
                and dm.get("wta_assignments")
                for dm in diag_msgs
            ):
                break
            step += 1
            time.sleep(0.10)

        four_trk = [
            dm for dm in diag_msgs
            if sum(1 for t in dm.get("tracks", []) if t.get("status") == "CONFIRMED") >= 4
        ]
        assert four_trk, f"Expected 4 confirmed tracks for K=4 > M=3, got {diag_msgs}"
        assert set(four_trk[-1]["wta_assignments"].keys()) == set(drones.keys())
    finally:
        _stop_planning(proc, ctx, node, executor)

    # Case 2: K=1 < M=3
    proc2, ctx2, node2, exec2 = _start_planning(
        83, {"drone_ids": list(drones.keys()), "confirm_m": 3, "confirm_n": 5}
    )
    diag2: List[Dict[str, Any]] = []
    try:
        node2.create_subscription(String, "/planning/diagnostics", lambda m: diag2.append(json.loads(m.data)), 10)
        vis_pub2 = node2.create_publisher(String, "/vision/targets", 10)
        cmd_pub2 = node2.create_publisher(String, "/strategy/cmd", 10)
        state_pubs2 = {d: node2.create_publisher(Odometry, f"/{d}/state", 10) for d in drones}
        _wait_ready(proc2, node2, exec2, list(drones.keys()))
        _arm_track(cmd_pub2, state_pubs2, drones, exec2)

        t_end = time.monotonic() + 2.5
        step = 0
        while time.monotonic() < t_end:
            ts = time.time()
            for d, xyz in drones.items():
                state_pubs2[d].publish(_make_odom(xyz, ts))
            vis_pub2.publish(_make_targets_json(ts, [
                {"det_id": 0, "class_id": "crimson_uav", "x": 20.0 + 0.2 * step, "y": 0.0, "z": 6.0,
                 "conf": 0.9, "stereo": True, "rays_used": 3, "parallax_deg": 18.0, "skew_rms_m": 0.1}
            ]))
            for _ in range(5):
                exec2.spin_once(timeout_sec=0.02)
            if step >= 4 and any(dm.get("wta_assignments") for dm in diag2):
                break
            step += 1
            time.sleep(0.10)
        with_wta = [dm for dm in diag2 if dm.get("wta_assignments")]
        assert with_wta, f"Expected WTA assignment for K=1 < M=3, got {diag2}"
        assert set(with_wta[-1]["wta_assignments"].keys()) == set(drones.keys())
    finally:
        _stop_planning(proc2, ctx2, node2, exec2)


def test_f06_bva_prolonged_dropout_beyond_max_coast_sec_kills_track() -> None:
    """Verify track is deleted cleanly when detection dropout exceeds max_coast_sec."""
    drones = {"drone_1": (0.0, 0.0, 9.5)}
    proc, ctx, node, executor = _start_planning(
        84, {"drone_ids": ["drone_1"], "confirm_m": 3, "confirm_n": 5, "max_coast_sec": 0.8}
    )
    diag_msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(String, "/planning/diagnostics", lambda m: diag_msgs.append(json.loads(m.data)), 10)
        vis_pub = node.create_publisher(String, "/vision/targets", 10)
        cmd_pub = node.create_publisher(String, "/strategy/cmd", 10)
        state_pubs = {"drone_1": node.create_publisher(Odometry, "/drone_1/state", 10)}
        _wait_ready(proc, node, executor, ["drone_1"])
        _arm_track(cmd_pub, state_pubs, drones, executor)

        for step in range(5):
            ts = time.time()
            state_pubs["drone_1"].publish(_make_odom(drones["drone_1"], ts))
            vis_pub.publish(_make_targets_json(ts, [
                {"det_id": 0, "class_id": "crimson_uav", "x": 18.0 + 0.3 * step, "y": 0.0, "z": 6.0,
                 "conf": 0.9, "stereo": True, "rays_used": 2, "parallax_deg": 18.0, "skew_rms_m": 0.1},
            ]))
            for _ in range(4):
                executor.spin_once(timeout_sec=0.03)
            time.sleep(0.09)

        # Wait > max_coast_sec (0.8s)
        diag_msgs.clear()
        t_end = time.monotonic() + 1.6
        while time.monotonic() < t_end:
            state_pubs["drone_1"].publish(_make_odom(drones["drone_1"], time.time()))
            executor.spin_once(timeout_sec=0.05)
            time.sleep(0.05)

        assert diag_msgs and diag_msgs[-1].get("tracks") == [], (
            f"Track survived beyond max_coast_sec=0.8s: {diag_msgs[-1]}"
        )
    finally:
        _stop_planning(proc, ctx, node, executor)


def test_f06_bva_high_g_step_turn_reversal_shifts_imm_mode_away_from_cv() -> None:
    """Verify sharp high-g target maneuver increases CTRV+JINK mode probability relative to CV."""
    drones = {"drone_1": (0.0, 0.0, 9.5)}
    proc, ctx, node, executor = _start_planning(
        85, {"drone_ids": ["drone_1"], "confirm_m": 3, "confirm_n": 5}
    )
    diag_msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(String, "/planning/diagnostics", lambda m: diag_msgs.append(json.loads(m.data)), 10)
        vis_pub = node.create_publisher(String, "/vision/targets", 10)
        cmd_pub = node.create_publisher(String, "/strategy/cmd", 10)
        state_pubs = {"drone_1": node.create_publisher(Odometry, "/drone_1/state", 10)}
        _wait_ready(proc, node, executor, ["drone_1"])
        _arm_track(cmd_pub, state_pubs, drones, executor)

        # Phase 1: Smooth straight-line CV motion (vx = 6 m/s)
        x_cur, y_cur = 18.0, 0.0
        t_cv = time.monotonic() + 2.0
        step = 0
        while time.monotonic() < t_cv:
            ts = time.time()
            x_cur += 0.6
            state_pubs["drone_1"].publish(_make_odom(drones["drone_1"], ts))
            vis_pub.publish(_make_targets_json(ts, [
                {"det_id": 0, "class_id": "crimson_uav", "x": x_cur, "y": y_cur, "z": 6.0,
                 "conf": 0.95, "stereo": True, "rays_used": 2, "parallax_deg": 20.0, "skew_rms_m": 0.05},
            ]))
            for _ in range(4):
                executor.spin_once(timeout_sec=0.02)
            if step >= 6 and any(
                t.get("status") == "CONFIRMED" for dm in diag_msgs for t in dm.get("tracks", [])
            ):
                break
            step += 1
            time.sleep(0.10)

        cv_tracks = [t for dm in diag_msgs for t in dm.get("tracks", []) if t.get("status") == "CONFIRMED"]
        assert cv_tracks, f"No confirmed track after CV phase: {diag_msgs}"
        cv_prob_before = cv_tracks[-1]["imm_probs"]["CV"]

        # Phase 2: Sharp lateral jink/reversal (high-g turn in -X / +Y)
        n0 = len(diag_msgs)
        t_jink = time.monotonic() + 2.5
        step = 0
        while time.monotonic() < t_jink:
            ts = time.time()
            x_cur -= 0.4
            y_cur += 0.7 + 0.12 * step
            state_pubs["drone_1"].publish(_make_odom(drones["drone_1"], ts))
            vis_pub.publish(_make_targets_json(ts, [
                {"det_id": 0, "class_id": "crimson_uav", "x": x_cur, "y": y_cur, "z": 6.0,
                 "conf": 0.95, "stereo": True, "rays_used": 2, "parallax_deg": 20.0, "skew_rms_m": 0.05},
            ]))
            for _ in range(4):
                executor.spin_once(timeout_sec=0.02)
            if step == 6:
                n0 = len(diag_msgs)
            if step >= 7 and any(
                t.get("status") == "CONFIRMED" for dm in diag_msgs[n0:] for t in dm.get("tracks", [])
            ):
                break
            step += 1
            time.sleep(0.10)

        jink_tracks = [
            t for dm in diag_msgs[n0:] for t in dm.get("tracks", []) if t.get("status") == "CONFIRMED"
        ]
        assert jink_tracks, f"Track lost during high-g jink: {diag_msgs}"
        probs_after = jink_tracks[-1]["imm_probs"]
        assert probs_after["CTRV"] + probs_after["JINK"] > 0.50 or probs_after["CV"] < cv_prob_before, (
            f"Expected IMM to shift away from CV after jink (before={cv_prob_before}, after={probs_after})"
        )
    finally:
        _stop_planning(proc, ctx, node, executor)


def test_f06_bva_nan_inf_malformed_detection_resilience_and_recovery() -> None:
    """Verify malformed JSON and NaN/Inf/bad detections do not crash planning_node and recovery succeeds."""
    drones = {"drone_1": (0.0, 0.0, 9.5)}
    proc, ctx, node, executor = _start_planning(
        86, {"drone_ids": ["drone_1"], "confirm_m": 3, "confirm_n": 5}
    )
    diag_msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(String, "/planning/diagnostics", lambda m: diag_msgs.append(json.loads(m.data)), 10)
        vis_pub = node.create_publisher(String, "/vision/targets", 10)
        cmd_pub = node.create_publisher(String, "/strategy/cmd", 10)
        state_pubs = {"drone_1": node.create_publisher(Odometry, "/drone_1/state", 10)}
        _wait_ready(proc, node, executor, ["drone_1"])
        _arm_track(cmd_pub, state_pubs, drones, executor)

        # Send malformed JSON and NaN/Inf raw JSON payloads
        bad_payloads = [
            "{not valid json",
            json.dumps({"stamp": "NaN", "frame_id": "world", "targets": []}),
            (
                '{"stamp": ' + str(time.time())
                + ', "frame_id": "world", "targets": [{"det_id": 0, "x": NaN, "y": Infinity, "z": 5.0, "conf": 0.9}]}'
            ),
            json.dumps({
                "stamp": time.time(),
                "frame_id": "world",
                "targets": [{"det_id": 0, "x": 1e9, "y": 0.0, "z": 5.0, "conf": -1.0}],
            }),
        ]
        for bad in bad_payloads:
            state_pubs["drone_1"].publish(_make_odom(drones["drone_1"], time.time()))
            vis_pub.publish(String(data=bad))
            executor.spin_once(timeout_sec=0.05)
            time.sleep(0.04)

        assert proc.poll() is None, "planning_node crashed on malformed/NaN /vision/targets"

        # Now feed clean detections continuously until CONFIRMED track appears in 2 Hz diagnostics
        diag_msgs.clear()
        t_end = time.monotonic() + 2.5
        step = 0
        while time.monotonic() < t_end:
            ts = time.time()
            state_pubs["drone_1"].publish(_make_odom(drones["drone_1"], ts))
            vis_pub.publish(_make_targets_json(ts, [
                {"det_id": 0, "class_id": "crimson_uav", "x": 18.0 + 0.3 * step, "y": 0.0, "z": 6.0,
                 "conf": 0.95, "stereo": True, "rays_used": 2, "parallax_deg": 18.0, "skew_rms_m": 0.1},
            ]))
            for _ in range(4):
                executor.spin_once(timeout_sec=0.02)
            if step >= 4 and any(
                t.get("status") == "CONFIRMED" for dm in diag_msgs for t in dm.get("tracks", [])
            ):
                break
            step += 1
            time.sleep(0.10)
        assert proc.poll() is None
        assert any(t.get("status") == "CONFIRMED" for dm in diag_msgs for t in dm.get("tracks", [])), (
            f"planning_node failed to recover after corrupt payloads: {diag_msgs}"
        )
    finally:
        _stop_planning(proc, ctx, node, executor)
