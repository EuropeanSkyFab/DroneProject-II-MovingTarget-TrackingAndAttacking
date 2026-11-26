"""Tier 1 functional tests for F6: Pure-NumPy Hungarian O(K^3) Solver & 3-Model IMM Tracker."""

from __future__ import annotations

import json
import math
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

pytestmark = [pytest.mark.tier1, pytest.mark.feature(id="F6"), pytest.mark.requires_m3]

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


def _make_targets_json(
    stamp_sec: float,
    targets: Sequence[Dict[str, Any]],
) -> String:
    msg = String()
    msg.data = json.dumps({
        "stamp": float(stamp_sec),
        "frame_id": "world",
        "targets": list(targets),
    })
    return msg


def _start_planning_harness(domain_id: int, params: Dict[str, Any]):
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
    node = Node(f"f6_probe_{domain_id}", context=ctx)
    executor = SingleThreadedExecutor(context=ctx)
    executor.add_node(node)
    return proc, ctx, node, executor


def _stop_planning_harness(
    proc: subprocess.Popen[str],
    ctx: Context,
    node: Node,
    executor: SingleThreadedExecutor,
) -> None:
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


def _wait_planning_ready(
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
        has_vision_sub = node.count_subscribers("/vision/targets") >= 1
        has_cmd_sub = node.count_subscribers("/strategy/cmd") >= 1
        has_diag_pub = node.count_publishers("/planning/diagnostics") >= 1
        has_plan_pub = node.count_publishers("/planning/target_pose") >= 1
        has_state_sub = all(node.count_subscribers(f"/{d}/state") >= 1 for d in drone_ids)
        if has_vision_sub and has_cmd_sub and has_diag_pub and has_plan_pub and has_state_sub:
            return
    pytest.fail(
        "Active install/ planning_node did not subscribe to /vision/targets within "
        f"{timeout_s}s (stale pre-M3 install/?)"
    )


def _send_takeoff_and_track(
    cmd_pub,
    state_pubs: Dict[str, Any],
    drone_poses: Dict[str, Tuple[float, float, float]],
    executor: SingleThreadedExecutor,
) -> None:
    now = time.time()
    for d, xyz in drone_poses.items():
        state_pubs[d].publish(_make_odom(xyz, now))
    cmd_pub.publish(String(data=json.dumps({"drone_id": "ALL", "action": "TAKEOFF"})))
    for _ in range(4):
        executor.spin_once(timeout_sec=0.03)
    time.sleep(0.08)
    now = time.time()
    for d, xyz in drone_poses.items():
        state_pubs[d].publish(_make_odom(xyz, now))
    cmd_pub.publish(String(data=json.dumps({"drone_id": "ALL", "action": "TRACK"})))
    for _ in range(4):
        executor.spin_once(timeout_sec=0.03)


def test_f06_planning_diagnostics_exposes_imm_tracks_cov_and_wta() -> None:
    """Verify /planning/diagnostics exposes tracks, imm_probs (CV/CTRV/JINK), covariance std, and WTA map."""
    drones = {
        "drone_1": (0.0, -4.0, 9.5),
        "drone_2": (0.0, 0.0, 9.5),
        "drone_3": (0.0, 4.0, 9.5),
    }
    proc, ctx, node, executor = _start_planning_harness(
        71, {"drone_ids": list(drones.keys()), "confirm_m": 3, "confirm_n": 5, "plan_hz": 20.0}
    )
    diag_msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(String, "/planning/diagnostics", lambda m: diag_msgs.append(json.loads(m.data)), 10)
        vis_pub = node.create_publisher(String, "/vision/targets", 10)
        cmd_pub = node.create_publisher(String, "/strategy/cmd", 10)
        state_pubs = {d: node.create_publisher(Odometry, f"/{d}/state", 10) for d in drones}

        _wait_planning_ready(proc, node, executor, list(drones.keys()))
        _send_takeoff_and_track(cmd_pub, state_pubs, drones, executor)

        # Publish continuously at 10 Hz until 2 Hz diagnostic pulse reports >=2 tracks + WTA
        deadline = time.monotonic() + 2.5
        step = 0
        while time.monotonic() < deadline:
            ts = time.time()
            for d, xyz in drones.items():
                state_pubs[d].publish(_make_odom(xyz, ts))
            vis_pub.publish(_make_targets_json(ts, [
                {"det_id": 0, "class_id": "crimson_uav", "x": 18.0 + 0.4 * step, "y": -3.0, "z": 6.0,
                 "conf": 0.92, "stereo": True, "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
                {"det_id": 1, "class_id": "amber_uav", "x": 22.0 + 0.4 * step, "y": 4.0, "z": 6.5,
                 "conf": 0.90, "stereo": True, "rays_used": 3, "parallax_deg": 18.0, "skew_rms_m": 0.1},
            ]))
            for _ in range(5):
                executor.spin_once(timeout_sec=0.02)
            if step >= 5 and any(
                len(dm.get("tracks", [])) >= 2 and dm.get("wta_assignments") for dm in diag_msgs
            ):
                break
            step += 1
            time.sleep(0.10)

        rich = [dm for dm in diag_msgs if len(dm.get("tracks", [])) >= 2 and dm.get("wta_assignments")]
        assert rich, f"Expected >=2 tracks & wta_assignments in /planning/diagnostics, got {diag_msgs}"
        dm = rich[-1]
        assert "tracks" in dm and "wta_assignments" in dm
        for trk in dm["tracks"]:
            assert set(trk["imm_probs"].keys()) == {"CV", "CTRV", "JINK"}
            assert math.isfinite(trk["pos_rmse_std_m"]) and trk["pos_rmse_std_m"] > 0.0
    finally:
        _stop_planning_harness(proc, ctx, node, executor)


def test_f06_imm_mode_probability_simplex_and_positive_covariance() -> None:
    """Verify IMM mode probabilities form a valid simplex (sum == 1.0 +- 0.05) and positive finite std."""
    drones = {"drone_1": (0.0, 0.0, 9.5), "drone_2": (0.0, 4.0, 9.5)}
    proc, ctx, node, executor = _start_planning_harness(
        72, {"drone_ids": list(drones.keys()), "confirm_m": 3, "confirm_n": 5, "plan_hz": 20.0}
    )
    diag_msgs: List[Dict[str, Any]] = []
    plan_msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(String, "/planning/diagnostics", lambda m: diag_msgs.append(json.loads(m.data)), 10)
        node.create_subscription(String, "/planning/target_pose", lambda m: plan_msgs.append(json.loads(m.data)), 20)
        vis_pub = node.create_publisher(String, "/vision/targets", 10)
        cmd_pub = node.create_publisher(String, "/strategy/cmd", 10)
        state_pubs = {d: node.create_publisher(Odometry, f"/{d}/state", 10) for d in drones}

        _wait_planning_ready(proc, node, executor, list(drones.keys()))
        _send_takeoff_and_track(cmd_pub, state_pubs, drones, executor)

        deadline = time.monotonic() + 2.5
        step = 0
        while time.monotonic() < deadline:
            ts = time.time()
            for d, xyz in drones.items():
                state_pubs[d].publish(_make_odom(xyz, ts))
            vis_pub.publish(_make_targets_json(ts, [
                {"det_id": 0, "class_id": "crimson_uav", "x": 16.0 + 0.5 * step, "y": 1.0, "z": 6.0,
                 "conf": 0.95, "stereo": True, "rays_used": 2, "parallax_deg": 18.0, "skew_rms_m": 0.1},
            ]))
            for _ in range(5):
                executor.spin_once(timeout_sec=0.02)
            has_confirmed = any(
                trk.get("status") == "CONFIRMED"
                for dm in diag_msgs
                for trk in dm.get("tracks", [])
            )
            if has_confirmed and any("imm_probs" in p for p in plan_msgs):
                break
            step += 1
            time.sleep(0.10)

        confirmed_diags = [
            trk
            for dm in diag_msgs
            for trk in dm.get("tracks", [])
            if trk.get("status") == "CONFIRMED"
        ]
        assert confirmed_diags, f"No CONFIRMED IMM tracks in diagnostics: {diag_msgs}"
        for trk in confirmed_diags:
            probs = trk["imm_probs"]
            assert set(probs.keys()) == {"CV", "CTRV", "JINK"}
            assert all(0.0 <= float(v) <= 1.0 for v in probs.values())
            assert abs(sum(float(v) for v in probs.values()) - 1.0) <= 0.05
            assert math.isfinite(trk["pos_rmse_std_m"]) and trk["pos_rmse_std_m"] > 0.0

        with_probs = [p for p in plan_msgs if "imm_probs" in p]
        assert with_probs, f"Expected imm_probs in /planning/target_pose, got {plan_msgs[:3]}"
        assert abs(sum(float(v) for v in with_probs[-1]["imm_probs"].values()) - 1.0) <= 0.05
    finally:
        _stop_planning_harness(proc, ctx, node, executor)


def test_f06_multi_horizon_pred_path_and_contract_validation() -> None:
    """Verify /planning/target_pose passes Contract #5 validation and outputs accurate pred_path."""
    drones = {"drone_1": (0.0, 0.0, 9.5), "drone_2": (0.0, 4.0, 9.5)}
    proc, ctx, node, executor = _start_planning_harness(
        73, {"drone_ids": list(drones.keys()), "confirm_m": 3, "confirm_n": 5, "plan_hz": 20.0}
    )
    plan_msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(String, "/planning/target_pose", lambda m: plan_msgs.append(json.loads(m.data)), 20)
        vis_pub = node.create_publisher(String, "/vision/targets", 10)
        cmd_pub = node.create_publisher(String, "/strategy/cmd", 10)
        state_pubs = {d: node.create_publisher(Odometry, f"/{d}/state", 10) for d in drones}

        _wait_planning_ready(proc, node, executor, list(drones.keys()))
        _send_takeoff_and_track(cmd_pub, state_pubs, drones, executor)

        # Constant velocity vx = +4.0 m/s along +X
        vx = 4.0
        t0 = time.time()
        last_x = 15.0
        for step in range(8):
            ts = time.time()
            dt = ts - t0
            last_x = 15.0 + vx * dt
            for d, xyz in drones.items():
                state_pubs[d].publish(_make_odom(xyz, ts))
            vis_pub.publish(_make_targets_json(ts, [
                {"det_id": 0, "class_id": "crimson_uav", "x": last_x, "y": 2.0, "z": 6.0,
                 "conf": 0.95, "stereo": True, "rays_used": 2, "parallax_deg": 20.0, "skew_rms_m": 0.05},
            ]))
            for _ in range(5):
                executor.spin_once(timeout_sec=0.03)
            time.sleep(0.10)
        for _ in range(4):
            executor.spin_once(timeout_sec=0.02)

        with_path = [p for p in plan_msgs if p.get("pred_path")]
        assert with_path, f"Expected non-empty pred_path in /planning/target_pose: {plan_msgs[:4]}"
        latest = with_path[-1]
        res = validate_planning_target_msg(latest)
        assert res.ok is True, f"Contract #5 validation failed: {res.errors}"
        pp = latest["pred_path"]
        assert len(pp) >= 3
        assert pp[-1][0] > pp[0][0], f"pred_path should extrapolate forward in +X: {pp}"
        # Check 1.0s horizon (index 3 in default horizons [0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
        if len(pp) >= 4:
            expected_1s_x = last_x + vx * 1.0
            err_1s = math.hypot(pp[3][0] - expected_1s_x, pp[3][1] - 2.0)
            assert err_1s <= 2.5, f"1s pred_path error {err_1s:.2f}m > 2.5m (pp[3]={pp[3]})"
    finally:
        _stop_planning_harness(proc, ctx, node, executor)


def test_f06_track_birth_tentative_confirmed_coast_delete_lifecycle() -> None:
    """Verify track lifecycle: 1 hit -> TENTATIVE, >=3 hits -> CONFIRMED, dropout -> COASTING -> deleted."""
    drones = {"drone_1": (0.0, 0.0, 9.5)}
    proc, ctx, node, executor = _start_planning_harness(
        74,
        {
            "drone_ids": ["drone_1"],
            "confirm_m": 3,
            "confirm_n": 5,
            "max_coast_sec": 1.1,
            "plan_hz": 20.0,
        },
    )
    diag_msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(String, "/planning/diagnostics", lambda m: diag_msgs.append(json.loads(m.data)), 10)
        vis_pub = node.create_publisher(String, "/vision/targets", 10)
        cmd_pub = node.create_publisher(String, "/strategy/cmd", 10)
        state_pubs = {"drone_1": node.create_publisher(Odometry, "/drone_1/state", 10)}

        _wait_planning_ready(proc, node, executor, ["drone_1"])
        _send_takeoff_and_track(cmd_pub, state_pubs, drones, executor)

        # 1) Single observation -> TENTATIVE
        ts = time.time()
        state_pubs["drone_1"].publish(_make_odom(drones["drone_1"], ts))
        vis_pub.publish(_make_targets_json(ts, [
            {"det_id": 0, "class_id": "crimson_uav", "x": 18.0, "y": 0.0, "z": 6.0,
             "conf": 0.9, "stereo": True, "rays_used": 2, "parallax_deg": 18.0, "skew_rms_m": 0.1},
        ]))
        t_wait = time.monotonic() + 1.0
        while time.monotonic() < t_wait and not any(dm.get("tracks") for dm in diag_msgs):
            executor.spin_once(timeout_sec=0.04)
            time.sleep(0.03)
        tentative_seen = any(
            t.get("status") == "TENTATIVE" for dm in diag_msgs for t in dm.get("tracks", [])
        )
        assert tentative_seen, f"Expected TENTATIVE status after 1 observation: {diag_msgs}"

        # 2) Confirm with continuous hits at 10 Hz until CONFIRMED appears in 2 Hz diagnostics
        diag_msgs.clear()
        t_confirm = time.monotonic() + 2.0
        step = 1
        while time.monotonic() < t_confirm:
            ts = time.time()
            state_pubs["drone_1"].publish(_make_odom(drones["drone_1"], ts))
            vis_pub.publish(_make_targets_json(ts, [
                {"det_id": 0, "class_id": "crimson_uav", "x": 18.0 + 0.2 * step, "y": 0.0, "z": 6.0,
                 "conf": 0.9, "stereo": True, "rays_used": 2, "parallax_deg": 18.0, "skew_rms_m": 0.1},
            ]))
            for _ in range(4):
                executor.spin_once(timeout_sec=0.02)
            if any(t.get("status") == "CONFIRMED" for dm in diag_msgs for t in dm.get("tracks", [])):
                break
            step += 1
            time.sleep(0.10)
        assert any(t.get("status") == "CONFIRMED" for dm in diag_msgs for t in dm.get("tracks", []))

        # 3) Stop vision detections -> COASTING then DELETED after max_coast_sec=1.1s
        diag_msgs.clear()
        t_drop = time.monotonic() + 2.0
        while time.monotonic() < t_drop:
            state_pubs["drone_1"].publish(_make_odom(drones["drone_1"], time.time()))
            executor.spin_once(timeout_sec=0.05)
            time.sleep(0.05)

        coasting_seen = any(
            t.get("status") == "COASTING" for dm in diag_msgs for t in dm.get("tracks", [])
        )
        assert coasting_seen, f"Expected COASTING status during dropout: {diag_msgs}"
        assert diag_msgs[-1].get("tracks") == [], (
            f"Expected track deletion after max_coast_sec=1.1s, got {diag_msgs[-1]}"
        )
    finally:
        _stop_planning_harness(proc, ctx, node, executor)


def test_f06_multi_target_k3_hungarian_association_and_wta() -> None:
    """Verify K>=3 targets undergo Hungarian association and unique track formation cleanly."""
    drones = {
        "drone_1": (0.0, -5.0, 9.5),
        "drone_2": (0.0, 0.0, 9.5),
        "drone_3": (0.0, 5.0, 9.5),
    }
    proc, ctx, node, executor = _start_planning_harness(
        75, {"drone_ids": list(drones.keys()), "confirm_m": 3, "confirm_n": 5, "plan_hz": 20.0}
    )
    diag_msgs: List[Dict[str, Any]] = []
    plan_msgs: List[Dict[str, Any]] = []
    try:
        node.create_subscription(String, "/planning/diagnostics", lambda m: diag_msgs.append(json.loads(m.data)), 10)
        node.create_subscription(String, "/planning/target_pose", lambda m: plan_msgs.append(json.loads(m.data)), 20)
        vis_pub = node.create_publisher(String, "/vision/targets", 10)
        cmd_pub = node.create_publisher(String, "/strategy/cmd", 10)
        state_pubs = {d: node.create_publisher(Odometry, f"/{d}/state", 10) for d in drones}

        _wait_planning_ready(proc, node, executor, list(drones.keys()))
        _send_takeoff_and_track(cmd_pub, state_pubs, drones, executor)

        t0 = time.monotonic()
        deadline = t0 + 2.0
        step = 0
        while time.monotonic() < deadline:
            ts = time.time()
            for d, xyz in drones.items():
                state_pubs[d].publish(_make_odom(xyz, ts))
            vis_pub.publish(_make_targets_json(ts, [
                {"det_id": 0, "class_id": "crimson_uav", "x": 18.0 + 0.3 * step, "y": -6.0, "z": 6.0,
                 "conf": 0.94, "stereo": True, "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
                {"det_id": 1, "class_id": "amber_uav", "x": 21.0 + 0.3 * step, "y": 0.0, "z": 7.0,
                 "conf": 0.93, "stereo": True, "rays_used": 3, "parallax_deg": 19.0, "skew_rms_m": 0.1},
                {"det_id": 2, "class_id": "slate_uav", "x": 24.0 + 0.3 * step, "y": 6.0, "z": 8.0,
                 "conf": 0.91, "stereo": True, "rays_used": 3, "parallax_deg": 18.0, "skew_rms_m": 0.1},
            ]))
            for _ in range(5):
                executor.spin_once(timeout_sec=0.02)
            if step >= 5 and any(
                sum(1 for t in dm.get("tracks", []) if t.get("status") == "CONFIRMED") == 3
                and dm.get("wta_assignments")
                for dm in diag_msgs
            ):
                break
            step += 1
            time.sleep(0.10)

        confirmed_three = [
            dm for dm in diag_msgs
            if sum(1 for t in dm.get("tracks", []) if t.get("status") == "CONFIRMED") == 3
        ]
        assert confirmed_three, f"Expected 3 CONFIRMED tracks via Hungarian GNN, got {diag_msgs}"
        wta = confirmed_three[-1]["wta_assignments"]
        assert set(wta.keys()) == set(drones.keys())
        assert len(set(wta.values())) == 3, f"Expected 1-to-1 WTA across 3 drones & 3 targets, got {wta}"
        assert time.monotonic() - t0 < 2.5
    finally:
        _stop_planning_harness(proc, ctx, node, executor)
