"""Tier 2 BVA & corner-case opaque-box tests for F8: 3D CBF Safety Shield & Control Bridge."""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import pytest

from helpers.arena_parser import load_obstacles_yaml, point_min_clearance
from helpers.ros_probe import run_command_with_gpu_guard

pytestmark = [
    pytest.mark.tier2,
    pytest.mark.feature(id="F8"),
    pytest.mark.requires_m4,
]


def _extract_probe_json(stdout: str) -> Dict[str, Any]:
    begin_marker = "__E2E_JSON_BEGIN__"
    end_marker = "__E2E_JSON_END__"
    assert begin_marker in stdout and end_marker in stdout, (
        f"Probe driver did not emit JSON envelope. stdout={stdout!r}"
    )
    raw = stdout.split(begin_marker, 1)[1].split(end_marker, 1)[0].strip()
    return json.loads(raw)


def _run_center_probe(
    tmp_path: Path,
    domain_id: int,
    node_params: List[str],
    drones: List[str],
    driver_body: str,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    script_path = tmp_path / f"f08_bva_center_{domain_id}.py"
    full_script = textwrap.dedent(
        f"""\
        import json
        import math
        import os
        import signal
        import subprocess
        import time

        import rclpy
        from geometry_msgs.msg import PoseStamped, TwistStamped
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from std_msgs.msg import String

        NODE_ARGS = ["ros2", "run", "center_node_pkg", "center_node", "--ros-args"] + {repr(node_params)}
        DRONES = {repr(drones)}


        class CenterBvaProbe(Node):
            def __init__(self):
                super().__init__("f08_bva_center_{domain_id}")
                self.setpoints = {{d: [] for d in DRONES}}
                self.ffs = {{d: [] for d in DRONES}}
                self.fleet_states = []
                self.state_pubs = {{}}
                self.plan_pub = self.create_publisher(String, "/planning/target_pose", 30)
                self.strat_pub = self.create_publisher(String, "/strategy/cmd", 10)
                self.create_subscription(String, "/fleet/state", self._on_fleet, 20)
                for d in DRONES:
                    self.state_pubs[d] = self.create_publisher(Odometry, f"/{{d}}/state", 10)
                    self.create_subscription(
                        PoseStamped,
                        f"/{{d}}/fmu/in/trajectory_setpoint",
                        lambda m, did=d: self._on_sp(did, m),
                        20,
                    )
                    self.create_subscription(
                        TwistStamped,
                        f"/{{d}}/fmu/in/trajectory_ff",
                        lambda m, did=d: self._on_ff(did, m),
                        20,
                    )

            def _on_fleet(self, msg):
                try:
                    self.fleet_states.append(json.loads(msg.data))
                except Exception:
                    pass

            def _on_sp(self, did, msg):
                p = msg.pose.position
                self.setpoints[did].append((float(p.x), float(p.y), float(p.z)))

            def _on_ff(self, did, msg):
                v = msg.twist.linear
                self.ffs[did].append((float(v.x), float(v.y), float(v.z)))

            def pub_state(self, did, x, y, z, vx=0.0, vy=0.0, vz=0.0):
                m = Odometry()
                m.header.stamp = self.get_clock().now().to_msg()
                m.header.frame_id = "world"
                m.pose.pose.position.x = float(x)
                m.pose.pose.position.y = float(y)
                m.pose.pose.position.z = float(z)
                m.pose.pose.orientation.w = 1.0
                m.twist.twist.linear.x = float(vx)
                m.twist.twist.linear.y = float(vy)
                m.twist.twist.linear.z = float(vz)
                self.state_pubs[did].publish(m)

            def pub_strategy(self, action, drone_id="ALL"):
                m = String()
                m.data = json.dumps({{"drone_id": drone_id, "action": action}})
                self.strat_pub.publish(m)

            def pub_plan(self, did, x, y, z, vx=0.0, vy=0.0, vz=0.0, action="TRACK"):
                m = String()
                m.data = json.dumps({{
                    "drone_id": did,
                    "stamp": time.time(),
                    "x": x,
                    "y": y,
                    "z": z,
                    "vx": vx,
                    "vy": vy,
                    "vz": vz,
                    "ax": 0.0,
                    "ay": 0.0,
                    "az": 0.0,
                    "action": action,
                    "t_go": 1.0,
                    "target_id": "tgt_1",
                }})
                self.plan_pub.publish(m)

            def spin_for(self, seconds):
                end = time.monotonic() + seconds
                while time.monotonic() < end:
                    rclpy.spin_once(self, timeout_sec=0.02)


        def wait_ready(node, timeout_s=8.0):
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                if (
                    node.count_subscribers("/planning/target_pose") >= 1
                    and node.count_subscribers("/strategy/cmd") >= 1
                    and node.count_publishers("/fleet/state") >= 1
                ):
                    return True
            return False


        def main():
            proc = subprocess.Popen(
                NODE_ARGS,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            rclpy.init()
            node = CenterBvaProbe()
            result = {{"ready": False, "node_rc": None}}
            try:
                ready = wait_ready(node, timeout_s=8.0)
                result["ready"] = ready
                if ready:
                    __DRIVER_BODY__
                result["node_rc"] = proc.poll()
            finally:
                node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=3.0)
                except Exception:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        pass
            print("__E2E_JSON_BEGIN__")
            print(json.dumps(result))
            print("__E2E_JSON_END__")


        if __name__ == "__main__":
            main()
        """
    ).replace(
        "            __DRIVER_BODY__",
        textwrap.indent(textwrap.dedent(driver_body).strip(), "            "),
    )
    compile(full_script, str(script_path), "exec")
    script_path.write_text(full_script, encoding="utf-8")
    res = run_command_with_gpu_guard(
        ["python3", str(script_path)],
        timeout_s=timeout_s,
        extra_env={"ROS_DOMAIN_ID": str(domain_id), "ROS_LOCALHOST_ONLY": "1"},
        source_env=True,
    )
    assert res.timed_out is False, f"Center BVA probe timed out: stderr={res.stderr}"
    assert res.returncode == 0, f"Center BVA probe failed rc={res.returncode}: stderr={res.stderr}"
    data = _extract_probe_json(res.stdout)
    assert data.get("ready") is True, f"center_node failed DDS discovery: {data}"
    assert data.get("node_rc") is None, f"center_node exited prematurely: rc={data.get('node_rc')}"
    return data


def _run_px4_bridge_probe(
    tmp_path: Path,
    domain_id: int,
    node_params: List[str],
    driver_body: str,
    timeout_s: float = 28.0,
) -> Dict[str, Any]:
    script_path = tmp_path / f"f08_bva_bridge_{domain_id}.py"
    full_script = textwrap.dedent(
        f"""\
        import json
        import math
        import os
        import signal
        import subprocess
        import time

        import rclpy
        from geometry_msgs.msg import PoseStamped, TwistStamped
        from px4_msgs.msg import (
            EstimatorStatusFlags,
            TrajectorySetpoint,
            VehicleCommand,
            VehicleLocalPosition,
            VehicleStatus,
        )
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Empty, String

        NODE_ARGS = ["ros2", "run", "flight_ctrl_pkg", "px4_bridge_node", "--ros-args"] + {repr(node_params)}


        class BridgeBvaProbe(Node):
            def __init__(self):
                super().__init__("f08_bva_bridge_{domain_id}")
                px4_qos = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                )
                self.px4_setpoints = []
                self.vehicle_cmds = []

                self.sp_pub = self.create_publisher(PoseStamped, "/drone_1/fmu/in/trajectory_setpoint", 10)
                self.ff_pub = self.create_publisher(TwistStamped, "/drone_1/fmu/in/trajectory_ff", 10)
                self.hb_pub = self.create_publisher(Empty, "/drone_1/heartbeat", 10)
                self.act_pub = self.create_publisher(String, "/drone_1/action", 10)

                self.lpos_pub = self.create_publisher(
                    VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", px4_qos
                )
                self.vstat_pub = self.create_publisher(
                    VehicleStatus, "/fmu/out/vehicle_status_v1", px4_qos
                )
                self.est_pub = self.create_publisher(
                    EstimatorStatusFlags, "/fmu/out/estimator_status_flags", px4_qos
                )

                self.create_subscription(
                    TrajectorySetpoint, "/fmu/in/trajectory_setpoint", self._on_px4_sp, px4_qos
                )
                self.create_subscription(
                    VehicleCommand, "/fmu/in/vehicle_command", self._on_vcmd, px4_qos
                )

            def _on_px4_sp(self, msg):
                self.px4_setpoints.append({{
                    "pos": [float(v) for v in msg.position],
                    "vel": [float(v) for v in msg.velocity],
                }})

            def _on_vcmd(self, msg):
                self.vehicle_cmds.append(int(msg.command))

            def pub_px4_telemetry(self, n=0.0, e=0.0, d=-6.0, vn=0.0, ve=0.0, vd=0.0):
                ts_us = int(self.get_clock().now().nanoseconds / 1000)
                lp = VehicleLocalPosition()
                lp.timestamp = ts_us
                lp.xy_valid = True
                lp.z_valid = True
                lp.v_xy_valid = True
                lp.v_z_valid = True
                lp.x = float(n)
                lp.y = float(e)
                lp.z = float(d)
                lp.vx = float(vn)
                lp.vy = float(ve)
                lp.vz = float(vd)
                lp.heading = 0.0
                self.lpos_pub.publish(lp)

                ef = EstimatorStatusFlags()
                ef.timestamp = ts_us
                ef.cs_tilt_align = True
                ef.cs_yaw_align = True
                self.est_pub.publish(ef)

                vs = VehicleStatus()
                vs.timestamp = ts_us
                vs.arming_state = VehicleStatus.ARMING_STATE_ARMED
                vs.nav_state = VehicleStatus.NAVIGATION_STATE_OFFBOARD
                vs.pre_flight_checks_pass = True
                self.vstat_pub.publish(vs)

            def pub_sp_enu(self, x, y, z):
                m = PoseStamped()
                m.header.stamp = self.get_clock().now().to_msg()
                m.header.frame_id = "world"
                m.pose.position.x = float(x)
                m.pose.position.y = float(y)
                m.pose.position.z = float(z)
                m.pose.orientation.w = 1.0
                self.sp_pub.publish(m)

            def pub_ff_enu(self, vx, vy, vz):
                m = TwistStamped()
                m.header.stamp = self.get_clock().now().to_msg()
                m.header.frame_id = "world"
                m.twist.linear.x = float(vx)
                m.twist.linear.y = float(vy)
                m.twist.linear.z = float(vz)
                self.ff_pub.publish(m)

            def pub_action(self, act):
                m = String()
                m.data = act
                self.act_pub.publish(m)

            def spin_for(self, seconds):
                end = time.monotonic() + seconds
                while time.monotonic() < end:
                    rclpy.spin_once(self, timeout_sec=0.02)


        def wait_ready(node, timeout_s=8.0):
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                if (
                    node.count_subscribers("/drone_1/fmu/in/trajectory_setpoint") >= 1
                    and node.count_publishers("/fmu/in/trajectory_setpoint") >= 1
                ):
                    return True
            return False


        def main():
            proc = subprocess.Popen(
                NODE_ARGS,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            rclpy.init()
            node = BridgeBvaProbe()
            result = {{"ready": False, "node_rc": None}}
            try:
                ready = wait_ready(node, timeout_s=8.0)
                result["ready"] = ready
                if ready:
                    __DRIVER_BODY__
                result["node_rc"] = proc.poll()
            finally:
                node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=3.0)
                except Exception:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        pass
            print("__E2E_JSON_BEGIN__")
            print(json.dumps(result))
            print("__E2E_JSON_END__")


        if __name__ == "__main__":
            main()
        """
    ).replace(
        "            __DRIVER_BODY__",
        textwrap.indent(textwrap.dedent(driver_body).strip(), "            "),
    )
    compile(full_script, str(script_path), "exec")
    script_path.write_text(full_script, encoding="utf-8")
    res = run_command_with_gpu_guard(
        ["python3", str(script_path)],
        timeout_s=timeout_s,
        extra_env={"ROS_DOMAIN_ID": str(domain_id), "ROS_LOCALHOST_ONLY": "1"},
        source_env=True,
    )
    assert res.timed_out is False, f"Bridge BVA probe timed out: stderr={res.stderr}"
    assert res.returncode == 0, f"Bridge BVA probe failed rc={res.returncode}: stderr={res.stderr}"
    data = _extract_probe_json(res.stdout)
    assert data.get("ready") is True, f"px4_bridge_node failed DDS discovery: {data}"
    assert data.get("node_rc") is None, f"px4_bridge_node exited prematurely: rc={data.get('node_rc')}"
    return data


def test_f08_bva_head_on_two_uav_collinear_symmetry_breaking(tmp_path: Path) -> None:
    """Verify CBF breaks exact collinear head-on symmetry without collision or deadlock."""
    body = """
    node.pub_state("drone_1", 19.4, 0.0, 8.0, vx=4.0)
    node.pub_state("drone_2", 20.6, 0.0, 8.0, vx=-4.0)
    node.spin_for(0.2)
    node.pub_strategy("TAKEOFF")
    node.spin_for(0.2)
    node.pub_strategy("INTERCEPT")

    for _ in range(15):
        node.pub_state("drone_1", 19.4, 0.0, 8.0, vx=4.0)
        node.pub_state("drone_2", 20.6, 0.0, 8.0, vx=-4.0)
        node.pub_plan("drone_1", 30.0, 0.0, 8.0, vx=4.0, action="INTERCEPT")
        node.pub_plan("drone_2", 10.0, 0.0, 8.0, vx=-4.0, action="INTERCEPT")
        node.spin_for(0.06)

    result["sp1"] = node.setpoints["drone_1"][-1] if node.setpoints["drone_1"] else None
    result["sp2"] = node.setpoints["drone_2"][-1] if node.setpoints["drone_2"] else None
    result["fleet"] = node.fleet_states[-1] if node.fleet_states else {{}}
    """
    data = _run_center_probe(
        tmp_path,
        domain_id=126,
        node_params=["-p", "swarm_ids:=['drone_1','drone_2']"],
        drones=["drone_1", "drone_2"],
        driver_body=body,
    )
    sp1, sp2 = data["sp1"], data["sp2"]
    assert sp1 is not None and sp2 is not None
    assert all(math.isfinite(v) for v in sp1 + sp2)
    dist = math.sqrt((sp1[0] - sp2[0]) ** 2 + (sp1[1] - sp2[1]) ** 2 + (sp1[2] - sp2[2]) ** 2)
    assert dist >= 1.0, f"Head-on collision separation violated: dist={dist:.3f}m"
    assert abs(sp1[1] - sp2[1]) > 0.05 or abs(sp1[2] - sp2[2]) > 0.05, (
        f"Expected lateral/vertical symmetry breaking on collinear head-on: sp1={sp1}, sp2={sp2}"
    )
    assert int(data["fleet"].get("cbf_active_count", 0)) >= 1


def test_f08_bva_trapped_inside_obstacle_nfz_junction_recovery(
    tmp_path: Path, ros2_ws_root: Path
) -> None:
    """Verify CBF recovers cleanly without NaN when UAV starts inside obstacle+NFZ junction."""
    cfg = load_obstacles_yaml(
        ros2_ws_root / "src" / "flight_ctrl_pkg" / "config" / "obstacles.yaml"
    )
    start_pt = (56.5, 15.5, 6.0)
    start_clr = point_min_clearance(start_pt, cfg, include_nfz=True)
    assert start_clr <= 0.0

    body = """
    node.pub_state("drone_1", 56.5, 15.5, 6.0)
    node.spin_for(0.2)
    node.pub_strategy("TAKEOFF")
    node.spin_for(0.2)
    node.pub_strategy("TRACK")

    for _ in range(15):
        node.pub_state("drone_1", 56.5, 15.5, 6.0)
        node.pub_plan("drone_1", 58.0, 15.5, 6.0, action="TRACK")
        node.spin_for(0.06)

    result["sp1"] = node.setpoints["drone_1"][-1] if node.setpoints["drone_1"] else None
    result["fleet"] = node.fleet_states[-1] if node.fleet_states else {{}}
    """
    data = _run_center_probe(
        tmp_path,
        domain_id=127,
        node_params=["-p", "swarm_ids:=['drone_1']"],
        drones=["drone_1"],
        driver_body=body,
    )
    sp1 = data["sp1"]
    assert sp1 is not None and all(math.isfinite(v) for v in sp1), f"Non-finite recovery: {sp1}"
    out_clr = point_min_clearance(sp1, cfg, include_nfz=True)
    assert out_clr > start_clr and out_clr > 0.0, (
        f"Expected CBF outward recovery from {start_clr:.3f}m, got {out_clr:.3f}m at {sp1}"
    )
    assert int(data["fleet"].get("cbf_active_count", 0)) >= 1


def test_f08_bva_px4_bridge_stale_and_nan_trajectory_ff_rejection(tmp_path: Path) -> None:
    """Verify px4_bridge_node rejects NaN trajectory_ff and >0.25s stale FF without NaN leak."""
    body = """
    node.pub_action("ATTACK")
    for _ in range(15):
        node.pub_px4_telemetry(n=0.0, e=0.0, d=-6.0)
        node.pub_sp_enu(0.0, 0.0, 6.0)
        node.hb_pub.publish(Empty())
        node.spin_for(0.04)

    # Fresh FF
    for _ in range(10):
        node.pub_px4_telemetry(n=0.0, e=0.0, d=-6.0)
        node.pub_sp_enu(0.0, 0.0, 6.0)
        node.pub_ff_enu(8.5, 0.0, 0.0)
        node.hb_pub.publish(Empty())
        node.spin_for(0.03)
    result["fresh"] = node.px4_setpoints[-1] if node.px4_setpoints else None

    # Inject NaN FF
    for _ in range(6):
        node.pub_px4_telemetry(n=0.0, e=0.0, d=-6.0)
        node.pub_sp_enu(0.0, 0.0, 6.0)
        node.pub_ff_enu(float("nan"), float("inf"), 0.0)
        node.hb_pub.publish(Empty())
        node.spin_for(0.03)
    result["after_nan"] = node.px4_setpoints[-1] if node.px4_setpoints else None

    # Stale >0.25s fallback to Shadow-FF
    t0 = time.monotonic()
    for _ in range(18):
        dt = time.monotonic() - t0
        node.pub_px4_telemetry(n=0.0, e=3.0 * dt, d=-6.0, ve=3.0)
        node.pub_sp_enu(3.0 * dt, 0.0, 6.0)
        node.hb_pub.publish(Empty())
        node.spin_for(0.03)
    result["after_stale"] = node.px4_setpoints[-1] if node.px4_setpoints else None
    """
    data = _run_px4_bridge_probe(
        tmp_path,
        domain_id=128,
        node_params=[
            "-p",
            "drone_id:=drone_1",
            "-p",
            "prestream_ticks:=3",
            "-p",
            "control_hz:=50.0",
        ],
        driver_body=body,
    )
    assert data["fresh"] is not None and data["fresh"]["vel"][1] > 6.0
    assert data["after_nan"] is not None
    assert all(math.isfinite(v) for v in data["after_nan"]["pos"] + data["after_nan"]["vel"]), (
        f"NaN leaked into TrajectorySetpoint: {data['after_nan']}"
    )
    assert data["after_stale"] is not None
    assert all(math.isfinite(v) for v in data["after_stale"]["vel"])
    assert data["after_stale"]["vel"][1] < 6.0


def test_f08_bva_zero_velocity_hover_and_geofence_altitude_clamps(tmp_path: Path) -> None:
    """Verify HOVER stationary hold / zero-velocity FF boundary, GROUND->ATTACK gate, and geofence clamps."""
    body = """
    node.pub_state("drone_1", 10.0, 0.0, 8.0)
    node.spin_for(0.2)
    # GROUND -> ATTACK must be rejected (no setpoint emitted)
    node.pub_strategy("ATTACK")
    node.spin_for(0.2)
    result["ground_sp_count"] = len(node.setpoints["drone_1"])

    node.pub_strategy("TAKEOFF")
    node.spin_for(0.2)
    ff_start = len(node.ffs["drone_1"])
    node.pub_strategy("HOVER")
    for _ in range(6):
        node.pub_state("drone_1", 10.0, 0.0, 8.0)
        node.spin_for(0.05)
    result["hover_sp"] = node.setpoints["drone_1"][-1] if node.setpoints["drone_1"] else None
    result["hover_ffs"] = node.ffs["drone_1"][ff_start:]

    # TRACK mode z=0.5m -> rejected or clamped >= 3.0m
    node.pub_strategy("TRACK")
    for _ in range(6):
        node.pub_state("drone_1", 10.0, 0.0, 8.0)
        node.pub_plan("drone_1", 12.0, 0.0, 4.0, action="TRACK")
        node.spin_for(0.05)
    for _ in range(6):
        node.pub_plan("drone_1", 12.0, 0.0, 0.5, action="TRACK")
        node.pub_plan("drone_1", 1200.0, 0.0, 8.0, action="TRACK")
        node.pub_plan("drone_1", float("nan"), 0.0, 8.0, action="TRACK")
        node.spin_for(0.05)
    result["track_sp"] = node.setpoints["drone_1"][-1] if node.setpoints["drone_1"] else None

    # ATTACK mode z=1.6m vs z=55.0m rejection/clamp
    node.pub_strategy("ATTACK")
    for _ in range(6):
        node.pub_state("drone_1", 10.0, 0.0, 8.0)
        node.pub_plan("drone_1", 14.0, 0.0, 1.8, action="ATTACK")
        node.spin_for(0.05)
    for _ in range(6):
        node.pub_plan("drone_1", 14.0, 0.0, 55.0, action="ATTACK")
        node.spin_for(0.05)
    result["attack_sp"] = node.setpoints["drone_1"][-1] if node.setpoints["drone_1"] else None
    """
    data = _run_center_probe(
        tmp_path,
        domain_id=129,
        node_params=["-p", "swarm_ids:=['drone_1']"],
        drones=["drone_1"],
        driver_body=body,
    )
    assert data["ground_sp_count"] == 0, "GROUND->ATTACK must be rejected"
    hover = data["hover_sp"]
    assert hover is not None and all(math.isfinite(v) for v in hover)
    assert hover[2] >= 3.0
    assert math.hypot(hover[0] - 10.0, hover[1] - 0.0) <= 0.1 and abs(hover[2] - 8.0) <= 0.1, (
        f"Expected HOVER stationary hold near (10.0, 0.0, 8.0), got {hover}"
    )
    for ff in data["hover_ffs"]:
        assert math.sqrt(ff[0] ** 2 + ff[1] ** 2 + ff[2] ** 2) <= 0.05, (
            f"Expected zero velocity FF in HOVER, got {ff}"
        )
    track = data["track_sp"]
    assert track is not None and track[2] >= 3.0 and abs(track[0]) <= 1000.0
    atk = data["attack_sp"]
    assert atk is not None and 1.5 <= atk[2] <= 50.0


def test_f08_bva_swarm_n8_cbf_cycle_latency_and_separation(tmp_path: Path) -> None:
    """Verify center_node 3D CBF scales to N=8 drones within real-time latency & >=1.0m separation."""
    drones = [f"drone_{i}" for i in range(1, 9)]
    body = """
    for i, d in enumerate(DRONES):
        ang = (2.0 * math.pi * i) / 8.0
        node.pub_state(d, 25.0 + 1.5 * math.cos(ang), 1.5 * math.sin(ang), 8.0)
    node.spin_for(0.25)
    node.pub_strategy("TAKEOFF")
    node.spin_for(0.2)
    node.pub_strategy("INTERCEPT")

    t0 = time.monotonic()
    for _ in range(20):
        for i, d in enumerate(DRONES):
            ang = (2.0 * math.pi * i) / 8.0
            node.pub_state(d, 25.0 + 1.5 * math.cos(ang), 1.5 * math.sin(ang), 8.0)
            node.pub_plan(d, 25.0, 0.0, 8.0, vx=2.0, action="INTERCEPT")
        node.spin_for(0.06)
    result["elapsed_s"] = time.monotonic() - t0
    result["counts"] = {d: len(node.setpoints[d]) for d in DRONES}
    result["last"] = {d: node.setpoints[d][-1] for d in DRONES if node.setpoints[d]}
    result["fleet"] = node.fleet_states[-1] if node.fleet_states else {}
    """
    data = _run_center_probe(
        tmp_path,
        domain_id=130,
        node_params=["-p", f"swarm_ids:={drones}"],
        drones=drones,
        driver_body=body,
    )
    assert all(data["counts"][d] >= 5 for d in drones), f"Starved drone setpoints: {data['counts']}"
    last = data["last"]
    assert len(last) == 8
    pts = list(last.values())
    for pt in pts:
        assert all(math.isfinite(v) for v in pt)
    min_pair = min(
        math.sqrt((pts[i][0] - pts[j][0]) ** 2 + (pts[i][1] - pts[j][1]) ** 2 + (pts[i][2] - pts[j][2]) ** 2)
        for i in range(len(pts))
        for j in range(i + 1, len(pts))
    )
    assert min_pair >= 1.0, f"N=8 CBF separation violated: min_pair={min_pair:.3f}m"
    fleet = data["fleet"]
    assert len(fleet.get("drones", {})) == 8, f"Expected 8 drones in /fleet/state: {fleet}"
    assert "min_inter_drone_dist_m" in fleet and "cbf_active_count" in fleet
