"""Tier 1 functional opaque-box tests for F8: 3D CBF Safety Shield & Feedforward Control Bridge."""

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
    pytest.mark.tier1,
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
    timeout_s: float = 28.0,
) -> Dict[str, Any]:
    script_path = tmp_path / f"f08_center_probe_{domain_id}.py"
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


        class CenterProbe(Node):
            def __init__(self):
                super().__init__("f08_center_probe_{domain_id}")
                self.setpoints = {{d: [] for d in DRONES}}
                self.ffs = {{d: [] for d in DRONES}}
                self.fleet_states = []
                self.state_pubs = {{}}
                self.plan_pub = self.create_publisher(String, "/planning/target_pose", 20)
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
                self.setpoints[did].append((float(p.x), float(p.y), float(p.z), msg.header.frame_id))

            def _on_ff(self, did, msg):
                v = msg.twist.linear
                self.ffs[did].append((time.monotonic(), float(v.x), float(v.y), float(v.z), msg.header.frame_id))

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
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "vx": float(vx),
                    "vy": float(vy),
                    "vz": float(vz),
                    "ax": 0.0,
                    "ay": 0.0,
                    "az": 0.0,
                    "action": action,
                    "t_go": 1.5,
                    "target_id": "tgt_1",
                    "focus_x": float(x),
                    "focus_y": float(y),
                    "focus_z": float(z),
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
            node = CenterProbe()
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
    assert res.timed_out is False, f"Center probe timed out: stderr={res.stderr}"
    assert res.returncode == 0, f"Center probe failed rc={res.returncode}: stderr={res.stderr}"
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
    script_path = tmp_path / f"f08_bridge_probe_{domain_id}.py"
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


        class BridgeProbe(Node):
            def __init__(self):
                super().__init__("f08_bridge_probe_{domain_id}")
                px4_qos = QoSProfile(
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                    history=HistoryPolicy.KEEP_LAST,
                    depth=1,
                )
                self.px4_setpoints = []
                self.drone_states = []
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
                self.create_subscription(Odometry, "/drone_1/state", self._on_state, 10)

            def _on_px4_sp(self, msg):
                self.px4_setpoints.append({{
                    "t": time.monotonic(),
                    "pos": [float(v) for v in msg.position],
                    "vel": [float(v) for v in msg.velocity],
                    "acc": [float(v) for v in msg.acceleration],
                    "yaw": float(msg.yaw),
                }})

            def _on_vcmd(self, msg):
                self.vehicle_cmds.append(int(msg.command))

            def _on_state(self, msg):
                p = msg.pose.pose.position
                v = msg.twist.twist.linear
                self.drone_states.append({{
                    "x": float(p.x), "y": float(p.y), "z": float(p.z),
                    "vx": float(v.x), "vy": float(v.y), "vz": float(v.z),
                }})

            def pub_px4_telemetry(self, n=0.0, e=0.0, d=-5.0, vn=0.0, ve=0.0, vd=0.0):
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
            node = BridgeProbe()
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
    assert res.timed_out is False, f"Bridge probe timed out: stderr={res.stderr}"
    assert res.returncode == 0, f"Bridge probe failed rc={res.returncode}: stderr={res.stderr}"
    data = _extract_probe_json(res.stdout)
    assert data.get("ready") is True, f"px4_bridge_node failed DDS discovery: {data}"
    assert data.get("node_rc") is None, f"px4_bridge_node exited prematurely: rc={data.get('node_rc')}"
    return data


def test_f08_cbf_3d_inter_uav_separation_and_fleet_state(tmp_path: Path) -> None:
    """Verify 3D CBF enforces inter-UAV separation >= 1.0m and emits CBF /fleet/state telemetry."""
    body = """
    node.pub_state("drone_1", 10.0, 0.0, 8.0)
    node.pub_state("drone_2", 10.8, 0.0, 8.0)
    node.spin_for(0.2)
    node.pub_strategy("TAKEOFF")
    node.spin_for(0.2)
    node.pub_strategy("TRACK")

    for _ in range(15):
        node.pub_state("drone_1", 10.0, 0.0, 8.0)
        node.pub_state("drone_2", 10.8, 0.0, 8.0)
        node.pub_plan("drone_1", 10.4, 0.0, 8.0, action="TRACK")
        node.pub_plan("drone_2", 10.4, 0.0, 8.0, action="TRACK")
        node.spin_for(0.06)

    result["sp1"] = node.setpoints["drone_1"][-1] if node.setpoints["drone_1"] else None
    result["sp2"] = node.setpoints["drone_2"][-1] if node.setpoints["drone_2"] else None
    result["fleet"] = node.fleet_states[-1] if node.fleet_states else {{}}
    """
    data = _run_center_probe(
        tmp_path,
        domain_id=121,
        node_params=["-p", "swarm_ids:=['drone_1','drone_2']"],
        drones=["drone_1", "drone_2"],
        driver_body=body,
    )
    sp1, sp2 = data["sp1"], data["sp2"]
    assert sp1 is not None and sp2 is not None, "Missing trajectory_setpoint outputs"
    dist = math.sqrt((sp1[0] - sp2[0]) ** 2 + (sp1[1] - sp2[1]) ** 2 + (sp1[2] - sp2[2]) ** 2)
    assert dist >= 1.0, f"CBF failed inter-UAV separation: dist={dist:.3f}m, sp1={sp1}, sp2={sp2}"

    fleet = data["fleet"]
    for key in ("min_inter_drone_dist_m", "min_obstacle_clearance_m", "cbf_active_count"):
        assert key in fleet, f"Missing Contract #6 key '{key}' in /fleet/state: {fleet}"
    assert int(fleet["cbf_active_count"]) >= 1, (
        f"Expected cbf_active_count >= 1 on converging pair, got {fleet}"
    )


def test_f08_cbf_3d_obstacle_and_nfz_clearance(
    tmp_path: Path, ros2_ws_root: Path
) -> None:
    """Verify 3D CBF shields setpoints clear of obstacles.yaml obstacles and no-fly zones."""
    cfg = load_obstacles_yaml(
        ros2_ws_root / "src" / "flight_ctrl_pkg" / "config" / "obstacles.yaml"
    )
    body = """
    # Place drone_1 near obs_comm_pylon_center (49, 1.5, r=1.4) and drone_2 near nfz_command_core (58, 15.5)
    node.pub_state("drone_1", 45.0, 1.5, 8.0)
    node.pub_state("drone_2", 50.0, 15.5, 8.0)
    node.spin_for(0.2)
    node.pub_strategy("TAKEOFF")
    node.spin_for(0.2)
    node.pub_strategy("INTERCEPT")

    for _ in range(15):
        node.pub_state("drone_1", 45.0, 1.5, 8.0)
        node.pub_state("drone_2", 50.0, 15.5, 8.0)
        node.pub_plan("drone_1", 49.0, 1.5, 8.0, vx=4.0, action="INTERCEPT")
        node.pub_plan("drone_2", 58.0, 15.5, 8.0, vx=4.0, action="INTERCEPT")
        node.spin_for(0.06)

    result["sp1"] = node.setpoints["drone_1"][-1] if node.setpoints["drone_1"] else None
    result["sp2"] = node.setpoints["drone_2"][-1] if node.setpoints["drone_2"] else None
    result["fleet"] = node.fleet_states[-1] if node.fleet_states else {{}}
    """
    data = _run_center_probe(
        tmp_path,
        domain_id=122,
        node_params=["-p", "swarm_ids:=['drone_1','drone_2']"],
        drones=["drone_1", "drone_2"],
        driver_body=body,
    )
    for idx, key in enumerate(("sp1", "sp2"), start=1):
        sp = data[key]
        assert sp is not None, f"Missing setpoint for drone_{idx}"
        clr = point_min_clearance((sp[0], sp[1], sp[2]), cfg, include_nfz=True)
        assert clr > 0.0, f"drone_{idx} setpoint {sp[:3]} entered obstacle/NFZ (clr={clr:.3f}m)"

    fleet = data["fleet"]
    assert "min_obstacle_clearance_m" in fleet and "cbf_active_count" in fleet, (
        f"Missing CBF obstacle metrics in /fleet/state: {fleet}"
    )
    assert int(fleet["cbf_active_count"]) >= 1, f"Expected CBF active count >= 1, got {fleet}"


def test_f08_center_publishes_trajectory_ff_and_bridge_timeout_fallback(
    tmp_path: Path,
) -> None:
    """Verify center_node emits /{d}/fmu/in/trajectory_ff and px4_bridge_node falls back <0.25s."""
    center_body = """
    node.pub_state("drone_1", 0.0, 0.0, 8.5)
    node.pub_state("drone_2", 0.0, 6.0, 8.5)
    node.spin_for(0.2)
    node.pub_strategy("TAKEOFF")
    node.spin_for(0.2)
    node.pub_strategy("INTERCEPT")

    for _ in range(18):
        node.pub_state("drone_1", 0.0, 0.0, 8.5)
        node.pub_state("drone_2", 0.0, 6.0, 8.5)
        node.pub_plan("drone_1", 12.0, -2.0, 9.0, vx=6.0, vy=-2.0, vz=0.5, action="INTERCEPT")
        node.pub_plan("drone_2", 12.0, 4.0, 9.0, vx=5.0, vy=-1.5, vz=0.3, action="INTERCEPT")
        node.spin_for(0.06)

    result["ff_d1"] = node.ffs["drone_1"]
    result["ff_d2"] = node.ffs["drone_2"]
    """
    cdata = _run_center_probe(
        tmp_path,
        domain_id=123,
        node_params=["-p", "swarm_ids:=['drone_1','drone_2']"],
        drones=["drone_1", "drone_2"],
        driver_body=center_body,
    )
    assert len(cdata["ff_d1"]) >= 5 and len(cdata["ff_d2"]) >= 5, (
        f"center_node did not publish /{{d}}/fmu/in/trajectory_ff: d1={len(cdata['ff_d1'])}"
    )
    last_ff = cdata["ff_d1"][-1]
    assert last_ff[4] == "world"
    assert all(math.isfinite(v) for v in last_ff[1:4])
    assert math.hypot(last_ff[1], last_ff[2]) > 0.5, f"Expected non-zero velocity FF: {last_ff}"

    bridge_body = """
    node.pub_action("ATTACK")
    # Enter AIRBORNE via prestream
    for _ in range(15):
        node.pub_px4_telemetry(n=0.0, e=0.0, d=-6.0)
        node.pub_sp_enu(0.0, 0.0, 6.0)
        node.hb_pub.publish(Empty())
        node.spin_for(0.04)

    # Phase 1: static position setpoint + fresh trajectory_ff (vx_enu=7.5, vy_enu=2.5, vz_enu=0.8)
    for _ in range(12):
        node.pub_px4_telemetry(n=0.0, e=0.0, d=-6.0)
        node.pub_sp_enu(0.0, 0.0, 6.0)
        node.pub_ff_enu(7.5, 2.5, 0.8)
        node.hb_pub.publish(Empty())
        node.spin_for(0.03)
    result["fresh_sp"] = node.px4_setpoints[-1] if node.px4_setpoints else None

    # Phase 2: stop trajectory_ff for >0.25s while moving position setpoint east at 3.0 m/s
    t_start = time.monotonic()
    for step in range(20):
        dt = time.monotonic() - t_start
        node.pub_px4_telemetry(n=0.0, e=3.0 * dt, d=-6.0, vn=0.0, ve=3.0, vd=0.0)
        node.pub_sp_enu(3.0 * dt, 0.0, 6.0)
        node.hb_pub.publish(Empty())
        node.spin_for(0.03)
    result["stale_sp"] = node.px4_setpoints[-1] if node.px4_setpoints else None
    """
    bdata = _run_px4_bridge_probe(
        tmp_path,
        domain_id=124,
        node_params=[
            "-p",
            "drone_id:=drone_1",
            "-p",
            "prestream_ticks:=3",
            "-p",
            "control_hz:=50.0",
        ],
        driver_body=bridge_body,
    )
    fresh = bdata["fresh_sp"]
    stale = bdata["stale_sp"]
    assert fresh is not None and stale is not None, "px4_bridge_node emitted no TrajectorySetpoint"
    # ENU (vx=7.5, vy=2.5, vz=0.8) -> NED (vn=2.5, ve=7.5, vd=-0.8)
    assert fresh["vel"][0] > 1.5 and fresh["vel"][1] > 5.0 and fresh["vel"][2] < -0.3, (
        f"px4_bridge_node ignored fresh trajectory_ff: {fresh}"
    )
    assert all(math.isfinite(v) for v in stale["vel"]), f"NaN in Shadow-FF fallback: {stale}"
    assert stale["vel"][1] < fresh["vel"][1] - 1.5, (
        f"Expected fallback from 7.5 m/s FF after >0.25s timeout, got {stale}"
    )


def test_f08_px4_bridge_enu_ned_roundtrip_consistency(tmp_path: Path) -> None:
    """Verify ENU<->NED coordinate transform consistency across px4_bridge_node topics."""
    body = """
    node.pub_action("INTERCEPT")
    for _ in range(15):
        node.pub_px4_telemetry(n=12.0, e=-5.0, d=-8.5, vn=1.5, ve=-2.0, vd=0.3)
        node.pub_sp_enu(-5.0, 12.0, 8.5)
        node.hb_pub.publish(Empty())
        node.spin_for(0.04)

    for _ in range(12):
        node.pub_px4_telemetry(n=12.0, e=-5.0, d=-8.5, vn=1.5, ve=-2.0, vd=0.3)
        node.pub_sp_enu(-5.0, 12.0, 8.5)
        node.pub_ff_enu(4.0, -1.0, 0.5)
        node.hb_pub.publish(Empty())
        node.spin_for(0.03)

    result["state"] = node.drone_states[-1] if node.drone_states else None
    result["px4_sp"] = node.px4_setpoints[-1] if node.px4_setpoints else None
    """
    data = _run_px4_bridge_probe(
        tmp_path,
        domain_id=125,
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
    st = data["state"]
    px4_sp = data["px4_sp"]
    assert st is not None and px4_sp is not None
    # NED (n=12, e=-5, d=-8.5, vn=1.5, ve=-2, vd=0.3) -> ENU (x=-5, y=12, z=8.5, vx=-2, vy=1.5, vz=-0.3)
    assert math.isclose(st["x"], -5.0, abs_tol=0.15)
    assert math.isclose(st["y"], 12.0, abs_tol=0.15)
    assert math.isclose(st["z"], 8.5, abs_tol=0.15)
    assert math.isclose(st["vx"], -2.0, abs_tol=0.15)
    assert math.isclose(st["vy"], 1.5, abs_tol=0.15)
    assert math.isclose(st["vz"], -0.3, abs_tol=0.15)
    # ENU sp (-5.0, 12.0, 8.5) -> NED pos (12.0, -5.0, -8.5)
    assert math.isclose(px4_sp["pos"][0], 12.0, abs_tol=0.15)
    assert math.isclose(px4_sp["pos"][1], -5.0, abs_tol=0.15)
    assert math.isclose(px4_sp["pos"][2], -8.5, abs_tol=0.15)
    # ENU ff (vx=4.0, vy=-1.0, vz=0.5) -> NED vel (vn=-1.0, ve=4.0, vd=-0.5)
    assert px4_sp["vel"][0] < -0.3 and px4_sp["vel"][1] > 2.5 and px4_sp["vel"][2] < -0.2, (
        f"Unexpected ENU->NED velocity setpoint: {px4_sp}"
    )
    assert math.isfinite(px4_sp["yaw"])


def test_f08_center_node_pkg_ament_lint_clean(ros2_ws_root: Path) -> None:
    """Verify center_node_pkg passes ament_copyright, ament_flake8, and ament_pep257."""
    pkg_dir = ros2_ws_root / "src" / "center_node_pkg"
    assert pkg_dir.is_dir(), f"Missing package directory: {pkg_dir}"

    for tool_cmd in (
        ["ament_copyright", ".", "test"],
        ["ament_flake8", "."],
        ["ament_pep257", ".", "test"],
    ):
        res = run_command_with_gpu_guard(
            tool_cmd,
            cwd=pkg_dir,
            timeout_s=30.0,
            source_env=True,
        )
        assert res.timed_out is False, f"{tool_cmd[0]} timed out"
        assert res.returncode == 0, (
            f"{tool_cmd[0]} failed (rc={res.returncode}):\n"
            f"stdout={res.stdout}\nstderr={res.stderr}"
        )
