"""Tier 2 BVA & corner-case opaque-box tests for F7: WTA, Encirclement & 3D APN."""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import pytest

from helpers.arena_parser import check_line_of_sight, load_obstacles_yaml, point_min_clearance
from helpers.metrics_validator import validate_planning_target_msg
from helpers.ros_probe import run_command_with_gpu_guard

pytestmark = [
    pytest.mark.tier2,
    pytest.mark.feature(id="F7"),
    pytest.mark.requires_m3,
]


def _extract_probe_json(stdout: str) -> Dict[str, Any]:
    begin_marker = "__E2E_JSON_BEGIN__"
    end_marker = "__E2E_JSON_END__"
    assert begin_marker in stdout and end_marker in stdout, (
        f"Probe driver did not emit JSON envelope. stdout={stdout!r}"
    )
    raw = stdout.split(begin_marker, 1)[1].split(end_marker, 1)[0].strip()
    return json.loads(raw)


def _run_planning_probe(
    tmp_path: Path,
    domain_id: int,
    node_params: List[str],
    driver_body: str,
    timeout_s: float = 28.0,
) -> Dict[str, Any]:
    script_path = tmp_path / f"f07_bva_probe_{domain_id}.py"
    params_repr = repr(node_params)
    full_script = textwrap.dedent(
        f"""\
        import json
        import math
        import os
        import signal
        import subprocess
        import time

        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from std_msgs.msg import String

        NODE_ARGS = ["ros2", "run", "planning_pkg", "planning_node", "--ros-args"] + {params_repr}


        class ProbeDriver(Node):
            def __init__(self):
                super().__init__("f07_bva_probe_{domain_id}")
                self.plan_msgs = []
                self.diag_msgs = []
                self.create_subscription(String, "/planning/target_pose", self._on_plan, 50)
                self.create_subscription(String, "/planning/diagnostics", self._on_diag, 20)
                self.vision_pub = self.create_publisher(String, "/vision/targets", 10)
                self.legacy_vis_pub = self.create_publisher(PoseStamped, "/vision/target_pose", 10)
                self.ext_pub = self.create_publisher(String, "/external/targets_detected", 10)
                self.strat_pub = self.create_publisher(String, "/strategy/cmd", 10)
                self.kill_pub = self.create_publisher(String, "/engagement/kill_event", 10)
                self.fleet_pub = self.create_publisher(String, "/fleet/state", 10)
                self.state_pubs = {{}}

            def _on_plan(self, msg):
                try:
                    self.plan_msgs.append(json.loads(msg.data))
                except Exception:
                    self.plan_msgs.append({{"_raw": msg.data}})

            def _on_diag(self, msg):
                try:
                    self.diag_msgs.append(json.loads(msg.data))
                except Exception:
                    self.diag_msgs.append({{"_raw": msg.data}})

            def pub_state(self, drone_id, x, y, z, vx=0.0, vy=0.0, vz=0.0):
                if drone_id not in self.state_pubs:
                    self.state_pubs[drone_id] = self.create_publisher(
                        Odometry, f"/{{drone_id}}/state", 10
                    )
                msg = Odometry()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "world"
                msg.pose.pose.position.x = float(x)
                msg.pose.pose.position.y = float(y)
                msg.pose.pose.position.z = float(z)
                msg.pose.pose.orientation.w = 1.0
                msg.twist.twist.linear.x = float(vx)
                msg.twist.twist.linear.y = float(vy)
                msg.twist.twist.linear.z = float(vz)
                self.state_pubs[drone_id].publish(msg)

            def pub_strategy(self, payload):
                m = String()
                m.data = json.dumps(payload)
                self.strat_pub.publish(m)

            def pub_vision_targets(self, targets, stamp=None):
                ts = float(time.time() if stamp is None else stamp)
                m = String()
                m.data = json.dumps({{"stamp": ts, "frame_id": "world", "targets": targets}})
                self.vision_pub.publish(m)

            def pub_vision_raw(self, raw_text):
                m = String()
                m.data = raw_text
                self.vision_pub.publish(m)

            def pub_legacy_vision_pose(self, x, y, z):
                p = PoseStamped()
                p.header.stamp = self.get_clock().now().to_msg()
                p.header.frame_id = "world"
                p.pose.position.x = float(x)
                p.pose.position.y = float(y)
                p.pose.position.z = float(z)
                self.legacy_vis_pub.publish(p)

            def pub_ext_targets(self, targets, stamp=None):
                ts = float(time.time() if stamp is None else stamp)
                m = String()
                m.data = json.dumps({{"stamp": ts, "frame_id": "world", "targets": targets}})
                self.ext_pub.publish(m)

            def pub_kill_event(self, payload):
                m = String()
                m.data = json.dumps(payload)
                self.kill_pub.publish(m)

            def pub_fleet_state(self, drones_dict):
                m = String()
                m.data = json.dumps({{"stamp": time.time(), "drones": drones_dict}})
                self.fleet_pub.publish(m)

            def spin_for(self, seconds):
                end = time.monotonic() + seconds
                while time.monotonic() < end:
                    rclpy.spin_once(self, timeout_sec=0.03)


        def wait_ready(node, timeout_s=8.0):
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                if (
                    node.count_subscribers("/strategy/cmd") >= 1
                    and node.count_publishers("/planning/target_pose") >= 1
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
            node = ProbeDriver()
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
    assert res.timed_out is False, f"Probe timed out: stderr={res.stderr}"
    assert res.returncode == 0, f"Probe failed rc={res.returncode}: stderr={res.stderr}"
    data = _extract_probe_json(res.stdout)
    assert data.get("ready") is True, f"planning_node failed DDS discovery: {data}"
    assert data.get("node_rc") is None, f"planning_node exited prematurely: rc={data.get('node_rc')}"
    return data


def _track_id_map(diag: Dict[str, Any]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for trk in diag.get("tracks", []):
        if isinstance(trk, dict):
            tid = str(trk.get("track_id", ""))
            ext = str(trk.get("external_id", ""))
            if tid:
                mapping[tid] = tid
                if ext:
                    mapping[ext] = tid
    return mapping


def test_f07_bva_all_targets_neutralized_return_transition(tmp_path: Path) -> None:
    """Verify neutralizing all targets clears WTA and cleanly transitions to RETURN."""
    body = """
    drones = ["drone_1", "drone_2", "drone_3"]
    for i, d in enumerate(drones):
        node.pub_state(d, 10.0, float(i * 3), 8.5)
    node.spin_for(0.25)
    node.pub_strategy({"drone_id": "ALL", "action": "TAKEOFF"})
    node.spin_for(0.2)
    node.pub_strategy({"drone_id": "ALL", "action": "ATTACK"})

    tgts = [
        {"det_id": 0, "target_id": "tgt_one", "class_id": "crimson_uav",
         "x": 26.0, "y": -5.0, "z": 7.5, "conf": 0.95, "stereo": True,
         "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
        {"det_id": 1, "target_id": "tgt_two", "class_id": "amber_uav",
         "x": 26.0, "y": 5.0, "z": 7.5, "conf": 0.95, "stereo": True,
         "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
    ]
    for _ in range(8):
        for i, d in enumerate(drones):
            node.pub_state(d, 10.0, float(i * 3), 8.5)
        node.pub_vision_targets(tgts)
        node.spin_for(0.12)

    node.pub_kill_event({
        "stamp": time.time(), "target_id": "tgt_one", "striker_id": "drone_1",
        "miss_dist_m": 0.3, "closing_speed_ms": 12.0, "x": 26.0, "y": -5.0, "z": 7.5,
    })
    node.pub_kill_event({
        "stamp": time.time(), "target_id": "tgt_two", "striker_id": "drone_2",
        "miss_dist_m": 0.4, "closing_speed_ms": 11.5, "x": 26.0, "y": 5.0, "z": 7.5,
    })
    node.pub_strategy({"drone_id": "ALL", "action": "RETURN", "neutralized_targets": ["tgt_one", "tgt_two"]})
    for _ in range(8):
        for i, d in enumerate(drones):
            node.pub_state(d, 10.0, float(i * 3), 8.5)
        node.spin_for(0.12)

    result["diag_after"] = node.diag_msgs[-1] if node.diag_msgs else {{}}
    result["plan_after"] = node.plan_msgs[-9:]
    """
    data = _run_planning_probe(
        tmp_path,
        domain_id=107,
        node_params=["-p", "drone_ids:=['drone_1','drone_2','drone_3']", "-p", "plan_hz:=20.0"],
        driver_body=body,
    )
    diag = data["diag_after"]
    assert diag.get("wta_assignments", None) == {}, (
        f"Expected empty WTA when all targets neutralized, got {diag.get('wta_assignments')}"
    )
    assert set(diag.get("neutralized_targets", [])) >= {"tgt_one", "tgt_two"}, (
        f"Expected both targets neutralized, got {diag.get('neutralized_targets')}"
    )
    assert data["plan_after"], "No RETURN planning outputs received"
    for msg in data["plan_after"]:
        v = validate_planning_target_msg(msg)
        assert v.ok is True, f"Invalid RETURN msg: {v.errors}"
        assert msg["action"] == "RETURN", f"Expected RETURN transition, got {msg}"
        assert msg.get("target_id", "") == ""


def test_f07_bva_starvation_n1_vs_m3_and_n2_vs_m4_priority_and_soc(tmp_path: Path) -> None:
    """Verify under-provisioned WTA boundaries (N=1 vs M=3 and N=2 vs M=4 with low battery SoC)."""
    # Probe A: N=1 vs M=3 starvation picks fastest/closest incoming threat
    body_a = """
    node.pub_state("drone_1", 0.0, 0.0, 8.5, vx=1.0)
    node.spin_for(0.25)
    node.pub_strategy({"drone_id": "ALL", "action": "TAKEOFF"})
    node.spin_for(0.2)
    node.pub_strategy({"drone_id": "ALL", "action": "ATTACK"})
    for step in range(10):
        node.pub_state("drone_1", 0.0, 0.0, 8.5, vx=1.0)
        tgts = [
            {"det_id": 0, "target_id": "tgt_near_fast", "class_id": "crimson_uav",
             "x": 14.0 - 0.4 * step, "y": 0.0, "z": 7.5, "conf": 0.95, "stereo": True,
             "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
            {"det_id": 1, "target_id": "tgt_far_1", "class_id": "amber_uav",
             "x": 68.0, "y": -15.0, "z": 7.5, "conf": 0.95, "stereo": True,
             "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
            {"det_id": 2, "target_id": "tgt_far_2", "class_id": "slate_uav",
             "x": 76.0, "y": 15.0, "z": 7.5, "conf": 0.95, "stereo": True,
             "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
        ]
        node.pub_vision_targets(tgts)
        node.pub_ext_targets(tgts)
        node.spin_for(0.12)
    result["diag"] = node.diag_msgs[-1] if node.diag_msgs else {{}}
    """
    data_a = _run_planning_probe(
        tmp_path,
        domain_id=108,
        node_params=["-p", "drone_ids:=['drone_1']", "-p", "plan_hz:=20.0"],
        driver_body=body_a,
    )
    diag_a = data_a["diag"]
    map_a = _track_id_map(diag_a)
    wta_a = diag_a.get("wta_assignments", {})
    assert len(wta_a) == 1, f"Expected 1 assignment for N=1 vs M=3, got {wta_a}"
    assert wta_a.get("drone_1") == map_a.get("tgt_near_fast"), (
        f"Expected drone_1 to prioritize tgt_near_fast, got wta={wta_a}, map={map_a}"
    )

    # Probe B: N=2 vs M=4 with co-located drones at battery SoC boundary (5% vs 100%)
    body_b = """
    for d in ("drone_1", "drone_2"):
        node.pub_state(d, 0.0, 0.0, 8.5, vx=1.0)
    node.spin_for(0.25)
    node.pub_strategy({"drone_id": "ALL", "action": "TAKEOFF"})
    node.spin_for(0.2)
    node.pub_strategy({"drone_id": "ALL", "action": "ATTACK"})
    for step in range(10):
        node.pub_fleet_state({
            "drone_1": {"battery_pct": 5.0},
            "drone_2": {"battery_pct": 100.0},
        })
        for d in ("drone_1", "drone_2"):
            node.pub_state(d, 0.0, 0.0, 8.5, vx=1.0)
        tgts = [
            {"det_id": 0, "target_id": "tgt_close", "class_id": "crimson_uav",
             "x": 12.0 - 0.3 * step, "y": -2.0, "z": 7.5, "conf": 0.95, "stereo": True,
             "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
            {"det_id": 1, "target_id": "tgt_mid", "class_id": "amber_uav",
             "x": 28.0 - 0.3 * step, "y": 2.0, "z": 7.5, "conf": 0.95, "stereo": True,
             "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
            {"det_id": 2, "target_id": "tgt_far_a", "class_id": "slate_uav",
             "x": 68.0, "y": -18.0, "z": 7.5, "conf": 0.95, "stereo": True,
             "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
            {"det_id": 3, "target_id": "tgt_far_b", "class_id": "slate_uav",
             "x": 74.0, "y": 18.0, "z": 7.5, "conf": 0.95, "stereo": True,
             "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
        ]
        node.pub_vision_targets(tgts)
        node.pub_ext_targets(tgts)
        node.spin_for(0.12)
    result["diag"] = node.diag_msgs[-1] if node.diag_msgs else {{}}
    result["plan"] = node.plan_msgs[-8:]
    """
    data_b = _run_planning_probe(
        tmp_path,
        domain_id=109,
        node_params=["-p", "drone_ids:=['drone_1','drone_2']", "-p", "plan_hz:=20.0"],
        driver_body=body_b,
    )
    diag_b = data_b["diag"]
    map_b = _track_id_map(diag_b)
    wta_b = diag_b.get("wta_assignments", {})
    assert len(wta_b) == 2, f"Expected 2 assignments for N=2 vs M=4, got {wta_b}"
    near_pair = {map_b.get("tgt_close"), map_b.get("tgt_mid")}
    assert set(wta_b.values()) == near_pair, (
        f"Expected N=2 to pick 2 high-threat near targets {near_pair}, got {wta_b}"
    )
    assert wta_b["drone_1"] == map_b.get("tgt_close"), (
        f"Expected low-SoC drone_1 assigned to closest target, got {wta_b}"
    )
    for m in data_b["plan"]:
        assert validate_planning_target_msg(m).ok is True


def test_f07_bva_narrow_corridor_pinch_waypoint_detour(tmp_path: Path) -> None:
    """Verify corridor planner threads narrow pinch gap between symmetric towers safely."""
    pinch_yaml = tmp_path / "pinch_obs.yaml"
    pinch_yaml.write_text(
        textwrap.dedent(
            """\
            arena_bounds:
              x_min: -10.0
              x_max: 50.0
              y_min: -25.0
              y_max: 25.0
              z_min: 0.0
              z_max: 40.0
            helipads:
              - id: "H1"
                drone_id: "drone_1"
                x: 0.0
                y: 0.0
                z: 0.0
                yaw: 0.0
            obstacles:
              - id: "pinch_north"
                type: "box"
                cx: 20.0
                cy: 5.2
                z_min: 0.0
                z_max: 25.0
                sx: 3.0
                sy: 6.0
                safety_margin_m: 0.8
                visual_style: "tower_n"
              - id: "pinch_south"
                type: "box"
                cx: 20.0
                cy: -5.2
                z_min: 0.0
                z_max: 25.0
                sx: 3.0
                sy: 6.0
                safety_margin_m: 0.8
                visual_style: "tower_s"
            no_fly_zones: []
            """
        ),
        encoding="utf-8",
    )
    cfg = load_obstacles_yaml(pinch_yaml)

    body = """
    node.pub_state("drone_1", 6.0, 0.0, 8.0, vx=2.0)
    node.spin_for(0.25)
    node.pub_strategy({"drone_id": "ALL", "action": "TAKEOFF"})
    node.spin_for(0.2)
    node.pub_strategy({"drone_id": "ALL", "action": "INTERCEPT"})
    tgt = [{
        "det_id": 0, "target_id": "tgt_corridor", "class_id": "crimson_uav",
        "x": 34.0, "y": 0.0, "z": 8.0, "conf": 0.95, "stereo": True,
        "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1,
    }]
    for _ in range(10):
        node.pub_state("drone_1", 6.0, 0.0, 8.0, vx=2.0)
        node.pub_vision_targets(tgt)
        node.spin_for(0.12)
    result["cmds"] = [m for m in node.plan_msgs if m.get("drone_id") == "drone_1" and m.get("target_id")][-5:]
    """
    data = _run_planning_probe(
        tmp_path,
        domain_id=110,
        node_params=[
            "-p",
            "drone_ids:=['drone_1']",
            "-p",
            "plan_hz:=20.0",
            "-p",
            f"obstacles_yaml_path:={pinch_yaml}",
        ],
        driver_body=body,
    )
    assert data["cmds"], "No pinch corridor command emitted"
    last = data["cmds"][-1]
    assert validate_planning_target_msg(last).ok is True
    pt = (float(last["x"]), float(last["y"]), float(last["z"]))
    assert point_min_clearance(pt, cfg, include_nfz=True) > 0.0, f"Pinch waypoint inside tower: {pt}"
    assert check_line_of_sight((6.0, 0.0, 8.0), pt, cfg) is True, (
        f"Segment to pinch waypoint {pt} hits tower"
    )


def test_f07_bva_simultaneous_multi_target_neutralization_retasking(tmp_path: Path) -> None:
    """Verify neutralizing 2 targets simultaneously retasks all 6 drones onto the single survivor."""
    body = """
    drones = [f"drone_{i}" for i in range(1, 7)]
    for i, d in enumerate(drones):
        node.pub_state(d, 0.0, -15.0 + 6.0 * i, 8.5, vx=1.0)
    node.spin_for(0.25)
    node.pub_strategy({"drone_id": "ALL", "action": "TAKEOFF"})
    node.spin_for(0.2)
    node.pub_strategy({"drone_id": "ALL", "action": "ATTACK"})

    tgts = [
        {"det_id": 0, "target_id": "tgt_alpha", "class_id": "crimson_uav",
         "x": 28.0, "y": -12.0, "z": 7.5, "conf": 0.95, "stereo": True,
         "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
        {"det_id": 1, "target_id": "tgt_beta", "class_id": "amber_uav",
         "x": 28.0, "y": 0.0, "z": 7.5, "conf": 0.95, "stereo": True,
         "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
        {"det_id": 2, "target_id": "tgt_gamma", "class_id": "slate_uav",
         "x": 28.0, "y": 12.0, "z": 7.5, "conf": 0.95, "stereo": True,
         "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
    ]
    for _ in range(10):
        for i, d in enumerate(drones):
            node.pub_state(d, 2.0, -15.0 + 6.0 * i, 8.5, vx=1.5)
        node.pub_vision_targets(tgts)
        node.pub_ext_targets(tgts)
        node.spin_for(0.12)

    result["diag_before"] = node.diag_msgs[-1] if node.diag_msgs else {{}}
    node.pub_strategy({"neutralized_targets": ["tgt_alpha", "tgt_beta"]})
    survivor = [tgts[2]]
    for _ in range(10):
        for i, d in enumerate(drones):
            node.pub_state(d, 4.0, -15.0 + 6.0 * i, 8.5, vx=1.5)
        node.pub_vision_targets(survivor)
        node.pub_ext_targets(survivor)
        node.spin_for(0.12)
    result["diag_after"] = node.diag_msgs[-1] if node.diag_msgs else {{}}
    """
    data = _run_planning_probe(
        tmp_path,
        domain_id=111,
        node_params=[
            "-p",
            "drone_ids:=['drone_1','drone_2','drone_3','drone_4','drone_5','drone_6']",
            "-p",
            "plan_hz:=20.0",
        ],
        driver_body=body,
    )
    after = data["diag_after"]
    map_b = _track_id_map(data["diag_before"])
    gamma_trk = map_b.get("tgt_gamma", "tgt_gamma")
    assert set(after.get("neutralized_targets", [])) >= {"tgt_alpha", "tgt_beta"}
    assert after.get("re_tasking_events", 0) >= 1
    wta = after.get("wta_assignments", {})
    assert len(wta) == 6, f"Expected all 6 drones converged on survivor, got {wta}"
    assert set(wta.values()) == {gamma_trk}, f"Expected all drones assigned to {gamma_trk}, got {wta}"


def test_f07_bva_evasive_high_g_reversal_during_terminal_apn(tmp_path: Path) -> None:
    """Verify terminal 3D APN adapts smoothly without NaN/divergence under abrupt high-g reversal."""
    body = """
    node.pub_state("drone_1", 10.0, 0.0, 8.0, vx=8.0)
    node.spin_for(0.25)
    node.pub_strategy({"drone_id": "ALL", "action": "TAKEOFF"})
    node.spin_for(0.2)
    node.pub_strategy({"drone_id": "ALL", "action": "ATTACK"})

    y = 0.0
    z = 7.0
    for step in range(6):
        y += 0.72  # +6 m/s lateral
        node.pub_state("drone_1", 10.0 + 0.8 * step, 0.0, 8.0, vx=8.0)
        node.pub_vision_targets([{
            "det_id": 0, "target_id": "tgt_evader", "class_id": "crimson_uav",
            "x": 26.0, "y": y, "z": z, "conf": 0.95, "stereo": True,
            "rays_used": 3, "parallax_deg": 24.0, "skew_rms_m": 0.1,
        }])
        node.spin_for(0.12)
    pre_msgs = [m for m in node.plan_msgs if m.get("drone_id") == "drone_1" and m.get("target_id")]

    for step in range(8):
        y -= 0.96  # abrupt reversal -8 m/s + climb
        z += 0.35
        node.pub_state("drone_1", 14.8 + 0.8 * step, 0.0, 8.0, vx=8.0)
        node.pub_vision_targets([{
            "det_id": 0, "target_id": "tgt_evader", "class_id": "crimson_uav",
            "x": 26.0, "y": y, "z": z, "conf": 0.95, "stereo": True,
            "rays_used": 3, "parallax_deg": 24.0, "skew_rms_m": 0.1,
        }])
        node.spin_for(0.12)
    post_msgs = [m for m in node.plan_msgs if m.get("drone_id") == "drone_1" and m.get("target_id")]
    result["pre"] = pre_msgs[-1] if pre_msgs else None
    result["post"] = post_msgs[-3:]
    """
    data = _run_planning_probe(
        tmp_path,
        domain_id=112,
        node_params=["-p", "drone_ids:=['drone_1']", "-p", "plan_hz:=20.0"],
        driver_body=body,
    )
    assert data["pre"] is not None and data["post"], "Missing pre/post reversal APN outputs"
    for m in [data["pre"]] + data["post"]:
        assert validate_planning_target_msg(m).ok is True
    last = data["post"][-1]
    assert float(last["vy"]) < float(data["pre"]["vy"]) or float(last["ay"]) < 0.0, (
        f"Expected APN reversal response: pre={data['pre']}, post={last}"
    )


def test_f07_bva_nan_oob_stale_and_coast_expiry_resilience(tmp_path: Path) -> None:
    """Verify coasting up to max_coast_sec, expiry after, and rejection of NaN/OOB/stale inputs."""
    body = """
    node.pub_state("drone_1", 0.0, 0.0, 8.5, vx=1.0)
    node.spin_for(0.25)
    node.pub_strategy({"drone_id": "ALL", "action": "TAKEOFF"})
    node.spin_for(0.2)
    node.pub_strategy({"drone_id": "ALL", "action": "TRACK"})

    # 1. Confirm a clean track, then verify coasting at +0.55s and expiry at +1.45s
    for _ in range(6):
        node.pub_state("drone_1", 0.0, 0.0, 8.5, vx=1.0)
        node.pub_vision_targets([{
            "det_id": 0, "target_id": "tgt_coast", "class_id": "crimson_uav",
            "x": 20.0, "y": 0.0, "z": 7.5, "conf": 0.95, "stereo": True,
            "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1,
        }])
        node.spin_for(0.10)

    node.spin_for(0.50)
    coast_msgs = [m for m in node.plan_msgs if m.get("drone_id") == "drone_1"]
    result["during_coast"] = coast_msgs[-1] if coast_msgs else None

    node.spin_for(0.95)
    exp_msgs = [m for m in node.plan_msgs if m.get("drone_id") == "drone_1"]
    result["after_expiry"] = exp_msgs[-1] if exp_msgs else None

    # 2. Inject OOB, stale, malformed JSON, and NaN inputs
    now_s = time.time()
    node.pub_vision_raw("{{not valid json")
    node.pub_vision_targets(
        [{"det_id": 0, "target_id": "tgt_stale", "class_id": "crimson_uav",
          "x": 20.0, "y": 0.0, "z": 7.5, "conf": 0.9, "stereo": True,
          "rays_used": 2, "parallax_deg": 10.0, "skew_rms_m": 0.1}],
        stamp=now_s - 2.5,
    )
    for _ in range(5):
        node.pub_vision_targets([
            {"det_id": 1, "target_id": "tgt_oob_far", "class_id": "crimson_uav",
             "x": 1e6, "y": 0.0, "z": 8.0, "conf": 0.9, "stereo": True,
             "rays_used": 2, "parallax_deg": 10.0, "skew_rms_m": 0.1},
            {"det_id": 2, "target_id": "tgt_oob_low", "class_id": "crimson_uav",
             "x": 20.0, "y": 0.0, "z": -2.0, "conf": 0.9, "stereo": True,
             "rays_used": 2, "parallax_deg": 10.0, "skew_rms_m": 0.1},
            {"det_id": 3, "target_id": "tgt_oob_high", "class_id": "crimson_uav",
             "x": 20.0, "y": 0.0, "z": 60.0, "conf": 0.9, "stereo": True,
             "rays_used": 2, "parallax_deg": 10.0, "skew_rms_m": 0.1},
        ])
        node.pub_legacy_vision_pose(float("nan"), 0.0, 8.0)
        node.spin_for(0.10)
    node.spin_for(0.65)

    result["diag_after_bad"] = node.diag_msgs[-1] if node.diag_msgs else {{}}
    result["bad_msgs"] = node.plan_msgs[-10:]
    """
    data = _run_planning_probe(
        tmp_path,
        domain_id=113,
        node_params=[
            "-p",
            "drone_ids:=['drone_1']",
            "-p",
            "plan_hz:=20.0",
            "-p",
            "max_coast_sec:=1.0",
        ],
        driver_body=body,
    )
    assert data["during_coast"] is not None
    assert data["during_coast"].get("target_id", "") != "", (
        f"Track dropped before max_coast_sec=1.0: {data['during_coast']}"
    )
    assert data["after_expiry"] is not None
    assert data["after_expiry"].get("target_id", "") == "", (
        f"Track did not expire after max_coast_sec=1.0: {data['after_expiry']}"
    )
    assert "tracks" in data["diag_after_bad"], (
        f"Missing /planning/diagnostics payload after bad inputs: {data['diag_after_bad']}"
    )
    assert data["diag_after_bad"].get("tracks", []) == [], (
        f"Expected 0 tracks from OOB/stale/NaN inputs, got {data['diag_after_bad']}"
    )
    for msg in data["bad_msgs"]:
        v = validate_planning_target_msg(msg)
        assert v.ok is True, f"Invalid planning msg under bad input stress: {v.errors}"
        assert math.isfinite(msg["x"]) and abs(msg["x"]) <= 1000.0
