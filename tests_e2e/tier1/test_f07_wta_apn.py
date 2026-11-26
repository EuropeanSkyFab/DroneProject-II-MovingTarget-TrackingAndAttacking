"""Tier 1 functional opaque-box tests for F7: Dynamic Sub-Swarm WTA, Encirclement & 3D APN."""

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
    pytest.mark.tier1,
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
    script_path = tmp_path / f"f07_probe_{domain_id}.py"
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
                super().__init__("f07_opaque_probe_{domain_id}")
                self.plan_msgs = []
                self.diag_msgs = []
                self.create_subscription(String, "/planning/target_pose", self._on_plan, 50)
                self.create_subscription(String, "/planning/diagnostics", self._on_diag, 20)
                self.vision_pub = self.create_publisher(String, "/vision/targets", 10)
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
    """Map external_id (or track_id) -> track_id from /planning/diagnostics."""
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


def test_f07_planning_target_pose_contract5_json_schema(tmp_path: Path) -> None:
    """Verify /planning/target_pose emits complete Contract #5 JSON fields and valid IMM outputs."""
    body = """
    drones = ["drone_1", "drone_2", "drone_3"]
    for d in drones:
        node.pub_state(d, 0.0, 0.0, 8.5, vx=1.0)
    node.spin_for(0.25)
    node.pub_strategy({"drone_id": "ALL", "action": "TAKEOFF"})
    node.spin_for(0.2)
    node.pub_strategy({"drone_id": "ALL", "action": "ATTACK"})
    for step in range(8):
        for i, d in enumerate(drones):
            node.pub_state(d, 2.0, float(i * 3), 8.5, vx=2.0)
        node.pub_vision_targets([
            {
                "det_id": 0,
                "target_id": "tgt_alpha",
                "class_id": "crimson_uav",
                "x": 22.0 + 0.6 * step,
                "y": 1.5 + 0.3 * step,
                "z": 7.0,
                "conf": 0.95,
                "stereo": True,
                "rays_used": 3,
                "parallax_deg": 22.0,
                "skew_rms_m": 0.1,
            }
        ])
        node.spin_for(0.12)
    result["plan_msgs"] = node.plan_msgs[-18:]
    """
    data = _run_planning_probe(
        tmp_path,
        domain_id=101,
        node_params=["-p", "drone_ids:=['drone_1','drone_2','drone_3']", "-p", "plan_hz:=20.0"],
        driver_body=body,
    )
    msgs = data["plan_msgs"]
    assert msgs, "No /planning/target_pose messages received"

    required_ext = {
        "drone_id",
        "stamp",
        "x",
        "y",
        "z",
        "vx",
        "vy",
        "vz",
        "ax",
        "ay",
        "az",
        "action",
        "t_go",
        "target_id",
        "focus_x",
        "focus_y",
        "focus_z",
        "imm_probs",
        "pred_path",
    }
    matched_ext = False
    for msg in msgs:
        v = validate_planning_target_msg(msg)
        assert v.ok is True, f"Contract #5 validation failed: {v.errors}, msg={msg}"
        if msg.get("target_id"):
            missing = required_ext - set(msg.keys())
            assert not missing, f"Missing Contract #5 fields {missing} in {msg}"
            assert isinstance(msg["pred_path"], list) and len(msg["pred_path"]) >= 1
            matched_ext = True
    assert matched_ext, f"No confirmed target planning messages with IMM/focus fields: {msgs[-3:]}"


def test_f07_multi_target_wta_pack_splitting_6v3(tmp_path: Path) -> None:
    """Verify dynamic WTA splits N=6 drones across M=3 distinct targets into balanced packs."""
    body = """
    drones = [f"drone_{i}" for i in range(1, 7)]
    for i, d in enumerate(drones):
        node.pub_state(d, 0.0, -15.0 + 6.0 * i, 8.5, vx=1.0)
    node.spin_for(0.25)
    node.pub_strategy({"drone_id": "ALL", "action": "TAKEOFF"})
    node.spin_for(0.2)
    node.pub_strategy({"drone_id": "ALL", "action": "ATTACK"})

    fleet = {d: {"battery_pct": 95.0} for d in drones}
    for step in range(12):
        node.pub_fleet_state(fleet)
        for i, d in enumerate(drones):
            node.pub_state(d, 2.0, -15.0 + 6.0 * i, 8.5, vx=1.5)
        tgts = [
            {"det_id": 0, "target_id": "tgt_alpha", "class_id": "crimson_uav",
             "x": 30.0, "y": -12.0, "z": 7.5, "conf": 0.94, "stereo": True,
             "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
            {"det_id": 1, "target_id": "tgt_beta", "class_id": "amber_uav",
             "x": 30.0, "y": 0.0, "z": 7.5, "conf": 0.94, "stereo": True,
             "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
            {"det_id": 2, "target_id": "tgt_gamma", "class_id": "slate_uav",
             "x": 30.0, "y": 12.0, "z": 7.5, "conf": 0.94, "stereo": True,
             "rays_used": 3, "parallax_deg": 20.0, "skew_rms_m": 0.1},
        ]
        node.pub_vision_targets(tgts)
        node.pub_ext_targets(tgts)
        node.spin_for(0.12)
    result["diag_msgs"] = node.diag_msgs[-5:]
    result["plan_msgs"] = node.plan_msgs[-24:]
    """
    data = _run_planning_probe(
        tmp_path,
        domain_id=102,
        node_params=[
            "-p",
            "drone_ids:=['drone_1','drone_2','drone_3','drone_4','drone_5','drone_6']",
            "-p",
            "plan_hz:=20.0",
        ],
        driver_body=body,
    )
    assert data["diag_msgs"], "No /planning/diagnostics messages received"
    last_diag = data["diag_msgs"][-1]
    wta = last_diag.get("wta_assignments", {})
    assert len(wta) == 6, f"Expected all 6 drones in wta_assignments, got {wta}"
    assigned_targets = set(wta.values())
    assert len(assigned_targets) == 3, (
        f"Expected pack splitting across all 3 targets, got wta={wta}"
    )
    for msg in data["plan_msgs"]:
        v = validate_planning_target_msg(msg)
        assert v.ok is True, f"Invalid planning message: {v.errors}"


def test_f07_dynamic_kill_retasking_via_strategy_and_kill_event(tmp_path: Path) -> None:
    """Verify kill_event + strategy neutralized_targets immediately retasks drones to survivors."""
    body = """
    drones = [f"drone_{i}" for i in range(1, 7)]
    for i, d in enumerate(drones):
        node.pub_state(d, 0.0, -15.0 + 6.0 * i, 8.5, vx=1.0)
    node.spin_for(0.25)
    node.pub_strategy({"drone_id": "ALL", "action": "TAKEOFF"})
    node.spin_for(0.2)
    node.pub_strategy({"drone_id": "ALL", "action": "ATTACK"})

    all_tgts = [
        {"det_id": 0, "target_id": "tgt_alpha", "class_id": "crimson_uav",
         "x": 28.0, "y": -12.0, "z": 7.5, "conf": 0.95, "stereo": True,
         "rays_used": 3, "parallax_deg": 21.0, "skew_rms_m": 0.1},
        {"det_id": 1, "target_id": "tgt_beta", "class_id": "amber_uav",
         "x": 28.0, "y": 0.0, "z": 7.5, "conf": 0.95, "stereo": True,
         "rays_used": 3, "parallax_deg": 21.0, "skew_rms_m": 0.1},
        {"det_id": 2, "target_id": "tgt_gamma", "class_id": "slate_uav",
         "x": 28.0, "y": 12.0, "z": 7.5, "conf": 0.95, "stereo": True,
         "rays_used": 3, "parallax_deg": 21.0, "skew_rms_m": 0.1},
    ]
    for _ in range(10):
        for i, d in enumerate(drones):
            node.pub_state(d, 2.0, -15.0 + 6.0 * i, 8.5, vx=1.5)
        node.pub_vision_targets(all_tgts)
        node.pub_ext_targets(all_tgts)
        node.spin_for(0.12)

    result["diag_before"] = node.diag_msgs[-1] if node.diag_msgs else {{}}

    node.pub_kill_event({
        "stamp": time.time(),
        "target_id": "tgt_alpha",
        "striker_id": "drone_1",
        "miss_dist_m": 0.45,
        "closing_speed_ms": 12.0,
        "x": 28.0,
        "y": -12.0,
        "z": 7.5,
    })
    node.pub_strategy({"neutralized_targets": ["tgt_alpha"]})

    surviving = all_tgts[1:]
    for _ in range(10):
        for i, d in enumerate(drones):
            node.pub_state(d, 4.0, -15.0 + 6.0 * i, 8.5, vx=1.5)
        node.pub_vision_targets(surviving)
        node.pub_ext_targets(surviving)
        node.spin_for(0.12)

    result["diag_after"] = node.diag_msgs[-1] if node.diag_msgs else {{}}
    """
    data = _run_planning_probe(
        tmp_path,
        domain_id=103,
        node_params=[
            "-p",
            "drone_ids:=['drone_1','drone_2','drone_3','drone_4','drone_5','drone_6']",
            "-p",
            "plan_hz:=20.0",
        ],
        driver_body=body,
    )
    before = data["diag_before"]
    after = data["diag_after"]
    assert len(before.get("wta_assignments", {})) == 6
    assert len(set(before["wta_assignments"].values())) == 3

    before_map = _track_id_map(before)
    alpha_trk = before_map.get("tgt_alpha", "tgt_alpha")

    neutralized = set(after.get("neutralized_targets", []))
    assert "tgt_alpha" in neutralized or alpha_trk in neutralized, (
        f"Expected tgt_alpha in neutralized_targets, got {after.get('neutralized_targets')}"
    )
    assert after.get("re_tasking_events", 0) >= 1, (
        f"Expected re_tasking_events >= 1, got {after.get('re_tasking_events')}"
    )
    wta_after = after.get("wta_assignments", {})
    assert len(wta_after) == 6, f"Expected all 6 drones reassigned, got {wta_after}"
    assert alpha_trk not in set(wta_after.values()) and "tgt_alpha" not in set(wta_after.values()), (
        f"Killed target still assigned after kill event: {wta_after}"
    )
    assert len(set(wta_after.values())) == 2, f"Expected 2 surviving targets, got {wta_after}"


def test_f07_generalized_k_drone_encirclement_non_crossing(tmp_path: Path) -> None:
    """Verify K=6 drones in TRACK mode encircle 1 target with non-crossing ~60-deg flanks."""
    body = """
    drones = [f"drone_{i}" for i in range(1, 7)]
    cx, cy, cz = 25.0, 0.0, 6.0
    for i, d in enumerate(drones):
        ang = (2.0 * math.pi * i) / 6.0
        node.pub_state(d, cx + 6.5 * math.cos(ang), cy + 6.5 * math.sin(ang), 9.0)
    node.spin_for(0.25)
    node.pub_strategy({"drone_id": "ALL", "action": "TAKEOFF"})
    node.spin_for(0.2)
    node.pub_strategy({"drone_id": "ALL", "action": "TRACK"})

    tgt = [{"det_id": 0, "target_id": "tgt_orbit", "class_id": "crimson_uav",
            "x": cx, "y": cy, "z": cz, "conf": 0.96, "stereo": True,
            "rays_used": 4, "parallax_deg": 26.0, "skew_rms_m": 0.08}]
    for _ in range(10):
        for i, d in enumerate(drones):
            ang = (2.0 * math.pi * i) / 6.0
            node.pub_state(d, cx + 6.5 * math.cos(ang), cy + 6.5 * math.sin(ang), 9.0)
        node.pub_vision_targets(tgt)
        node.spin_for(0.12)

    latest = {}
    for m in node.plan_msgs:
        if m.get("drone_id") in drones and m.get("target_id"):
            latest[m["drone_id"]] = m
    result["latest"] = latest
    """
    data = _run_planning_probe(
        tmp_path,
        domain_id=104,
        node_params=[
            "-p",
            "drone_ids:=['drone_1','drone_2','drone_3','drone_4','drone_5','drone_6']",
            "-p",
            "plan_hz:=20.0",
        ],
        driver_body=body,
    )
    latest = data["latest"]
    assert len(latest) == 6, f"Expected 6 encirclement commands, got {list(latest.keys())}"

    angles: List[float] = []
    for d_id, msg in sorted(latest.items()):
        v = validate_planning_target_msg(msg)
        assert v.ok is True, f"{d_id} invalid msg: {v.errors}"
        assert msg["action"] == "TRACK"
        assert msg["z"] >= 3.2
        r = math.hypot(msg["x"] - 25.0, msg["y"] - 0.0)
        assert 4.2 <= r <= 8.0, f"{d_id} orbit radius {r:.2f}m out of expected encirclement band"
        angles.append(math.atan2(msg["y"] - 0.0, msg["x"] - 25.0))

    angles.sort()
    diffs = [angles[i + 1] - angles[i] for i in range(len(angles) - 1)]
    diffs.append((angles[0] + 2.0 * math.pi) - angles[-1])
    min_sep_deg = math.degrees(min(diffs))
    assert min_sep_deg >= 40.0, (
        f"Expected >= 40 deg angular separation for 6-drone encirclement, got {min_sep_deg:.1f} deg"
    )


def test_f07_3d_apn_terminal_guidance_lead_and_feedforward(tmp_path: Path) -> None:
    """Verify 3D APN emits finite velocity/accel feedforward with lateral lead on jinking target."""
    body = """
    node.pub_state("drone_1", 0.0, 0.0, 8.0, vx=6.0)
    node.spin_for(0.25)
    node.pub_strategy({"drone_id": "ALL", "action": "TAKEOFF"})
    node.spin_for(0.2)
    node.pub_strategy({"drone_id": "ALL", "action": "ATTACK"})

    t0 = time.monotonic()
    for _ in range(12):
        dt = time.monotonic() - t0
        node.pub_state("drone_1", 6.0 * dt, 0.0, 8.0, vx=6.0)
        tgt = [{
            "det_id": 0,
            "target_id": "tgt_jink",
            "class_id": "crimson_uav",
            "x": 24.0,
            "y": 2.0 + 4.5 * dt,
            "z": 7.0 + 0.8 * dt,
            "conf": 0.96,
            "stereo": True,
            "rays_used": 3,
            "parallax_deg": 24.0,
            "skew_rms_m": 0.1,
        }]
        node.pub_vision_targets(tgt)
        node.spin_for(0.10)

    apn_msgs = [m for m in node.plan_msgs if m.get("drone_id") == "drone_1" and m.get("target_id")]
    result["apn_msgs"] = apn_msgs[-6:]
    """
    data = _run_planning_probe(
        tmp_path,
        domain_id=105,
        node_params=["-p", "drone_ids:=['drone_1']", "-p", "plan_hz:=20.0"],
        driver_body=body,
    )
    apn_msgs = data["apn_msgs"]
    assert apn_msgs, "No APN target_pose messages emitted for drone_1"
    last = apn_msgs[-1]
    v = validate_planning_target_msg(last)
    assert v.ok is True, f"APN msg failed Contract #5: {v.errors}"
    assert last["action"] == "ATTACK"
    assert 0.0 < float(last["t_go"]) < 5.0, f"Unexpected t_go={last['t_go']}"
    for k in ("vx", "vy", "vz", "ax", "ay", "az"):
        assert math.isfinite(float(last[k])), f"Non-finite feedforward {k}={last[k]}"
    assert float(last["vy"]) > 0.5 or float(last["ay"]) > 0.1, (
        f"Expected positive lateral APN lead, got vy={last.get('vy')}, ay={last.get('ay')}"
    )


def test_f07_obstacle_aware_3d_corridor_detour_guidance(tmp_path: Path) -> None:
    """Verify planning_node detours around a blocking 3D obstacle loaded via obstacles_yaml_path."""
    obs_yaml = tmp_path / "detour_obs.yaml"
    obs_yaml.write_text(
        textwrap.dedent(
            """\
            arena_bounds:
              x_min: -10.0
              x_max: 60.0
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
              - id: "obs_wall"
                type: "box"
                cx: 20.0
                cy: 0.0
                z_min: 0.0
                z_max: 25.0
                sx: 3.0
                sy: 16.0
                safety_margin_m: 1.5
                visual_style: "test_wall"
            no_fly_zones: []
            """
        ),
        encoding="utf-8",
    )
    cfg = load_obstacles_yaml(obs_yaml)

    body = """
    node.pub_state("drone_1", 5.0, 0.0, 8.0, vx=2.0)
    node.spin_for(0.25)
    node.pub_strategy({"drone_id": "ALL", "action": "TAKEOFF"})
    node.spin_for(0.2)
    node.pub_strategy({"drone_id": "ALL", "action": "INTERCEPT"})

    tgt = [{
        "det_id": 0,
        "target_id": "tgt_behind_wall",
        "class_id": "crimson_uav",
        "x": 35.0,
        "y": 0.0,
        "z": 8.0,
        "conf": 0.95,
        "stereo": True,
        "rays_used": 3,
        "parallax_deg": 20.0,
        "skew_rms_m": 0.1,
    }]
    for _ in range(10):
        node.pub_state("drone_1", 5.0, 0.0, 8.0, vx=2.0)
        node.pub_vision_targets(tgt)
        node.spin_for(0.12)

    cmds = [m for m in node.plan_msgs if m.get("drone_id") == "drone_1" and m.get("target_id")]
    result["cmds"] = cmds[-5:]
    """
    data = _run_planning_probe(
        tmp_path,
        domain_id=106,
        node_params=[
            "-p",
            "drone_ids:=['drone_1']",
            "-p",
            "plan_hz:=20.0",
            "-p",
            f"obstacles_yaml_path:={obs_yaml}",
        ],
        driver_body=body,
    )
    cmds = data["cmds"]
    assert cmds, "No detour command emitted for drone_1"
    last = cmds[-1]
    v = validate_planning_target_msg(last)
    assert v.ok is True, f"Detour command invalid: {v.errors}"
    pt = (float(last["x"]), float(last["y"]), float(last["z"]))
    clr = point_min_clearance(pt, cfg, include_nfz=True)
    assert clr > 0.0, f"Commanded waypoint {pt} penetrates obstacle (clr={clr:.3f}m)"
    assert abs(pt[1]) > 2.0 or pt[2] > 25.0, (
        f"Expected lateral or vertical detour around wall, got {pt}"
    )
    assert check_line_of_sight((5.0, 0.0, 8.0), pt, cfg) is True, (
        f"Segment from drone (5,0,8) to waypoint {pt} intersects obstacle"
    )
