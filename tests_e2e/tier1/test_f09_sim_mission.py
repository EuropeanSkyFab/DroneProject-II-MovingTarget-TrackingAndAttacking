"""Tier 1 functional opaque-box tests for F9: Multi-Target Sim & Event-Driven Strike/BDA."""

from __future__ import annotations

import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import pytest

from helpers.arena_parser import (
    load_obstacles_yaml,
    parse_sdf_collisions,
    point_min_clearance,
)
from helpers.metrics_validator import (
    validate_kill_event_msg,
    validate_mission_summary,
    validate_mission_trace,
)
from helpers.ros_probe import run_command_with_gpu_guard

pytestmark = [
    pytest.mark.tier1,
    pytest.mark.feature(id="F9"),
]


def _extract_probe_json(stdout: str) -> Dict[str, Any]:
    begin_marker = "__E2E_JSON_BEGIN__"
    end_marker = "__E2E_JSON_END__"
    assert begin_marker in stdout and end_marker in stdout, (
        f"Probe driver did not emit JSON envelope. stdout={stdout!r}"
    )
    raw = stdout.split(begin_marker, 1)[1].split(end_marker, 1)[0].strip()
    return json.loads(raw)


def _run_f9_probe(
    tmp_path: Path,
    domain_id: int,
    node_cmds: List[List[str]],
    driver_body: str,
    timeout_s: float = 35.0,
) -> Dict[str, Any]:
    script_path = tmp_path / f"f09_probe_{domain_id}.py"
    full_script = textwrap.dedent(
        f"""\
        import json
        import math
        import os
        import signal
        import subprocess
        import time

        import rclpy
        from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
        from std_msgs.msg import String

        NODE_CMDS = {repr(node_cmds)}


        class F9Probe(Node):
            def __init__(self):
                super().__init__("f09_probe_{domain_id}")
                self.targets_gt = []
                self.radar_gt = []
                self.kill_events = []
                self.strategy_cmds = []
                self.legacy_gt_count = 0
                self.state_pubs = {{}}
                self.fc_pubs = {{}}

                self.create_subscription(String, "/sim/targets_gt", self._on_targets_gt, 30)
                self.create_subscription(String, "/external/targets_detected", self._on_radar, 20)
                self.create_subscription(String, "/engagement/kill_event", self._on_kill, 20)
                self.create_subscription(String, "/strategy/cmd", self._on_strat, 30)
                self._legacy_sub = None

                self.vision_pub = self.create_publisher(String, "/vision/targets", 10)
                self.vis_diag_pub = self.create_publisher(String, "/vision/diagnostics", 10)
                self.plan_diag_pub = self.create_publisher(String, "/planning/diagnostics", 10)
                self.fleet_pub = self.create_publisher(String, "/fleet/state", 10)

            def _on_targets_gt(self, msg):
                try:
                    self.targets_gt.append((time.monotonic(), json.loads(msg.data)))
                except Exception:
                    pass

            def _on_radar(self, msg):
                try:
                    self.radar_gt.append((time.monotonic(), json.loads(msg.data)))
                except Exception:
                    pass

            def _on_kill(self, msg):
                try:
                    self.kill_events.append(json.loads(msg.data))
                except Exception:
                    self.kill_events.append({{"_raw": msg.data}})

            def _on_strat(self, msg):
                try:
                    self.strategy_cmds.append(json.loads(msg.data))
                except Exception:
                    pass

            def _on_legacy_odom(self, _msg):
                self.legacy_gt_count += 1

            def _on_legacy_pose(self, _msg):
                self.legacy_gt_count += 1

            def pub_drone_state(self, did, x, y, z, vx=0.0, vy=0.0, vz=0.0):
                if did not in self.state_pubs:
                    self.state_pubs[did] = self.create_publisher(Odometry, f"/{{did}}/state", 10)
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

            def pub_fc_status(self, did):
                if did not in self.fc_pubs:
                    self.fc_pubs[did] = self.create_publisher(
                        DiagnosticStatus, f"/{{did}}/fmu/out/fc_status", 10
                    )
                st = DiagnosticStatus()
                st.level = DiagnosticStatus.OK
                st.name = "PX4 Nominal"
                st.hardware_id = did
                st.values = [KeyValue(key="ekf_aligned", value="True")]
                self.fc_pubs[did].publish(st)

            def pub_vision_targets(self, targets):
                m = String()
                m.data = json.dumps({{"stamp": time.time(), "frame_id": "world", "targets": targets}})
                self.vision_pub.publish(m)

            def spin_for(self, seconds):
                end = time.monotonic() + seconds
                while time.monotonic() < end:
                    rclpy.spin_once(self, timeout_sec=0.02)


        def wait_topics(node, timeout_s=5.0):
            has_sim = any("target_sim_node" in " ".join(cmd) for cmd in NODE_CMDS)
            has_cmd = any("mission_commander_node" in " ".join(cmd) for cmd in NODE_CMDS)
            deadline = time.monotonic() + timeout_s
            t_map = {{}}
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
                t_map = {{name: list(types) for name, types in node.get_topic_names_and_types()}}
                sim_ok = (not has_sim) or (
                    "/sim/targets_gt" in t_map and "/sim/target_gt" in t_map
                )
                cmd_ok = (not has_cmd) or (
                    "/strategy/cmd" in t_map
                    and node.count_subscribers("/vision/targets") >= 1
                )
                if sim_ok and cmd_ok:
                    return t_map
            return t_map


        def main():
            procs = [
                subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    start_new_session=True,
                )
                for cmd in NODE_CMDS
            ]
            rclpy.init()
            node = F9Probe()
            result = {{"node_rcs": [], "topics": []}}
            try:
                t_map = wait_topics(node, timeout_s=5.0)
                legacy_types = t_map.get("/sim/target_gt", [])
                if "nav_msgs/msg/Odometry" in legacy_types:
                    node._legacy_sub = node.create_subscription(
                        Odometry, "/sim/target_gt", node._on_legacy_odom, 10
                    )
                elif "geometry_msgs/msg/PoseStamped" in legacy_types:
                    node._legacy_sub = node.create_subscription(
                        PoseStamped, "/sim/target_gt", node._on_legacy_pose, 10
                    )
                result["topics"] = [[name, types] for name, types in t_map.items()]
                __DRIVER_BODY__
                result["node_rcs"] = [p.poll() for p in procs]
            finally:
                node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
                for p in procs:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGINT)
                        p.wait(timeout=3.5)
                    except Exception:
                        try:
                            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                        except Exception:
                            pass
            print("__E2E_JSON_BEGIN__")
            print(json.dumps(result))
            print("__E2E_JSON_END__")


        if __name__ == "__main__":
            main()
        """
    ).replace(
        "        __DRIVER_BODY__",
        textwrap.indent(textwrap.dedent(driver_body).strip(), "        "),
    )
    compile(full_script, str(script_path), "exec")
    script_path.write_text(full_script, encoding="utf-8")
    res = run_command_with_gpu_guard(
        ["python3", str(script_path)],
        timeout_s=timeout_s,
        extra_env={"ROS_DOMAIN_ID": str(domain_id), "ROS_LOCALHOST_ONLY": "1"},
        source_env=True,
    )
    assert res.timed_out is False, f"F9 probe timed out: stderr={res.stderr}"
    assert res.returncode == 0, f"F9 probe failed rc={res.returncode}: stderr={res.stderr}"
    data = _extract_probe_json(res.stdout)
    assert all(rc is None for rc in data["node_rcs"]), (
        f"ROS node exited unexpectedly: {data['node_rcs']}"
    )
    return data


@pytest.mark.requires_m4
def test_f09_target_sim_publishes_multi_target_gt_radar_and_legacy_compat(
    tmp_path: Path,
) -> None:
    """Verify /sim/targets_gt (~25Hz), /external/targets_detected (~5Hz) JSON & legacy /sim/target_gt."""
    body = """
    node.spin_for(2.2)
    result["topics"] = [[name, list(types)] for name, types in node.get_topic_names_and_types()]
    result["gt_count"] = len(node.targets_gt)
    result["radar_count"] = len(node.radar_gt)
    result["last_gt"] = node.targets_gt[-1][1] if node.targets_gt else None
    result["last_radar"] = node.radar_gt[-1][1] if node.radar_gt else None
    result["legacy_gt_count"] = node.legacy_gt_count
    """
    data = _run_f9_probe(
        tmp_path,
        domain_id=131,
        node_cmds=[
            ["ros2", "run", "flight_ctrl_pkg", "target_sim_node", "--ros-args", "-p", "sim_backend:=kinematic"]
        ],
        driver_body=body,
    )
    topics = {name: types for name, types in data["topics"]}
    assert "/sim/targets_gt" in topics, f"Missing /sim/targets_gt topic: {topics}"
    assert "/external/targets_detected" in topics, f"Missing /external/targets_detected: {topics}"
    assert "/sim/target_gt" in topics, f"Missing legacy /sim/target_gt: {topics}"
    assert any(t in topics["/sim/target_gt"] for t in ("nav_msgs/msg/Odometry", "geometry_msgs/msg/PoseStamped"))

    assert data["gt_count"] >= 20, f"Expected ~25Hz /sim/targets_gt, got {data['gt_count']} msgs"
    assert data["radar_count"] >= 4, f"Expected ~5Hz /external/targets_detected, got {data['radar_count']} msgs"
    assert data["legacy_gt_count"] >= 10

    gt = data["last_gt"]
    assert isinstance(gt, dict) and float(gt.get("stamp", 0.0)) > 0.0
    tgts = gt.get("targets", [])
    assert len(tgts) >= 3, f"Expected >= 3 targets in /sim/targets_gt, got {gt}"
    ids = set()
    for t in tgts:
        assert t.get("target_id") and t.get("class_id")
        ids.add(t["target_id"])
        assert t.get("status") == "ACTIVE"
        for k in ("x", "y", "z", "vx", "vy", "vz"):
            assert math.isfinite(float(t[k]))
    assert len(ids) >= 3

    radar = data["last_radar"]
    assert isinstance(radar, dict) and len(radar.get("targets", [])) >= 3


@pytest.mark.requires_m4
def test_f09_target_sim_evasive_trajectories_zero_obstacle_penetration(
    tmp_path: Path, ros2_ws_root: Path
) -> None:
    """Verify multi-target evasive trajectories stay inside bounds and clear of obstacles."""
    cfg = load_obstacles_yaml(
        ros2_ws_root / "src" / "flight_ctrl_pkg" / "config" / "obstacles.yaml"
    )
    body = """
    node.spin_for(3.0)
    result["samples"] = [payload for _, payload in node.targets_gt]
    """
    data = _run_f9_probe(
        tmp_path,
        domain_id=132,
        node_cmds=[
            ["ros2", "run", "flight_ctrl_pkg", "target_sim_node", "--ros-args", "-p", "sim_backend:=kinematic"]
        ],
        driver_body=body,
    )
    samples = data["samples"]
    assert len(samples) >= 20, "Missing /sim/targets_gt samples"
    bounds = cfg["arena_bounds"]
    for frame in samples:
        for t in frame.get("targets", []):
            x, y, z = float(t["x"]), float(t["y"]), float(t["z"])
            assert bounds["x_min"] <= x <= bounds["x_max"]
            assert bounds["y_min"] <= y <= bounds["y_max"]
            assert bounds["z_min"] <= z <= bounds["z_max"]
            clr = point_min_clearance((x, y, z), cfg, include_nfz=False)
            assert clr > 0.0, f"Target {t.get('target_id')} penetrated obstacle: {(x, y, z)}, clr={clr:.3f}m"


@pytest.mark.requires_m4
def test_f09_kill_event_schema_and_target_neutralization_radar_exclusion(
    tmp_path: Path,
) -> None:
    """Verify hit neutralization updates /sim/targets_gt status, emits kill_event, and excludes radar."""
    body = """
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not node.targets_gt:
        node.spin_for(0.05)
    assert node.targets_gt, "No initial /sim/targets_gt received"
    tgt0 = node.targets_gt[-1][1]["targets"][0]
    t0_id = tgt0["target_id"]

    # Close drone_1 onto target_0 to trigger proximity/strike hit
    for _ in range(25):
        if node.targets_gt:
            cur = [t for t in node.targets_gt[-1][1]["targets"] if t["target_id"] == t0_id][0]
            node.pub_drone_state("drone_1", cur["x"], cur["y"], cur["z"], vx=12.0)
        node.spin_for(0.05)

    node.spin_for(0.8)
    result["killed_id"] = t0_id
    result["last_gt"] = node.targets_gt[-1][1]
    result["last_radar"] = node.radar_gt[-1][1] if node.radar_gt else {{}}
    result["kill_events"] = node.kill_events
    """
    data = _run_f9_probe(
        tmp_path,
        domain_id=133,
        node_cmds=[
            ["ros2", "run", "flight_ctrl_pkg", "target_sim_node", "--ros-args", "-p", "sim_backend:=kinematic"]
        ],
        driver_body=body,
    )
    killed_id = data["killed_id"]
    gt_by_id = {t["target_id"]: t for t in data["last_gt"]["targets"]}
    assert gt_by_id[killed_id]["status"] == "NEUTRALIZED", (
        f"Expected {killed_id} NEUTRALIZED after close strike, got {gt_by_id}"
    )
    active_others = [t for tid, t in gt_by_id.items() if tid != killed_id and t["status"] == "ACTIVE"]
    assert len(active_others) >= 2, f"Expected surviving targets to remain ACTIVE, got {gt_by_id}"

    radar_ids = {t["target_id"] for t in data["last_radar"].get("targets", [])}
    assert killed_id not in radar_ids, (
        f"Neutralized target {killed_id} still emitted on /external/targets_detected: {radar_ids}"
    )
    assert data["kill_events"], "Expected /engagement/kill_event on target neutralization"
    for kev in data["kill_events"]:
        v = validate_kill_event_msg(kev)
        assert v.ok is True, f"Invalid /engagement/kill_event: {v.errors}, kev={kev}"


@pytest.mark.requires_m4
def test_f09_mission_commander_event_driven_fsm_and_retask(
    tmp_path: Path, ros2_ws_root: Path
) -> None:
    """Verify mission_commander_node executes event-driven KILL/BDA->RETASK->RETURN->COMPLETE."""
    cfg = load_obstacles_yaml(
        ros2_ws_root / "src" / "flight_ctrl_pkg" / "config" / "obstacles.yaml"
    )
    summary_path = tmp_path / "f09_summary.json"
    trace_path = tmp_path / "f09_trace.jsonl"
    drones = [f"drone_{i}" for i in range(1, 7)]
    helipads = {hp["drone_id"]: (hp["x"], hp["y"], hp["z"]) for hp in cfg["helipads"]}

    body = f"""
    drones = {repr(drones)}
    helipads = {repr(helipads)}
    for d in drones:
        hx, hy, hz = helipads[d]
        node.pub_fc_status(d)
        node.pub_drone_state(d, hx, hy, 0.05)
    node.spin_for(0.5)

    # Takeoff & climb
    for _ in range(10):
        for d in drones:
            hx, hy, _ = helipads[d]
            node.pub_fc_status(d)
            node.pub_drone_state(d, hx, hy, 8.5, vx=1.0)
        if node.targets_gt:
            vis = [
                {{"det_id": idx, "target_id": t["target_id"], "class_id": t["class_id"],
                  "x": t["x"], "y": t["y"], "z": t["z"], "conf": 0.95, "stereo": True,
                  "rays_used": 3, "parallax_deg": 22.0, "skew_rms_m": 0.1}}
                for idx, t in enumerate(node.targets_gt[-1][1]["targets"])
                if t.get("status") == "ACTIVE"
            ]
            if vis:
                node.pub_vision_targets(vis)
        node.spin_for(0.15)

    # Sequentially strike active targets with >= 0.65s pacing between kills
    seen_targets = False
    last_kill_t = time.monotonic() - 0.65
    last_active_count = None
    strike_deadline = time.monotonic() + 12.0
    while time.monotonic() < strike_deadline:
        for d in drones:
            node.pub_fc_status(d)
        if not node.targets_gt:
            node.spin_for(0.10)
            continue
        active = [t for t in node.targets_gt[-1][1]["targets"] if t.get("status") == "ACTIVE"]
        if active:
            seen_targets = True
        if seen_targets and not active:
            break
        if last_active_count is not None and len(active) < last_active_count:
            last_kill_t = time.monotonic()
        last_active_count = len(active)

        if active and (time.monotonic() - last_kill_t) >= 0.65:
            t0 = active[0]
            node.pub_drone_state("drone_1", t0["x"], t0["y"], t0["z"], vx=10.0)
        else:
            node.pub_drone_state("drone_1", 10.0, -10.0, 10.0, vx=2.0)
        for idx, d in enumerate(drones[1:], start=1):
            node.pub_drone_state(d, 20.0, -10.0 + 4.0 * idx, 10.0, vx=2.0)
        vis = [
            {{"det_id": idx, "target_id": t["target_id"], "class_id": t["class_id"],
              "x": t["x"], "y": t["y"], "z": t["z"], "conf": 0.95, "stereo": True,
              "rays_used": 3, "parallax_deg": 22.0, "skew_rms_m": 0.1}}
            for idx, t in enumerate(active)
        ]
        if vis:
            node.pub_vision_targets(vis)
        node.spin_for(0.10)

    # Wait up to 6.0s for RETURN on /strategy/cmd before landing
    ret_deadline = time.monotonic() + 6.0
    while time.monotonic() < ret_deadline:
        if any(c.get("action") == "RETURN" for c in node.strategy_cmds):
            break
        for d in drones:
            node.pub_fc_status(d)
        node.pub_drone_state("drone_1", 10.0, -10.0, 10.0, vx=1.0)
        for idx, d in enumerate(drones[1:], start=1):
            node.pub_drone_state(d, 20.0, -10.0 + 4.0 * idx, 10.0, vx=1.0)
        node.spin_for(0.10)

    # Land all drones on their helipads and poll summary_path before SIGINT
    summary_file = {repr(str(summary_path))}
    land_deadline = time.monotonic() + 6.0
    while time.monotonic() < land_deadline:
        for d in drones:
            hx, hy, _ = helipads[d]
            node.pub_fc_status(d)
            node.pub_drone_state(d, hx, hy, 0.05, vx=0.0, vy=0.0, vz=0.0)
        node.spin_for(0.10)
        if os.path.isfile(summary_file) and os.path.getsize(summary_file) > 0:
            node.spin_for(0.15)
            break

    result["strategy_cmds"] = node.strategy_cmds
    result["kill_events"] = node.kill_events
    """
    data = _run_f9_probe(
        tmp_path,
        domain_id=134,
        node_cmds=[
            ["ros2", "run", "flight_ctrl_pkg", "target_sim_node", "--ros-args", "-p", "sim_backend:=kinematic"],
            [
                "ros2",
                "run",
                "flight_ctrl_pkg",
                "mission_commander_node",
                "--ros-args",
                "-p",
                f"drone_ids:={drones}",
                "-p",
                f"summary_json_path:={summary_path}",
                "-p",
                f"log_jsonl_path:={trace_path}",
                "-p",
                "exit_on_complete:=false",
            ],
        ],
        driver_body=body,
        timeout_s=40.0,
    )
    actions = [c.get("action") for c in data["strategy_cmds"]]
    assert "RETURN" in actions, f"Expected RETURN on /strategy/cmd before landing, got {actions}"
    assert summary_path.is_file(), f"mission_commander_node did not write {summary_path}"
    assert trace_path.is_file(), f"mission_commander_node did not write {trace_path}"
    sres = validate_mission_summary(summary_path, min_drones=6, min_targets=3, min_obstacles=6)
    assert sres.ok is True, f"Contract #7 summary failed: {sres.errors}"
    kill_ids = [k["target_id"] for k in sres.data["target_kill_details"]]
    assert len(set(kill_ids)) == sres.data["targets_neutralized"]

    tres = validate_mission_trace(trace_path, obstacles_cfg=cfg, summary_data=sres.data)
    assert tres.ok is True, f"Contract #7 trace failed: {tres.errors}"
    assert any(p in tres.phases_observed for p in ("ATTACK", "KILL", "BDA", "RETASK"))


@pytest.mark.requires_m4
@pytest.mark.requires_sitl
def test_f09_contract7_summary_and_trace_validation_sitl(ros2_ws_root: Path) -> None:
    """Validate live/recorded 6v3 SITL summary.json & trace.jsonl against Contract #7."""
    summary_path = Path(
        os.environ.get("E2E_SITL_SUMMARY_PATH", "/tmp/project2_mission_summary.json")
    )
    trace_path = Path(
        os.environ.get("E2E_SITL_TRACE_PATH", "/tmp/project2_mission_trace.jsonl")
    )
    cfg = load_obstacles_yaml(
        ros2_ws_root / "src" / "flight_ctrl_pkg" / "config" / "obstacles.yaml"
    )
    sres = validate_mission_summary(
        summary_path, min_drones=6, min_targets=3, min_obstacles=6
    )
    assert sres.ok is True, f"SITL summary Contract #7 failed: {sres.errors}"
    kill_ids = [k["target_id"] for k in sres.data["target_kill_details"]]
    assert len(set(kill_ids)) == sres.data["targets_neutralized"], (
        f"Duplicate target_id entries in target_kill_details: {kill_ids}"
    )

    tres = validate_mission_trace(trace_path, obstacles_cfg=cfg, summary_data=sres.data)
    assert tres.ok is True, f"SITL trace validation failed: {tres.errors}"


@pytest.mark.requires_m1
def test_f09_urban_arena_sdf_has_strike_burst_emitters(ros2_ws_root: Path) -> None:
    """Verify urban_counter_swarm_arena.sdf defines >= 3 strike particle-emitter models."""
    sdf_path = (
        ros2_ws_root
        / "src"
        / "flight_ctrl_pkg"
        / "sim"
        / "worlds"
        / "urban_counter_swarm_arena.sdf"
    )
    audit = parse_sdf_collisions(sdf_path)
    burst_models = [m["name"] for m in audit.models if m.get("has_particle_emitter")]
    assert len(burst_models) >= 3, (
        f"Expected >= 3 particle-emitter strike burst models, found {burst_models}"
    )
