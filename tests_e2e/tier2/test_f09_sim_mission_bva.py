"""Tier 2 BVA & corner-case opaque-box tests for F9: Sim & Event-Driven Strike/BDA."""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import pytest

from helpers.arena_parser import load_obstacles_yaml, point_min_clearance
from helpers.metrics_validator import (
    validate_kill_event_msg,
    validate_mission_summary,
    validate_mission_trace,
)
from helpers.ros_probe import run_command_with_gpu_guard

pytestmark = [
    pytest.mark.tier2,
    pytest.mark.feature(id="F9"),
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


def _run_f9_bva_probe(
    tmp_path: Path,
    domain_id: int,
    node_cmds: List[List[str]],
    driver_body: str,
    timeout_s: float = 35.0,
) -> Dict[str, Any]:
    script_path = tmp_path / f"f09_bva_probe_{domain_id}.py"
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


        class F9BvaProbe(Node):
            def __init__(self):
                super().__init__("f09_bva_probe_{domain_id}")
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

                self.gt_pub = self.create_publisher(String, "/sim/targets_gt", 10)
                self.vision_pub = self.create_publisher(String, "/vision/targets", 10)
                self.kill_pub = self.create_publisher(String, "/engagement/kill_event", 10)

            def _on_targets_gt(self, msg):
                try:
                    self.targets_gt.append(json.loads(msg.data))
                except Exception:
                    pass

            def _on_radar(self, msg):
                try:
                    self.radar_gt.append(json.loads(msg.data))
                except Exception:
                    pass

            def _on_kill(self, msg):
                try:
                    self.kill_events.append(json.loads(msg.data))
                except Exception:
                    pass

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

            def pub_vision_raw(self, raw_text):
                m = String()
                m.data = raw_text
                self.vision_pub.publish(m)

            def pub_targets_gt(self, targets):
                m = String()
                m.data = json.dumps({{"stamp": time.time(), "targets": targets}})
                self.gt_pub.publish(m)

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
            node = F9BvaProbe()
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
    assert res.timed_out is False, f"F9 BVA probe timed out: stderr={res.stderr}"
    assert res.returncode == 0, f"F9 BVA probe failed rc={res.returncode}: stderr={res.stderr}"
    data = _extract_probe_json(res.stdout)
    assert all(rc is None for rc in data["node_rcs"]), (
        f"ROS node exited prematurely: {data['node_rcs']}"
    )
    return data


def test_f09_bva_zero_targets_immediate_return_and_zero_speed_scale(
    tmp_path: Path,
) -> None:
    """Verify speed_scale=0.0 stationary target boundary and zero-active-target RETURN transition."""
    # Probe A: target_sim_node speed_scale:=0.0
    body_a = """
    node.spin_for(1.2)
    result["gt_msgs"] = node.targets_gt[-12:]
    result["radar_count"] = len(node.radar_gt)
    result["legacy_count"] = node.legacy_gt_count
    """
    data_a = _run_f9_bva_probe(
        tmp_path,
        domain_id=135,
        node_cmds=[
            [
                "ros2",
                "run",
                "flight_ctrl_pkg",
                "target_sim_node",
                "--ros-args",
                "-p",
                "sim_backend:=kinematic",
                "-p",
                "speed_scale:=0.0",
            ]
        ],
        driver_body=body_a,
    )
    assert data_a["gt_msgs"], "Missing /sim/targets_gt under speed_scale=0.0"
    assert data_a["radar_count"] >= 2 and data_a["legacy_count"] >= 5
    first_t = data_a["gt_msgs"][0]["targets"][0]
    last_t = data_a["gt_msgs"][-1]["targets"][0]
    disp = math.sqrt(
        (first_t["x"] - last_t["x"]) ** 2
        + (first_t["y"] - last_t["y"]) ** 2
        + (first_t["z"] - last_t["z"]) ** 2
    )
    assert disp < 0.05, f"Target drifted with speed_scale=0.0: disp={disp:.4f}m"
    assert math.sqrt(last_t["vx"] ** 2 + last_t["vy"] ** 2 + last_t["vz"] ** 2) < 0.05

    # Probe B: mission_commander_node with zero active targets transitions promptly to RETURN
    summary_p = tmp_path / "zero_tgt_summary.json"
    trace_p = tmp_path / "zero_tgt_trace.jsonl"
    drones = [f"drone_{i}" for i in range(1, 7)]
    body_b = f"""
    drones = {repr(drones)}
    for i, d in enumerate(drones):
        node.pub_fc_status(d)
        node.pub_drone_state(d, 0.0, float(i * 3), 8.5)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        for i, d in enumerate(drones):
            node.pub_fc_status(d)
            node.pub_drone_state(d, 0.0, float(i * 3), 8.5)
        node.pub_targets_gt([])
        node.spin_for(0.10)
        if any(c.get("action") == "RETURN" for c in node.strategy_cmds):
            break
    result["cmds"] = node.strategy_cmds
    """
    data_b = _run_f9_bva_probe(
        tmp_path,
        domain_id=136,
        node_cmds=[
            [
                "ros2",
                "run",
                "flight_ctrl_pkg",
                "mission_commander_node",
                "--ros-args",
                "-p",
                f"drone_ids:={drones}",
                "-p",
                f"summary_json_path:={summary_p}",
                "-p",
                f"log_jsonl_path:={trace_p}",
                "-p",
                "exit_on_complete:=false",
            ]
        ],
        driver_body=body_b,
    )
    actions = [c.get("action") for c in data_b["cmds"]]
    assert "RETURN" in actions, (
        f"Expected zero-target mission to transition to RETURN, got strategy_cmds={data_b['cmds']}"
    )


def test_f09_bva_simultaneous_multi_kill_bda_ordering_and_idempotency(
    tmp_path: Path,
) -> None:
    """Verify repeated/simultaneous kills remain idempotent with unique target_kill_details."""
    summary_p = tmp_path / "idempotent_summary.json"
    trace_p = tmp_path / "idempotent_trace.jsonl"
    drones = [f"drone_{i}" for i in range(1, 7)]
    body = f"""
    drones = {repr(drones)}
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not node.targets_gt:
        node.spin_for(0.05)
    assert node.targets_gt, "No /sim/targets_gt received"
    tgts = node.targets_gt[-1]["targets"]
    t0_id = tgts[0]["target_id"]
    t1_id = tgts[1]["target_id"]

    for d in drones:
        node.pub_fc_status(d)
        node.pub_drone_state(d, 0.0, 0.0, 8.5)

    # Trigger double-hit on t0_id and simultaneous hit on t1_id
    for _ in range(15):
        cur = {{t["target_id"]: t for t in node.targets_gt[-1]["targets"]}}
        node.pub_drone_state("drone_1", cur[t0_id]["x"], cur[t0_id]["y"], cur[t0_id]["z"], vx=10.0)
        node.pub_drone_state("drone_2", cur[t1_id]["x"], cur[t1_id]["y"], cur[t1_id]["z"], vx=10.0)
        node.spin_for(0.06)

    # Repeat kill proximity on already-neutralized t0_id
    for _ in range(10):
        cur = {{t["target_id"]: t for t in node.targets_gt[-1]["targets"]}}
        node.pub_drone_state("drone_3", cur[t0_id]["x"], cur[t0_id]["y"], cur[t0_id]["z"], vx=10.0)
        node.spin_for(0.06)

    result["t0_id"] = t0_id
    result["t1_id"] = t1_id
    result["last_gt"] = node.targets_gt[-1]
    result["kill_events"] = node.kill_events
    """
    data = _run_f9_bva_probe(
        tmp_path,
        domain_id=137,
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
                f"summary_json_path:={summary_p}",
                "-p",
                f"log_jsonl_path:={trace_p}",
                "-p",
                "exit_on_complete:=false",
            ],
        ],
        driver_body=body,
    )
    gt_map = {t["target_id"]: t for t in data["last_gt"]["targets"]}
    assert gt_map[data["t0_id"]]["status"] == "NEUTRALIZED"
    assert gt_map[data["t1_id"]]["status"] == "NEUTRALIZED"
    for kev in data["kill_events"]:
        assert validate_kill_event_msg(kev).ok is True

    assert summary_p.is_file(), "mission_commander_node did not write summary.json"
    summary = json.loads(summary_p.read_text(encoding="utf-8"))
    assert "targets_neutralized" in summary and "target_kill_details" in summary
    assert int(summary["targets_neutralized"]) <= int(summary["target_count"])
    detail_ids = [d["target_id"] for d in summary["target_kill_details"]]
    assert len(detail_ids) == len(set(detail_ids)), (
        f"Double-kill created duplicate target_kill_details: {detail_ids}"
    )


def test_f09_bva_target_skirting_obstacle_boundary_clearance(
    tmp_path: Path, ros2_ws_root: Path
) -> None:
    """Verify target_sim_node trajectories stay >= 0.0m near obstacle surfaces."""
    cfg = load_obstacles_yaml(
        ros2_ws_root / "src" / "flight_ctrl_pkg" / "config" / "obstacles.yaml"
    )
    # Exact boundary check on obs_comm_pylon_center (cx=49.0, cy=1.5, r=1.4, z_max=16.5)
    assert point_min_clearance((50.41, 1.5, 8.0), cfg, include_nfz=False) > 0.0
    assert point_min_clearance((50.39, 1.5, 8.0), cfg, include_nfz=False) < 0.0

    body = """
    node.spin_for(3.5)
    result["frames"] = node.targets_gt
    """
    data = _run_f9_bva_probe(
        tmp_path,
        domain_id=138,
        node_cmds=[
            ["ros2", "run", "flight_ctrl_pkg", "target_sim_node", "--ros-args", "-p", "sim_backend:=kinematic"]
        ],
        driver_body=body,
    )
    assert len(data["frames"]) >= 25
    for frame in data["frames"]:
        for t in frame.get("targets", []):
            clr = point_min_clearance((t["x"], t["y"], t["z"]), cfg, include_nfz=False)
            assert clr >= 0.0, f"Target skirted into obstacle: {t}, clr={clr:.4f}m"


def test_f09_bva_late_stage_target_escape_reacquisition(tmp_path: Path) -> None:
    """Verify near-miss escape (> lethal radius) avoids premature kill and sustains tracking."""
    body = """
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and not node.targets_gt:
        node.spin_for(0.05)
    assert node.targets_gt
    t0 = node.targets_gt[-1]["targets"][0]
    t0_id = t0["target_id"]

    # Approach to ~2.5m (outside lethal hit radius) then pull back >12m
    for _ in range(10):
        cur = [t for t in node.targets_gt[-1]["targets"] if t["target_id"] == t0_id][0]
        node.pub_drone_state("drone_1", cur["x"] - 2.5, cur["y"], cur["z"], vx=4.0)
        node.spin_for(0.06)
    for _ in range(10):
        cur = [t for t in node.targets_gt[-1]["targets"] if t["target_id"] == t0_id][0]
        node.pub_drone_state("drone_1", cur["x"] - 14.0, cur["y"], cur["z"], vx=-2.0)
        node.spin_for(0.06)

    result["t0_id"] = t0_id
    result["last_gt"] = node.targets_gt[-1]
    result["kill_events"] = node.kill_events
    """
    data = _run_f9_bva_probe(
        tmp_path,
        domain_id=139,
        node_cmds=[
            ["ros2", "run", "flight_ctrl_pkg", "target_sim_node", "--ros-args", "-p", "sim_backend:=kinematic"]
        ],
        driver_body=body,
    )
    gt_map = {t["target_id"]: t for t in data["last_gt"]["targets"]}
    assert gt_map[data["t0_id"]]["status"] == "ACTIVE", (
        f"Target falsely neutralized at 2.5m near-miss escape: {gt_map}"
    )
    assert data["kill_events"] == [], f"Unexpected spurious kill_event on escape: {data['kill_events']}"


def test_f09_bva_corrupt_telemetry_incomplete_landing_and_tamper_rejection(
    tmp_path: Path, ros2_ws_root: Path
) -> None:
    """Verify resilience to corrupt telemetry, unlanded completed=False, and verifier BVA rejections."""
    cfg = load_obstacles_yaml(
        ros2_ws_root / "src" / "flight_ctrl_pkg" / "config" / "obstacles.yaml"
    )
    summary_p = tmp_path / "incomplete_summary.json"
    trace_p = tmp_path / "incomplete_trace.jsonl"
    drones = [f"drone_{i}" for i in range(1, 7)]

    body = f"""
    drones = {repr(drones)}
    node.pub_vision_raw("{{corrupt_json")
    for _ in range(12):
        for i, d in enumerate(drones):
            node.pub_fc_status(d)
            # Drones hover at z=5.0m (never land < 0.38m)
            node.pub_drone_state(d, 5.0, float(i * 3), 5.0, vx=0.0, vy=0.0, vz=0.0)
        node.pub_vision_raw(json.dumps({{
            "stamp": time.time(), "frame_id": "world",
            "targets": [{{"det_id": 0, "x": float("nan"), "y": 0.0, "z": 5.0}}]
        }}))
        node.spin_for(0.10)
    """
    _run_f9_bva_probe(
        tmp_path,
        domain_id=140,
        node_cmds=[
            [
                "ros2",
                "run",
                "flight_ctrl_pkg",
                "mission_commander_node",
                "--ros-args",
                "-p",
                f"drone_ids:={drones}",
                "-p",
                f"summary_json_path:={summary_p}",
                "-p",
                f"log_jsonl_path:={trace_p}",
                "-p",
                "exit_on_complete:=false",
            ]
        ],
        driver_body=body,
    )
    assert summary_p.is_file(), "mission_commander_node failed to emit summary.json on shutdown"
    raw_sum = json.loads(summary_p.read_text(encoding="utf-8"))
    for req_key in ("target_count", "targets_neutralized", "completed", "gpu3_violation"):
        assert req_key in raw_sum, f"Missing Contract #7 key '{req_key}' in {raw_sum}"
    assert raw_sum["completed"] is False, (
        f"Expected completed=False when drones never landed, got {raw_sum['completed']}"
    )
    assert validate_mission_summary(summary_p).ok is False

    # Validate Contract #4 kill_event BVA boundaries
    valid_kill = {
        "stamp": 1.0,
        "target_id": "tgt_1",
        "striker_id": "drone_1",
        "miss_dist_m": 0.0,
        "closing_speed_ms": 0.0,
        "x": 10.0,
        "y": 0.0,
        "z": 5.0,
    }
    assert validate_kill_event_msg(valid_kill).ok is True
    bad_kill = dict(valid_kill, miss_dist_m=-1e-4)
    assert validate_kill_event_msg(bad_kill).ok is False

    # Validate trace tamper & penetration rejection
    bad_trace = tmp_path / "bad_trace.jsonl"
    bad_trace.write_text(
        "\n".join(
            [
                json.dumps({
                    "timestamp": 2.0,
                    "trace_id": "t1",
                    "span_id": "s1",
                    "phase": "ATTACK",
                    "drones": {
                        "drone_1": {"pos": [49.0, 1.5, 8.0], "vel": [0.0, 0.0, 0.0]},
                        "drone_2": {"pos": [20.0, 0.0, 8.0], "vel": [0.0, 0.0, 0.0]},
                    },
                }),
                json.dumps({
                    "timestamp": 1.0,  # non-monotonic
                    "trace_id": "t1",
                    "span_id": "s2",
                    "phase": "RETURN",
                    "drones": {
                        "drone_1": {"pos": [10.0, 0.0, 8.0], "vel": [0.0, 0.0, 0.0]},
                        "drone_2": {"pos": [20.0, 0.0, 8.0], "vel": [0.0, 0.0, 0.0]},
                    },
                }),
            ]
        ),
        encoding="utf-8",
    )
    tres = validate_mission_trace(bad_trace, obstacles_cfg=cfg)
    assert tres.ok is False and len(tres.errors) >= 2
