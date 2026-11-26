# -*- coding: utf-8 -*-
"""planning_pkg.planning_node 集成测试：多目标 WTA、击毁重分配、ROS 2 接口契约与单目标兼容。"""

from __future__ import annotations

import json
import math

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from std_msgs.msg import String

from planning_pkg.planning_node import PlanningNode


def _make_odom(drone_id: str, x: float, y: float, z: float) -> Odometry:
    msg = Odometry()
    msg.header.frame_id = "world"
    msg.child_frame_id = f"{drone_id}_base_link"
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.position.z = float(z)
    msg.pose.pose.orientation.w = 1.0
    return msg


def test_planning_node_multi_target_wta_6_drones_and_kill_retasking():
    rclpy.init()
    node = PlanningNode()
    probe = rclpy.create_node("planning_contract_probe")
    captured_plans = []
    captured_diags = []

    probe.create_subscription(String, "/planning/target_pose", lambda m: captured_plans.append(json.loads(m.data)), 50)
    probe.create_subscription(String, "/planning/diagnostics", lambda m: captured_diags.append(json.loads(m.data)), 20)

    def spin_both(rounds=6):
        for _ in range(rounds):
            rclpy.spin_once(probe, timeout_sec=0.005)

    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.005)
        rclpy.spin_once(probe, timeout_sec=0.005)
        if node.plan_pub.get_subscription_count() >= 1 and node.diag_pub.get_subscription_count() >= 1:
            break

    try:
        # 1. 注册 6 架无人机并切换为 ATTACK
        drones6 = [f"drone_{i + 1}" for i in range(6)]
        for idx, d_id in enumerate(drones6):
            node._ensure_drone_subscription(d_id)
            node._make_state_cb(d_id)(_make_odom(d_id, 0.0, idx * 3.0, 9.0))

        cmd = String()
        cmd.data = json.dumps({"drone_id": "ALL", "action": "TAKEOFF"})
        node._strategy_cb(cmd)
        cmd.data = json.dumps({"drone_id": "ALL", "action": "ATTACK"})
        node._strategy_cb(cmd)

        # 2. 注入 /fleet/state 电量
        fleet_msg = String()
        fleet_msg.data = json.dumps({
            "drones": {d: {"battery_pct": 90.0 - i * 5.0} for i, d in enumerate(drones6)}
        })
        node._fleet_state_cb(fleet_msg)

        # 3. 连续注入 4 帧 /vision/targets (2 个目标，满足 3-of-5 确认) + /external/targets_detected
        now_s = node._now_sec()
        for k in range(4):
            tk = now_s - 0.15 + k * 0.04
            v_msg = String()
            v_msg.data = json.dumps({
                "stamp": tk,
                "frame_id": "world",
                "targets": [
                    {"det_id": 0, "class_id": "crimson_uav", "x": 22.0 + 3.0 * k * 0.04, "y": -4.0, "z": 6.5, "conf": 0.94, "stereo": True},
                    {"det_id": 1, "class_id": "amber_uav", "x": 26.0 + 2.0 * k * 0.04, "y": 14.0, "z": 7.0, "conf": 0.91, "stereo": True},
                ],
            })
            node._vision_targets_cb(v_msg)

        ext_msg = String()
        ext_msg.data = json.dumps({
            "stamp": now_s,
            "targets": [
                {"target_id": "tgt_alpha", "class_id": "crimson_uav", "x": 22.4, "y": -4.0, "z": 6.5},
                {"target_id": "tgt_beta", "class_id": "amber_uav", "x": 26.2, "y": 14.0, "z": 7.0},
            ],
        })
        node._external_targets_cb(ext_msg)

        node._last_diag_sec = 0.0
        node._plan_tick()
        for _ in range(120):
            rclpy.spin_once(probe, timeout_sec=0.005)
            if len({p["drone_id"] for p in captured_plans}) == 6 and len(captured_diags) >= 1:
                break

        assert len(captured_plans) >= 6
        latest_by_drone = {p["drone_id"]: p for p in captured_plans}
        assert set(latest_by_drone.keys()) == set(drones6)
        assigned_tids = set()
        for p in latest_by_drone.values():
            for req in ("x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "target_id", "imm_probs", "pred_path", "focus_x"):
                assert req in p
            assert math.isfinite(p["vx"]) and math.isfinite(p["ax"])
            assert abs(sum(p["imm_probs"].values()) - 1.0) < 1e-3
            assert len(p["pred_path"]) == 6
            assigned_tids.add(p["target_id"])
        assert len(assigned_tids) == 2

        assert len(captured_diags) >= 1
        diag = captured_diags[-1]
        for k in ("tracks", "wta_assignments", "neutralized_targets", "re_tasking_events"):
            assert k in diag
        assert len(diag["wta_assignments"]) == 6

        # 4. 注入 /engagement/kill_event 击毁 tgt_alpha -> 验证全部 6 架即时重分配到剩余目标
        spin_both(25)
        kill_ts = node._now_sec()
        kill_msg = String()
        kill_msg.data = json.dumps({
            "stamp": kill_ts,
            "target_id": "tgt_alpha",
            "striker_id": "drone_1",
            "x": 22.4,
            "y": -4.0,
            "z": 6.5,
        })
        node._kill_event_cb(kill_msg)
        captured_plans.clear()
        node._last_diag_sec = 0.0
        node._plan_tick()
        for _ in range(120):
            rclpy.spin_once(probe, timeout_sec=0.005)
            post_kill = [p for p in captured_plans if p["stamp"] >= kill_ts]
            if len({p["drone_id"] for p in post_kill}) == 6 and len(captured_diags) >= 1 and "tgt_alpha" in captured_diags[-1].get("neutralized_targets", []):
                break

        post_kill_plans = [p for p in captured_plans if p["stamp"] >= kill_ts]
        latest_after = {p["drone_id"]: p for p in post_kill_plans}
        assert set(latest_after.keys()) == set(drones6)
        rem_tids = {p["target_id"] for p in latest_after.values()}
        assert len(rem_tids) == 1
        assert rem_tids != assigned_tids
        assert captured_diags[-1]["re_tasking_events"] >= 1
        assert "tgt_alpha" in captured_diags[-1]["neutralized_targets"]

        # 5. 通过 /strategy/cmd neutralized_targets 击毁剩余目标 tgt_beta -> 集群安全切换为空目标巡逻
        strat_kill = String()
        strat_kill.data = json.dumps({"neutralized_targets": ["tgt_beta"]})
        node._strategy_cb(strat_kill)
        assert "tgt_beta" in node.planner.neutralized_targets
        assert len(node.track_bank.get_confirmed_tracks(node._now_sec())) == 0
        assert not node.estimator.is_valid(node._now_sec())

        captured_plans.clear()
        post_final_ts = node._now_sec()
        node._plan_tick()
        for _ in range(120):
            rclpy.spin_once(probe, timeout_sec=0.005)
            post_final = [p for p in captured_plans if p["stamp"] >= post_final_ts]
            if len({p["drone_id"] for p in post_final}) == 6:
                break
        post_final_plans = {
            p["drone_id"]: p for p in captured_plans if p["stamp"] >= post_final_ts
        }
        assert set(post_final_plans.keys()) == set(drones6)
        for p in post_final_plans.values():
            assert p["target_id"] == ""
            assert p["action"] != "ATTACK"
            assert p["action"] == "TRACK"
    finally:
        probe.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


def test_planning_node_legacy_single_target_compatibility():
    rclpy.init()
    node = PlanningNode()
    probe = rclpy.create_node("planning_legacy_probe")
    captured_plans = []
    probe.create_subscription(String, "/planning/target_pose", lambda m: captured_plans.append(json.loads(m.data)), 30)

    try:
        assert node.drone_ids == ["drone_1", "drone_2", "drone_3"]
        for idx, d_id in enumerate(node.drone_ids):
            node._make_state_cb(d_id)(_make_odom(d_id, 0.0, idx * 3.0, 9.0))
        for act in ("TAKEOFF", "TRACK"):
            cmd = String()
            cmd.data = json.dumps({"drone_id": "ALL", "action": act})
            node._strategy_cb(cmd)

        # 传统单目标 /vision/target_pose 仅输入 1 帧即有效
        vis = PoseStamped()
        vis.header.stamp = node.get_clock().now().to_msg()
        vis.pose.position.x = 18.0
        vis.pose.position.y = 2.0
        vis.pose.position.z = 6.0
        node._vision_cb(vis)
        assert node.estimator.is_valid(node._now_sec())

        node._plan_tick()
        for _ in range(8):
            rclpy.spin_once(node, timeout_sec=0.005)
            rclpy.spin_once(probe, timeout_sec=0.005)

        assert len(captured_plans) >= 3
        latest = {p["drone_id"]: p for p in captured_plans[-3:]}
        assert set(latest.keys()) == {"drone_1", "drone_2", "drone_3"}
        for p in latest.values():
            assert p["action"] == "TRACK"
            assert "focus_x" in p and "imm_probs" in p and "pred_path" in p
    finally:
        probe.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


def test_planning_node_vision_only_neutralized_targets_and_no_legacy_ghost():
    rclpy.init()
    node = PlanningNode()
    probe = rclpy.create_node("planning_vision_kill_probe")
    captured_plans = []
    probe.create_subscription(String, "/planning/target_pose", lambda m: captured_plans.append(json.loads(m.data)), 50)

    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.005)
        rclpy.spin_once(probe, timeout_sec=0.005)
        if node.plan_pub.get_subscription_count() >= 1:
            break

    try:
        for idx, d_id in enumerate(node.drone_ids):
            node._make_state_cb(d_id)(_make_odom(d_id, 0.0, idx * 3.0, 9.0))
        for act in ("TAKEOFF", "ATTACK"):
            cmd = String()
            cmd.data = json.dumps({"drone_id": "ALL", "action": act})
            node._strategy_cb(cmd)

        # 纯 /vision/targets（不注入 /external/targets_detected）：
        # 目标1仅含 det_id（分配 track_id=trk_1），目标2附带显式 target_id="tgt_vis_2"
        now_s = node._now_sec()
        for k in range(4):
            tk = now_s - 0.12 + k * 0.04
            v_msg = String()
            v_msg.data = json.dumps({
                "stamp": tk,
                "frame_id": "world",
                "targets": [
                    {"det_id": 0, "class_id": "crimson_uav", "x": 20.0 + k * 0.1, "y": -5.0, "z": 6.5, "conf": 0.95, "stereo": True},
                    {"det_id": 1, "target_id": "tgt_vis_2", "class_id": "amber_uav", "x": 24.0 + k * 0.1, "y": 9.0, "z": 7.0, "conf": 0.92, "stereo": True},
                ],
            })
            node._vision_targets_cb(v_msg)

        node._plan_tick()
        for _ in range(40):
            rclpy.spin_once(probe, timeout_sec=0.005)
        confirmed = node.track_bank.get_confirmed_tracks(node._now_sec())
        assert len(confirmed) == 2
        trk_no_ext = next(t.track_id for t in confirmed if not t.external_id)

        # 1) 通过 dict {"track_id": trk_no_ext} 击毁纯视觉航迹
        kill_1 = String()
        kill_1.data = json.dumps({"neutralized_targets": [{"track_id": trk_no_ext}]})
        node._strategy_cb(kill_1)
        rem = node.track_bank.get_confirmed_tracks(node._now_sec())
        assert len(rem) == 1 and rem[0].external_id == "tgt_vis_2"

        # 2) 通过字符串 external_id 击毁最后一个纯视觉航迹 -> 不得复活 "legacy_target"
        kill_2 = String()
        kill_2.data = json.dumps({"neutralized_targets": ["tgt_vis_2"]})
        node._strategy_cb(kill_2)
        assert len(node.track_bank.get_confirmed_tracks(node._now_sec())) == 0
        assert not node.estimator.is_valid(node._now_sec())

        captured_plans.clear()
        ts_after = node._now_sec()
        node._plan_tick()
        for _ in range(40):
            rclpy.spin_once(probe, timeout_sec=0.005)
        latest = {p["drone_id"]: p for p in captured_plans if p["stamp"] >= ts_after}
        assert set(latest.keys()) == set(node.drone_ids)
        for p in latest.values():
            assert p["target_id"] == ""
            assert p["action"] != "ATTACK"
    finally:
        probe.destroy_node()
        node.destroy_node()
        rclpy.shutdown()


def test_planning_node_future_stamp_poison_resilience():
    rclpy.init()
    node = PlanningNode()
    try:
        t0 = node._now_sec()
        assert node._extract_json_stamp({"stamp": t0 + 20.0}, t0) <= t0 + 0.5
        assert node._extract_json_stamp({"stamp": t0 - 5.0}, t0) == t0 - 5.0

        ps = PoseStamped()
        ps.header.stamp.sec = int(t0 + 20.0)
        ps.header.stamp.nanosec = 0
        assert node._extract_stamp_sec(ps, t0) <= t0 + 0.5

        for k in range(4):
            vm = String()
            vm.data = json.dumps({
                "stamp": t0 + k * 0.04,
                "targets": [{"det_id": 0, "x": 15.0 + k * 0.1, "y": 4.0, "z": 6.0}],
            })
            node._vision_targets_cb(vm)

        trks = node.track_bank.get_confirmed_tracks(t0 + 0.2)
        assert len(trks) == 1
        upd_before = trks[0].update_count

        # 注入 +20s 未来时间戳脏包
        vm_future = String()
        vm_future.data = json.dumps({
            "stamp": t0 + 20.0,
            "targets": [{"det_id": 0, "x": 15.5, "y": 4.0, "z": 6.0}],
        })
        node._vision_targets_cb(vm_future)

        # 紧接着发送正常时间戳帧，必须持续更新而不能被冻结
        for k in range(4):
            vm_next = String()
            vm_next.data = json.dumps({
                "stamp": t0 + 0.20 + k * 0.04,
                "targets": [{"det_id": 0, "x": 15.8 + k * 0.15, "y": 4.0, "z": 6.0}],
            })
            node._vision_targets_cb(vm_next)

        trks_after = node.track_bank.get_confirmed_tracks(t0 + 0.35)
        assert len(trks_after) == 1
        assert trks_after[0].update_count > upd_before
    finally:
        node.destroy_node()
        rclpy.shutdown()
