# -*- coding: utf-8 -*-
"""
planning_node.py —— 课题二路径规划与多机协同拦截制导主节点（接口 C 提供方）

严格遵循《课题二内部接口规范 V3.0》与《课题二-课题三双向接口规范 V2.0》：
  订阅：
    - 接口 A: /vision/target_pose         (geometry_msgs/PoseStamped, 机载视觉高精度目标位姿)
    - 接口 B: /external/target_detected   (geometry_msgs/PoseStamped, 外部宏观雷达引导)
    - 接口 D: /strategy/cmd               (std_msgs/String JSON, 课题三战术指令，支持个体与 ALL)
    - 接口 E: /drone_<id>/state           (nav_msgs/Odometry, 己方各机 ENU 位姿与速度)
  发布：
    - 接口 C: /planning/target_pose       (std_msgs/String JSON, 带 drone_id/stamp/x/y/z/action)
    - 观测诊断: /planning/diagnostics     (std_msgs/String JSON, 估计器协方差与制导状态追踪)
"""

from __future__ import annotations

import copy
import json
import math
from typing import Dict, List, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from planning_pkg.estimator import ManeuveringTargetEstimator
from planning_pkg.guidance import CooperativeGuidancePlanner, DynamicsProfile

_VALID_ACTIONS = {
    "TAKEOFF",
    "HOVER",
    "RETURN",
    "TRACK",
    "INTERCEPT",
    "ATTACK",
    "RESUME_PATROL",
}


class PlanningNode(Node):
    """多源融合机动目标预测与三机协同制导规划节点。"""

    def __init__(self) -> None:
        super().__init__("planning_node")

        # -------------------- 参数声明 --------------------
        self.declare_parameter("drone_ids", ["drone_1", "drone_2", "drone_3"])
        self.declare_parameter("plan_hz", 20.0)
        self.declare_parameter("v_cruise", 5.2)
        self.declare_parameter("v_intercept", 10.5)
        self.declare_parameter("v_attack", 12.2)
        self.declare_parameter("a_max", 10.5)
        self.declare_parameter("min_alt", 3.2)
        self.declare_parameter("max_alt", 35.0)
        self.declare_parameter("track_radius", 6.0)
        self.declare_parameter("track_alt_offset", 3.5)
        self.declare_parameter("max_coast_sec", 3.0)
        self.declare_parameter("sigma_vision", 0.36)
        self.declare_parameter("sigma_external", 0.55)
        self.declare_parameter("max_xy_dist", 950.0)

        p = self.get_parameter
        self.drone_ids: List[str] = [str(d) for d in p("drone_ids").value]
        self.plan_hz: float = max(2.0, float(p("plan_hz").value))
        self.max_xy_dist: float = float(p("max_xy_dist").value)

        profile = DynamicsProfile(
            v_cruise=float(p("v_cruise").value),
            v_intercept=float(p("v_intercept").value),
            v_attack=float(p("v_attack").value),
            a_max=float(p("a_max").value),
            min_alt=float(p("min_alt").value),
            max_alt=float(p("max_alt").value),
            track_radius=float(p("track_radius").value),
            track_alt_offset=float(p("track_alt_offset").value),
        )
        self.estimator = ManeuveringTargetEstimator(
            sigma_vision=float(p("sigma_vision").value),
            sigma_external=float(p("sigma_external").value),
            max_coast_sec=float(p("max_coast_sec").value),
        )
        self.planner = CooperativeGuidancePlanner(profile=profile)

        # -------------------- 内部集群状态缓存 --------------------
        self.drone_modes: Dict[str, str] = {d: "GROUND" for d in self.drone_ids}
        self.drone_kinematics: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._last_diag_sec: float = 0.0

        # -------------------- 订阅器（接口 A, B, D, E） --------------------
        self.create_subscription(PoseStamped, "/vision/target_pose", self._vision_cb, 10)
        self.create_subscription(PoseStamped, "/external/target_detected", self._external_cb, 10)
        self.create_subscription(String, "/strategy/cmd", self._strategy_cb, 10)
        for d_id in self.drone_ids:
            self.create_subscription(
                Odometry,
                f"/{d_id}/state",
                self._make_state_cb(d_id),
                10,
            )

        # -------------------- 发布器（接口 C + 诊断） --------------------
        self.plan_pub = self.create_publisher(String, "/planning/target_pose", 20)
        self.diag_pub = self.create_publisher(String, "/planning/diagnostics", 10)

        self.timer = self.create_timer(1.0 / self.plan_hz, self._plan_tick)
        self.get_logger().info(
            f"planning_node 已启动({self.plan_hz:.0f}Hz): 集群={self.drone_ids}, "
            f"V_max={profile.v_attack:.1f}m/s, R_track={profile.track_radius:.1f}m"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _extract_stamp_sec(self, msg: PoseStamped, fallback_now: float) -> float:
        s = int(msg.header.stamp.sec)
        ns = int(msg.header.stamp.nanosec)
        if s == 0 and ns == 0:
            return fallback_now
        t = s + ns * 1e-9
        # 若上游误发仿真零点戳，回退至当前墙钟
        if abs(fallback_now - t) > 60.0:
            return fallback_now
        return t

    def _sanitize_target_pose(self, msg: PoseStamped) -> Tuple[bool, Tuple[float, float, float]]:
        pos = msg.pose.position
        x, y, z = float(pos.x), float(pos.y), float(pos.z)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return False, (0.0, 0.0, 0.0)
        if abs(x) > self.max_xy_dist or abs(y) > self.max_xy_dist or z < -0.5 or z > 55.0:
            return False, (0.0, 0.0, 0.0)
        return True, (x, y, z)

    def _vision_cb(self, msg: PoseStamped) -> None:
        m = copy.deepcopy(msg)
        ok, xyz = self._sanitize_target_pose(m)
        if not ok:
            self.get_logger().warn("[规划] 拒绝非法视觉坐标输入", throttle_duration_sec=2.0)
            return
        now_s = self._now_sec()
        ts = self._extract_stamp_sec(m, now_s)
        if now_s - ts > 0.55:
            return
        self.estimator.update(xyz, ts, source="vision")

    def _external_cb(self, msg: PoseStamped) -> None:
        m = copy.deepcopy(msg)
        ok, xyz = self._sanitize_target_pose(m)
        if not ok:
            self.get_logger().warn("[规划] 拒绝非法外部雷达坐标输入", throttle_duration_sec=2.0)
            return
        now_s = self._now_sec()
        ts = self._extract_stamp_sec(m, now_s)
        if now_s - ts > 1.0:
            return
        self.estimator.update(xyz, ts, source="external")

    def _make_state_cb(self, drone_id: str):
        def _cb(msg: Odometry) -> None:
            m = copy.deepcopy(msg)
            p = m.pose.pose.position
            v = m.twist.twist.linear
            pos = np.array([float(p.x), float(p.y), float(p.z)], dtype=float)
            vel = np.array([float(v.x), float(v.y), float(v.z)], dtype=float)
            if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))):
                return
            self.drone_kinematics[drone_id] = (pos, vel)
            # 返航触地后自动与中枢同步回 GROUND
            if self.drone_modes.get(drone_id) == "RETURN":
                if pos[2] < 0.32 and float(np.linalg.norm(vel)) < 0.22:
                    self.drone_modes[drone_id] = "GROUND"

        return _cb

    def _strategy_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().error(f"[规划] 解析 /strategy/cmd 失败: {exc}")
            return

        raw_id = str(data.get("drone_id", "drone_1"))
        action = str(data.get("action", "HOVER")).upper().strip()
        if action not in _VALID_ACTIONS:
            self.get_logger().warn(f"[规划] 忽略未知战术动作: {action}")
            return

        targets = self.drone_ids if raw_id.upper() == "ALL" else [raw_id]
        for d_id in targets:
            if d_id not in self.drone_modes:
                continue
            cur_mode = self.drone_modes[d_id]
            airborne_now = (
                d_id in self.drone_kinematics and float(self.drone_kinematics[d_id][0][2]) > 1.2
            )
            if cur_mode == "GROUND" and not airborne_now:
                if action != "TAKEOFF":
                    continue
            elif action == "TAKEOFF" and cur_mode != "GROUND":
                continue
            self.drone_modes[d_id] = action
            self.get_logger().info(f"[规划] {d_id} 战术状态切换 -> {action}")

    def _plan_tick(self) -> None:
        now_s = self._now_sec()
        target_valid = self.estimator.is_valid(now_s)
        if target_valid:
            self.estimator.propagate_to(now_s)
            t_pos, t_vel, t_acc = self.estimator.get_kinematics_at(now_s)
        else:
            t_pos = np.zeros(3, dtype=float)
            t_vel = np.zeros(3, dtype=float)
            t_acc = np.zeros(3, dtype=float)

        cmds = self.planner.compute_swarm_targets(
            now_sec=now_s,
            swarm_ids=self.drone_ids,
            drone_modes=self.drone_modes,
            drone_states=self.drone_kinematics,
            target_valid=target_valid,
            target_pos=t_pos,
            target_vel=t_vel,
            target_acc=t_acc,
            predict_fn=lambda tau: self.estimator.predict_pv_ahead(tau),
        )

        pub_stamp = self._now_sec()
        for d_id, payload in cmds.items():
            x = max(-self.max_xy_dist, min(self.max_xy_dist, float(payload["x"])))
            y = max(-self.max_xy_dist, min(self.max_xy_dist, float(payload["y"])))
            z = float(payload["z"])
            out_json = {
                "drone_id": d_id,
                "stamp": pub_stamp,
                "x": round(x, 4),
                "y": round(y, 4),
                "z": round(z, 4),
                "action": str(payload["action"]),
                "t_go": payload.get("t_go", 0.0),
            }
            if target_valid:
                out_json["focus_x"] = round(float(t_pos[0]), 4)
                out_json["focus_y"] = round(float(t_pos[1]), 4)
                out_json["focus_z"] = round(float(t_pos[2]), 4)
            out_msg = String()
            out_msg.data = json.dumps(out_json)
            self.plan_pub.publish(out_msg)

        if now_s - self._last_diag_sec >= 0.5:
            self._last_diag_sec = now_s
            diag = {
                "stamp": round(now_s, 3),
                "modes": self.drone_modes,
                "estimator": self.estimator.diagnostics(now_s),
                "active_plans": {
                    k: {"t_go": v.get("t_go"), "reachable": v.get("reachable")}
                    for k, v in cmds.items()
                },
            }
            d_msg = String()
            d_msg.data = json.dumps(diag)
            self.diag_pub.publish(d_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
