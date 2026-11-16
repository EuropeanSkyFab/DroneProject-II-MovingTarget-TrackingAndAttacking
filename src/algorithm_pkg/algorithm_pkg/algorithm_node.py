"""ROS 2 适配层：把纯算法核心接到课题二现有话题上。

本节点只做三件事：
1. 收目标位置（视觉优先，外部检测兜底）和己方状态；
2. 调 ``algorithm_core`` 估速、解拦截/跟踪点；
3. 按中枢约定的 JSON 发布 ``/planning/target_pose``。

纯算法仍然放在 ``algorithm_core.py``，不 import rclpy；
本文件只负责 ROS 消息、参数、定时器和日志。
"""

from __future__ import annotations

import json
import math
from collections import deque
from typing import Deque, Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from algorithm_pkg.algorithm_core import (
    TargetSample,
    apply_constraints,
    predict_target,
    solve_intercept,
)

# 只有这三种模式需要规划节点发点；其余模式由中枢自己生成坐标。
_PLAN_ACTIONS = ("TRACK", "INTERCEPT", "ATTACK")


class AlgorithmNode(Node):
    """规划算法 ROS 2 节点。"""

    def __init__(self) -> None:
        super().__init__("algorithm_node")

        # ------------------------------ 参数 ------------------------------
        self.declare_parameter("drone_ids", ["drone_1", "drone_2", "drone_3"])
        self.declare_parameter("publish_rate_hz", 20.0)
        self.declare_parameter("v_max", 6.0)
        self.declare_parameter("t_go_max", 8.0)
        self.declare_parameter("max_age", 1.0)
        self.declare_parameter("max_speed", 15.0)
        self.declare_parameter("min_distance", 0.5)
        self.declare_parameter("d_tail", 4.0)
        self.declare_parameter("min_alt", 3.0)
        self.declare_parameter("safe_alt", 5.0)
        self.declare_parameter("max_alt", 50.0)
        self.declare_parameter("max_xy", 1000.0)
        self.declare_parameter("max_step", 0.0)  # 0 表示不限制
        self.declare_parameter("state_timeout", 1.0)
        self.declare_parameter("target_timeout", 0.5)
        self.declare_parameter("history_sec", 3.0)
        self.declare_parameter("default_action", "HOVER")

        p = self.get_parameter
        self.drone_ids = list(p("drone_ids").value)
        self.publish_rate_hz = float(p("publish_rate_hz").value)
        self.v_max = float(p("v_max").value)
        self.t_go_max = float(p("t_go_max").value)
        self.max_age = float(p("max_age").value)
        self.max_speed = float(p("max_speed").value)
        self.min_distance = float(p("min_distance").value)
        self.d_tail = float(p("d_tail").value)
        self.min_alt = float(p("min_alt").value)
        self.safe_alt = float(p("safe_alt").value)
        self.max_alt = float(p("max_alt").value)
        self.max_xy = float(p("max_xy").value)
        self.state_timeout = float(p("state_timeout").value)
        self.target_timeout = float(p("target_timeout").value)
        self.history_sec = float(p("history_sec").value)

        max_step = float(p("max_step").value)
        self.max_step: Optional[float] = max_step if max_step > 0.0 else None

        if self.publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        if not self.drone_ids:
            raise ValueError("drone_ids must not be empty")

        self.default_action = str(p("default_action").value).strip().upper()
        if self.default_action not in (
            "TAKEOFF",
            "HOVER",
            "RETURN",
            "TRACK",
            "INTERCEPT",
            "ATTACK",
            "RESUME_PATROL",
        ):
            self.default_action = "HOVER"

        # ------------------------------ 缓存 ------------------------------
        # 视觉和外部检测分开存，避免两个来源的时间戳混在一起拟合。
        self._vision_samples: Deque[TargetSample] = deque(maxlen=200)
        self._external_samples: Deque[TargetSample] = deque(maxlen=200)
        self._last_target_rx = {"vision": -1.0, "external": -1.0}

        self._states: dict[str, Optional[Odometry]] = {
            drone_id: None for drone_id in self.drone_ids
        }
        self._state_rx: dict[str, float] = {
            drone_id: -1.0 for drone_id in self.drone_ids
        }
        self._modes: dict[str, str] = {
            drone_id: self.default_action for drone_id in self.drone_ids
        }

        # ------------------------------ 订阅 ------------------------------
        self.create_subscription(
            PoseStamped, "/vision/target_pose", self._on_vision, 10
        )
        self.create_subscription(
            PoseStamped,
            "/external/target_detected",
            self._on_external,
            10,
        )
        self.create_subscription(String, "/strategy/cmd", self._on_strategy, 10)

        for drone_id in self.drone_ids:
            self.create_subscription(
                Odometry,
                f"/{drone_id}/state",
                lambda msg, d_id=drone_id: self._on_state(msg, d_id),
                10,
            )

        # ------------------------------ 发布 ------------------------------
        self._plan_pubs = {
            drone_id: self.create_publisher(
                String, "/planning/target_pose", 10
            )
            for drone_id in self.drone_ids
        }

        period = 1.0 / self.publish_rate_hz
        self.create_timer(period, self._tick)

        self.get_logger().info(
            "algorithm_node 已启动："
            f"drones={self.drone_ids}, v_max={self.v_max:.1f} m/s, "
            f"rate={self.publish_rate_hz:.1f} Hz, "
            f"default_action={self.default_action}"
        )

    # ------------------------------ 工具函数 ------------------------------
    def _now(self) -> float:
        """返回当前 ROS 时钟秒数。"""
        return self.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def _stamp_seconds(msg: PoseStamped) -> float:
        """把 PoseStamped.header.stamp 转成秒。"""
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _record_target(
        self,
        samples: Deque[TargetSample],
        msg: PoseStamped,
        source: str,
    ) -> None:
        """缓存一帧目标位置；时间戳异常时按收到时刻处理。"""
        now = self._now()
        stamp = self._stamp_seconds(msg)
        if stamp <= 0.0 or stamp > now:
            # 图像时间戳缺失或轻微超前时，用收到时刻兜底，避免整帧被丢。
            stamp = now

        # 太老的帧不进入拟合窗口。
        if now - stamp > self.history_sec:
            return

        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        z = float(msg.pose.position.z)
        if not all(math.isfinite(value) for value in (x, y, z)):
            return

        samples.append(TargetSample(stamp=stamp, x=x, y=y, z=z))
        self._last_target_rx[source] = now

    def _select_samples(self, now: float) -> Deque[TargetSample]:
        """视觉新鲜时用视觉，否则用外部检测。"""
        vision_fresh = (
            len(self._vision_samples) > 0
            and now - self._last_target_rx["vision"] <= self.target_timeout
        )
        if vision_fresh:
            return self._vision_samples

        external_fresh = (
            len(self._external_samples) > 0
            and now - self._last_target_rx["external"] <= self.target_timeout
        )
        if external_fresh:
            return self._external_samples

        return self._vision_samples

    def _own_position(self, drone_id: str, now: float) -> Optional[tuple[float, float, float]]:
        """取己方当前位置；状态缺失或过期时返回 None。"""
        state = self._states.get(drone_id)
        if state is None:
            return None
        if now - self._state_rx[drone_id] > self.state_timeout:
            return None

        pos = state.pose.pose.position
        own = (float(pos.x), float(pos.y), float(pos.z))
        if not all(math.isfinite(value) for value in own):
            return None
        return own

    # ------------------------------ 回调 ------------------------------
    def _on_vision(self, msg: PoseStamped) -> None:
        self._record_target(self._vision_samples, msg, "vision")

    def _on_external(self, msg: PoseStamped) -> None:
        self._record_target(self._external_samples, msg, "external")

    def _on_state(self, msg: Odometry, drone_id: str) -> None:
        self._states[drone_id] = msg
        self._state_rx[drone_id] = self._now()

    def _on_strategy(self, msg: String) -> None:
        """跟踪课题三的战术指令；TAKEOFF/HOVER/RETURN 不发规划点。"""
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().error(f"解析 /strategy/cmd 失败: {exc}")
            return

        drone_id = str(data.get("drone_id", "drone_1"))
        if drone_id not in self._modes:
            self.get_logger().warn(f"忽略未知无人机 {drone_id} 的战术指令")
            return

        action = str(data.get("action", "HOVER")).strip().upper()
        if action not in (
            "TAKEOFF",
            "HOVER",
            "RETURN",
            "TRACK",
            "INTERCEPT",
            "ATTACK",
            "RESUME_PATROL",
        ):
            self.get_logger().warn(f"忽略未知动作 {action}")
            return

        self._modes[drone_id] = action
        self.get_logger().info(f"{drone_id} 规划模式切换为 {action}")

    # ------------------------------ 主循环 ------------------------------
    def _tick(self) -> None:
        now = self._now()

        samples = self._select_samples(now)
        if not samples:
            return

        target = predict_target(
            samples,
            now,
            max_age=self.max_age,
            max_speed=self.max_speed,
        )
        if target is None or not target.valid:
            self.get_logger().warn(
                "目标估计不可用（样本不足、过期或超速），本帧不发布",
                throttle_duration_sec=2.0,
            )
            return

        for drone_id in self.drone_ids:
            action = self._modes.get(drone_id, self.default_action)
            if action not in _PLAN_ACTIONS:
                continue

            # 拦截/攻击依赖目标速度。只有一帧时 predict_target 会给出
            # velocity_valid=False（速度按 0 处理），此时发点等于瞄准
            # 目标当前位置，容易在联调初期产生误导性的“拦截点”。
            if action in ("INTERCEPT", "ATTACK") and not target.velocity_valid:
                self.get_logger().warn(
                    f"{drone_id} {action} 等待目标速度估计（至少两帧）",
                    throttle_duration_sec=1.0,
                )
                continue

            own = self._own_position(drone_id, now)
            if own is None:
                self.get_logger().warn(
                    f"{drone_id} 状态缺失或过期，本帧不发布",
                    throttle_duration_sec=2.0,
                )
                continue

            solution = solve_intercept(
                own,
                target,
                self.v_max,
                mode=action,
                t_go_max=self.t_go_max,
                min_distance=self.min_distance,
                d_tail=self.d_tail,
            )
            if solution.point is None:
                # ATTACK 无解时中枢会保留上一帧最多 2 秒；
                # 这里只记录原因，不伪造尾随点。
                self.get_logger().warn(
                    f"{drone_id} {action} 无可用点: {solution.reason}",
                    throttle_duration_sec=1.0,
                )
                continue

            point = apply_constraints(
                solution.point,
                action,
                own=own,
                min_alt=self.min_alt,
                safe_alt=self.safe_alt,
                max_alt=self.max_alt,
                max_xy=self.max_xy,
                max_step=self.max_step,
            )
            if point is None:
                continue

            self._publish_plan(drone_id, action, point, now)

    def _publish_plan(
        self,
        drone_id: str,
        action: str,
        point: tuple[float, float, float],
        now: float,
    ) -> None:
        """按 /planning/target_pose 的 JSON 契约发布一个点。"""
        payload = {
            "drone_id": drone_id,
            "stamp": now,
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2]),
            "action": action,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self._plan_pubs[drone_id].publish(msg)


def main(args: Optional[list[str]] = None) -> None:
    """节点入口。"""
    rclpy.init(args=args)
    node = AlgorithmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
