"""ROS 2 适配层：把纯算法核心接到课题二现有话题上。

本节点只做三件事：
1. 收目标位置（视觉优先，外部检测兜底）和己方状态；
2. 调 ``algorithm_core`` 估速，再调 ``plan_policy.plan`` 决定本帧发不发点；
3. 按中枢约定的 JSON 发布 ``/planning/target_pose``。

纯算法放在 ``algorithm_core.py`` / ``plan_*.py``，都不 import rclpy；
本文件只负责 ROS 消息、参数、定时器、日志和“是否真的发布”。

ROS 参数说明（长度 m，速度 m/s，时间 s；默认值与 ``declare_parameter``
一致）：

- ``drone_ids``：参与规划和发布的无人机 ID 列表；同时决定订阅哪些
  ``/<id>/state``、创建哪些发布器。
- ``publish_rate_hz``：主循环频率，默认 20 Hz；应与飞控控制周期一致。
- ``v_max``：兜底最大速度，默认 6.0 m/s；只在不使用专用速度时生效。
- ``v_max_intercept``：INTERCEPT 专用速度，默认 7.6 m/s。
- ``v_max_attack``：ATTACK 专用速度，默认 12.4 m/s。
- ``t_go_max``：拦截/攻击解的最大预测时间，默认 8.0 s；超过视为不可达。
- ``max_age``：P1 估速允许的最大目标年龄，默认 1.0 s。
- ``max_pred_age``：P2 发点前的最终年龄门限，默认 0.5 s；比 ``max_age``
  更严格，避免拿旧目标算拦截。
- ``max_speed``：目标速度上限，默认 15.0 m/s；超过判为估计无效。
- ``min_distance``：视为“已到达/命中”的距离，默认 0.5 m。
- ``d_tail``：INTERCEPT 不可达时尾随点与目标的距离，默认 4.0 m。
- ``min_alt``：中枢允许的最低高度，默认 3.0 m；也是 ATTACK 的下限。
- ``safe_alt``：TRACK/INTERCEPT 的安全高度，默认 5.0 m。
- ``max_alt``：最高高度，默认 50.0 m。
- ``max_xy``：水平边界，默认 1000.0 m。
- ``max_step``：单帧最大位移，默认 0.0（不限制）；仅调试/保守测试用。
- ``state_timeout``：己方状态过期时间，默认 1.0 s。
- ``target_timeout``：视觉/外部检测源的“新鲜度”，默认 0.5 s；只决定
  用哪个源，不是最终发点年龄门限。
- ``history_sec``：目标样本保留/拟合窗口，默认 3.0 s。
- ``default_action``：每架无人机启动时的模式，默认 HOVER；非法值回退
  HOVER。

注意：
- INTERCEPT/ATTACK 的 7.6/12.4 m/s 来自飞控 20 Hz 的 ``max_step``
  （0.38/0.62 m/帧），不是规划节点的发布频率。
- ``max_step`` 为 0 时不做单帧裁剪；真正的速度/加速度约束仍在飞控。
"""

from __future__ import annotations

import json
import math
from collections import deque
from typing import Deque, Optional, Union

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from algorithm_pkg.algorithm_core import TargetSample, predict_target
from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.plan_policy import plan
from algorithm_pkg.plan_types import PLAN_ACTIONS


class AlgorithmNode(Node):
    """规划算法 ROS 2 节点。

    参数含义见模块 docstring 的“ROS 参数说明”；本类只负责读取参数、
    建立订阅/发布器和按定时器调用纯策略。
    """

    def __init__(self) -> None:
        """读取参数、建立缓存/订阅/发布器，并启动主循环定时器。"""
        super().__init__("algorithm_node")

        # ------------------------------ 参数 ------------------------------
        # 参与规划的无人机；同时决定状态订阅和规划发布器。
        self.declare_parameter("drone_ids", ["drone_1", "drone_2", "drone_3"])
        # 主循环频率；20Hz 与飞控控制周期对齐。
        self.declare_parameter("publish_rate_hz", 20.0)

        # 速度参数（m/s）：v_max 是兜底，INTERCEPT/ATTACK 各用专用上限。
        # 7.6/12.4 对应飞控 20Hz 的 0.38/0.62 m/帧。
        self.declare_parameter("v_max", 6.0)
        self.declare_parameter("v_max_intercept", 7.6)
        self.declare_parameter("v_max_attack", 12.4)
        # 拦截解的最大预测时间（s）；超过则认为不可达。
        self.declare_parameter("t_go_max", 8.0)

        # 目标年龄门限（s）：max_age 给 P1 估速，max_pred_age 是发点前
        # 的最终门限，后者更严格。
        self.declare_parameter("max_age", 1.0)
        self.declare_parameter("max_pred_age", 0.5)
        # 目标速度上限（m/s）；超过判为估计无效。
        self.declare_parameter("max_speed", 15.0)

        # 几何/安全边界（m）：min_distance 命中距离，d_tail 尾随距离。
        self.declare_parameter("min_distance", 0.5)
        self.declare_parameter("d_tail", 4.0)
        # 高度边界（m）：ATTACK 用 min_alt，TRACK/INTERCEPT 用 safe_alt。
        self.declare_parameter("min_alt", 3.0)
        self.declare_parameter("safe_alt", 5.0)
        self.declare_parameter("max_alt", 50.0)
        # 水平边界（m）。
        self.declare_parameter("max_xy", 1000.0)
        # 单帧最大位移（m/帧）；0 表示不限制，仅调试用。
        self.declare_parameter("max_step", 0.0)

        # 状态/目标时效（s）：state_timeout 管己方状态，
        # target_timeout 只负责选择视觉/外部源。
        self.declare_parameter("state_timeout", 1.0)
        self.declare_parameter("target_timeout", 0.5)
        # 目标样本保留窗口（s）。
        self.declare_parameter("history_sec", 3.0)
        # 启动模式；非法值在下面回退为 HOVER。
        self.declare_parameter("default_action", "HOVER")

        p = self.get_parameter
        self.drone_ids = list(p("drone_ids").value)
        self.publish_rate_hz = float(p("publish_rate_hz").value)
        self.v_max = float(p("v_max").value)
        self.v_max_intercept = float(p("v_max_intercept").value)
        self.v_max_attack = float(p("v_max_attack").value)
        self.t_go_max = float(p("t_go_max").value)
        self.max_age = float(p("max_age").value)
        self.max_pred_age = float(p("max_pred_age").value)
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
        if self.v_max <= 0.0:
            raise ValueError("v_max must be positive")
        if self.v_max_intercept <= 0.0:
            self.v_max_intercept = self.v_max
        if self.v_max_attack <= 0.0:
            self.v_max_attack = self.v_max
        if not self.drone_ids:
            raise ValueError("drone_ids must not be empty")

        # P2 的参数对象在启动时构造一次；非法值直接让节点启动失败，
        # 比飞到一半才发现速度为 0 更安全。
        self.plan_params = PlanParams(
            v_max=self.v_max,
            v_max_intercept=self.v_max_intercept,
            v_max_attack=self.v_max_attack,
            t_go_max=self.t_go_max,
            min_distance=self.min_distance,
            d_tail=self.d_tail,
            min_alt=self.min_alt,
            safe_alt=self.safe_alt,
            max_alt=self.max_alt,
            max_xy=self.max_xy,
            max_step=self.max_step,
            max_pred_age=self.max_pred_age,
        )

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
        # 记录 state.header.stamp；center 会把同一 topic 的旧 state 转发回来，
        # 不能只靠接收时刻判断新鲜度。
        self._state_stamp: dict[str, float] = {
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
        self.create_subscription(
            String, "/strategy/cmd", self._on_strategy, 10
        )

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
            f"v_intercept={self.v_max_intercept:.1f} m/s, "
            f"v_attack={self.v_max_attack:.1f} m/s, "
            f"max_pred_age={self.max_pred_age:.2f}s, "
            f"rate={self.publish_rate_hz:.1f} Hz, "
            f"default_action={self.default_action}"
        )

    # ------------------------------ 工具函数 ------------------------------
    def _now(self) -> float:
        """返回当前 ROS 时钟秒数。"""
        return self.get_clock().now().nanoseconds / 1e9

    @staticmethod
    def _stamp_seconds(msg: Union[PoseStamped, Odometry]) -> float:
        """把带 header 的 ROS 消息时间戳转成秒。"""
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
        vision_fresh = len(self._vision_samples) > 0 and (
            now - self._last_target_rx["vision"] <= self.target_timeout
        )
        if vision_fresh:
            return self._vision_samples

        external_fresh = len(self._external_samples) > 0 and (
            now - self._last_target_rx["external"] <= self.target_timeout
        )
        if external_fresh:
            return self._external_samples

        return self._vision_samples

    def _own_position(
        self,
        drone_id: str,
        now: float,
    ) -> Optional[tuple[float, float, float]]:
        """取己方当前位置；状态缺失或过期时返回 None。"""
        state = self._states.get(drone_id)
        if state is None:
            return None

        stamp = self._state_stamp.get(drone_id, -1.0)
        reference = stamp if stamp > 0.0 else self._state_rx[drone_id]
        if now - reference > self.state_timeout:
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
        """缓存己方状态；用 header 时间戳拒绝被中枢回灌的旧状态。"""
        now = self._now()
        stamp = self._stamp_seconds(msg)
        last_stamp = self._state_stamp[drone_id]

        if stamp <= 0.0 or stamp > now:
            # mock/旧桥接可能不带时间戳；退回按接收时刻判断。
            # 但一旦收到过有效 header，就不能让“无时间戳/未来时间戳”
            # 的帧把新状态覆盖回旧状态。
            if last_stamp > 0.0:
                return
            stamp = 0.0

        if stamp > 0.0 and last_stamp > 0.0 and stamp + 1e-6 < last_stamp:
            # center 会把同一 topic 的 state 再转发一次；旧帧不能覆盖新帧。
            return

        self._state_stamp[drone_id] = stamp
        self._state_rx[drone_id] = now
        self._states[drone_id] = msg

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
        target = None
        if samples:
            target = predict_target(
                samples,
                now,
                max_age=self.max_age,
                max_speed=self.max_speed,
            )

        if target is None or not target.valid:
            self.get_logger().warn(
                "目标估计不可用（样本不足、过期或超速）；"
                "TRACK/INTERCEPT 本帧不发点，ATTACK 会发保持点",
                throttle_duration_sec=2.0,
            )

        for drone_id in self.drone_ids:
            action = self._modes.get(drone_id, self.default_action)
            if action not in PLAN_ACTIONS:
                # TAKEOFF/HOVER/RETURN 由中枢生成；RESUME_PATROL 未定义。
                continue

            own = self._own_position(drone_id, now)
            result = plan(action, own, target, self.plan_params)

            if result.point is None:
                # ATTACK 无解时，中枢会保留上一帧最多 2 秒。纯 P2 只返回
                # “不发点”，适配层再补一个安全保持点，避免飞机继续飞向
                # 已经失效的旧攻击点；这不是 P2 的尾随点。
                if action == "ATTACK" and own is not None:
                    hold = self._hold_point(own)
                    self._publish_plan(drone_id, "ATTACK", hold, now)
                    self.get_logger().warn(
                        f"{drone_id} ATTACK 无解({result.reason})，发当前位置保持",
                        throttle_duration_sec=1.0,
                    )
                else:
                    self.get_logger().warn(
                        f"{drone_id} {action} 本帧不发点: {result.reason}",
                        throttle_duration_sec=1.0,
                    )
                continue

            self._publish_plan(drone_id, result.action, result.point, now)

    def _hold_point(
        self,
        own: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        """把“当前位置保持点”收进中枢允许的安全边界。"""
        x, y, z = own
        x = min(max(x, -self.max_xy), self.max_xy)
        y = min(max(y, -self.max_xy), self.max_xy)
        z = min(max(z, self.min_alt), self.max_alt)
        return (x, y, z)

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
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
