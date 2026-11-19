"""ROS 2 适配层：只搬运消息、调用 P1/P2/P3，不在节点里写算法逻辑。

订阅：

- ``/vision/target_pose``（``geometry_msgs/PoseStamped``）：视觉目标，优先；
- ``/external/target_detected``（``geometry_msgs/PoseStamped``）：视觉不新鲜时兜底；
- ``/strategy/cmd``（``std_msgs/String`` JSON）：``drone_id`` + ``action``；
- ``/drone_<id>/state``（``nav_msgs/Odometry``）：己方 ENU 位置与源时间戳。

发布：

- ``/planning/target_pose``（``std_msgs/String`` JSON）：``drone_id``、``stamp``、
  ``x``、``y``、``z``、``action``；``stamp`` 是发布时刻的 ROS clock 秒数。

ROS 参数（launch 里的默认值）：

====================================  =========  ==========================================
参数                                  默认值      说明
====================================  =========  ==========================================
``drone_ids``                         drone_1..3  必须与中枢 ``swarm_config`` 一致
``publish_rate_hz``                   20.0       规划定时器频率，Hz
``v_max``                             6.0        宪章 V_max，m/s
``v_max_intercept``                   6.0        INTERCEPT 速度，m/s；宪章下同为 V_max
``v_max_attack``                      6.0        ATTACK 速度，m/s；宪章下同为 V_max
``t_go_max``                          8.0        最大拦截时间，s
``max_age``                           1.0        目标采样最大年龄，s
``max_pred_age``                      0.5        预测点最大年龄，s
``max_speed``                         15.0       目标估计速度上限，m/s
``min_distance``                      0.5        与目标最小距离，m
``d_tail``                            4.0        INTERCEPT 尾随距离，m
``min_alt``                           3.0        ATTACK 最低高度，m
``safe_alt``                          5.0        TRACK/INTERCEPT 最低高度，m
``max_alt``                           50.0       最高高度，m
``max_xy``                            1000.0     水平边界，m
``max_step``                          0.30       单帧最大跳变，m；0 表示不限制
``target_timeout``                    0.5        视觉/外部源切换阈值，s
``history_sec``                       3.0        目标采样窗口，s
``default_action``                    HOVER      未收到战术指令时的模式
``t_extrap_max``                      1.0        P3 外推上限，s
``n_enter``                           2          P3 进入更差级别所需帧数
``n_exit``                            1          P3 L3→L2 所需帧数
``n_recover``                         3          P3 回 NORMAL 所需帧数
``max_step_recover``                  0.2        恢复第一帧跳变上限，m
``n_recover_step``                    5          恢复期放开帧数
``freq_anomaly_enabled``              false      是否启用丢帧率诊断
``freq_anomaly_threshold``            0.5        丢帧率阈值
``freq_window_size``                  20         丢帧率窗口
``hold_point_source``                 last_valid L2 保持点来源
``ego_timeout``                       1.0        己方状态最大年龄，s；替代旧 state_timeout
``target_min_alt``                    0.0        目标输入最低高度，m
====================================  =========  ==========================================
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String

from algorithm_pkg.adapter_core import (
    DroneRuntime,
    TargetBuffer,
    choose_target,
    level_transition,
    payload,
    plan_drone,
    should_publish,
)
from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.plan_types import PLAN_ACTIONS
from algorithm_pkg.validators import CmdCode, validate_cmd


@dataclass(frozen=True)
class StateSnapshot:
    """一帧己方状态。

    Attributes:
        position: ENU 位置，米。
        stamp: 状态源时间戳，秒；无 header.stamp 时用接收时刻。
        received_at: 本地接收时刻，秒；仅用于日志排查。
    """

    position: tuple[float, float, float]
    stamp: float
    received_at: float


class AlgorithmNode(Node):
    """规划适配节点：订阅 A/B/D/E，发布 C。"""

    def __init__(self) -> None:
        super().__init__("algorithm_node")

        self._drone_ids = [
            str(item) for item in self.declare_parameter(
                "drone_ids",
                ["drone_1", "drone_2", "drone_3"],
            ).value
        ]
        self._publish_rate_hz = float(
            self.declare_parameter("publish_rate_hz", 20.0).value
        )
        self._max_age = float(self.declare_parameter("max_age", 1.0).value)
        self._max_speed = float(self.declare_parameter("max_speed", 15.0).value)
        self._target_timeout = float(
            self.declare_parameter("target_timeout", 0.5).value
        )
        self._history_sec = float(
            self.declare_parameter("history_sec", 3.0).value
        )
        self._default_action = str(
            self.declare_parameter("default_action", "HOVER").value
        ).strip().upper()

        # ! max_step 默认 0.30 = 宪章 V_max 6.0 m/s × 0.05s。
        # ! 0 仍保留为“不限幅”的兼容入口；联调/交付不要用 0。
        max_step_value = float(self.declare_parameter("max_step", 0.30).value)
        max_step = max_step_value if max_step_value > 0.0 else None

        self._params = PlanParams(
            v_max=float(self.declare_parameter("v_max", 6.0).value),
            v_max_intercept=float(
                self.declare_parameter("v_max_intercept", 6.0).value
            ),
            v_max_attack=float(
                self.declare_parameter("v_max_attack", 6.0).value
            ),
            t_go_max=float(self.declare_parameter("t_go_max", 8.0).value),
            min_distance=float(
                self.declare_parameter("min_distance", 0.5).value
            ),
            d_tail=float(self.declare_parameter("d_tail", 4.0).value),
            min_alt=float(self.declare_parameter("min_alt", 3.0).value),
            safe_alt=float(self.declare_parameter("safe_alt", 5.0).value),
            max_alt=float(self.declare_parameter("max_alt", 50.0).value),
            max_xy=float(self.declare_parameter("max_xy", 1000.0).value),
            max_step=max_step,
            max_pred_age=float(
                self.declare_parameter("max_pred_age", 0.5).value
            ),
            t_extrap_max=float(
                self.declare_parameter("t_extrap_max", 1.0).value
            ),
            n_enter=int(self.declare_parameter("n_enter", 2).value),
            n_exit=int(self.declare_parameter("n_exit", 1).value),
            n_recover=int(self.declare_parameter("n_recover", 3).value),
            max_step_recover=float(
                self.declare_parameter("max_step_recover", 0.2).value
            ),
            n_recover_step=int(
                self.declare_parameter("n_recover_step", 5).value
            ),
            freq_anomaly_enabled=bool(
                self.declare_parameter("freq_anomaly_enabled", False).value
            ),
            freq_anomaly_threshold=float(
                self.declare_parameter("freq_anomaly_threshold", 0.5).value
            ),
            freq_window_size=int(
                self.declare_parameter("freq_window_size", 20).value
            ),
            hold_point_source=str(
                self.declare_parameter("hold_point_source", "last_valid").value
            ),
            ego_timeout=float(
                self.declare_parameter("ego_timeout", 1.0).value
            ),
            target_min_alt=float(
                self.declare_parameter("target_min_alt", 0.0).value
            ),
        )

        self._vision = TargetBuffer(history_sec=self._history_sec)
        self._external = TargetBuffer(history_sec=self._history_sec)
        self._states: Dict[str, StateSnapshot] = {}
        self._modes: Dict[str, str] = {
            drone_id: self._default_action for drone_id in self._drone_ids
        }
        self._runtimes: Dict[str, DroneRuntime] = {
            drone_id: DroneRuntime.create(self._params)
            for drone_id in self._drone_ids
        }
        self._plan_pubs: Dict[str, object] = {}

        self.create_subscription(
            PoseStamped,
            "/vision/target_pose",
            self._on_vision,
            10,
        )
        self.create_subscription(
            PoseStamped,
            "/external/target_detected",
            self._on_external,
            10,
        )
        self.create_subscription(
            String,
            "/strategy/cmd",
            self._on_strategy,
            10,
        )

        for drone_id in self._drone_ids:
            self.create_subscription(
                Odometry,
                f"/{drone_id}/state",
                lambda msg, drone_id=drone_id: self._on_state(drone_id, msg),
                10,
            )
            self._plan_pubs[drone_id] = self.create_publisher(
                String,
                "/planning/target_pose",
                10,
            )

        self.create_timer(1.0 / self._publish_rate_hz, self._tick)

        self.get_logger().info(
            "algorithm_node 已启动：drone_ids=%s, max_step=%s, rate=%.1fHz"
            % (self._drone_ids, self._params.max_step, self._publish_rate_hz)
        )

    # ------------------------------------------------------------------
    # 订阅回调：只做消息→纯 Python 输入的搬运
    # ------------------------------------------------------------------
    def _on_vision(self, msg: PoseStamped) -> None:
        """接收视觉目标；非法时间戳帧直接丢弃。"""
        now = self._now()
        point = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        )
        problem = self._vision.offer(self._stamp_of(msg), point, now)
        if problem is not None:
            self.get_logger().warn(
                f"丢弃视觉帧：{problem}",
                throttle_duration_sec=1.0,
            )

    def _on_external(self, msg: PoseStamped) -> None:
        """接收外部引导目标；规则与视觉一致。"""
        now = self._now()
        point = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        )
        problem = self._external.offer(self._stamp_of(msg), point, now)
        if problem is not None:
            self.get_logger().warn(
                f"丢弃外部目标帧：{problem}",
                throttle_duration_sec=1.0,
            )

    def _on_state(self, drone_id: str, msg: Odometry) -> None:
        """接收己方状态。

        # ! 不在回调里拒绝“时间戳倒退”的状态：把原始 stamp 交给 P3 的
        # ! validate_ego 分类，才能走 P3 的 frame_discarded_time_reversed
        # ! 分支，并保证该帧不参与降级计数（P3 降级策略 §3）。
        # ! 只有 stamp 缺失（0）时才退回接收时刻，否则 E/F 回灌的旧帧
        # ! 会把己方状态一直“保活”。
        """
        now = self._now()
        stamp = self._stamp_of(msg)
        if stamp <= 0.0:
            stamp = now
            self.get_logger().warn(
                f"{drone_id} 的 state 没有 header.stamp，改用接收时刻判新鲜",
                throttle_duration_sec=5.0,
            )
        position = msg.pose.pose.position
        self._states[drone_id] = StateSnapshot(
            position=(
                float(position.x),
                float(position.y),
                float(position.z),
            ),
            stamp=stamp,
            received_at=now,
        )

    def _on_strategy(self, msg: String) -> None:
        """接收课题三战术指令；非法 JSON/动作一律拒绝，不改当前模式。"""
        status = validate_cmd(msg.data)
        if status.code is not CmdCode.FRESH:
            self.get_logger().warn(
                f"忽略非法指令：{status.reason}",
                throttle_duration_sec=2.0,
            )
            return

        drone_id = status.drone_id
        if drone_id not in self._modes:
            self.get_logger().warn(
                f"忽略未知无人机的指令：{drone_id}",
                throttle_duration_sec=2.0,
            )
            return

        previous = self._modes[drone_id]
        self._modes[drone_id] = status.action
        self.get_logger().info(
            f"{drone_id} 模式切换：{previous}->{status.action}"
        )

    # ------------------------------------------------------------------
    # 定时器：一帧规划
    # ------------------------------------------------------------------
    def _tick(self) -> None:
        """对所有飞机各跑一帧 P3，按需发布接口 C。"""
        now = self._now()
        target = choose_target(
            self._vision,
            self._external,
            now,
            self._target_timeout,
            self._max_age,
            self._max_speed,
        )
        if target is None and any(
            action in PLAN_ACTIONS for action in self._modes.values()
        ):
            self.get_logger().warn(
                "没有可用目标：视觉/外部都没有有效采样",
                throttle_duration_sec=2.0,
            )

        for drone_id in self._drone_ids:
            action = self._modes.get(drone_id, self._default_action)
            snapshot = self._states.get(drone_id)
            own = snapshot.position if snapshot is not None else None
            ego_stamp = snapshot.stamp if snapshot is not None else None

            runtime = self._runtimes[drone_id]
            result = plan_drone(
                runtime,
                action,
                own,
                target,
                ego_stamp,
                now,
                self._params,
            )

            # ! 中枢动作（TAKEOFF/HOVER/RETURN/RESUME_PATROL）由 P3 透传，
            # ! 不驱动 FSM；它们的 level 固定 NORMAL，不能拿来做级别变化
            # ! 日志，否则会误报“HOLDING->NORMAL”。
            if action in PLAN_ACTIONS:
                transition = level_transition(runtime.last_level, result)
                if transition is not None:
                    self.get_logger().info(
                        f"{drone_id} 降级级别变化：{transition}",
                        # ! 级别变化本身很少，但恢复期可能 0.15s 内两级；
                        # ! 1.0s 节流会吞掉 HOLDING->NORMAL，用 0.1s。
                        throttle_duration_sec=0.1,
                    )
                runtime.last_level = result.level

            if not should_publish(result):
                # 中枢动作（TAKEOFF/HOVER/RETURN）本来就不发点，不刷日志。
                if action in PLAN_ACTIONS:
                    self.get_logger().warn(
                        f"{drone_id} 本帧不发点：level={result.level} "
                        f"reason={result.reason}",
                        throttle_duration_sec=1.0,
                    )
                continue

            if result.guard_notes:
                self.get_logger().debug(
                    f"{drone_id} 护栏裁剪：{'; '.join(result.guard_notes)}",
                    throttle_duration_sec=5.0,
                )

            self._publish_plan(drone_id, now, result)

    def _publish_plan(self, drone_id: str, now: float, result) -> None:
        """发布接口 C；``stamp`` 固定用发布时刻。"""
        data = payload(drone_id, now, result)
        self._plan_pubs[drone_id].publish(
            String(data=json.dumps(data, ensure_ascii=False))
        )

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _stamp_of(msg) -> float:
        """把 ROS 消息头时间戳转成秒。"""
        return float(msg.header.stamp.sec) + float(
            msg.header.stamp.nanosec
        ) * 1e-9

    def _now(self) -> float:
        """当前 ROS 时钟，秒。"""
        return self.get_clock().now().nanoseconds * 1e-9


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
        rclpy.shutdown()


if __name__ == "__main__":
    main()
