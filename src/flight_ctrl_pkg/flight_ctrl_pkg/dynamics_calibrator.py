# -*- coding: utf-8 -*-
"""
dynamics_calibrator.py —— 机型物理动力学阶跃响应实测脚本

用途（配合《内部分工细节 V2.0》）：
  在 PX4 + Gazebo 仿真启动后，通过向 /drone_1/fmu/in/trajectory_setpoint 与 /drone_1/action
  发布标准阶跃指令，实时记录 /drone_1/state 响应曲线，自动测算并打印：
    - V_cruise（常规 TRACK/HOVER 限幅最大平飞速度）
    - V_max（ATTACK 解除限幅最大平飞速度）
    - a_max（从悬停加速到巡航/极速时的最大有效加速度）
    - yaw_rate（偏航角速度上限）
    - R_min（最小转弯半径近似估算 V_cruise^2 / a_lat 或 V_cruise / yaw_rate）
"""

import copy
import math
import time
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Empty, String


class DynamicsCalibrator(Node):
    """自动阶跃响应标定节点。"""

    def __init__(self) -> None:
        super().__init__('dynamics_calibrator')

        self.declare_parameter('drone_id', 'drone_1')
        self.declare_parameter('cruise_alt', 8.0)
        self.declare_parameter('step_dist', 30.0)

        self.drone_id: str = str(self.get_parameter('drone_id').value)
        self.cruise_alt: float = float(self.get_parameter('cruise_alt').value)
        self.step_dist: float = float(self.get_parameter('step_dist').value)

        self.state: Optional[Odometry] = None
        self.samples: List[Tuple[float, float, float, float, float, float, float]] = []

        self.create_subscription(Odometry, f'/{self.drone_id}/state', self._state_cb, 20)
        self.target_pub = self.create_publisher(
            PoseStamped, f'/{self.drone_id}/fmu/in/trajectory_setpoint', 10
        )
        self.action_pub = self.create_publisher(String, f'/{self.drone_id}/action', 10)
        self.hb_pub = self.create_publisher(Empty, f'/{self.drone_id}/heartbeat', 10)

        self.phase: str = 'WAIT_STATE'
        self.phase_start: float = time.time()
        self.origin_xy: Tuple[float, float] = (0.0, 0.0)
        self.metrics: dict = {}

        self.create_timer(0.05, self._tick)
        self.get_logger().info(f'[{self.drone_id}] 动力学标定节点已启动，等待 /{self.drone_id}/state...')

    def _state_cb(self, msg: Odometry) -> None:
        self.state = copy.deepcopy(msg)
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.samples.append((time.time(), p.x, p.y, p.z, v.x, v.y, yaw))

    def _send_target(self, x: float, y: float, z: float, action: str) -> None:
        act = String()
        act.data = action
        self.action_pub.publish(act)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        self.target_pub.publish(msg)

    def _analyze_segment(self, t0: float, t1: float) -> Tuple[float, float, float]:
        seg = [s for s in self.samples if t0 <= s[0] <= t1]
        if len(seg) < 5:
            return 0.0, 0.0, 0.0
        speeds = [math.hypot(s[4], s[5]) for s in seg]
        max_v = max(speeds)
        max_a = 0.0
        max_yaw_rate = 0.0
        prev_idx = 0
        for i in range(1, len(seg)):
            dt = seg[i][0] - seg[prev_idx][0]
            if dt < 0.10:
                continue
            acc = abs(speeds[i] - speeds[prev_idx]) / dt
            if acc > max_a:
                max_a = acc
            dyaw = (seg[i][6] - seg[prev_idx][6] + math.pi) % (2.0 * math.pi) - math.pi
            yr = abs(dyaw) / dt
            if yr > max_yaw_rate:
                max_yaw_rate = yr
            prev_idx = i
        return max_v, max_a, max_yaw_rate

    def _tick(self) -> None:
        self.hb_pub.publish(Empty())
        now = time.time()
        if self.state is None:
            return

        pos = self.state.pose.pose.position
        elapsed = now - self.phase_start

        if self.phase == 'WAIT_STATE':
            self.origin_xy = (pos.x, pos.y)
            self.phase = 'TAKEOFF'
            self.phase_start = now
            self.get_logger().info(f'阶段 1/4：起飞至 {self.cruise_alt:.1f}m...')
            return

        if self.phase == 'TAKEOFF':
            self._send_target(self.origin_xy[0], self.origin_xy[1], self.cruise_alt, 'TAKEOFF')
            if abs(pos.z - self.cruise_alt) < 0.5 and elapsed > 4.0:
                self.phase = 'STEP_TRACK'
                self.phase_start = now
                self.get_logger().info('阶段 2/4：测试 TRACK 巡航阶跃响应...')
            return

        if self.phase == 'STEP_TRACK':
            self._send_target(
                self.origin_xy[0] + self.step_dist,
                self.origin_xy[1],
                self.cruise_alt,
                'TRACK',
            )
            if elapsed > 10.0:
                v_c, a_c, yr_c = self._analyze_segment(self.phase_start, now)
                self.metrics['V_cruise'] = v_c
                self.metrics['a_cruise'] = a_c
                self.metrics['yaw_rate_c'] = yr_c
                self.phase = 'STEP_ATTACK'
                self.phase_start = now
                self.get_logger().info('阶段 3/4：测试 ATTACK 全速冲刺阶跃响应...')
            return

        if self.phase == 'STEP_ATTACK':
            self._send_target(
                self.origin_xy[0] + self.step_dist,
                self.origin_xy[1] + self.step_dist,
                self.cruise_alt,
                'ATTACK',
            )
            if elapsed > 8.0:
                v_m, a_m, yr_m = self._analyze_segment(self.phase_start, now)
                self.metrics['V_max'] = v_m
                self.metrics['a_max'] = max(self.metrics.get('a_cruise', 0.0), a_m)
                self.metrics['yaw_rate'] = max(self.metrics.get('yaw_rate_c', 0.0), yr_m)
                self.phase = 'RETURN'
                self.phase_start = now
                self._print_report()
                self.get_logger().info('阶段 4/4：返航降落...')
            return

        if self.phase == 'RETURN':
            if elapsed < 6.0:
                self._send_target(self.origin_xy[0], self.origin_xy[1], self.cruise_alt, 'RETURN')
            else:
                self._send_target(self.origin_xy[0], self.origin_xy[1], 0.0, 'RETURN')
            speed_xy = math.hypot(
                self.state.twist.twist.linear.x, self.state.twist.twist.linear.y
            )
            if (pos.z < 0.38 and speed_xy < 0.25 and elapsed > 8.0) or elapsed > 20.0:
                self.get_logger().info('标定完成，安全落地！')
                raise SystemExit(0)

    def _print_report(self) -> None:
        v_cruise = self.metrics.get('V_cruise', 0.0)
        v_max = self.metrics.get('V_max', 0.0)
        a_max = self.metrics.get('a_max', 0.0)
        yr_deg = math.degrees(self.metrics.get('yaw_rate', 0.0))
        r_min = (v_cruise ** 2 / max(a_max, 0.5)) if v_cruise > 0.1 else 0.0
        report = (
            '\n==================== 实测动力学参数报告 ====================\n'
            f'  V_cruise : {v_cruise:.2f} m/s\n'
            f'  V_max    : {v_max:.2f} m/s\n'
            f'  a_max    : {a_max:.2f} m/s^2\n'
            f'  yaw_rate : {yr_deg:.1f} deg/s\n'
            f'  R_min(估): {r_min:.2f} m\n'
            '============================================================'
        )
        print(report, flush=True)
        self.get_logger().info(report)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DynamicsCalibrator()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
