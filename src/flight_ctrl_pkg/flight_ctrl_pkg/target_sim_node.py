# -*- coding: utf-8 -*-
"""
target_sim_node.py —— 红方高机动三维逃逸靶机控制器与外部宏观雷达模拟器（接口 B / topic1_adapter）

核心职责：
  1. 通过 gz.transport13 订阅 Gazebo 中 red_moving_target 的实时物理真值位姿，
     执行 25Hz 闭环航向与三维速度伺服控制 (/model/red_moving_target/cmd_vel)，
     支持 evasive_weave / figure8 / racetrack / circle / hover 五种标准机动剖面；
     同时支持无 Gazebo 环境下的纯运动学模式 (sim_backend='kinematic')。
  2. 发布精确地面真值话题 /sim/target_gt (nav_msgs/Odometry, ENU) 供科学评测与脱靶量审计。
  3. 作为《课题二内部接口规范 V3.0》中的 topic1_adapter，以可配置频率和受控随机种子高斯噪声
     发布接口 B 话题 /external/target_detected (geometry_msgs/PoseStamped, ENU)。
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _quat_to_yaw_enu(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_to_quat_enu(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


class TargetSimNode(Node):
    """三维机动红方靶机闭环驱动与外部雷达引导发布节点。"""

    def __init__(self) -> None:
        super().__init__('target_sim_node')

        self.declare_parameter('sim_backend', 'auto')  # 'auto' | 'gazebo' | 'kinematic'
        self.declare_parameter('gz_world', 'moving_target_arena')
        self.declare_parameter('target_model', 'red_moving_target')
        self.declare_parameter('profile', 'evasive_weave')  # evasive_weave | figure8 | circle | hover
        self.declare_parameter('speed_scale', 1.0)
        self.declare_parameter('center_x', 20.5)
        self.declare_parameter('center_y', 2.5)
        self.declare_parameter('base_z', 5.7)
        self.declare_parameter('amp_z', 0.85)
        self.declare_parameter('control_hz', 25.0)
        self.declare_parameter('radar_hz', 5.0)
        self.declare_parameter('enable_radar', True)
        self.declare_parameter('radar_noise_xy', 0.28)
        self.declare_parameter('radar_noise_z', 0.32)
        self.declare_parameter('random_seed', 42)

        p = self.get_parameter
        self.sim_backend: str = str(p('sim_backend').value)
        self.gz_world: str = str(p('gz_world').value)
        self.target_model: str = str(p('target_model').value)
        self.profile: str = str(p('profile').value)
        self.speed_scale: float = max(0.0, float(p('speed_scale').value))
        self.cx: float = float(p('center_x').value)
        self.cy: float = float(p('center_y').value)
        self.base_z: float = float(p('base_z').value)
        self.amp_z: float = float(p('amp_z').value)
        self.control_hz: float = max(5.0, float(p('control_hz').value))
        self.radar_hz: float = max(0.5, float(p('radar_hz').value))
        self.enable_radar: bool = bool(p('enable_radar').value)
        self.noise_xy: float = max(0.0, float(p('radar_noise_xy').value))
        self.noise_z: float = max(0.0, float(p('radar_noise_z').value))
        self.rng = np.random.default_rng(int(p('random_seed').value))

        self._lock = threading.Lock()
        self._gt_pos = np.array([6.8, 0.5, 5.5], dtype=float)
        self._gt_vel = np.zeros(3, dtype=float)
        self._gt_yaw: float = 0.0
        self._prev_gz_pos: Optional[np.ndarray] = None
        self._prev_gz_time: float = 0.0
        self._sim_t0: Optional[float] = None
        self._latest_sim_t: float = 0.0
        self._gz_pose_received: bool = False
        self._t0: float = time.time()
        self._last_radar_pub: float = 0.0

        self.gt_pub = self.create_publisher(Odometry, '/sim/target_gt', 20)
        self.radar_pub = self.create_publisher(PoseStamped, '/external/target_detected', 10)

        self._gz_node = None
        self._gz_cmd_pub = None
        self._twist_cls = None
        if self.sim_backend in ('auto', 'gazebo'):
            self._init_gz_interface()

        self.dt = 1.0 / self.control_hz
        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f'target_sim_node 已启动: profile={self.profile}, backend={self.sim_backend}, '
            f'radar={self.enable_radar}({self.radar_hz:.1f}Hz, seed={int(p("random_seed").value)})'
        )

    def _init_gz_interface(self) -> None:
        try:
            import gz.transport13 as gzt
            from gz.msgs10 import pose_v_pb2, twist_pb2

            self._twist_cls = twist_pb2.Twist
            self._gz_node = gzt.Node()
            pose_topic = f'/world/{self.gz_world}/dynamic_pose/info'
            cmd_topic = f'/model/{self.target_model}/cmd_vel'
            self._gz_node.subscribe(pose_v_pb2.Pose_V, pose_topic, self._on_gz_poses)
            self._gz_cmd_pub = self._gz_node.advertise(cmd_topic, twist_pb2.Twist)
            self.get_logger().info(f'已挂载 Gazebo 靶机总线: {cmd_topic}')
        except Exception as exc:
            self.get_logger().warn(f'Gazebo transport 不可用，降级为纯运动学靶机模式: {exc}')
            self.sim_backend = 'kinematic'

    def _on_gz_poses(self, msg) -> None:
        now = time.time()
        sim_t = float(msg.header.stamp.sec) + float(msg.header.stamp.nsec) * 1e-9
        ref_t = sim_t if sim_t > 0.0 else now
        for pose in msg.pose:
            if pose.name == self.target_model:
                pos = np.array([
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                ], dtype=float)
                yaw = _quat_to_yaw_enu(
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                )
                with self._lock:
                    if sim_t > 0.0:
                        if self._sim_t0 is None:
                            self._sim_t0 = sim_t
                        self._latest_sim_t = sim_t
                    if self._prev_gz_pos is not None and (ref_t - self._prev_gz_time) > 0.008:
                        dt = ref_t - self._prev_gz_time
                        raw_v = (pos - self._prev_gz_pos) / dt
                        if float(np.linalg.norm(raw_v)) < 35.0:
                            self._gt_vel = 0.65 * self._gt_vel + 0.35 * raw_v
                    self._prev_gz_pos = pos
                    self._prev_gz_time = ref_t
                    self._gt_pos = pos
                    self._gt_yaw = yaw
                    self._gz_pose_received = True
                break

    def reference_trajectory(self, t_rel: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算 t_rel 秒时刻红方靶机期望三维参考位置与参考速度（严格约束在无障碍中央空域）。"""
        s = self.speed_scale
        if self.profile == 'hover' or s <= 1e-4:
            return np.array([self.cx, self.cy, self.base_z], dtype=float), np.zeros(3, dtype=float)

        if self.profile == 'circle':
            w = 0.32 * s
            rx, ry = 9.0, 5.5
            x = self.cx + rx * math.cos(w * t_rel)
            y = self.cy + ry * math.sin(w * t_rel)
            vx = -rx * w * math.sin(w * t_rel)
            vy = ry * w * math.cos(w * t_rel)
            wz = 0.25 * s
            z = self.base_z + self.amp_z * math.sin(wz * t_rel)
            vz = self.amp_z * wz * math.cos(wz * t_rel)
            return np.array([x, y, z]), np.array([vx, vy, vz])

        if self.profile == 'figure8':
            w = 0.26 * s
            rx, ry = 10.5, 5.2
            x = self.cx + rx * math.sin(w * t_rel)
            y = self.cy + ry * math.sin(2.0 * w * t_rel)
            vx = rx * w * math.cos(w * t_rel)
            vy = 2.0 * ry * w * math.cos(2.0 * w * t_rel)
            wz = 0.30 * s
            z = self.base_z + self.amp_z * math.sin(wz * t_rel)
            vz = self.amp_z * wz * math.cos(wz * t_rel)
            return np.array([x, y, z]), np.array([vx, vy, vz])

        # 默认 evasive_weave：三维复合蛇形机动（兼具大曲率拐弯与高度起伏，检验立体三角化与 CTRV 预测）
        w = 0.28 * s
        rx, ry = 10.2, 5.0
        x = self.cx - rx * math.cos(w * t_rel)
        y = self.cy + ry * math.sin(w * t_rel) + 1.4 * math.sin(2.3 * w * t_rel)
        vx = rx * w * math.sin(w * t_rel)
        vy = ry * w * math.cos(w * t_rel) + 1.4 * 2.3 * w * math.cos(2.3 * w * t_rel)
        wz = 0.34 * s
        z = self.base_z + self.amp_z * math.sin(wz * t_rel)
        vz = self.amp_z * wz * math.cos(wz * t_rel)
        return np.array([x, y, z], dtype=float), np.array([vx, vy, vz], dtype=float)

    def _tick(self) -> None:
        now = time.time()
        use_gz = (
            self.sim_backend == 'gazebo'
            or (self.sim_backend == 'auto' and self._gz_pose_received)
        )
        with self._lock:
            sim_t0 = self._sim_t0
            sim_now = self._latest_sim_t
        if use_gz and sim_t0 is not None and sim_now >= sim_t0:
            t_rel = sim_now - sim_t0
        else:
            t_rel = now - self._t0
        p_ref, v_ref = self.reference_trajectory(t_rel)

        if use_gz and self._gz_cmd_pub is not None and self._twist_cls is not None:
            with self._lock:
                cur_pos = self._gt_pos.copy()
                cur_yaw = self._gt_yaw

            err_pos = p_ref - cur_pos
            v_world = v_ref + np.array([1.25, 1.25, 1.50]) * err_pos
            v_h = math.hypot(float(v_world[0]), float(v_world[1]))
            if v_h > 8.0:
                v_world[0] *= 8.0 / v_h
                v_world[1] *= 8.0 / v_h
            v_world[2] = max(-2.5, min(2.5, float(v_world[2])))

            des_yaw = math.atan2(float(v_world[1]), float(v_world[0])) if v_h > 0.25 else cur_yaw
            yaw_err = _wrap_pi(des_yaw - cur_yaw)
            yaw_rate = max(-1.8, min(1.8, 2.4 * yaw_err))

            # 世界 ENU 线速度旋转至靶机本体 FLU 坐标系下发给 VelocityControl 插件
            c_y, s_y = math.cos(-cur_yaw), math.sin(-cur_yaw)
            vx_body = c_y * float(v_world[0]) - s_y * float(v_world[1])
            vy_body = s_y * float(v_world[0]) + c_y * float(v_world[1])

            cmd = self._twist_cls()
            cmd.linear.x = float(vx_body)
            cmd.linear.y = float(vy_body)
            cmd.linear.z = float(v_world[2])
            cmd.angular.z = float(yaw_rate)
            self._gz_cmd_pub.publish(cmd)

            with self._lock:
                gt_pos = self._gt_pos.copy()
                gt_vel = self._gt_vel.copy()
                gt_yaw = self._gt_yaw
        else:
            # 纯运动学模式（无 Gazebo 时平滑跟踪参考轨迹）
            with self._lock:
                err = p_ref - self._gt_pos
                v_cmd = v_ref + 1.5 * err
                self._gt_pos = self._gt_pos + v_cmd * self.dt
                self._gt_vel = v_cmd
                if math.hypot(float(v_cmd[0]), float(v_cmd[1])) > 0.2:
                    self._gt_yaw = math.atan2(float(v_cmd[1]), float(v_cmd[0]))
                gt_pos = self._gt_pos.copy()
                gt_vel = self._gt_vel.copy()
                gt_yaw = self._gt_yaw

        stamp = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'world'
        odom.child_frame_id = 'red_moving_target'
        odom.pose.pose.position.x = float(gt_pos[0])
        odom.pose.pose.position.y = float(gt_pos[1])
        odom.pose.pose.position.z = float(gt_pos[2])
        qx, qy, qz, qw = _yaw_to_quat_enu(gt_yaw)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(gt_vel[0])
        odom.twist.twist.linear.y = float(gt_vel[1])
        odom.twist.twist.linear.z = float(gt_vel[2])
        self.gt_pub.publish(odom)

        if self.enable_radar and (now - self._last_radar_pub) >= (1.0 / self.radar_hz):
            self._last_radar_pub = now
            radar_msg = PoseStamped()
            radar_msg.header.stamp = stamp
            radar_msg.header.frame_id = 'world'
            radar_msg.pose.position.x = float(gt_pos[0] + self.rng.normal(0.0, self.noise_xy))
            radar_msg.pose.position.y = float(gt_pos[1] + self.rng.normal(0.0, self.noise_xy))
            radar_msg.pose.position.z = max(3.2, float(gt_pos[2] + self.rng.normal(0.0, self.noise_z)))
            radar_msg.pose.orientation.w = 1.0
            self.radar_pub.publish(radar_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TargetSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
