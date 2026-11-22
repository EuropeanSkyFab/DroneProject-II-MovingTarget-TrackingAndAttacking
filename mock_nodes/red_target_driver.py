#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""红方靶机驱动 + 攻击效果判定

作用（一个节点搞定四件事）：
  1. 从 Gazebo 读靶机真值位姿：/world/<world>/pose/info
  2. 以 10Hz 把靶机当前位置发布到 /planning/target_pose（JSON），喂给中枢的 ATTACK 分支
  3. 订阅我方 /<drone>/state，按**航迹线段**计算最小脱靶量，距离小于阈值判定命中
  4. 可选：通过 /model/<model>/cmd_vel 驱动靶机绕场巡航

为什么命中要靠几何判定而不是碰撞：
  靶机模型 red_target_uav 里没有任何 <collision>（注释写明“无碰撞，确保 5.5~8.5m
  立体缠斗安全”），撞上去会直接穿过去，物理引擎不产生接触事件。只能按距离判定。

为什么脱靶量用“点到线段距离”而不是采样点：
  /<drone>/state 只有 20Hz，攻击末段速度可达 10~23 m/s，相邻两帧相隔 0.5~1.2 米。
  若只比较采样点，真实最近点很可能落在两帧之间，测出的脱靶量会偏大且不稳定。
  这里把相邻两帧连成线段，求靶机到该线段的距离，20Hz 也能还原真实最近点。

坐标系说明：
  中枢/桥接输出的 ENU 是**相对出生点**的，而 Gazebo pose/info 是**世界坐标**。
  x500 出生在 (0, 0, 0.5)，故世界 z = 桥接 z + 0.5，用 --drone-spawn-z 校正。

用法：
  python3 red_target_driver.py                       # 靶机悬停，只测脱靶量（先跑这个）
  python3 red_target_driver.py --pattern circle      # 靶机绕场盘旋（进阶）
  python3 red_target_driver.py --drone drone_1 --hit-radius 0.5
"""
import argparse
import json
import math
import threading
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from gz.transport13 import Node as GzNode
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.twist_pb2 import Twist


def _point_segment_distance(p, a, b):
    """点 p 到线段 ab 的最短距离（三维）。"""
    abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    apx, apy, apz = p[0] - a[0], p[1] - a[1], p[2] - a[2]
    l2 = abx * abx + aby * aby + abz * abz
    if l2 < 1e-9:
        return math.dist(p, a)
    t = (apx * abx + apy * aby + apz * abz) / l2
    t = max(0.0, min(1.0, t))
    q = (a[0] + t * abx, a[1] + t * aby, a[2] + t * abz)
    return math.dist(p, q)


class RedTargetDriver(Node):
    def __init__(self, args):
        super().__init__('red_target_driver')
        self.args = args
        self.lock = threading.Lock()

        self.target = None          # 靶机世界坐标 (x, y, z)
        self.drone = None           # 我方世界坐标 (x, y, z)，已做出生点校正
        self.prev_drone = None      # 上一帧位置，用于线段法求最近点
        self.min_dist = float('inf')
        self.hit = False
        self.hit_time = None
        self.odom_count = 0
        self._last_log = 0.0
        self._start_time = time.time()

        # ---------------- Gazebo 侧 ----------------
        self.gz = GzNode()
        pose_topic = f'/world/{args.world}/pose/info'
        if self.gz.subscribe(Pose_V, pose_topic, self._gz_pose_cb):
            self.get_logger().info(f'已订阅 Gazebo 真值话题：{pose_topic}')
        else:
            self.get_logger().error(f'订阅失败（Gazebo 没启动？）：{pose_topic}')

        self.cmd_pub = None
        if args.pattern == 'circle':
            self.cmd_pub = self.gz.advertise(f'/model/{args.model}/cmd_vel', Twist)
            self.get_logger().info(
                f'巡航模式：向 /model/{args.model}/cmd_vel 下发速度（{args.speed:.1f} m/s）')

        # ---------------- ROS 侧 ----------------
        self.plan_pub = self.create_publisher(String, '/planning/target_pose', 10)
        self.create_subscription(Odometry, f'/{args.drone}/state', self._state_cb, 10)
        self.create_timer(1.0 / args.publish_hz, self._tick)

        self.get_logger().info(
            f'靶机驱动就绪 | model={args.model} | 目标机={args.drone} | '
            f'命中半径={args.hit_radius:.2f}m | 模式={args.pattern}')

    # ---------------- 回调 ----------------
    def _gz_pose_cb(self, msg):
        for p in msg.pose:
            if p.name == self.args.model:
                with self.lock:
                    self.target = (p.position.x, p.position.y, p.position.z)
                return

    def _state_cb(self, msg):
        """我方位姿回调：20Hz，在这里做线段法脱靶量统计，比定时器采样准得多。"""
        pos = msg.pose.pose.position
        cur = (pos.x, pos.y, pos.z + self.args.drone_spawn_z)

        with self.lock:
            prev = self.prev_drone
            self.prev_drone = cur
            self.drone = cur
            self.odom_count += 1
            tgt = self.target

            if tgt is None:
                return
            # 有上一帧就连成线段求距离，否则退化为点到点
            d = _point_segment_distance(tgt, prev, cur) if prev is not None else math.dist(tgt, cur)

            if d < self.min_dist:
                self.min_dist = d
            first_hit = (not self.hit) and d <= self.args.hit_radius
            if first_hit:
                self.hit = True
                self.hit_time = time.time() - self._start_time

        if first_hit:
            self.get_logger().error(
                f'★ 命中！最近距离 {d:.3f}m <= {self.args.hit_radius:.2f}m，'
                f'用时 {self.hit_time:.1f}s')

    # ---------------- 主循环（10Hz） ----------------
    def _tick(self):
        with self.lock:
            tgt = self.target
            drone = self.drone

        if tgt is None:
            self.get_logger().warn('等待 Gazebo 靶机位姿...（确认仿真已启动、模型名正确）',
                                   throttle_duration_sec=3.0)
            return

        # 1) 喂给中枢的 ATTACK 分支
        out = String()
        out.data = json.dumps({
            'drone_id': self.args.drone,
            'stamp': self.get_clock().now().nanoseconds / 1e9,   # 必须刷新，否则被保鲜期丢弃
            'x': round(tgt[0], 3),
            'y': round(tgt[1], 3),
            'z': round(tgt[2], 3),
            'action': 'ATTACK',
        })
        self.plan_pub.publish(out)

        # 2) 巡航驱动（可选）
        if self.cmd_pub is not None:
            self._drive_circle(tgt)

        # 3) 状态打印（1Hz）
        if drone is None:
            return
        now = time.time()
        if now - self._last_log >= 1.0:
            self._last_log = now
            dist = math.dist(drone, tgt)
            with self.lock:
                md = self.min_dist
                hit = self.hit
                n = self.odom_count
            self.get_logger().info(
                f'靶机({tgt[0]:6.1f},{tgt[1]:6.1f},{tgt[2]:5.1f})  '
                f'我方({drone[0]:6.1f},{drone[1]:6.1f},{drone[2]:5.1f})  '
                f'当前={dist:6.2f}m  最小={md:6.2f}m  位姿帧={n}'
                + ('  [已命中]' if hit else ''))

    def _drive_circle(self, tgt):
        """绕 (center_x, center_y) 做切向巡航，并对高度做简单比例保持。"""
        tx, ty, tz = tgt
        rx = tx - self.args.center_x
        ry = ty - self.args.center_y
        r = math.hypot(rx, ry)

        tw = Twist()
        if r < 0.5:
            tw.linear.x = self.args.speed          # 太靠近圆心，先推出去，避免除零
        else:
            tw.linear.x = -ry / r * self.args.speed
            tw.linear.y = rx / r * self.args.speed
        tw.linear.z = max(-1.5, min(1.5, (self.args.altitude - tz) * 0.8))
        self.cmd_pub.publish(tw)


def main():
    parser = argparse.ArgumentParser(description='红方靶机驱动 + 攻击效果判定')
    parser.add_argument('--world', default='moving_target_arena', help='Gazebo 世界名')
    parser.add_argument('--model', default='red_moving_target', help='靶机模型名')
    parser.add_argument('--drone', default='drone_1', help='我方无人机 ID')
    parser.add_argument('--hit-radius', type=float, default=0.5, help='命中判定半径（米）')
    parser.add_argument('--drone-spawn-z', type=float, default=0.5,
                        help='我方出生高度，用于把桥接的 ENU 换算回世界坐标')
    parser.add_argument('--publish-hz', type=float, default=10.0)
    parser.add_argument('--pattern', choices=['hover', 'circle'], default='hover')
    parser.add_argument('--speed', type=float, default=2.0, help='巡航切向速度（m/s）')
    parser.add_argument('--altitude', type=float, default=5.5, help='巡航目标高度（米）')
    parser.add_argument('--center-x', type=float, default=0.0)
    parser.add_argument('--center-y', type=float, default=0.0)
    args, _ = parser.parse_known_args()

    rclpy.init(args=None)
    node = RedTargetDriver(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        with node.lock:
            md = node.min_dist
            hit = node.hit
            ht = node.hit_time
            n = node.odom_count
        print()
        print('=' * 58)
        print('  攻击效果汇总')
        print('-' * 58)
        print(f'  最小脱靶量 : '
              f'{"未测到（我方从未上报过位姿）" if md == float("inf") else f"{md:.3f} m"}')
        print(f'  命中判定   : {"命中" if hit else "未命中"}'
              f'（阈值 {args.hit_radius:.2f} m）')
        if hit and ht is not None:
            print(f'  命中时刻   : 开跑后 {ht:.1f} s')
        print(f'  位姿帧数   : {n}')
        print('=' * 58)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
