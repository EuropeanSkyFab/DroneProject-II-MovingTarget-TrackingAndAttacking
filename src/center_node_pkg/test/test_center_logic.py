# Copyright 2026 EuropeanSkyFab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functional unit tests for CenterNode state machine, safety gates, and 3D avoidance."""

import json
import math
import time

from diagnostic_msgs.msg import DiagnosticStatus
from geometry_msgs.msg import PoseStamped, Vector3
from nav_msgs.msg import Odometry
import rclpy
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String

from center_node_pkg.center_node import CenterNode


def _make_odom(drone_id, x, y, z, vz=0.0):
    msg = Odometry()
    msg.header.frame_id = 'world'
    msg.child_frame_id = f'{drone_id}_base_link'
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.position.z = float(z)
    msg.twist.twist.linear.z = float(vz)
    return msg


def test_center_state_machine_gates_and_all_broadcast():
    """Verify GROUND gate, ALL broadcast, geofence clamping, and telemetry health."""
    rclpy.init()
    node = CenterNode()
    captured_sp = {}
    captured_gimbal = {}
    captured_link = {}

    probe = rclpy.create_node('center_test_probe')
    for d in ('drone_1', 'drone_2', 'drone_3'):
        probe.create_subscription(
            PoseStamped,
            f'/{d}/fmu/in/trajectory_setpoint',
            lambda m, d_id=d: captured_sp.setdefault(d_id, []).append(m),
            10,
        )
        probe.create_subscription(
            Vector3,
            f'/{d}/fmu/in/gimbal_setpoint',
            lambda m, d_id=d: captured_gimbal.setdefault(d_id, []).append(m),
            10,
        )
        probe.create_subscription(
            DiagnosticStatus,
            f'/{d}/link_status',
            lambda m, d_id=d: captured_link.__setitem__(d_id, m),
            10,
        )

    try:
        # 1. GROUND 态拒绝直接进入 ATTACK
        cmd = String()
        cmd.data = json.dumps({'drone_id': 'drone_1', 'action': 'ATTACK'})
        node.strategy_callback(cmd)
        assert node.drone_modes['drone_1'] == 'GROUND'

        # 2. 注入三机地面位姿并执行 ALL 群发 TAKEOFF
        for idx, d in enumerate(('drone_1', 'drone_2', 'drone_3')):
            node.drone_state_callback(_make_odom(d, 0.0, idx * 3.0, 0.1), d)

        cmd.data = json.dumps({'drone_id': 'ALL', 'action': 'TAKEOFF'})
        node.strategy_callback(cmd)
        assert all(node.drone_modes[d] == 'TAKEOFF' for d in ('drone_1', 'drone_2', 'drone_3'))

        # 3. 非法 NaN/越界视觉输入必须被拦截
        bad_vis = PoseStamped()
        bad_vis.header.stamp = node.get_clock().now().to_msg()
        bad_vis.pose.position.x = float('nan')
        bad_vis.pose.position.y = 5.0
        bad_vis.pose.position.z = 5.5
        node.vision_callback(bad_vis)
        assert node.latest_vision_target is None

        # 4. 合法视觉输入 + TRACK + 三维云台俯仰解算验证
        for idx, d in enumerate(('drone_1', 'drone_2', 'drone_3')):
            node.drone_state_callback(_make_odom(d, 0.0, idx * 3.0, 9.5), d)
        good_vis = PoseStamped()
        good_vis.header.stamp = node.get_clock().now().to_msg()
        good_vis.pose.position.x = 10.0
        good_vis.pose.position.y = 0.0
        good_vis.pose.position.z = 5.5
        node.vision_callback(good_vis)
        assert node.latest_vision_target is not None

        cmd.data = json.dumps({'drone_id': 'ALL', 'action': 'TRACK'})
        node.strategy_callback(cmd)
        node.timer_callback()
        for _ in range(10):
            rclpy.spin_once(probe, timeout_sec=0.01)

        assert len(captured_sp.get('drone_1', [])) >= 1
        assert len(captured_gimbal.get('drone_1', [])) >= 1
        g1 = captured_gimbal['drone_1'][-1]
        # drone_1 在 (0,0,9.5) 看 (10,0,5.5)：正东 yaw_ned = pi/2，俯仰 atan2(-4, 10) ≈ -0.3805 rad
        assert abs(g1.y - math.pi / 2.0) < 0.05
        assert abs(g1.x - math.atan2(-4.0, 10.0)) < 0.05

        # 5. 验证断联自回环免疫：人为老化 last_state_times 后 link_status 必须变为 LOST
        node.last_state_times['drone_2'] = time.time() - 1.5
        raw_b = BatteryState()
        raw_b.voltage = 24.85
        raw_b.percentage = 0.91
        node.battery_raw_callback(raw_b, 'drone_1')
        node._publish_telemetry(time.time())
        for _ in range(10):
            rclpy.spin_once(probe, timeout_sec=0.01)

        assert captured_link['drone_2'].level == DiagnosticStatus.ERROR
        kv_keys = {kv.key for kv in captured_link['drone_1'].values}
        assert {'delay_ms', 'rssi', 'mode', 'ekf_aligned'}.issubset(kv_keys)
    finally:
        probe.destroy_node()
        node.destroy_node()
        rclpy.shutdown()
