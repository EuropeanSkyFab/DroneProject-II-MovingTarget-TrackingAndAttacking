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

"""Chaos and fault-injection tests for anti-corruption gates and optical dropout coasting."""

import json
import math
import time

from diagnostic_msgs.msg import DiagnosticStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from std_msgs.msg import String

from center_node_pkg.center_node import CenterNode
from planning_pkg.planning_node import PlanningNode


def _make_odom(drone_id, x, y, z):
    """Build a valid ENU Odometry message for testing."""
    msg = Odometry()
    msg.header.frame_id = 'world'
    msg.child_frame_id = f'{drone_id}_base_link'
    msg.pose.pose.position.x = float(x)
    msg.pose.pose.position.y = float(y)
    msg.pose.pose.position.z = float(z)
    msg.pose.pose.orientation.w = 1.0
    return msg


def test_fault_injection_gates_and_optical_dropout_coasting():
    """Verify NaN/Inf/OOB blocking, 1.5s vision dropout coasting, and link degradation."""
    rclpy.init()
    center = CenterNode()
    planner = PlanningNode()
    probe = rclpy.create_node('fault_injection_probe')

    captured_sp = {d: [] for d in ('drone_1', 'drone_2', 'drone_3')}
    captured_link = {}

    for d_id in ('drone_1', 'drone_2', 'drone_3'):
        probe.create_subscription(
            PoseStamped,
            f'/{d_id}/fmu/in/trajectory_setpoint',
            lambda m, key=d_id: captured_sp[key].append(m),
            10,
        )
        probe.create_subscription(
            DiagnosticStatus,
            f'/{d_id}/link_status',
            lambda m, key=d_id: captured_link.__setitem__(key, m),
            10,
        )

    def spin_cluster(rounds=6):
        for _ in range(rounds):
            rclpy.spin_once(planner, timeout_sec=0.005)
            rclpy.spin_once(center, timeout_sec=0.005)
            rclpy.spin_once(probe, timeout_sec=0.005)

    try:
        # 1. Initialize 3 airborne UAVs in TRACK mode
        for idx, d_id in enumerate(('drone_1', 'drone_2', 'drone_3')):
            odom = _make_odom(d_id, 0.0, idx * 3.0, 9.5)
            center.drone_state_callback(odom, d_id)
            planner.drone_kinematics[d_id] = (
                planner.drone_kinematics.get(d_id, (None, None))[0]
                if d_id in planner.drone_kinematics
                else None
            )
            planner._make_state_cb(d_id)(odom)

        cmd = String()
        cmd.data = json.dumps({'drone_id': 'ALL', 'action': 'TAKEOFF'})
        center.strategy_callback(cmd)
        planner._strategy_cb(cmd)

        cmd.data = json.dumps({'drone_id': 'ALL', 'action': 'TRACK'})
        center.strategy_callback(cmd)
        planner._strategy_cb(cmd)

        # 2. Fault Injection A: NaN, Inf, 10000m OOB, and stale timestamps on Vision & Plan
        corrupt_coords = [
            (float('nan'), 2.0, 5.5),
            (12.0, float('inf'), 5.5),
            (10000.0, 0.0, 5.5),
            (12.0, 2.0, -5.0),
        ]
        for bx, by, bz in corrupt_coords:
            v_bad = PoseStamped()
            v_bad.header.stamp = center.get_clock().now().to_msg()
            v_bad.pose.position.x = bx
            v_bad.pose.position.y = by
            v_bad.pose.position.z = bz
            center.vision_callback(v_bad)
            planner._vision_cb(v_bad)

            p_bad = String()
            now_s = center.get_clock().now().nanoseconds / 1e9
            p_bad.data = json.dumps({
                'drone_id': 'drone_1',
                'stamp': now_s,
                'x': bx if not math.isnan(bx) else 10000.0,
                'y': by if math.isfinite(by) else 10000.0,
                'z': bz,
                'action': 'TRACK',
            })
            center.plan_callback(p_bad)

        stale_vis = PoseStamped()
        stale_ns = center.get_clock().now().nanoseconds - int(2.0 * 1e9)
        stale_vis.header.stamp.sec = int(stale_ns // 1_000_000_000)
        stale_vis.header.stamp.nanosec = int(stale_ns % 1_000_000_000)
        stale_vis.pose.position.x = 15.0
        stale_vis.pose.position.y = 3.0
        stale_vis.pose.position.z = 5.5
        center.vision_callback(stale_vis)

        assert center.latest_vision_target is None
        assert center.latest_plan_targets['drone_1'] is None

        # 3. Feed 15 clean visual frames along a curving trajectory, then cut vision for 1.5s!
        t_base = planner._now_sec() - 1.5
        for k in range(15):
            tk = t_base + k * 0.10
            px_k = 16.0 + 3.2 * (k * 0.10)
            py_k = 2.0 + 1.5 * math.sin(0.5 * k * 0.10)
            planner.estimator.update((px_k, py_k, 5.6), tk, source='vision')

        # Simulate 1.5s complete optical blackout: advance estimator & run planner + center
        t_blackout = t_base + 1.4 + 1.5
        assert planner.estimator.is_valid(t_blackout)
        pred_pos, _ = planner.estimator.predict_pv_ahead(1.5)
        assert math.isfinite(float(pred_pos[0])) and pred_pos[0] > 23.5

        planner._plan_tick()
        spin_cluster(8)
        center.timer_callback()
        spin_cluster(8)

        for d_id in ('drone_1', 'drone_2', 'drone_3'):
            assert len(captured_sp[d_id]) >= 1
            sp = captured_sp[d_id][-1].pose.position
            assert math.isfinite(sp.x) and math.isfinite(sp.y) and math.isfinite(sp.z)
            assert abs(sp.x) <= center.MAX_XY_DIST and sp.z >= center.MIN_ALT

        # 4. Fault Injection C: Bridge Telemetry Degradation (GOOD -> WEAK -> LOST)
        now_wall = time.time()
        center.last_state_times['drone_1'] = now_wall - 0.05
        center.last_state_times['drone_2'] = now_wall - 0.45
        center.last_state_times['drone_3'] = now_wall - 1.25
        center._publish_telemetry(now_wall)
        spin_cluster(8)

        assert captured_link['drone_1'].level == DiagnosticStatus.OK
        assert captured_link['drone_2'].level == DiagnosticStatus.WARN
        assert captured_link['drone_3'].level == DiagnosticStatus.ERROR
    finally:
        probe.destroy_node()
        planner.destroy_node()
        center.destroy_node()
        rclpy.shutdown()
