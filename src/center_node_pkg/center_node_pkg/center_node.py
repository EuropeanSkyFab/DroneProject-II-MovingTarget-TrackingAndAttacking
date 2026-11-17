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

"""Central tactical state machine, safety geofence, and 3D deconfliction node."""

import copy
import json
import math
import time

from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, Vector3
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Empty, String

VALID_ACTIONS = {
    'TAKEOFF',
    'HOVER',
    'RETURN',
    'TRACK',
    'INTERCEPT',
    'ATTACK',
    'RESUME_PATROL',
}


class CenterNode(Node):
    """Multi-UAV tactical scheduler, anti-corruption gate, and 3D collision avoidance node."""

    def __init__(self):
        """Initialize subscriptions, publishers, safety thresholds, and 20Hz control timer."""
        super().__init__('center_node')

        self.declare_parameter('swarm_ids', ['drone_1', 'drone_2', 'drone_3'])
        self.declare_parameter('takeoff_alt', 9.5)
        self.declare_parameter('track_safe_dist', 5.0)
        self.declare_parameter('home_spacing_y', 3.0)
        self.declare_parameter('max_vision_delay', 0.5)
        self.declare_parameter('max_plan_delay', 0.8)

        self.swarm_config = [str(d) for d in self.get_parameter('swarm_ids').value]
        self.takeoff_alt = float(self.get_parameter('takeoff_alt').value)
        self.TRACK_SAFE_DIST = float(self.get_parameter('track_safe_dist').value)
        self.home_spacing_y = float(self.get_parameter('home_spacing_y').value)
        self.max_vision_delay = float(self.get_parameter('max_vision_delay').value)
        self.max_plan_delay = float(self.get_parameter('max_plan_delay').value)

        self.heartbeat_pubs = {}
        self.fmu_pubs = {}
        self.gimbal_pubs = {}
        self.action_pubs = {}
        self.battery_pubs = {}
        self.link_status_pubs = {}

        self.drone_modes = {d_id: 'GROUND' for d_id in self.swarm_config}
        self.drone_states = {d_id: None for d_id in self.swarm_config}
        self.last_state_times = {d_id: 0.0 for d_id in self.swarm_config}
        self.raw_batteries = {d_id: None for d_id in self.swarm_config}
        self.fc_diags = {d_id: {} for d_id in self.swarm_config}
        self.takeoff_anchors = {}

        self.latest_plan_targets = {d_id: None for d_id in self.swarm_config}
        self.latest_plan_focus = {d_id: None for d_id in self.swarm_config}
        self.last_plan_times = {d_id: 0.0 for d_id in self.swarm_config}

        self.current_mode = 'GROUND'
        self.last_vision_time = 0.0
        self.latest_vision_target = None
        self.last_battery_pub_time = 0.0
        self.last_fleet_pub_time = 0.0

        self.MAX_XY_DIST = 1000.0
        self.MAX_ALT = 50.0
        self.MIN_ALT = 3.0
        self.ATTACK_MIN_ALT = 1.5
        self.LAND_ALT_THRESHOLD = 0.32
        self.LAND_SPEED_THRESHOLD = 0.25
        self.COLLISION_WARN_DIST = 5.8
        self.COLLISION_DANGER_DIST = 3.8

        self.plan_sub = self.create_subscription(
            String, '/planning/target_pose', self.plan_callback, 20
        )
        self.vision_sub = self.create_subscription(
            PoseStamped, '/vision/target_pose', self.vision_callback, 10
        )
        self.strategy_sub = self.create_subscription(
            String, '/strategy/cmd', self.strategy_callback, 10
        )

        self.fleet_state_pub = self.create_publisher(String, '/fleet/state', 10)

        for drone_id in self.swarm_config:
            self.create_subscription(
                Odometry,
                f'/{drone_id}/state',
                lambda msg, d_id=drone_id: self.drone_state_callback(msg, d_id),
                10,
            )
            self.create_subscription(
                BatteryState,
                f'/{drone_id}/fmu/out/battery_raw',
                lambda msg, d_id=drone_id: self.battery_raw_callback(msg, d_id),
                10,
            )
            self.create_subscription(
                DiagnosticStatus,
                f'/{drone_id}/fmu/out/fc_status',
                lambda msg, d_id=drone_id: self.fc_status_callback(msg, d_id),
                10,
            )

            self.fmu_pubs[drone_id] = self.create_publisher(
                PoseStamped, f'/{drone_id}/fmu/in/trajectory_setpoint', 10
            )
            self.gimbal_pubs[drone_id] = self.create_publisher(
                Vector3, f'/{drone_id}/fmu/in/gimbal_setpoint', 10
            )
            self.action_pubs[drone_id] = self.create_publisher(
                String, f'/{drone_id}/action', 10
            )
            self.battery_pubs[drone_id] = self.create_publisher(
                BatteryState, f'/{drone_id}/battery', 10
            )
            self.link_status_pubs[drone_id] = self.create_publisher(
                DiagnosticStatus, f'/{drone_id}/link_status', 10
            )
            self.heartbeat_pubs[drone_id] = self.create_publisher(
                Empty, f'/{drone_id}/heartbeat', 10
            )

        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info(f'中枢节点已启动(20Hz): 集群={self.swarm_config}')

    def send_single_target(self, drone_id, target_msg):
        """Clamp target setpoint within electronic geofence and publish to bridge."""
        if target_msg is None or drone_id not in self.fmu_pubs:
            return

        out_msg = copy.deepcopy(target_msg)
        out_msg.header.stamp = self.get_clock().now().to_msg()
        out_msg.header.frame_id = 'world'

        out_msg.pose.position.x = max(
            -self.MAX_XY_DIST, min(self.MAX_XY_DIST, float(out_msg.pose.position.x))
        )
        out_msg.pose.position.y = max(
            -self.MAX_XY_DIST, min(self.MAX_XY_DIST, float(out_msg.pose.position.y))
        )

        mode = self.drone_modes.get(drone_id, 'GROUND')
        if out_msg.pose.position.z > self.MAX_ALT + 1.0:
            out_msg.pose.position.z = self.MAX_ALT
            self.get_logger().warn(
                f'[{drone_id}] 目标高度超上限，拦截在 {self.MAX_ALT}m',
                throttle_duration_sec=2.0,
            )

        if mode == 'ATTACK':
            if out_msg.pose.position.z < self.ATTACK_MIN_ALT:
                out_msg.pose.position.z = self.ATTACK_MIN_ALT
        elif mode not in ('TAKEOFF', 'RETURN'):
            if out_msg.pose.position.z < self.MIN_ALT:
                out_msg.pose.position.z = self.MIN_ALT + 0.1
                self.get_logger().warn(
                    f'[{drone_id}] 目标高度过低，拉高至 {self.MIN_ALT + 0.1:.1f}m',
                    throttle_duration_sec=2.0,
                )

        self.fmu_pubs[drone_id].publish(out_msg)

    def publish_gimbal_target(self, drone_id, target_msg, my_state):
        """Compute 3D pitch and NED yaw pointing angles and publish gimbal setpoint."""
        if target_msg is None or my_state is None or drone_id not in self.gimbal_pubs:
            return

        dx = float(target_msg.pose.position.x) - float(my_state.pose.pose.position.x)
        dy = float(target_msg.pose.position.y) - float(my_state.pose.pose.position.y)
        dz = float(target_msg.pose.position.z) - float(my_state.pose.pose.position.z)
        horiz_dist = math.hypot(dx, dy)

        yaw_ned = math.atan2(dx, dy)
        pitch_rad = math.atan2(dz, max(horiz_dist, 0.35))
        pitch_rad = max(math.radians(-55.0), min(math.radians(12.0), pitch_rad))

        gimbal_cmd = Vector3()
        gimbal_cmd.x = float(pitch_rad)
        gimbal_cmd.y = float(yaw_ned)
        gimbal_cmd.z = 0.0
        self.gimbal_pubs[drone_id].publish(gimbal_cmd)

    def drone_state_callback(self, msg, drone_id):
        """Store deep-copied bridge odometry and update link freshness timestamp."""
        p = msg.pose.pose.position
        if not (math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.z)):
            return
        self.drone_states[drone_id] = copy.deepcopy(msg)
        self.last_state_times[drone_id] = time.time()

    def battery_raw_callback(self, msg, drone_id):
        """Cache real 6S battery telemetry reported by px4_bridge_node."""
        self.raw_batteries[drone_id] = copy.deepcopy(msg)

    def fc_status_callback(self, msg, drone_id):
        """Cache flight controller diagnostic key-values."""
        self.fc_diags[drone_id] = {kv.key: kv.value for kv in msg.values}

    def apply_collision_avoidance(self, drone_id, my_state, target, my_mode):
        """Apply dynamic-priority 3D tangential/vertical collision avoidance."""
        my_index = self.swarm_config.index(drone_id)

        for other_id in self.swarm_config:
            if other_id == drone_id:
                continue

            other_state = self.drone_states.get(other_id)
            other_mode = self.drone_modes.get(other_id, 'GROUND')
            if other_state is None or other_mode == 'GROUND':
                continue

            odx = float(my_state.pose.pose.position.x - other_state.pose.pose.position.x)
            ody = float(my_state.pose.pose.position.y - other_state.pose.pose.position.y)
            odz = float(my_state.pose.pose.position.z - other_state.pose.pose.position.z)
            dist = math.sqrt(odx * odx + ody * ody + odz * odz)

            if not (0.01 < dist < self.COLLISION_WARN_DIST):
                continue

            if my_mode in ('TAKEOFF', 'RETURN') and dist >= 2.2:
                continue

            if my_mode == 'ATTACK' and other_mode != 'ATTACK':
                is_higher_priority = True
            elif my_mode != 'ATTACK' and other_mode == 'ATTACK':
                is_higher_priority = False
            else:
                is_higher_priority = my_index < self.swarm_config.index(other_id)

            if is_higher_priority:
                if dist < 3.6:
                    horiz = max(math.hypot(odx, ody), 1e-4)
                    target.pose.position.x += (odx / horiz) * 2.4
                    target.pose.position.y += (ody / horiz) * 2.4
                continue

            other_vz = float(other_state.twist.twist.linear.z)
            horiz = max(math.hypot(odx, ody), 1e-4)
            ux, uy = odx / horiz, ody / horiz
            if abs(other_vz) < 0.5 and abs(odz) < 2.2:
                target.pose.position.z = min(float(target.pose.position.z) + 2.4, 25.0)
                if dist >= 4.2:
                    self.get_logger().warn(
                        f'[{drone_id}] 友机平飞接近({dist:.1f}m)，执行战术拉升避让',
                        throttle_duration_sec=1.5,
                    )
                    continue

            if dist < self.COLLISION_DANGER_DIST:
                push = 5.6
                target.pose.position.x += ux * push
                target.pose.position.y += uy * push
                self.get_logger().warn(
                    f'[{drone_id}] 进入危险间距({dist:.1f}m)，紧急排斥分离',
                    throttle_duration_sec=1.0,
                )
            else:
                tangent_x, tangent_y = -uy, ux
                push = (self.COLLISION_WARN_DIST - dist) * 1.15
                target.pose.position.x += (0.65 * ux + 0.75 * tangent_x) * push
                target.pose.position.y += (0.65 * uy + 0.75 * tangent_y) * push

    def _is_valid_xyz(self, x, y, z, min_z):
        if math.isnan(x) or math.isnan(y) or math.isnan(z):
            return False
        if math.isinf(x) or math.isinf(y) or math.isinf(z):
            return False
        if abs(x) > self.MAX_XY_DIST or abs(y) > self.MAX_XY_DIST:
            return False
        if z > self.MAX_ALT or z < min_z:
            return False
        return True

    def plan_callback(self, msg):
        """Validate and store per-drone trajectory target from planning_pkg."""
        try:
            data = json.loads(msg.data)
            drone_id = str(data.get('drone_id', 'drone_1'))
            if drone_id not in self.swarm_config:
                return

            msg_time = float(data.get('stamp', 0.0))
            current_time = self.get_clock().now().nanoseconds / 1e9
            if current_time - msg_time > self.max_plan_delay:
                return

            x = float(data.get('x', 0.0))
            y = float(data.get('y', 0.0))
            z = float(data.get('z', 0.0))
            act = str(data.get('action', 'TRACK')).upper()
            min_z = self.ATTACK_MIN_ALT if act == 'ATTACK' else self.MIN_ALT

            if not self._is_valid_xyz(x, y, z, min_z):
                self.get_logger().warn('规划数据越界或含非法浮点数，已拦截', throttle_duration_sec=1.5)
                return

            target = PoseStamped()
            target.header.stamp = self.get_clock().now().to_msg()
            target.header.frame_id = 'world'
            target.pose.position.x = x
            target.pose.position.y = y
            target.pose.position.z = z
            target.pose.orientation.w = 1.0

            self.latest_plan_targets[drone_id] = target
            if 'focus_x' in data and 'focus_y' in data and 'focus_z' in data:
                focus = PoseStamped()
                focus.header = target.header
                focus.pose.position.x = float(data['focus_x'])
                focus.pose.position.y = float(data['focus_y'])
                focus.pose.position.z = float(data['focus_z'])
                focus.pose.orientation.w = 1.0
                self.latest_plan_focus[drone_id] = focus
            else:
                self.latest_plan_focus[drone_id] = None
            self.last_plan_times[drone_id] = time.time()
        except Exception as exc:
            self.get_logger().error(f'解析规划 JSON 失败: {exc}')

    def vision_callback(self, msg):
        """Validate timestamp latency and physical bounds of visual target pose."""
        current_time = self.get_clock().now()
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        delay = (current_time - msg_time).nanoseconds / 1e9
        if delay > self.max_vision_delay:
            self.get_logger().warn(
                f'[视觉] 数据延迟过高 ({delay:.3f}s)，已丢弃', throttle_duration_sec=1.5
            )
            return

        x = float(msg.pose.position.x)
        y = float(msg.pose.position.y)
        z = float(msg.pose.position.z)
        if not self._is_valid_xyz(x, y, z, self.ATTACK_MIN_ALT):
            self.get_logger().warn(
                f'[视觉] 目标超出安全围栏或非法 ({x:.2f}, {y:.2f}, {z:.2f})，已拦截',
                throttle_duration_sec=1.5,
            )
            return

        self.latest_vision_target = copy.deepcopy(msg)
        self.last_vision_time = time.time()

    def strategy_callback(self, msg):
        """Parse strategic commands (supporting single drone_id or ALL broadcast)."""
        try:
            data = json.loads(msg.data)
            raw_id = str(data.get('drone_id', 'drone_1')).strip()
            action = str(data.get('action', 'HOVER')).upper().strip()

            if action not in VALID_ACTIONS:
                self.get_logger().warn(f'拒绝未知战术指令: {action}')
                return

            target_ids = self.swarm_config if raw_id.upper() == 'ALL' else [raw_id]
            for d_id in target_ids:
                if d_id not in self.swarm_config:
                    self.get_logger().warn(f'拒绝执行：系统中不存在 {d_id}')
                    continue

                cur_mode = self.drone_modes.get(d_id, 'GROUND')
                if cur_mode == 'GROUND' and action != 'TAKEOFF':
                    self.get_logger().warn(f'拒绝执行 {action}：{d_id} 处于 GROUND，请先 TAKEOFF')
                    continue
                if cur_mode != 'GROUND' and action == 'TAKEOFF':
                    self.get_logger().warn(f'拒绝重复 TAKEOFF：{d_id} 当前处于 {cur_mode}')
                    continue

                if action == 'TAKEOFF':
                    st = self.drone_states.get(d_id)
                    if st is not None:
                        self.takeoff_anchors[d_id] = (
                            float(st.pose.pose.position.x),
                            float(st.pose.pose.position.y),
                        )
                    else:
                        idx = self.swarm_config.index(d_id)
                        self.takeoff_anchors[d_id] = (0.0, idx * self.home_spacing_y)

                self.drone_modes[d_id] = action
                self.current_mode = action
                if d_id in self.action_pubs:
                    act_msg = String()
                    act_msg.data = 'TRACK' if action == 'RESUME_PATROL' else action
                    self.action_pubs[d_id].publish(act_msg)

                self.get_logger().info(f'战术状态切换: {d_id} -> {action}')
        except Exception as exc:
            self.get_logger().error(f'解析中控指令失败: {exc}')

    def _compute_builtin_patrol(self, drone_id, now_s):
        idx = self.swarm_config.index(drone_id)
        phase = 0.18 * now_s + (2.0 * math.pi * idx) / max(len(self.swarm_config), 1)
        target = PoseStamped()
        target.header.stamp = self.get_clock().now().to_msg()
        target.header.frame_id = 'world'
        target.pose.position.x = 18.0 + 11.0 * math.cos(phase)
        target.pose.position.y = 3.0 + 7.5 * math.sin(phase)
        target.pose.position.z = 9.0 + 0.6 * idx
        target.pose.orientation.w = 1.0
        return target

    def timer_callback(self):
        """Execute 20Hz state machine routing, collision avoidance, and telemetry broadcast."""
        now = time.time()

        for drone_id in self.heartbeat_pubs:
            self.heartbeat_pubs[drone_id].publish(Empty())

        if now - self.last_vision_time > 2.0:
            self.latest_vision_target = None

        for d_id in self.swarm_config:
            if now - self.last_plan_times[d_id] > 2.0:
                self.latest_plan_targets[d_id] = None
                self.latest_plan_focus[d_id] = None

        for drone_id in self.swarm_config:
            mode = self.drone_modes.get(drone_id, 'GROUND')
            my_state = self.drone_states.get(drone_id)
            if mode == 'GROUND' or my_state is None:
                continue

            my_idx = self.swarm_config.index(drone_id)
            cur_x = float(my_state.pose.pose.position.x)
            cur_y = float(my_state.pose.pose.position.y)
            cur_z = float(my_state.pose.pose.position.z)

            if mode == 'TAKEOFF':
                ax, ay = self.takeoff_anchors.get(drone_id, (cur_x, cur_y))
                target = PoseStamped()
                target.header.frame_id = 'world'
                target.pose.position.x = ax
                target.pose.position.y = ay
                target.pose.position.z = self.takeoff_alt + 0.4 * my_idx
                self.apply_collision_avoidance(drone_id, my_state, target, mode)
                self.send_single_target(drone_id, target)

            elif mode == 'HOVER':
                tx, ty, tz = cur_x, cur_y, max(cur_z, self.MIN_ALT + 0.5)
                for other_id in self.swarm_config:
                    if other_id == drone_id:
                        continue
                    other_state = self.drone_states.get(other_id)
                    other_mode = self.drone_modes.get(other_id, 'GROUND')
                    if other_state is not None and other_mode != 'GROUND':
                        odx = tx - float(other_state.pose.pose.position.x)
                        ody = ty - float(other_state.pose.pose.position.y)
                        odz = tz - float(other_state.pose.pose.position.z)
                        dist = math.sqrt(odx * odx + ody * ody + odz * odz)
                        if 0.01 < dist < 6.0:
                            horiz = max(math.hypot(odx, ody), 1e-4)
                            push = (6.0 - dist) * 1.4
                            tx += (odx / horiz) * push
                            ty += (ody / horiz) * push
                target = PoseStamped()
                target.header.frame_id = 'world'
                target.pose.position.x = tx
                target.pose.position.y = ty
                target.pose.position.z = tz
                self.send_single_target(drone_id, target)

            elif mode == 'RESUME_PATROL':
                my_plan = self.latest_plan_targets.get(drone_id)
                target = (
                    copy.deepcopy(my_plan)
                    if my_plan is not None
                    else self._compute_builtin_patrol(drone_id, now)
                )
                self.apply_collision_avoidance(drone_id, my_state, target, mode)
                self.send_single_target(drone_id, target)
                if self.latest_vision_target is not None:
                    self.publish_gimbal_target(drone_id, self.latest_vision_target, my_state)

            elif mode == 'RETURN':
                home_x = 0.0
                home_y = my_idx * self.home_spacing_y
                horiz_dist = math.hypot(cur_x - home_x, cur_y - home_y)
                vx = float(my_state.twist.twist.linear.x)
                vy = float(my_state.twist.twist.linear.y)
                vz = float(my_state.twist.twist.linear.z)
                horiz_spd = math.hypot(vx, vy)
                speed = math.sqrt(vx * vx + vy * vy + vz * vz)

                target = PoseStamped()
                target.header.frame_id = 'world'
                if horiz_dist > 0.32 or (cur_z > 2.0 and horiz_spd > 0.38):
                    # 分层返航高度走廊 (7.5m / 9.0m / 10.5m) + 停机坪正上方悬停刹停对准后再落
                    ret_alt = (
                        max(cur_z, 7.5 + 1.5 * my_idx) if horiz_dist > 1.5 else max(cur_z, 4.5)
                    )
                    target.pose.position.x = home_x
                    target.pose.position.y = home_y
                    target.pose.position.z = min(ret_alt, 20.0)
                    if horiz_dist > 1.5:
                        self.apply_collision_avoidance(drone_id, my_state, target, mode)
                else:
                    target.pose.position.x = home_x
                    target.pose.position.y = home_y
                    target.pose.position.z = 0.0
                self.send_single_target(drone_id, target)

                if cur_z < self.LAND_ALT_THRESHOLD and speed < self.LAND_SPEED_THRESHOLD:
                    self.drone_modes[drone_id] = 'GROUND'
                    if drone_id in self.action_pubs:
                        act_msg = String()
                        act_msg.data = 'HOVER'
                        self.action_pubs[drone_id].publish(act_msg)
                    self.get_logger().info(f'[{drone_id}] 已安全触地停机坪，重置为 GROUND')

            elif mode == 'TRACK':
                my_plan = self.latest_plan_targets.get(drone_id)
                plan_fresh = (now - self.last_plan_times[drone_id]) <= 1.2

                if my_plan is not None and plan_fresh:
                    # 优先采用 planning_pkg 输出的 120° 锁相合围与视场高度优化点
                    target = copy.deepcopy(my_plan)
                elif self.latest_vision_target is not None:
                    # 无规划器独立运行兜底：围绕视觉目标做分相安全距离环绕 + 俯视高度抬升
                    raw_v = self.latest_vision_target
                    target = copy.deepcopy(raw_v)
                    target.pose.position.z = max(
                        self.MIN_ALT + 3.0, float(raw_v.pose.position.z) + 3.4 + 0.4 * my_idx
                    )
                    dx = float(raw_v.pose.position.x) - cur_x
                    dy = float(raw_v.pose.position.y) - cur_y
                    h_dist = math.hypot(dx, dy)
                    if 0.01 < h_dist < self.TRACK_SAFE_DIST:
                        ux, uy = dx / h_dist, dy / h_dist
                        push = (self.TRACK_SAFE_DIST - h_dist) * 1.5
                        target.pose.position.x = float(raw_v.pose.position.x) - ux * (
                            self.TRACK_SAFE_DIST * 0.85
                        ) - uy * push
                        target.pose.position.y = float(raw_v.pose.position.y) - uy * (
                            self.TRACK_SAFE_DIST * 0.85
                        ) + ux * push
                else:
                    continue

                self.apply_collision_avoidance(drone_id, my_state, target, mode)
                self.send_single_target(drone_id, target)
                focus_pt = self.latest_plan_focus.get(drone_id)
                gimbal_ref = (
                    self.latest_vision_target
                    if self.latest_vision_target is not None
                    else (focus_pt if focus_pt is not None else target)
                )
                self.publish_gimbal_target(drone_id, gimbal_ref, my_state)

            elif mode in ('INTERCEPT', 'ATTACK'):
                my_plan = self.latest_plan_targets.get(drone_id)
                plan_fresh = (now - self.last_plan_times[drone_id]) <= 1.2
                if my_plan is not None and plan_fresh:
                    target = copy.deepcopy(my_plan)
                elif self.latest_vision_target is not None:
                    target = copy.deepcopy(self.latest_vision_target)
                    if mode == 'INTERCEPT':
                        target.pose.position.z = float(target.pose.position.z) + 1.5
                else:
                    continue

                self.apply_collision_avoidance(drone_id, my_state, target, mode)
                self.send_single_target(drone_id, target)
                focus_pt = self.latest_plan_focus.get(drone_id)
                gimbal_ref = (
                    self.latest_vision_target
                    if self.latest_vision_target is not None
                    else (focus_pt if focus_pt is not None else target)
                )
                self.publish_gimbal_target(drone_id, gimbal_ref, my_state)

        self._publish_telemetry(now)

    def _publish_telemetry(self, now):
        """Publish link health, real 6S battery state, and aggregated /fleet/state."""
        fleet_snapshot = {'stamp': round(now, 3), 'drones': {}}

        for drone_id in self.swarm_config:
            my_state = self.drone_states.get(drone_id)
            last_t = self.last_state_times.get(drone_id, 0.0)
            if my_state is None or last_t <= 0.0:
                continue

            delay = max(0.0, now - last_t)
            pos = my_state.pose.pose.position
            dist_home = math.hypot(float(pos.x), float(pos.y))

            status = DiagnosticStatus()
            status.hardware_id = drone_id
            if delay < 0.25:
                status.level = DiagnosticStatus.OK
                status.name = 'Link Status: GOOD'
                status.message = 'All systems nominal'
                rssi_dbm = int(round(-50.0 - 0.18 * min(dist_home, 150.0)))
            elif delay < 0.80:
                status.level = DiagnosticStatus.WARN
                status.name = 'Link Status: WEAK'
                status.message = 'High telemetry latency'
                rssi_dbm = -82
            else:
                status.level = DiagnosticStatus.ERROR
                status.name = 'Link Status: LOST'
                status.message = 'Bridge telemetry timeout'
                rssi_dbm = -110

            ekf_str = self.fc_diags.get(drone_id, {}).get('ekf_aligned', 'True')
            status.values = [
                KeyValue(key='delay_ms', value=str(round(delay * 1000.0, 1))),
                KeyValue(key='rssi', value=str(rssi_dbm)),
                KeyValue(key='mode', value=self.drone_modes.get(drone_id, 'GROUND')),
                KeyValue(key='ekf_aligned', value=str(ekf_str)),
            ]
            self.link_status_pubs[drone_id].publish(status)

            raw_batt = self.raw_batteries.get(drone_id)
            batt_pct = float(raw_batt.percentage) if raw_batt is not None else 1.0
            batt_v = float(raw_batt.voltage) if raw_batt is not None else 25.2
            fleet_snapshot['drones'][drone_id] = {
                'mode': self.drone_modes.get(drone_id, 'GROUND'),
                'pos': [round(float(pos.x), 2), round(float(pos.y), 2), round(float(pos.z), 2)],
                'battery_pct': round(batt_pct, 3),
                'voltage_v': round(batt_v, 2),
                'delay_ms': round(delay * 1000.0, 1),
                'rssi': rssi_dbm,
            }

        if now - self.last_fleet_pub_time >= 0.2 and fleet_snapshot['drones']:
            self.last_fleet_pub_time = now
            f_msg = String()
            f_msg.data = json.dumps(fleet_snapshot)
            self.fleet_state_pub.publish(f_msg)

        if now - self.last_battery_pub_time >= 0.5:
            self.last_battery_pub_time = now
            stamp_msg = self.get_clock().now().to_msg()
            for drone_id in self.swarm_config:
                raw = self.raw_batteries.get(drone_id)
                if raw is not None:
                    out_batt = copy.deepcopy(raw)
                    out_batt.header.stamp = stamp_msg
                    out_batt.header.frame_id = f'{drone_id}_base_link'
                else:
                    out_batt = BatteryState()
                    out_batt.header.stamp = stamp_msg
                    out_batt.header.frame_id = f'{drone_id}_base_link'
                    out_batt.percentage = 1.0
                    out_batt.voltage = 25.2
                    out_batt.temperature = 28.0
                    out_batt.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
                self.battery_pubs[drone_id].publish(out_batt)


def main(args=None):
    """Run the central integration ROS 2 node."""
    rclpy.init(args=args)
    node = CenterNode()
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
