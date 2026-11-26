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

"""Central tactical state machine, 3D CBF safety shield, and velocity feedforward node."""

import copy
import json
import math
import os
import time

from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Empty, String
import yaml

VALID_ACTIONS = {
    'TAKEOFF',
    'HOVER',
    'RETURN',
    'TRACK',
    'INTERCEPT',
    'ATTACK',
    'RESUME_PATROL',
}


def _discover_default_obstacles_yaml():
    """Locate default obstacles.yaml from flight_ctrl_pkg share or source tree."""
    try:
        from ament_index_python.packages import get_package_share_directory
        cand = os.path.join(
            get_package_share_directory('flight_ctrl_pkg'), 'config', 'obstacles.yaml'
        )
        if os.path.isfile(cand):
            return cand
    except Exception:
        pass
    rel_cand = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            '..',
            'flight_ctrl_pkg',
            'config',
            'obstacles.yaml',
        )
    )
    if os.path.isfile(rel_cand):
        return rel_cand
    ws_cand = '/home/hkw/flying/ros2_ws/src/flight_ctrl_pkg/config/obstacles.yaml'
    if os.path.isfile(ws_cand):
        return ws_cand
    return ''


def _load_arena_config(yaml_path):
    """Load arena_bounds, helipads, obstacles, and no_fly_zones from YAML."""
    path = yaml_path if (yaml_path and os.path.isfile(yaml_path)) else (
        _discover_default_obstacles_yaml()
    )
    if not path or not os.path.isfile(path):
        return {'arena_bounds': {}, 'helipads': [], 'obstacles': [], 'no_fly_zones': []}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return {'arena_bounds': {}, 'helipads': [], 'obstacles': [], 'no_fly_zones': []}
        return {
            'arena_bounds': raw.get('arena_bounds') or {},
            'helipads': raw.get('helipads') or [],
            'obstacles': raw.get('obstacles') or [],
            'no_fly_zones': raw.get('no_fly_zones') or [],
        }
    except Exception:
        return {'arena_bounds': {}, 'helipads': [], 'obstacles': [], 'no_fly_zones': []}


def _volume_clearance_and_push(px, py, pz, vol, is_nfz=False, extra_buf=0.0):
    """Compute signed 3D clearance to an obstacle/NFZ and outward unit push vector."""
    vtype = 'cylinder' if is_nfz else str(vol.get('type', 'box')).lower()
    cx = float(vol.get('cx', 0.0))
    cy = float(vol.get('cy', 0.0))
    z_min = float(vol.get('z_min', 0.0))
    z_max = float(vol.get('z_max', 20.0))
    margin = 0.0 if is_nfz else float(vol.get('safety_margin_m', 0.0))
    z_top = z_max + margin

    if vtype == 'cylinder':
        r_eff = float(vol.get('radius', 1.0)) + margin
        dx = px - cx
        dy = py - cy
        d_xy = math.hypot(dx, dy)
        d_rad = d_xy - r_eff
        if d_xy > 1e-4:
            ux, uy = dx / d_xy, dy / d_xy
        else:
            ux, uy = -1.0, 0.0
        if pz < z_min:
            clr = math.hypot(max(d_rad, 0.0), z_min - pz)
        elif pz > z_top:
            clr = (pz - z_top) if d_rad <= 0.0 else math.hypot(d_rad, pz - z_top)
        else:
            clr = d_rad

        need_r = r_eff + extra_buf
        if pz <= z_top + extra_buf and d_xy < need_r:
            return clr, (cx + ux * need_r, cy + uy * need_r, pz)
        return clr, None

    sx = float(vol.get('sx', 2.0))
    sy = float(vol.get('sy', 2.0))
    yaw = float(vol.get('yaw', 0.0))
    hx = 0.5 * sx + margin
    hy = 0.5 * sy + margin
    c_y = math.cos(-yaw)
    s_y = math.sin(-yaw)
    rx = px - cx
    ry = py - cy
    lx = c_y * rx - s_y * ry
    ly = s_y * rx + c_y * ry
    dx = abs(lx) - hx
    dy = abs(ly) - hy
    outside_xy = math.hypot(max(dx, 0.0), max(dy, 0.0))
    inside_xy = min(max(dx, dy), 0.0)
    d_xy_box = outside_xy + inside_xy

    if pz < z_min:
        clr = (z_min - pz) if d_xy_box <= 0.0 else math.hypot(d_xy_box, z_min - pz)
    elif pz > z_top:
        clr = (pz - z_top) if d_xy_box <= 0.0 else math.hypot(d_xy_box, pz - z_top)
    else:
        if d_xy_box <= 0.0:
            clr = max(d_xy_box, z_min - pz, pz - z_top)
        else:
            clr = d_xy_box

    hx_b = hx + extra_buf
    hy_b = hy + extra_buf
    if pz <= z_top + extra_buf and abs(lx) < hx_b and abs(ly) < hy_b:
        pen_x = hx_b - abs(lx)
        pen_y = hy_b - abs(ly)
        pen_z = (z_top + extra_buf) - pz if (not is_nfz and pz > z_top - 1.8) else 1e9
        if pen_z < min(pen_x, pen_y):
            return clr, (px, py, z_top + extra_buf)
        if pen_x <= pen_y:
            lx_new = hx_b if lx >= 0.0 else -hx_b
            ly_new = ly
        else:
            lx_new = lx
            ly_new = hy_b if ly >= 0.0 else -hy_b
        c_w = math.cos(yaw)
        s_w = math.sin(yaw)
        wx = cx + c_w * lx_new - s_w * ly_new
        wy = cy + s_w * lx_new + c_w * ly_new
        return clr, (wx, wy, pz)

    return clr, None


def _point_min_clearance(px, py, pz, arena_cfg):
    """Compute minimum signed 3D clearance to all obstacles and NFZs."""
    min_c = 999.0
    for obs in arena_cfg.get('obstacles', []):
        c, _ = _volume_clearance_and_push(px, py, pz, obs, is_nfz=False, extra_buf=0.0)
        if c < min_c:
            min_c = c
    for nfz in arena_cfg.get('no_fly_zones', []):
        c, _ = _volume_clearance_and_push(px, py, pz, nfz, is_nfz=True, extra_buf=0.0)
        if c < min_c:
            min_c = c
    return min_c


class CenterNode(Node):
    """Multi-UAV tactical scheduler, anti-corruption gate, and 3D CBF safety node."""

    def __init__(self):
        """Initialize subscriptions, publishers, 3D CBF shield, and 20Hz control timer."""
        super().__init__('center_node')

        self.declare_parameter('swarm_ids', ['drone_1', 'drone_2', 'drone_3'])
        self.declare_parameter('obstacles_yaml_path', '')
        self.declare_parameter('takeoff_alt', 9.5)
        self.declare_parameter('track_safe_dist', 5.0)
        self.declare_parameter('home_spacing_y', 3.0)
        self.declare_parameter('max_vision_delay', 0.5)
        self.declare_parameter('max_plan_delay', 0.8)
        self.declare_parameter('cbf_safe_dist', 2.2)

        self.swarm_config = [str(d) for d in self.get_parameter('swarm_ids').value]
        self.obstacles_yaml_path = str(self.get_parameter('obstacles_yaml_path').value).strip()
        self.arena_cfg = _load_arena_config(self.obstacles_yaml_path)
        self.takeoff_alt = float(self.get_parameter('takeoff_alt').value)
        self.TRACK_SAFE_DIST = float(self.get_parameter('track_safe_dist').value)
        self.home_spacing_y = float(self.get_parameter('home_spacing_y').value)
        self.max_vision_delay = float(self.get_parameter('max_vision_delay').value)
        self.max_plan_delay = float(self.get_parameter('max_plan_delay').value)
        self.cbf_safe_dist = max(1.6, float(self.get_parameter('cbf_safe_dist').value))

        self.helipads = {}
        for pad in self.arena_cfg.get('helipads', []):
            d_id = str(pad.get('drone_id', '')).strip()
            if d_id:
                self.helipads[d_id] = (float(pad.get('x', 0.0)), float(pad.get('y', 0.0)))

        self.heartbeat_pubs = {}
        self.fmu_pubs = {}
        self.ff_pubs = {}
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
        self.latest_plan_vels = {d_id: (0.0, 0.0, 0.0) for d_id in self.swarm_config}
        self.latest_plan_focus = {d_id: None for d_id in self.swarm_config}
        self.last_plan_times = {d_id: 0.0 for d_id in self.swarm_config}

        self.current_mode = 'GROUND'
        self.last_vision_time = 0.0
        self.latest_vision_target = None
        self.last_battery_pub_time = 0.0
        self.last_fleet_pub_time = 0.0
        self.last_cbf_active_count = 0

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
            self.ff_pubs[drone_id] = self.create_publisher(
                TwistStamped, f'/{drone_id}/fmu/in/trajectory_ff', 10
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
        self.get_logger().info(
            f'中枢节点已启动(20Hz): 集群={self.swarm_config}, '
            f'障碍物={len(self.arena_cfg.get("obstacles", []))}'
        )

    def _get_home_xy(self, drone_id):
        """Return helipad home (x, y) for drone_id."""
        if drone_id in self.helipads:
            return self.helipads[drone_id]
        idx = self.swarm_config.index(drone_id) if drone_id in self.swarm_config else 0
        return (0.0, idx * self.home_spacing_y)

    def _project_outside_obstacles(self, px, py, pz, cur_x, cur_y, cur_z, buf=0.65):
        """Project candidate setpoint (px, py, pz) outside all 3D obstacles and NFZs."""
        x, y, z = float(px), float(py), float(pz)
        modified = False
        all_vols = [
            (obs, False) for obs in self.arena_cfg.get('obstacles', [])
        ] + [
            (nfz, True) for nfz in self.arena_cfg.get('no_fly_zones', [])
        ]
        if not all_vols:
            return x, y, z, False

        # Check dense ray samples from current pos to target so obstacles ahead are not skipped
        for alpha in (0.15, 0.32, 0.50, 0.72, 1.0):
            sx = cur_x + alpha * (x - cur_x)
            sy = cur_y + alpha * (y - cur_y)
            sz = cur_z + alpha * (z - cur_z)
            for vol, is_nfz in all_vols:
                _, pushed = _volume_clearance_and_push(
                    sx, sy, sz, vol, is_nfz=is_nfz, extra_buf=max(buf, 0.95)
                )
                if pushed is not None:
                    modified = True
                    # Deflect target toward the pushed safe boundary point
                    x, y, z = pushed[0], pushed[1], max(z, pushed[2])

        for _ in range(3):
            any_push = False
            for vol, is_nfz in all_vols:
                _, pushed = _volume_clearance_and_push(
                    x, y, z, vol, is_nfz=is_nfz, extra_buf=max(buf, 0.95)
                )
                if pushed is not None:
                    x, y, z = pushed
                    modified = True
                    any_push = True
            if not any_push:
                break
        return x, y, z, modified

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
        elif mode not in ('TAKEOFF', 'RETURN', 'GROUND'):
            if out_msg.pose.position.z < self.MIN_ALT:
                out_msg.pose.position.z = self.MIN_ALT + 0.1
                self.get_logger().warn(
                    f'[{drone_id}] 目标高度过低，拉高至 {self.MIN_ALT + 0.1:.1f}m',
                    throttle_duration_sec=2.0,
                )

        self.fmu_pubs[drone_id].publish(out_msg)

    def publish_velocity_ff(self, drone_id, vx, vy, vz):
        """Publish 3D velocity feedforward TwistStamped in world ENU frame."""
        if drone_id not in self.ff_pubs:
            return
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.twist.linear.x = float(vx) if math.isfinite(float(vx)) else 0.0
        msg.twist.linear.y = float(vy) if math.isfinite(float(vy)) else 0.0
        msg.twist.linear.z = float(vz) if math.isfinite(float(vz)) else 0.0
        self.ff_pubs[drone_id].publish(msg)

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
        """Apply 3D CBF inter-UAV separation, symmetry breaking, and obstacle shield."""
        my_index = self.swarm_config.index(drone_id)
        cur_x = float(my_state.pose.pose.position.x)
        cur_y = float(my_state.pose.pose.position.y)
        cur_z = float(my_state.pose.pose.position.z)
        cbf_triggered = False

        for other_id in self.swarm_config:
            if other_id == drone_id:
                continue

            other_state = self.drone_states.get(other_id)
            other_mode = self.drone_modes.get(other_id, 'GROUND')
            if other_state is None or other_mode == 'GROUND':
                continue

            other_idx = self.swarm_config.index(other_id)
            ox = float(other_state.pose.pose.position.x)
            oy = float(other_state.pose.pose.position.y)
            oz = float(other_state.pose.pose.position.z)
            odx = cur_x - ox
            ody = cur_y - oy
            odz = cur_z - oz
            dist = math.sqrt(odx * odx + ody * ody + odz * odz)

            if not (0.001 < dist < self.COLLISION_WARN_DIST):
                continue

            if my_mode in ('TAKEOFF', 'RETURN') and dist >= 2.2:
                continue

            cbf_triggered = True
            horiz = math.hypot(odx, ody)
            if horiz > 1e-4:
                ux, uy = odx / horiz, ody / horiz
            else:
                ang = (2.0 * math.pi * my_index) / max(len(self.swarm_config), 1)
                ux, uy = math.cos(ang), math.sin(ang)

            # Right-hand lateral normal breaks exact collinear head-on symmetry
            lat_x, lat_y = -uy, ux

            if dist < self.COLLISION_DANGER_DIST:
                push = max(2.2, (self.COLLISION_DANGER_DIST - dist) * 2.2)
                lat_push = max(1.4, (self.COLLISION_DANGER_DIST - dist) * 1.5)
                z_sign = 1.0 if my_index < other_idx else -1.0
                target.pose.position.x = cur_x + ux * push + lat_x * lat_push
                target.pose.position.y = cur_y + uy * push + lat_y * lat_push
                target.pose.position.z = max(
                    self.MIN_ALT + 0.2,
                    min(self.MAX_ALT, cur_z + z_sign * 0.8),
                )
            else:
                push = (self.COLLISION_WARN_DIST - dist) * 1.15
                target.pose.position.x += (0.65 * ux + 0.75 * lat_x) * push
                target.pose.position.y += (0.65 * uy + 0.75 * lat_y) * push
                if abs(odz) < 2.0 and my_index > other_idx:
                    target.pose.position.z = min(
                        float(target.pose.position.z) + 1.6, 25.0
                    )

        px, py, pz, obs_mod = self._project_outside_obstacles(
            target.pose.position.x,
            target.pose.position.y,
            target.pose.position.z,
            cur_x,
            cur_y,
            cur_z,
            buf=0.65,
        )
        if obs_mod:
            target.pose.position.x = px
            target.pose.position.y = py
            target.pose.position.z = pz
            cbf_triggered = True

        return cbf_triggered

    def _joint_cbf_deconflict_setpoints(self, pending_targets, cbf_flags):
        """Enforce pairwise 3D setpoint separation and obstacle clearance across all UAVs."""
        active_ids = list(pending_targets.keys())
        if not active_ids:
            return

        min_sp_dist = self.cbf_safe_dist
        n_act = len(active_ids)
        for _ in range(6):
            for i in range(n_act):
                id_a = active_ids[i]
                pa = pending_targets[id_a].pose.position
                st_a = self.drone_states.get(id_a)
                for j in range(i + 1, n_act):
                    id_b = active_ids[j]
                    pb = pending_targets[id_b].pose.position
                    st_b = self.drone_states.get(id_b)

                    dx = float(pa.x - pb.x)
                    dy = float(pa.y - pb.y)
                    dz = float(pa.z - pb.z)
                    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if dist < min_sp_dist:
                        cbf_flags[id_a] = True
                        cbf_flags[id_b] = True
                        if dist > 1e-3:
                            ux, uy, uz = dx / dist, dy / dist, dz / dist
                        elif st_a is not None and st_b is not None:
                            sdx = float(
                                st_a.pose.pose.position.x - st_b.pose.pose.position.x
                            )
                            sdy = float(
                                st_a.pose.pose.position.y - st_b.pose.pose.position.y
                            )
                            sdz = float(
                                st_a.pose.pose.position.z - st_b.pose.pose.position.z
                            )
                            sdist = math.sqrt(sdx * sdx + sdy * sdy + sdz * sdz)
                            if sdist > 1e-3:
                                ux, uy, uz = sdx / sdist, sdy / sdist, sdz / sdist
                            else:
                                ang = (2.0 * math.pi * i) / max(n_act, 1)
                                ux, uy, uz = math.cos(ang), math.sin(ang), 0.0
                        else:
                            ang = (2.0 * math.pi * i) / max(n_act, 1)
                            ux, uy, uz = math.cos(ang), math.sin(ang), 0.0

                        corr = 0.52 * (min_sp_dist - dist)
                        pa.x = float(pa.x + ux * corr)
                        pa.y = float(pa.y + uy * corr)
                        pa.z = float(pa.z + uz * corr * 0.4)
                        pb.x = float(pb.x - ux * corr)
                        pb.y = float(pb.y - uy * corr)
                        pb.z = float(pb.z - uz * corr * 0.4)

                    # Also prevent setpoint A from sitting on top of drone B's current pose
                    if st_b is not None:
                        bx = float(st_b.pose.pose.position.x)
                        by = float(st_b.pose.pose.position.y)
                        bz = float(st_b.pose.pose.position.z)
                        adx = float(pa.x - bx)
                        ady = float(pa.y - by)
                        adz = float(pa.z - bz)
                        adist = math.sqrt(adx * adx + ady * ady + adz * adz)
                        if 1e-4 < adist < min_sp_dist:
                            cbf_flags[id_a] = True
                            push = min_sp_dist - adist
                            pa.x = float(pa.x + (adx / adist) * push)
                            pa.y = float(pa.y + (ady / adist) * push)

                    if st_a is not None:
                        ax = float(st_a.pose.pose.position.x)
                        ay = float(st_a.pose.pose.position.y)
                        az = float(st_a.pose.pose.position.z)
                        bdx = float(pb.x - ax)
                        bdy = float(pb.y - ay)
                        bdz = float(pb.z - az)
                        bdist = math.sqrt(bdx * bdx + bdy * bdy + bdz * bdz)
                        if 1e-4 < bdist < min_sp_dist:
                            cbf_flags[id_b] = True
                            push = min_sp_dist - bdist
                            pb.x = float(pb.x + (bdx / bdist) * push)
                            pb.y = float(pb.y + (bdy / bdist) * push)

            for d_id in active_ids:
                p = pending_targets[d_id].pose.position
                st = self.drone_states.get(d_id)
                cx = float(st.pose.pose.position.x) if st is not None else float(p.x)
                cy = float(st.pose.pose.position.y) if st is not None else float(p.y)
                cz = float(st.pose.pose.position.z) if st is not None else float(p.z)
                nx, ny, nz, mod = self._project_outside_obstacles(
                    p.x, p.y, p.z, cx, cy, cz, buf=0.65
                )
                if mod:
                    p.x, p.y, p.z = nx, ny, nz
                    cbf_flags[d_id] = True

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
        """Validate and store per-drone trajectory target and velocity from planning_pkg."""
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
            min_z = 0.0 if act == 'RETURN' else (
                self.ATTACK_MIN_ALT if act == 'ATTACK' else self.MIN_ALT
            )

            if not self._is_valid_xyz(x, y, z, min_z):
                self.get_logger().warn(
                    '规划数据越界或含非法浮点数，已拦截', throttle_duration_sec=1.5
                )
                return

            vx = float(data.get('vx', 0.0))
            vy = float(data.get('vy', 0.0))
            vz = float(data.get('vz', 0.0))
            if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz)):
                vx, vy, vz = 0.0, 0.0, 0.0

            target = PoseStamped()
            target.header.stamp = self.get_clock().now().to_msg()
            target.header.frame_id = 'world'
            target.pose.position.x = x
            target.pose.position.y = y
            target.pose.position.z = z
            target.pose.orientation.w = 1.0

            self.latest_plan_targets[drone_id] = target
            self.latest_plan_vels[drone_id] = (vx, vy, vz)
            if 'focus_x' in data and 'focus_y' in data and 'focus_z' in data:
                fx = float(data['focus_x'])
                fy = float(data['focus_y'])
                fz = float(data['focus_z'])
                if math.isfinite(fx) and math.isfinite(fy) and math.isfinite(fz):
                    focus = PoseStamped()
                    focus.header = target.header
                    focus.pose.position.x = fx
                    focus.pose.position.y = fy
                    focus.pose.position.z = fz
                    focus.pose.orientation.w = 1.0
                    self.latest_plan_focus[drone_id] = focus
                else:
                    self.latest_plan_focus[drone_id] = None
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
            if not isinstance(data, dict) or 'action' not in data:
                return
            raw_id = str(data.get('drone_id', 'ALL')).strip()
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
                        self.takeoff_anchors[d_id] = self._get_home_xy(d_id)

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
        """Execute 20Hz state machine routing, 3D CBF safety shield, and telemetry."""
        now = time.time()

        for drone_id in self.heartbeat_pubs:
            self.heartbeat_pubs[drone_id].publish(Empty())

        if now - self.last_vision_time > 2.0:
            self.latest_vision_target = None

        for d_id in self.swarm_config:
            if now - self.last_plan_times[d_id] > 2.0:
                self.latest_plan_targets[d_id] = None
                self.latest_plan_vels[d_id] = (0.0, 0.0, 0.0)
                self.latest_plan_focus[d_id] = None

        pending_targets = {}
        pending_gimbal = {}
        cbf_flags = {d_id: False for d_id in self.swarm_config}

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
                if self.apply_collision_avoidance(drone_id, my_state, target, mode):
                    cbf_flags[drone_id] = True
                pending_targets[drone_id] = target

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
                        if 0.01 < dist < 3.6:
                            cbf_flags[drone_id] = True
                            horiz = max(math.hypot(odx, ody), 1e-4)
                            push = (3.6 - dist) * 1.2
                            tx += (odx / horiz) * push
                            ty += (ody / horiz) * push
                px, py, pz, obs_mod = self._project_outside_obstacles(
                    tx, ty, tz, cur_x, cur_y, cur_z, buf=0.65
                )
                if obs_mod:
                    tx, ty, tz = px, py, pz
                    cbf_flags[drone_id] = True
                target = PoseStamped()
                target.header.frame_id = 'world'
                target.pose.position.x = tx
                target.pose.position.y = ty
                target.pose.position.z = tz
                pending_targets[drone_id] = target

            elif mode == 'RESUME_PATROL':
                my_plan = self.latest_plan_targets.get(drone_id)
                target = (
                    copy.deepcopy(my_plan)
                    if my_plan is not None
                    else self._compute_builtin_patrol(drone_id, now)
                )
                if self.apply_collision_avoidance(drone_id, my_state, target, mode):
                    cbf_flags[drone_id] = True
                pending_targets[drone_id] = target
                if self.latest_vision_target is not None:
                    pending_gimbal[drone_id] = (self.latest_vision_target, my_state)

            elif mode == 'RETURN':
                home_x, home_y = self._get_home_xy(drone_id)
                horiz_dist = math.hypot(cur_x - home_x, cur_y - home_y)
                vx = float(my_state.twist.twist.linear.x)
                vy = float(my_state.twist.twist.linear.y)
                vz = float(my_state.twist.twist.linear.z)
                horiz_spd = math.hypot(vx, vy)
                speed = math.sqrt(vx * vx + vy * vy + vz * vz)

                target = PoseStamped()
                target.header.frame_id = 'world'
                if horiz_dist > 0.32 or (cur_z > 2.0 and horiz_spd > 0.38):
                    ret_alt = (
                        max(cur_z, 10.5 + 1.4 * my_idx)
                        if horiz_dist > 2.2
                        else max(cur_z, 4.5)
                    )
                    if horiz_dist > 5.0:
                        ux = (home_x - cur_x) / horiz_dist
                        uy = (home_y - cur_y) / horiz_dist
                        target.pose.position.x = cur_x + ux * 5.0
                        target.pose.position.y = cur_y + uy * 5.0
                    else:
                        target.pose.position.x = home_x
                        target.pose.position.y = home_y
                    target.pose.position.z = min(ret_alt, 22.0)
                    if horiz_dist > 1.5:
                        if self.apply_collision_avoidance(drone_id, my_state, target, mode):
                            cbf_flags[drone_id] = True
                else:
                    target.pose.position.x = home_x
                    target.pose.position.y = home_y
                    target.pose.position.z = 0.0
                pending_targets[drone_id] = target

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
                    target = copy.deepcopy(my_plan)
                elif self.latest_vision_target is not None:
                    raw_v = self.latest_vision_target
                    target = copy.deepcopy(raw_v)
                    target.pose.position.z = max(
                        self.MIN_ALT + 3.0,
                        float(raw_v.pose.position.z) + 3.4 + 0.4 * my_idx,
                    )
                    dx = float(raw_v.pose.position.x) - cur_x
                    dy = float(raw_v.pose.position.y) - cur_y
                    h_dist = math.hypot(dx, dy)
                    if 0.01 < h_dist < self.TRACK_SAFE_DIST:
                        ux, uy = dx / h_dist, dy / h_dist
                        push = (self.TRACK_SAFE_DIST - h_dist) * 1.5
                        target.pose.position.x = (
                            float(raw_v.pose.position.x)
                            - ux * (self.TRACK_SAFE_DIST * 0.85)
                            - uy * push
                        )
                        target.pose.position.y = (
                            float(raw_v.pose.position.y)
                            - uy * (self.TRACK_SAFE_DIST * 0.85)
                            + ux * push
                        )
                else:
                    continue

                if self.apply_collision_avoidance(drone_id, my_state, target, mode):
                    cbf_flags[drone_id] = True
                pending_targets[drone_id] = target
                focus_pt = self.latest_plan_focus.get(drone_id)
                gimbal_ref = (
                    self.latest_vision_target
                    if self.latest_vision_target is not None
                    else (focus_pt if focus_pt is not None else target)
                )
                pending_gimbal[drone_id] = (gimbal_ref, my_state)

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

                if self.apply_collision_avoidance(drone_id, my_state, target, mode):
                    cbf_flags[drone_id] = True
                pending_targets[drone_id] = target
                focus_pt = self.latest_plan_focus.get(drone_id)
                gimbal_ref = (
                    self.latest_vision_target
                    if self.latest_vision_target is not None
                    else (focus_pt if focus_pt is not None else target)
                )
                pending_gimbal[drone_id] = (gimbal_ref, my_state)

        self._joint_cbf_deconflict_setpoints(pending_targets, cbf_flags)
        self.last_cbf_active_count = sum(1 for v in cbf_flags.values() if v)

        for drone_id, target in pending_targets.items():
            self.send_single_target(drone_id, target)
            mode = self.drone_modes.get(drone_id, 'GROUND')
            my_state = self.drone_states.get(drone_id)
            if mode == 'HOVER':
                self.publish_velocity_ff(drone_id, 0.0, 0.0, 0.0)
            elif my_state is not None:
                pvx, pvy, pvz = self.latest_plan_vels.get(drone_id, (0.0, 0.0, 0.0))
                plan_fresh = (now - self.last_plan_times[drone_id]) <= 1.2
                dx = float(target.pose.position.x - my_state.pose.pose.position.x)
                dy = float(target.pose.position.y - my_state.pose.pose.position.y)
                dz = float(target.pose.position.z - my_state.pose.pose.position.z)
                if plan_fresh and not cbf_flags.get(drone_id, False) and (
                    math.hypot(pvx, pvy) > 0.05 or abs(pvz) > 0.05
                ):
                    vx_cmd = pvx + 0.8 * dx
                    vy_cmd = pvy + 0.8 * dy
                    vz_cmd = pvz + 0.8 * dz
                else:
                    gain = 2.2 if mode == 'ATTACK' else (1.6 if mode == 'INTERCEPT' else 1.2)
                    vx_cmd = gain * dx + (0.35 * pvx if plan_fresh else 0.0)
                    vy_cmd = gain * dy + (0.35 * pvy if plan_fresh else 0.0)
                    vz_cmd = 1.2 * dz + (0.35 * pvz if plan_fresh else 0.0)
                h_spd = math.hypot(vx_cmd, vy_cmd)
                v_cap = 13.5 if mode == 'ATTACK' else (10.5 if mode == 'INTERCEPT' else 6.5)
                if h_spd > v_cap:
                    vx_cmd *= v_cap / h_spd
                    vy_cmd *= v_cap / h_spd
                vz_cmd = max(-4.0, min(4.0, vz_cmd))
                self.publish_velocity_ff(drone_id, vx_cmd, vy_cmd, vz_cmd)

        for drone_id, (gimbal_ref, my_state) in pending_gimbal.items():
            self.publish_gimbal_target(drone_id, gimbal_ref, my_state)

        self._publish_telemetry(now)

    def _publish_telemetry(self, now):
        """Publish link health, real 6S battery state, and aggregated /fleet/state."""
        fleet_snapshot = {'stamp': round(now, 3), 'drones': {}}
        active_positions = []
        min_obs_clr = 999.0

        for drone_id in self.swarm_config:
            my_state = self.drone_states.get(drone_id)
            last_t = self.last_state_times.get(drone_id, 0.0)
            if my_state is None or last_t <= 0.0:
                continue

            delay = max(0.0, now - last_t)
            pos = my_state.pose.pose.position
            px, py, pz = float(pos.x), float(pos.y), float(pos.z)
            active_positions.append((px, py, pz))
            clr = _point_min_clearance(px, py, pz, self.arena_cfg)
            if clr < min_obs_clr:
                min_obs_clr = clr
            dist_home = math.hypot(px, py)

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
                'pos': [round(px, 2), round(py, 2), round(pz, 2)],
                'battery_pct': round(batt_pct, 3),
                'voltage_v': round(batt_v, 2),
                'delay_ms': round(delay * 1000.0, 1),
                'rssi': rssi_dbm,
            }

        min_inter_dist = 999.0
        for i in range(len(active_positions)):
            for j in range(i + 1, len(active_positions)):
                d_ij = math.dist(active_positions[i], active_positions[j])
                if d_ij < min_inter_dist:
                    min_inter_dist = d_ij

        fleet_snapshot['min_inter_drone_dist_m'] = round(min_inter_dist, 4)
        fleet_snapshot['min_obstacle_clearance_m'] = round(min_obs_clr, 4)
        fleet_snapshot['cbf_active_count'] = int(self.last_cbf_active_count)

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
