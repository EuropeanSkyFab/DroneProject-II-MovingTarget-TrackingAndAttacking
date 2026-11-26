# -*- coding: utf-8 -*-
"""
target_sim_node.py —— 多目标高机动三维逃逸靶机控制器、击毁物理裁决器与外部雷达模拟器

核心职责：
  1. 支持同时管理 3 架敌方高机动无人机靶标：
       - red_moving_target   (class_id: crimson_uav)
       - amber_escort_target (class_id: amber_uav)
       - slate_recon_target  (class_id: slate_uav)
     支持 Gazebo Harmonic (gz.transport13 闭环 VelocityControl + Strike Burst 粒子爆裂特效)
     及无 Gazebo 环境下的纯运动学模式 (sim_backend='kinematic')。
  2. 严格约束全部靶机位于 obstacles.yaml 的 arena_bounds 与障碍物安全包线之外。
  3. 实时监测 /drone_<id>/state 与 ACTIVE 靶机间距，命中即触发幂等击毁：
       - 更新靶机 status = "NEUTRALIZED" 并冻结速度 / 触发爆裂烟火；
       - 发布 Contract #4 /engagement/kill_event (std_msgs/String JSON)；
       - 仅将剩余 ACTIVE 靶机发布至 /external/targets_detected；
       - 全量靶机状态持续发布至 /sim/targets_gt (25 Hz)，并保留 /sim/target_gt 及
         /external/target_detected 向后兼容。
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
import yaml


_DRONE_STATE_RE = re.compile(r'^/(drone_\d+)/state$')


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _quat_to_yaw_enu(qx: float, qy: float, qz: float, qw: float) -> float:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def _yaw_to_quat_enu(yaw: float) -> Tuple[float, float, float, float]:
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def _discover_default_obstacles_yaml() -> str:
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
        os.path.join(os.path.dirname(__file__), '..', 'config', 'obstacles.yaml')
    )
    if os.path.isfile(rel_cand):
        return rel_cand
    ws_cand = '/home/hkw/flying/ros2_ws/src/flight_ctrl_pkg/config/obstacles.yaml'
    if os.path.isfile(ws_cand):
        return ws_cand
    return ''


def _load_obstacles_cfg(yaml_path: str) -> Dict[str, Any]:
    path = yaml_path if (yaml_path and os.path.isfile(yaml_path)) else _discover_default_obstacles_yaml()
    default_cfg: Dict[str, Any] = {
        'arena_bounds': {
            'x_min': -16.0,
            'x_max': 96.0,
            'y_min': -34.0,
            'y_max': 34.0,
            'z_min': 0.0,
            'z_max': 45.0,
        },
        'obstacles': [],
    }
    if not path or not os.path.isfile(path):
        return default_cfg
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return default_cfg
        return {
            'arena_bounds': raw.get('arena_bounds') or default_cfg['arena_bounds'],
            'obstacles': raw.get('obstacles') or [],
        }
    except Exception:
        return default_cfg


def _clamp_and_clear_obstacles(
    pos: np.ndarray,
    cfg: Dict[str, Any],
    extra_buf: float = 0.65,
) -> Tuple[np.ndarray, bool]:
    """Clamp position inside arena_bounds and push outside all 3D obstacles + safety margin."""
    bounds = cfg.get('arena_bounds') or {}
    x_min = float(bounds.get('x_min', -16.0)) + 1.0
    x_max = float(bounds.get('x_max', 96.0)) - 1.0
    y_min = float(bounds.get('y_min', -34.0)) + 1.0
    y_max = float(bounds.get('y_max', 34.0)) - 1.0
    z_min = max(3.0, float(bounds.get('z_min', 0.0)) + 3.0)
    z_max = float(bounds.get('z_max', 45.0)) - 1.5

    x = max(x_min, min(x_max, float(pos[0])))
    y = max(y_min, min(y_max, float(pos[1])))
    z = max(z_min, min(z_max, float(pos[2])))
    pushed = False

    obstacles = cfg.get('obstacles') or []
    for _ in range(3):
        any_hit = False
        for obs in obstacles:
            otype = str(obs.get('type', 'box')).lower()
            cx = float(obs.get('cx', 0.0))
            cy = float(obs.get('cy', 0.0))
            oz_min = float(obs.get('z_min', 0.0))
            oz_max = float(obs.get('z_max', 20.0))
            margin = float(obs.get('safety_margin_m', 0.0))
            z_top = oz_max + margin + extra_buf
            if z < oz_min - extra_buf or z > z_top:
                continue

            if otype == 'cylinder':
                r_req = float(obs.get('radius', 1.0)) + margin + extra_buf
                dx = x - cx
                dy = y - cy
                d_xy = math.hypot(dx, dy)
                if d_xy < r_req:
                    ux, uy = (dx / d_xy, dy / d_xy) if d_xy > 1e-4 else (1.0, 0.0)
                    x = cx + ux * r_req
                    y = cy + uy * r_req
                    pushed = True
                    any_hit = True
            else:
                sx = float(obs.get('sx', 2.0))
                sy = float(obs.get('sy', 2.0))
                yaw = float(obs.get('yaw', 0.0))
                hx = 0.5 * sx + margin + extra_buf
                hy = 0.5 * sy + margin + extra_buf
                c_y = math.cos(-yaw)
                s_y = math.sin(-yaw)
                rx, ry = x - cx, y - cy
                lx = c_y * rx - s_y * ry
                ly = s_y * rx + c_y * ry
                if abs(lx) < hx and abs(ly) < hy:
                    pen_x = hx - abs(lx)
                    pen_y = hy - abs(ly)
                    if pen_x <= pen_y:
                        lx = hx if lx >= 0.0 else -hx
                    else:
                        ly = hy if ly >= 0.0 else -hy
                    c_w = math.cos(yaw)
                    s_w = math.sin(yaw)
                    x = cx + c_w * lx - s_w * ly
                    y = cy + s_w * lx + c_w * ly
                    pushed = True
                    any_hit = True
        if not any_hit:
            break

    return np.array([x, y, z], dtype=float), pushed


@dataclass
class _TargetEntity:
    target_id: str
    class_id: str
    center_x: float
    center_y: float
    base_z: float
    amp_z: float
    rx: float
    ry: float
    omega: float
    phase: float
    burst_idx: int
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    yaw: float = 0.0
    status: str = 'ACTIVE'
    gz_pose_received: bool = False
    prev_gz_pos: Optional[np.ndarray] = None
    prev_gz_time: float = 0.0


class TargetSimNode(Node):
    """多目标三维机动靶机驱动、击毁判定与外部雷达发布节点。"""

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
        self.declare_parameter('enable_multi_target', True)
        self.declare_parameter('obstacles_yaml_path', '')
        self.declare_parameter('hit_radius_m', 1.25)

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
        self.enable_multi_target: bool = bool(p('enable_multi_target').value)
        self.obstacles_cfg: Dict[str, Any] = _load_obstacles_cfg(
            str(p('obstacles_yaml_path').value).strip()
        )
        self.hit_radius_m: float = max(0.3, float(p('hit_radius_m').value))

        self._lock = threading.Lock()
        self._sim_t0: Optional[float] = None
        self._latest_sim_t: float = 0.0
        self._t0: float = time.time()
        self._last_radar_pub: float = 0.0

        # Legacy single-target attributes preserved
        self._gt_pos = np.array([self.cx - 10.2, self.cy, self.base_z], dtype=float)
        self._gt_vel = np.zeros(3, dtype=float)
        self._gt_yaw: float = 0.0
        self._gz_pose_received: bool = False

        self._targets: List[_TargetEntity] = [
            _TargetEntity(
                target_id=self.target_model,
                class_id='crimson_uav',
                center_x=self.cx,
                center_y=self.cy,
                base_z=self.base_z,
                amp_z=self.amp_z,
                rx=10.2,
                ry=5.0,
                omega=0.28,
                phase=0.0,
                burst_idx=1,
            )
        ]
        if self.enable_multi_target:
            self._targets.extend([
                _TargetEntity(
                    target_id='amber_escort_target',
                    class_id='amber_uav',
                    center_x=23.0,
                    center_y=-3.0,
                    base_z=6.8,
                    amp_z=0.75,
                    rx=7.5,
                    ry=4.0,
                    omega=0.31,
                    phase=2.094,
                    burst_idx=2,
                ),
                _TargetEntity(
                    target_id='slate_recon_target',
                    class_id='slate_uav',
                    center_x=44.0,
                    center_y=-2.5,
                    base_z=8.0,
                    amp_z=0.90,
                    rx=6.0,
                    ry=5.5,
                    omega=0.25,
                    phase=4.188,
                    burst_idx=3,
                ),
            ])

        for ent in self._targets:
            p0, _ = self._entity_reference(ent, 0.0)
            p0_safe, _ = _clamp_and_clear_obstacles(p0, self.obstacles_cfg, extra_buf=0.75)
            ent.pos = p0_safe
            ent.vel = np.zeros(3, dtype=float)
        self._gt_pos = self._targets[0].pos.copy()

        # Publishers
        self.gt_pub = self.create_publisher(Odometry, '/sim/target_gt', 20)
        self.radar_pub = self.create_publisher(PoseStamped, '/external/target_detected', 10)
        self.multi_gt_pub = self.create_publisher(String, '/sim/targets_gt', 20)
        self.multi_radar_pub = self.create_publisher(String, '/external/targets_detected', 10)
        self.kill_pub = self.create_publisher(String, '/engagement/kill_event', 20)

        # Subscribe to friendlies for proximity strike kill evaluation
        self._drone_states: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._subscribed_drones: Set[str] = set()
        for idx in range(1, 9):
            self._ensure_drone_sub(f'drone_{idx}')
        self.create_timer(1.0, self._discover_drones)

        # Gazebo transport (gracefully falls back to kinematic if offline)
        self._gz_node = None
        self._gz_cmd_pubs: Dict[str, Any] = {}
        self._gz_fx_pubs: Dict[int, Any] = {}
        self._twist_cls = None
        self._bool_cls = None
        if self.sim_backend in ('auto', 'gazebo'):
            self._init_gz_interface()

        self.dt = 1.0 / self.control_hz
        self.timer = self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f'target_sim_node 已启动: backend={self.sim_backend}, '
            f'targets={[t.target_id for t in self._targets]}, hit_r={self.hit_radius_m:.2f}m'
        )

    def _ensure_drone_sub(self, drone_id: str) -> None:
        if drone_id in self._subscribed_drones:
            return
        self._subscribed_drones.add(drone_id)
        self.create_subscription(
            Odometry,
            f'/{drone_id}/state',
            lambda msg, d_id=drone_id: self._on_drone_state(d_id, msg),
            10,
        )

    def _discover_drones(self) -> None:
        try:
            topics = self.get_topic_names_and_types()
        except Exception:
            return
        for name, types in topics:
            m = _DRONE_STATE_RE.match(name)
            if m and any('Odometry' in t for t in types):
                self._ensure_drone_sub(m.group(1))

    def _on_drone_state(self, drone_id: str, msg: Odometry) -> None:
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        px, py, pz = float(p.x), float(p.y), float(p.z)
        vx, vy, vz = float(v.x), float(v.y), float(v.z)
        if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
            return
        if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz)):
            vx, vy, vz = 0.0, 0.0, 0.0
        with self._lock:
            self._drone_states[drone_id] = (
                np.array([px, py, pz], dtype=float),
                np.array([vx, vy, vz], dtype=float),
            )

    def _init_gz_interface(self) -> None:
        try:
            import gz.transport13 as gzt
            from gz.msgs10 import boolean_pb2, pose_v_pb2, twist_pb2

            self._twist_cls = twist_pb2.Twist
            self._bool_cls = boolean_pb2.Boolean
            self._gz_node = gzt.Node()
            pose_topic = f'/world/{self.gz_world}/dynamic_pose/info'
            self._gz_node.subscribe(pose_v_pb2.Pose_V, pose_topic, self._on_gz_poses)
            for ent in self._targets:
                cmd_topic = f'/model/{ent.target_id}/cmd_vel'
                self._gz_cmd_pubs[ent.target_id] = self._gz_node.advertise(
                    cmd_topic, twist_pb2.Twist
                )
                self._gz_fx_pubs[ent.burst_idx] = self._gz_node.advertise(
                    f'/fx/strike_burst_{ent.burst_idx}/cmd', boolean_pb2.Boolean
                )
            self.get_logger().info(
                f'已挂载 Gazebo 靶机总线: {list(self._gz_cmd_pubs.keys())}'
            )
        except Exception as exc:
            self.get_logger().warn(f'Gazebo transport 不可用，降级为纯运动学靶机模式: {exc}')
            self.sim_backend = 'kinematic'

    def _on_gz_poses(self, msg) -> None:
        now = time.time()
        sim_t = float(msg.header.stamp.sec) + float(msg.header.stamp.nsec) * 1e-9
        ref_t = sim_t if sim_t > 0.0 else now
        by_name = {ent.target_id: ent for ent in self._targets}
        with self._lock:
            if sim_t > 0.0:
                if self._sim_t0 is None:
                    self._sim_t0 = sim_t
                self._latest_sim_t = sim_t
            for pose in msg.pose:
                ent = by_name.get(pose.name)
                if ent is None:
                    continue
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
                if ent.prev_gz_pos is not None and (ref_t - ent.prev_gz_time) > 0.008:
                    dt = ref_t - ent.prev_gz_time
                    raw_v = (pos - ent.prev_gz_pos) / dt
                    if float(np.linalg.norm(raw_v)) < 35.0:
                        ent.vel = 0.65 * ent.vel + 0.35 * raw_v
                ent.prev_gz_pos = pos
                ent.prev_gz_time = ref_t
                ent.pos = pos
                ent.yaw = yaw
                ent.gz_pose_received = True
                if ent.target_id == self.target_model:
                    self._gt_pos = pos.copy()
                    self._gt_vel = ent.vel.copy()
                    self._gt_yaw = yaw
                    self._gz_pose_received = True

    def reference_trajectory(self, t_rel: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算 t_rel 秒时刻主红方靶机期望三维参考位置与参考速度（严格约束在无障碍走廊）。"""
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

        # 默认 evasive_weave
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

    def _entity_reference(self, ent: _TargetEntity, t_rel: float) -> Tuple[np.ndarray, np.ndarray]:
        if ent.target_id == self.target_model:
            p_raw, v_raw = self.reference_trajectory(t_rel)
            p_safe, pushed = _clamp_and_clear_obstacles(p_raw, self.obstacles_cfg, extra_buf=0.75)
            return p_safe, (np.zeros(3, dtype=float) if (pushed and self.speed_scale <= 1e-4) else v_raw)

        s = self.speed_scale
        if self.profile == 'hover' or s <= 1e-4:
            p0 = np.array([ent.center_x, ent.center_y, ent.base_z], dtype=float)
            p_safe, _ = _clamp_and_clear_obstacles(p0, self.obstacles_cfg, extra_buf=0.75)
            return p_safe, np.zeros(3, dtype=float)

        w = ent.omega * s
        th = w * t_rel + ent.phase
        if ent.target_id == 'slate_recon_target':
            # Figure-8 weave in central-east canyon corridor
            x = ent.center_x + ent.rx * math.sin(th)
            y = ent.center_y + ent.ry * math.sin(2.0 * th)
            vx = ent.rx * w * math.cos(th)
            vy = 2.0 * ent.ry * w * math.cos(2.0 * th)
        else:
            # Elliptical weave in west/south corridor
            x = ent.center_x - ent.rx * math.cos(th)
            y = ent.center_y + ent.ry * math.sin(th) + 1.1 * math.sin(2.1 * th)
            vx = ent.rx * w * math.sin(th)
            vy = ent.ry * w * math.cos(th) + 1.1 * 2.1 * w * math.cos(2.1 * th)
        wz = 0.30 * s
        z = ent.base_z + ent.amp_z * math.sin(wz * t_rel + ent.phase)
        vz = ent.amp_z * wz * math.cos(wz * t_rel + ent.phase)
        p_safe, pushed = _clamp_and_clear_obstacles(
            np.array([x, y, z], dtype=float), self.obstacles_cfg, extra_buf=0.75
        )
        v_out = np.array([vx, vy, vz], dtype=float)
        if pushed:
            v_out *= 0.85
        return p_safe, v_out

    def _trigger_strike_fx(self, ent: _TargetEntity) -> None:
        if self._bool_cls is None:
            return
        fx_pub = self._gz_fx_pubs.get(ent.burst_idx)
        if fx_pub is not None:
            try:
                bmsg = self._bool_cls()
                bmsg.data = True
                fx_pub.publish(bmsg)
            except Exception:
                pass
        gz_pub = self._gz_cmd_pubs.get(ent.target_id)
        if gz_pub is not None and self._twist_cls is not None:
            try:
                stop_cmd = self._twist_cls()
                gz_pub.publish(stop_cmd)
            except Exception:
                pass

    def _evaluate_strikes_locked(self, now: float) -> List[Dict[str, Any]]:
        """Evaluate proximity hits between friendly drones and ACTIVE targets (under self._lock)."""
        events: List[Dict[str, Any]] = []
        if not self._drone_states:
            return events

        for ent in self._targets:
            if ent.status != 'ACTIVE':
                continue
            best_did: Optional[str] = None
            best_dist = float('inf')
            best_closing = 0.0
            for d_id, (d_pos, d_vel) in self._drone_states.items():
                rel = ent.pos - d_pos
                dist = float(np.linalg.norm(rel))
                if dist <= self.hit_radius_m and dist < best_dist:
                    best_dist = dist
                    rel_vel = d_vel - ent.vel
                    if dist > 1e-4:
                        closing = max(0.0, float(np.dot(rel_vel, rel / dist)))
                    else:
                        closing = max(0.0, float(np.linalg.norm(rel_vel)))
                    best_closing = closing
                    best_did = d_id

            if best_did is not None:
                ent.status = 'NEUTRALIZED'
                ent.vel = np.zeros(3, dtype=float)
                ev = {
                    'stamp': round(now, 6),
                    'target_id': ent.target_id,
                    'class_id': ent.class_id,
                    'striker_id': best_did,
                    'miss_dist_m': round(best_dist, 4),
                    'closing_speed_ms': round(best_closing, 3),
                    'x': round(float(ent.pos[0]), 4),
                    'y': round(float(ent.pos[1]), 4),
                    'z': round(float(ent.pos[2]), 4),
                }
                events.append(ev)
        return events

    def _tick(self) -> None:
        now = time.time()
        with self._lock:
            sim_t0 = self._sim_t0
            sim_now = self._latest_sim_t
        if sim_t0 is not None and sim_now >= sim_t0:
            t_rel = sim_now - sim_t0
        else:
            t_rel = now - self._t0

        stationary = self.profile == 'hover' or self.speed_scale <= 1e-4

        for ent in self._targets:
            use_gz_ent = (
                self.sim_backend == 'gazebo'
                or (self.sim_backend == 'auto' and ent.gz_pose_received)
            )
            gz_pub = self._gz_cmd_pubs.get(ent.target_id)

            if ent.status == 'NEUTRALIZED':
                ent.vel = np.zeros(3, dtype=float)
                if use_gz_ent and gz_pub is not None and self._twist_cls is not None:
                    gz_pub.publish(self._twist_cls())
                continue

            p_ref, v_ref = self._entity_reference(ent, t_rel)

            if use_gz_ent and gz_pub is not None and self._twist_cls is not None:
                with self._lock:
                    cur_pos = ent.pos.copy()
                    cur_yaw = ent.yaw
                if stationary:
                    v_world = np.zeros(3, dtype=float)
                    yaw_rate = 0.0
                else:
                    err_pos = p_ref - cur_pos
                    v_world = v_ref + np.array([1.25, 1.25, 1.50]) * err_pos
                    v_h = math.hypot(float(v_world[0]), float(v_world[1]))
                    if v_h > 8.0:
                        v_world[0] *= 8.0 / v_h
                        v_world[1] *= 8.0 / v_h
                    v_world[2] = max(-2.5, min(2.5, float(v_world[2])))
                    des_yaw = math.atan2(float(v_world[1]), float(v_world[0])) if v_h > 0.25 else cur_yaw
                    yaw_rate = max(-1.8, min(1.8, 2.4 * _wrap_pi(des_yaw - cur_yaw)))

                c_y, s_y = math.cos(-cur_yaw), math.sin(-cur_yaw)
                cmd = self._twist_cls()
                cmd.linear.x = float(c_y * float(v_world[0]) - s_y * float(v_world[1]))
                cmd.linear.y = float(s_y * float(v_world[0]) + c_y * float(v_world[1]))
                cmd.linear.z = float(v_world[2])
                cmd.angular.z = float(yaw_rate)
                gz_pub.publish(cmd)
            else:
                with self._lock:
                    if stationary:
                        ent.vel = np.zeros(3, dtype=float)
                        ent.pos, _ = _clamp_and_clear_obstacles(
                            ent.pos, self.obstacles_cfg, extra_buf=0.75
                        )
                    else:
                        err = p_ref - ent.pos
                        v_cmd = v_ref + 1.5 * err
                        cand_pos = ent.pos + v_cmd * self.dt
                        safe_pos, pushed = _clamp_and_clear_obstacles(
                            cand_pos, self.obstacles_cfg, extra_buf=0.75
                        )
                        actual_vel = (safe_pos - ent.pos) / max(self.dt, 1e-3)
                        ent.pos = safe_pos
                        ent.vel = actual_vel if pushed else v_cmd
                        if math.hypot(float(ent.vel[0]), float(ent.vel[1])) > 0.2:
                            ent.yaw = math.atan2(float(ent.vel[1]), float(ent.vel[0]))

        with self._lock:
            kill_events = self._evaluate_strikes_locked(now)
            targets_snap = [
                {
                    'target_id': ent.target_id,
                    'class_id': ent.class_id,
                    'x': round(float(ent.pos[0]), 4),
                    'y': round(float(ent.pos[1]), 4),
                    'z': round(float(ent.pos[2]), 4),
                    'vx': round(float(ent.vel[0]), 4),
                    'vy': round(float(ent.vel[1]), 4),
                    'vz': round(float(ent.vel[2]), 4),
                    'yaw': round(float(ent.yaw), 4),
                    'status': ent.status,
                }
                for ent in self._targets
            ]
            lead_ent = next((e for e in self._targets if e.status == 'ACTIVE'), self._targets[0])
            self._gt_pos = lead_ent.pos.copy()
            self._gt_vel = lead_ent.vel.copy()
            self._gt_yaw = lead_ent.yaw

        for kev in kill_events:
            k_msg = String()
            k_msg.data = json.dumps(kev)
            self.kill_pub.publish(k_msg)
            self.get_logger().info(
                f"[击毁判定] striker={kev['striker_id']} 击毁 {kev['target_id']} "
                f"(miss={kev['miss_dist_m']:.3f}m, Vc={kev['closing_speed_ms']:.2f}m/s)"
            )
            for ent in self._targets:
                if ent.target_id == kev['target_id']:
                    self._trigger_strike_fx(ent)

        # Publish Contract /sim/targets_gt (25 Hz)
        m_gt = String()
        m_gt.data = json.dumps({
            'stamp': round(now, 6),
            'frame_id': 'world',
            'targets': targets_snap,
        })
        self.multi_gt_pub.publish(m_gt)

        # Legacy /sim/target_gt (Odometry)
        stamp = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = 'world'
        odom.child_frame_id = lead_ent.target_id
        odom.pose.pose.position.x = float(lead_ent.pos[0])
        odom.pose.pose.position.y = float(lead_ent.pos[1])
        odom.pose.pose.position.z = float(lead_ent.pos[2])
        qx, qy, qz, qw = _yaw_to_quat_enu(lead_ent.yaw)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(lead_ent.vel[0])
        odom.twist.twist.linear.y = float(lead_ent.vel[1])
        odom.twist.twist.linear.z = float(lead_ent.vel[2])
        self.gt_pub.publish(odom)

        # Publish /external/targets_detected (ACTIVE targets only) & legacy /external/target_detected
        if self.enable_radar and (now - self._last_radar_pub) >= (1.0 / self.radar_hz):
            self._last_radar_pub = now
            active_radar: List[Dict[str, Any]] = []
            for det_idx, t in enumerate(targets_snap):
                if t['status'] != 'ACTIVE':
                    continue
                rx = float(t['x'] + self.rng.normal(0.0, self.noise_xy))
                ry = float(t['y'] + self.rng.normal(0.0, self.noise_xy))
                rz = max(3.0, float(t['z'] + self.rng.normal(0.0, self.noise_z)))
                active_radar.append({
                    'det_id': det_idx,
                    'target_id': t['target_id'],
                    'class_id': t['class_id'],
                    'x': round(rx, 4),
                    'y': round(ry, 4),
                    'z': round(rz, 4),
                    'conf': 0.82,
                    'source': 'radar',
                })

            m_rad = String()
            m_rad.data = json.dumps({
                'stamp': round(now, 6),
                'frame_id': 'world',
                'targets': active_radar,
            })
            self.multi_radar_pub.publish(m_rad)

            if active_radar:
                r0 = active_radar[0]
                radar_msg = PoseStamped()
                radar_msg.header.stamp = stamp
                radar_msg.header.frame_id = 'world'
                radar_msg.pose.position.x = float(r0['x'])
                radar_msg.pose.position.y = float(r0['y'])
                radar_msg.pose.position.z = float(r0['z'])
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
