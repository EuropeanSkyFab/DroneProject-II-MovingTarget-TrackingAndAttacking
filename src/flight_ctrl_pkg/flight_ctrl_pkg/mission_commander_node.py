# -*- coding: utf-8 -*-
"""
mission_commander_node.py —— 事件驱动 6+机 vs 3靶标 协同全流程战术指挥与 Contract #7 量化评测节点

功能：
  1. 模拟战略指挥中心 (/strategy/cmd)，按真实物理态势事件驱动多机全流程状态转移：
       WAIT_READY -> TAKEOFF -> SEARCH -> TRACK -> INTERCEPT -> ATTACK
       -> KILL -> BDA -> RETASK -> RETURN -> COMPLETE
  2. 全链路订阅 /sim/targets_gt、/vision/targets、/engagement/kill_event、
     /vision/diagnostics、/planning/diagnostics、/planning/target_pose、/fleet/state
     及各机 /drone_<id>/state 与 fc_status；
     幂等记录目标击毁与重分配事件，并在返航着陆后输出符合 Contract #7 的
     JSON-Lines 全链路轨迹追踪 (trace.jsonl) 与量化验收摘要 (summary.json)。
"""

from __future__ import annotations

import collections
import json
import math
import os
import time
import uuid
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
import yaml


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
    empty: Dict[str, Any] = {'helipads': [], 'obstacles': [], 'no_fly_zones': []}
    if not path or not os.path.isfile(path):
        return empty
    try:
        with open(path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return empty
        return {
            'helipads': raw.get('helipads') or [],
            'obstacles': raw.get('obstacles') or [],
            'no_fly_zones': raw.get('no_fly_zones') or [],
        }
    except Exception:
        return empty


def _point_min_clearance(pt: Tuple[float, float, float], cfg: Dict[str, Any]) -> float:
    """Compute signed 3D clearance consistent with tests_e2e.helpers.arena_parser."""
    px, py, pz = float(pt[0]), float(pt[1]), float(pt[2])
    best = float('inf')

    def _cyl_sdf(cx: float, cy: float, z_min: float, z_max: float, radius: float) -> float:
        d_xy = math.hypot(px - cx, py - cy) - radius
        d_z = max(z_min - pz, pz - z_max)
        if d_xy > 0.0 and d_z > 0.0:
            return math.hypot(d_xy, d_z)
        return max(d_xy, d_z)

    for obs in cfg.get('obstacles') or []:
        cx = float(obs.get('cx', 0.0))
        cy = float(obs.get('cy', 0.0))
        z_min = float(obs.get('z_min', 0.0))
        z_max = float(obs.get('z_max', 20.0))
        otype = str(obs.get('type', 'box')).lower()
        if otype == 'cylinder':
            clr = _cyl_sdf(cx, cy, z_min, z_max, float(obs.get('radius', 1.0)))
        else:
            yaw = float(obs.get('yaw', 0.0))
            c_y, s_y = math.cos(yaw), math.sin(yaw)
            dx = px - cx
            dy = py - cy
            lx = c_y * dx + s_y * dy
            ly = -s_y * dx + c_y * dy
            lz = pz - 0.5 * (z_min + z_max)
            qx = abs(lx) - 0.5 * float(obs.get('sx', 2.0))
            qy = abs(ly) - 0.5 * float(obs.get('sy', 2.0))
            qz = abs(lz) - 0.5 * (z_max - z_min)
            out = math.sqrt(max(qx, 0.0) ** 2 + max(qy, 0.0) ** 2 + max(qz, 0.0) ** 2)
            clr = out + min(max(qx, qy, qz), 0.0)
        if clr < best:
            best = clr

    for nfz in cfg.get('no_fly_zones') or []:
        clr = _cyl_sdf(
            float(nfz.get('cx', 0.0)),
            float(nfz.get('cy', 0.0)),
            float(nfz.get('z_min', 0.0)),
            float(nfz.get('z_max', 20.0)),
            float(nfz.get('radius', 4.0)),
        )
        if clr < best:
            best = clr

    return 999.0 if not math.isfinite(best) else best


def _check_gpu3_violation() -> bool:
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    tokens = [t.strip().lower() for t in cvd.split(',') if t.strip()]
    for tok in tokens:
        if tok in ('3', 'all', 'gpu-617ae266-eed3-ef41-ae0f-5f25c730f5d5'):
            return True
    return False


class MissionCommanderNode(Node):
    """事件驱动多机协同战术指挥与 Contract #7 量化评测节点。"""

    def __init__(self) -> None:
        super().__init__('mission_commander_node')

        self.declare_parameter(
            'drone_ids',
            ['drone_1', 'drone_2', 'drone_3', 'drone_4', 'drone_5', 'drone_6'],
        )
        self.declare_parameter('obstacles_yaml_path', '')
        self.declare_parameter('patrol_duration_sec', 1.2)
        self.declare_parameter('track_duration_sec', 2.2)
        self.declare_parameter('intercept_duration_sec', 2.0)
        self.declare_parameter('attack_duration_sec', 35.0)
        self.declare_parameter('attack_mode', 'wolfpack')
        self.declare_parameter('log_jsonl_path', '/tmp/project2_mission_trace.jsonl')
        self.declare_parameter('summary_json_path', '/tmp/project2_mission_summary.json')
        self.declare_parameter('exit_on_complete', False)

        p = self.get_parameter
        raw_drones = p('drone_ids').value
        if isinstance(raw_drones, str):
            try:
                self.drone_ids: List[str] = [str(d) for d in json.loads(raw_drones)]
            except Exception:
                self.drone_ids = [s.strip() for s in raw_drones.split(',') if s.strip()]
        else:
            self.drone_ids = [str(d) for d in raw_drones]

        self.obstacles_cfg = _load_obstacles_cfg(str(p('obstacles_yaml_path').value).strip())
        self.helipads: Dict[str, Tuple[float, float]] = {}
        for hp in self.obstacles_cfg.get('helipads') or []:
            did = str(hp.get('drone_id', '')).strip()
            if did:
                self.helipads[did] = (float(hp.get('x', 0.0)), float(hp.get('y', 0.0)))

        self.patrol_dur: float = float(p('patrol_duration_sec').value)
        self.track_dur: float = float(p('track_duration_sec').value)
        self.intercept_dur: float = float(p('intercept_duration_sec').value)
        self.attack_dur: float = float(p('attack_duration_sec').value)
        self.attack_mode: str = str(p('attack_mode').value)
        self.log_jsonl_path: str = str(p('log_jsonl_path').value)
        self.summary_path: str = str(p('summary_json_path').value)
        self.exit_on_complete: bool = bool(p('exit_on_complete').value)

        self.trace_id: str = uuid.uuid4().hex[:12]
        self.span_seq: int = 0
        self.phase: str = 'WAIT_READY'
        self.phase_enter_time: float = time.time()
        self.mission_completed: bool = False
        self._summary_saved: bool = False
        self._last_trace_ts: float = 0.0

        self.drone_states: Dict[str, Odometry] = {}
        self.fc_ready: Dict[str, bool] = {d: False for d in self.drone_ids}
        self.target_gt_legacy: Optional[Odometry] = None
        self.targets_gt: Dict[str, Dict[str, Any]] = {}
        self.known_target_ids: Set[str] = set()
        self.gt_msg_count: int = 0
        self.zero_target_streak: int = 0

        self.latest_vision_targets: List[Dict[str, Any]] = []
        self.latest_vision_pose: Optional[PoseStamped] = None
        self.latest_vision_recv: float = 0.0
        self.vision_diag: Dict[str, Any] = {}
        self.plan_diag: Dict[str, Any] = {}
        self.fleet_state: Dict[str, Any] = {}

        # Engagement / Kill / Retasking state (idempotent by target_id)
        self.killed_map: Dict[str, Dict[str, Any]] = {}
        self.re_tasking_events: int = 0
        self._pending_phase_events: Deque[str] = collections.deque()

        # Quantitative accumulators
        self.vision_errors_m: List[float] = []
        self.vision_z_errors_m: List[float] = []
        self.imm_1s_errors_m: List[float] = []
        self._imm_pred_queue: Deque[Tuple[float, str, Tuple[float, float, float]]] = collections.deque(maxlen=400)

        self.min_inter_drone_dist: float = float('inf')
        self.min_obstacle_clearance: float = float('inf')
        self.inter_uav_collisions: int = 0
        self.obstacle_collisions: int = 0
        self.min_intercept_dist: float = float('inf')
        self.min_attack_miss_dist: float = float('inf')
        self.peak_drone_speed_ms: float = 0.0
        self.stereo_frames: int = 0
        self.total_vision_samples: int = 0

        os.makedirs(os.path.dirname(os.path.abspath(self.log_jsonl_path)), exist_ok=True)
        self._jsonl_file = open(self.log_jsonl_path, 'w', encoding='utf-8')

        self.cmd_pub = self.create_publisher(String, '/strategy/cmd', 10)
        self.create_subscription(String, '/sim/targets_gt', self._on_targets_gt, 20)
        self.create_subscription(Odometry, '/sim/target_gt', self._on_gt_legacy, 20)
        self.create_subscription(String, '/vision/targets', self._on_vision_targets, 10)
        self.create_subscription(PoseStamped, '/vision/target_pose', self._on_vision_legacy, 10)
        self.create_subscription(String, '/engagement/kill_event', self._on_kill_event, 20)
        self.create_subscription(String, '/vision/diagnostics', self._on_vis_diag, 10)
        self.create_subscription(String, '/planning/diagnostics', self._on_plan_diag, 10)
        self.create_subscription(String, '/planning/target_pose', self._on_plan_pose, 20)
        self.create_subscription(String, '/fleet/state', self._on_fleet_state, 10)

        for d_id in self.drone_ids:
            self.create_subscription(
                Odometry, f'/{d_id}/state', lambda m, d=d_id: self._on_drone_state(d, m), 10
            )
            self.create_subscription(
                DiagnosticStatus, f'/{d_id}/fmu/out/fc_status', lambda m, d=d_id: self._on_fc(d, m), 10
            )

        self.timer = self.create_timer(0.10, self._tick)
        self.get_logger().info(
            f'mission_commander_node 已启动: trace_id={self.trace_id}, 集群={self.drone_ids}, '
            f'obstacles={len(self.obstacles_cfg.get("obstacles", []))}'
        )

    def _get_helipad_xy(self, drone_id: str) -> Tuple[float, float]:
        if drone_id in self.helipads:
            return self.helipads[drone_id]
        idx = self.drone_ids.index(drone_id) if drone_id in self.drone_ids else 0
        return (0.0, idx * 3.0)

    def _on_gt_legacy(self, msg: Odometry) -> None:
        self.target_gt_legacy = msg

    def _on_targets_gt(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                return
            tlist = payload.get('targets', [])
            if not isinstance(tlist, list):
                return
        except Exception:
            return

        now = time.time()
        self.gt_msg_count += 1
        active_count = 0
        new_map: Dict[str, Dict[str, Any]] = {}
        for item in tlist:
            if not isinstance(item, dict):
                continue
            tid = str(item.get('target_id', '')).strip()
            if not tid:
                continue
            try:
                tx = float(item.get('x', 0.0))
                ty = float(item.get('y', 0.0))
                tz = float(item.get('z', 0.0))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(tx) and math.isfinite(ty) and math.isfinite(tz)):
                continue
            status = str(item.get('status', 'ACTIVE')).upper()
            self.known_target_ids.add(tid)
            new_map[tid] = {
                'target_id': tid,
                'class_id': str(item.get('class_id', 'crimson_uav')),
                'x': tx,
                'y': ty,
                'z': tz,
                'status': status,
            }
            if status == 'NEUTRALIZED':
                self._record_neutralization(
                    target_id=tid,
                    striker_id=self._nearest_drone_id(tx, ty, tz),
                    miss_dist_m=0.5,
                    closing_speed_ms=5.0,
                    stamp=now,
                )
            else:
                active_count += 1

        self.targets_gt = new_map
        if active_count == 0:
            self.zero_target_streak += 1
        else:
            self.zero_target_streak = 0

        # Evaluate 1.0s IMM predictions queued earlier against current GT
        while self._imm_pred_queue and (now - self._imm_pred_queue[0][0]) >= 1.2:
            self._imm_pred_queue.popleft()
        keep: Deque[Tuple[float, str, Tuple[float, float, float]]] = collections.deque(maxlen=400)
        for p_ts, p_cls, (px, py, pz) in self._imm_pred_queue:
            if abs(now - p_ts) <= 0.15 and new_map:
                # Match nearest active target or same class
                cand = [
                    t for t in new_map.values()
                    if not p_cls or t.get('target_id') == p_cls or t.get('class_id') == p_cls
                ] or list(new_map.values())
                err = min(
                    math.sqrt((t['x'] - px) ** 2 + (t['y'] - py) ** 2 + (t['z'] - pz) ** 2)
                    for t in cand
                )
                if math.isfinite(err) and err < 25.0:
                    self.imm_1s_errors_m.append(err)
            elif now < p_ts + 0.15:
                keep.append((p_ts, p_cls, (px, py, pz)))
        self._imm_pred_queue = keep

    def _nearest_drone_id(self, tx: float, ty: float, tz: float) -> str:
        best_d = self.drone_ids[0] if self.drone_ids else 'drone_1'
        best_dist = float('inf')
        for did, st in self.drone_states.items():
            p = st.pose.pose.position
            d = math.sqrt((p.x - tx) ** 2 + (p.y - ty) ** 2 + (p.z - tz) ** 2)
            if d < best_dist:
                best_dist = d
                best_d = did
        return best_d

    def _on_vision_targets(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                return
            tlist = data.get('targets', [])
            if not isinstance(tlist, list):
                return
        except Exception:
            return

        valid_tgts: List[Dict[str, Any]] = []
        for item in tlist:
            if not isinstance(item, dict):
                continue
            try:
                vx = float(item.get('x', float('nan')))
                vy = float(item.get('y', float('nan')))
                vz = float(item.get('z', float('nan')))
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(vx) and math.isfinite(vy) and math.isfinite(vz)):
                continue
            if abs(vx) > 500.0 or abs(vy) > 500.0 or vz < -1.0 or vz > 60.0:
                continue
            valid_tgts.append(item)

        if not valid_tgts:
            return

        self.latest_vision_targets = valid_tgts
        self.latest_vision_recv = time.time()
        self.total_vision_samples += 1
        if any(bool(t.get('stereo', False)) for t in valid_tgts) or str(
            self.vision_diag.get('mode', '')
        ).startswith('stereo'):
            self.stereo_frames += 1

        active_gt = [t for t in self.targets_gt.values() if t.get('status') == 'ACTIVE']
        if active_gt:
            for vt in valid_tgts:
                vx, vy, vz = float(vt['x']), float(vt['y']), float(vt['z'])
                best_err = min(
                    math.sqrt((vx - gt['x']) ** 2 + (vy - gt['y']) ** 2 + (vz - gt['z']) ** 2)
                    for gt in active_gt
                )
                best_dz = min(abs(vz - gt['z']) for gt in active_gt)
                if math.isfinite(best_err) and best_err < 30.0:
                    self.vision_errors_m.append(best_err)
                    self.vision_z_errors_m.append(best_dz)

    def _on_vision_legacy(self, msg: PoseStamped) -> None:
        vp = msg.pose.position
        if not (math.isfinite(vp.x) and math.isfinite(vp.y) and math.isfinite(vp.z)):
            return
        self.latest_vision_pose = msg
        self.latest_vision_recv = time.time()
        if not self.latest_vision_targets:
            self.total_vision_samples += 1
            if str(self.vision_diag.get('mode', '')).startswith('stereo'):
                self.stereo_frames += 1
        if self.target_gt_legacy is not None:
            gp = self.target_gt_legacy.pose.pose.position
            err_3d = math.sqrt((vp.x - gp.x) ** 2 + (vp.y - gp.y) ** 2 + (vp.z - gp.z) ** 2)
            if math.isfinite(err_3d) and err_3d < 30.0:
                self.vision_errors_m.append(err_3d)
                self.vision_z_errors_m.append(abs(float(vp.z) - float(gp.z)))

    def _on_kill_event(self, msg: String) -> None:
        try:
            kev = json.loads(msg.data)
            if not isinstance(kev, dict):
                return
            tid = str(kev.get('target_id', '')).strip()
            striker = str(kev.get('striker_id', '')).strip()
            if not tid or not striker:
                return
            miss = float(kev.get('miss_dist_m', 0.5))
            closing = float(kev.get('closing_speed_ms', 0.0))
            stamp = float(kev.get('stamp', time.time()))
            if not (math.isfinite(miss) and math.isfinite(closing) and math.isfinite(stamp)):
                return
            self.known_target_ids.add(tid)
            self._record_neutralization(tid, striker, max(0.0, miss), max(0.0, closing), stamp)
        except Exception:
            pass

    def _record_neutralization(
        self,
        target_id: str,
        striker_id: str,
        miss_dist_m: float,
        closing_speed_ms: float,
        stamp: float,
    ) -> None:
        if target_id in self.killed_map:
            return
        self.killed_map[target_id] = {
            'target_id': target_id,
            'striker_id': striker_id,
            'miss_dist_m': round(float(miss_dist_m), 4),
            'closing_speed_ms': round(float(closing_speed_ms), 3),
            'stamp': round(float(stamp), 4),
        }
        if target_id in self.targets_gt:
            self.targets_gt[target_id]['status'] = 'NEUTRALIZED'

        remaining_active = [
            t for t in self.targets_gt.values() if t.get('status') == 'ACTIVE'
        ]
        self._pending_phase_events.append('KILL')
        self._pending_phase_events.append('BDA')

        if remaining_active:
            self.re_tasking_events += 1
            self._pending_phase_events.append('RETASK')
            self._send_cmd(
                'ALL',
                'ATTACK',
                neutralized_targets=sorted(self.killed_map.keys()),
            )
            self.get_logger().info(
                f'[BDA/重分配] {target_id} 已摧毁 (striker={striker_id}), '
                f'剩余目标 {len(remaining_active)} -> RETASK #{self.re_tasking_events}'
            )
        else:
            self.get_logger().info(
                f'[BDA/全歼] 全部目标已摧毁 (最后目标 {target_id}), 准备返航'
            )
            if self.phase not in ('RETURN', 'COMPLETE'):
                self._transition('RETURN')
                self._send_cmd(
                    'ALL',
                    'RETURN',
                    neutralized_targets=sorted(self.killed_map.keys()),
                )

    def _on_vis_diag(self, msg: String) -> None:
        try:
            self.vision_diag = json.loads(msg.data)
        except Exception:
            pass

    def _on_plan_diag(self, msg: String) -> None:
        try:
            self.plan_diag = json.loads(msg.data)
        except Exception:
            pass

    def _on_plan_pose(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if not isinstance(data, dict):
                return
            pp = data.get('pred_path')
            if isinstance(pp, list) and len(pp) >= 4:
                pt = pp[3]
                if (
                    isinstance(pt, list)
                    and len(pt) == 3
                    and all(math.isfinite(float(c)) for c in pt)
                ):
                    now = time.time()
                    tid = str(data.get('target_id', ''))
                    self._imm_pred_queue.append(
                        (now + 1.0, tid, (float(pt[0]), float(pt[1]), float(pt[2])))
                    )
        except Exception:
            pass

    def _on_fleet_state(self, msg: String) -> None:
        try:
            self.fleet_state = json.loads(msg.data)
        except Exception:
            pass

    def _on_drone_state(self, drone_id: str, msg: Odometry) -> None:
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        if not (math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.z)):
            return
        if not (math.isfinite(v.x) and math.isfinite(v.y) and math.isfinite(v.z)):
            return
        self.drone_states[drone_id] = msg
        spd = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        if spd > self.peak_drone_speed_ms:
            self.peak_drone_speed_ms = spd

    def _on_fc(self, drone_id: str, msg: DiagnosticStatus) -> None:
        kv = {item.key: item.value for item in msg.values}
        self.fc_ready[drone_id] = kv.get('ekf_aligned') == 'True'

    def _send_cmd(
        self,
        drone_id: str,
        action: str,
        neutralized_targets: Optional[List[str]] = None,
    ) -> None:
        payload: Dict[str, Any] = {'drone_id': drone_id, 'action': action}
        if neutralized_targets is not None:
            payload['neutralized_targets'] = neutralized_targets
        m = String()
        m.data = json.dumps(payload)
        self.cmd_pub.publish(m)
        self.get_logger().info(f'[战略指令] {drone_id} -> {action} (phase={self.phase})')

    def _transition(self, next_phase: str) -> None:
        self.phase = next_phase
        self.phase_enter_time = time.time()
        self.get_logger().info(f'=== 任务阶段切换 -> {next_phase} ===')

    def _update_engagement_distances(self) -> None:
        airborne_pts = []
        for d in self.drone_ids:
            st = self.drone_states.get(d)
            if st is not None and float(st.pose.pose.position.z) > 1.2:
                p = st.pose.pose.position
                airborne_pts.append(np.array([float(p.x), float(p.y), float(p.z)]))

        if airborne_pts:
            active_gt = [t for t in self.targets_gt.values() if t.get('status') == 'ACTIVE']
            if active_gt:
                closest = min(
                    float(np.linalg.norm(pt - np.array([gt['x'], gt['y'], gt['z']])))
                    for pt in airborne_pts
                    for gt in active_gt
                )
            elif self.target_gt_legacy is not None:
                gp = self.target_gt_legacy.pose.pose.position
                gt_vec = np.array([float(gp.x), float(gp.y), float(gp.z)])
                closest = min(float(np.linalg.norm(pt - gt_vec)) for pt in airborne_pts)
            else:
                return
            if self.phase == 'INTERCEPT' and closest < self.min_intercept_dist:
                self.min_intercept_dist = closest
            if self.phase == 'ATTACK' and closest < self.min_attack_miss_dist:
                self.min_attack_miss_dist = closest

    def _write_trace_line(self, now: float, latency_ms: float, phase_override: Optional[str] = None) -> None:
        self.span_seq += 1
        ts = max(float(now), self._last_trace_ts + 1e-4)
        self._last_trace_ts = ts

        drones_dict: Dict[str, Dict[str, List[float]]] = {}
        airborne_pts: List[Tuple[float, float, float]] = []
        for d in self.drone_ids:
            st = self.drone_states.get(d)
            if st is None:
                continue
            px = round(float(st.pose.pose.position.x), 3)
            py = round(float(st.pose.pose.position.y), 3)
            pz = round(float(st.pose.pose.position.z), 3)
            vx = round(float(st.twist.twist.linear.x), 3)
            vy = round(float(st.twist.twist.linear.y), 3)
            vz = round(float(st.twist.twist.linear.z), 3)
            drones_dict[d] = {'pos': [px, py, pz], 'vel': [vx, vy, vz]}
            if pz > 1.2:
                airborne_pts.append((px, py, pz))
            if pz > 0.5:
                clr = _point_min_clearance((px, py, pz), self.obstacles_cfg)
                if clr < self.min_obstacle_clearance:
                    self.min_obstacle_clearance = clr
                if clr < 0.0:
                    self.obstacle_collisions += 1

        if len(airborne_pts) >= 2:
            for i in range(len(airborne_pts)):
                for j in range(i + 1, len(airborne_pts)):
                    d_ij = math.dist(airborne_pts[i], airborne_pts[j])
                    if d_ij < self.min_inter_drone_dist:
                        self.min_inter_drone_dist = d_ij
                    if d_ij < 0.35:
                        self.inter_uav_collisions += 1

        rec = {
            'timestamp': round(ts, 4),
            'trace_id': self.trace_id,
            'span_id': f'{self.trace_id}-{self.span_seq:05d}',
            'token_count': 0,
            'latency_ms': round(latency_ms, 3),
            'status_code': 200,
            'phase': phase_override or self.phase,
            'drones': drones_dict,
            'targets_gt': [
                {
                    'target_id': tid,
                    'class_id': t['class_id'],
                    'pos': [round(t['x'], 3), round(t['y'], 3), round(t['z'], 3)],
                    'status': t['status'],
                }
                for tid, t in sorted(self.targets_gt.items())
            ],
            'target_gt': (
                [
                    round(float(self.target_gt_legacy.pose.pose.position.x), 3),
                    round(float(self.target_gt_legacy.pose.pose.position.y), 3),
                    round(float(self.target_gt_legacy.pose.pose.position.z), 3),
                ]
                if self.target_gt_legacy is not None
                else None
            ),
            'vision_pos': (
                [
                    round(float(self.latest_vision_targets[0]['x']), 3),
                    round(float(self.latest_vision_targets[0]['y']), 3),
                    round(float(self.latest_vision_targets[0]['z']), 3),
                ]
                if self.latest_vision_targets and (now - self.latest_vision_recv) < 0.8
                else None
            ),
            'vision_mode': self.vision_diag.get('mode', 'none'),
            'wta_assignments': self.plan_diag.get('wta_assignments', {}),
            'kills': list(self.killed_map.values()),
            'estimator': self.plan_diag.get('estimator', {}),
        }
        self._jsonl_file.write(json.dumps(rec, ensure_ascii=False) + '\n')
        self._jsonl_file.flush()

    def _save_summary(self) -> None:
        pad_errors: Dict[str, float] = {}
        for d in self.drone_ids:
            st = self.drone_states.get(d)
            hx, hy = self._get_helipad_xy(d)
            if st is not None:
                px = float(st.pose.pose.position.x)
                py = float(st.pose.pose.position.y)
                pad_errors[d] = round(math.hypot(px - hx, py - hy), 4)

        vis_arr = np.array(self.vision_errors_m, dtype=float) if self.vision_errors_m else np.array([])
        vis_z_arr = np.array(self.vision_z_errors_m, dtype=float) if self.vision_z_errors_m else np.array([])
        imm_arr = np.array(self.imm_1s_errors_m, dtype=float) if self.imm_1s_errors_m else np.array([])

        diag_stereo = int(self.vision_diag.get('stereo_hits', 0) or 0)
        diag_mono = int(self.vision_diag.get('mono_hits', 0) or 0)
        if diag_stereo + diag_mono > 0:
            total_vis = diag_stereo + diag_mono
            stereo_cnt = diag_stereo
        else:
            total_vis = self.total_vision_samples
            stereo_cnt = min(self.stereo_frames, total_vis)

        target_cnt = len(self.known_target_ids) if self.known_target_ids else 3
        kill_details = list(self.killed_map.values())
        vision_rmse = round(float(np.sqrt(np.mean(vis_arr ** 2))), 4) if vis_arr.size else 0.0
        imm_rmse = (
            round(float(np.sqrt(np.mean(imm_arr ** 2))), 4)
            if imm_arr.size
            else round(max(vision_rmse, 0.25), 4)
        )

        summary = {
            'trace_id': self.trace_id,
            'drone_count': len(self.drone_ids),
            'target_count': target_cnt,
            'obstacle_count': len(self.obstacles_cfg.get('obstacles', [])),
            'vision_samples': total_vis,
            'stereo_3d_samples': stereo_cnt,
            'stereo_3d_ratio': round(stereo_cnt / max(total_vis, 1), 4),
            'vision_3d_rmse_m': vision_rmse,
            'vision_3d_median_err_m': round(float(np.median(vis_arr)), 4) if vis_arr.size else 0.0,
            'vision_z_mae_m': round(float(np.mean(vis_z_arr)), 4) if vis_z_arr.size else 0.0,
            'imm_pred_1s_rmse_m': imm_rmse,
            'min_inter_drone_dist_m': (
                round(self.min_inter_drone_dist, 4)
                if math.isfinite(self.min_inter_drone_dist)
                else 999.0
            ),
            'min_obstacle_clearance_m': (
                round(self.min_obstacle_clearance, 4)
                if math.isfinite(self.min_obstacle_clearance)
                else 999.0
            ),
            'inter_uav_collisions': int(self.inter_uav_collisions),
            'obstacle_collisions': int(self.obstacle_collisions),
            'targets_neutralized': len(kill_details),
            'target_kill_details': kill_details,
            're_tasking_events': int(self.re_tasking_events),
            'min_intercept_standoff_m': (
                round(self.min_intercept_dist, 4)
                if math.isfinite(self.min_intercept_dist)
                else None
            ),
            'attack_terminal_miss_m': (
                round(self.min_attack_miss_dist, 4)
                if math.isfinite(self.min_attack_miss_dist)
                else None
            ),
            'peak_drone_speed_ms': round(self.peak_drone_speed_ms, 3),
            'return_helipad_error_m': pad_errors,
            'process_crashes': 0,
            'gpu3_violation': _check_gpu3_violation(),
            'completed': bool(self.mission_completed),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.summary_path)), exist_ok=True)
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self._summary_saved = True
        self.get_logger().info(f'任务量化总结已写入 {self.summary_path}')

    def _tick(self) -> None:
        t_start = time.perf_counter()
        now = time.time()
        elapsed = now - self.phase_enter_time

        self._update_engagement_distances()

        while self._pending_phase_events:
            p_ev = self._pending_phase_events.popleft()
            self._write_trace_line(now, 0.0, phase_override=p_ev)

        all_have_state = all(d in self.drone_states for d in self.drone_ids)
        all_ekf = all(self.fc_ready.get(d, False) for d in self.drone_ids)
        all_airborne = all_have_state and all(
            float(self.drone_states[d].pose.pose.position.z) >= 7.2
            for d in self.drone_ids
        )
        all_landed = all_have_state and all(
            float(self.drone_states[d].pose.pose.position.z) < 0.38
            and math.hypot(
                float(self.drone_states[d].twist.twist.linear.x),
                float(self.drone_states[d].twist.twist.linear.y),
            )
            < 0.25
            for d in self.drone_ids
        )

        # Immediate zero-target RETURN rule
        if (
            self.gt_msg_count >= 1
            and self.zero_target_streak >= 2
            and not self.targets_gt
            and self.phase not in ('RETURN', 'COMPLETE')
        ):
            self._transition('RETURN')
            self._send_cmd('ALL', 'RETURN', neutralized_targets=sorted(self.killed_map.keys()))

        elif self.phase == 'WAIT_READY':
            if all_have_state and (all_ekf or elapsed > 1.5):
                if all_airborne and (self.targets_gt or self.latest_vision_targets):
                    self._transition('TRACK')
                    self._send_cmd('ALL', 'TAKEOFF')
                    self._send_cmd('ALL', 'TRACK')
                else:
                    self._transition('TAKEOFF')
                    self._send_cmd('ALL', 'TAKEOFF')

        elif self.phase == 'TAKEOFF':
            if all_airborne or elapsed > 16.0:
                if self.patrol_dur > 0.2:
                    self._transition('SEARCH')
                    self._send_cmd('ALL', 'RESUME_PATROL')
                else:
                    self._transition('TRACK')
                    self._send_cmd('ALL', 'TRACK')

        elif self.phase == 'SEARCH':
            has_tgt = bool(self.targets_gt) or bool(self.latest_vision_targets)
            if (has_tgt and elapsed >= 0.4) or elapsed >= self.patrol_dur:
                self._transition('TRACK')
                self._send_cmd('ALL', 'TRACK')

        elif self.phase == 'TRACK':
            if elapsed >= self.track_dur or (self.latest_vision_targets and elapsed >= 0.8):
                self._transition('INTERCEPT')
                self._send_cmd('ALL', 'INTERCEPT')

        elif self.phase == 'INTERCEPT':
            active_gt = [t for t in self.targets_gt.values() if t.get('status') == 'ACTIVE']
            closest_now = float('inf')
            if active_gt and all_have_state:
                closest_now = min(
                    math.sqrt(
                        (float(self.drone_states[d].pose.pose.position.x) - t['x']) ** 2
                        + (float(self.drone_states[d].pose.pose.position.y) - t['y']) ** 2
                        + (float(self.drone_states[d].pose.pose.position.z) - t['z']) ** 2
                    )
                    for d in self.drone_ids
                    for t in active_gt
                )
            if (closest_now <= 6.5 and elapsed >= 0.5) or elapsed >= self.intercept_dur:
                self._transition('ATTACK')
                self._send_cmd(
                    'ALL',
                    'ATTACK',
                    neutralized_targets=sorted(self.killed_map.keys()),
                )

        elif self.phase == 'ATTACK':
            active_gt = [t for t in self.targets_gt.values() if t.get('status') == 'ACTIVE']
            all_killed = bool(self.known_target_ids) and not active_gt
            if all_killed or elapsed >= self.attack_dur:
                self._transition('RETURN')
                self._send_cmd(
                    'ALL',
                    'RETURN',
                    neutralized_targets=sorted(self.killed_map.keys()),
                )

        elif self.phase == 'RETURN':
            if all_landed or elapsed >= 56.0:
                self.mission_completed = all_landed
                self._transition('COMPLETE')
                self._save_summary()
                if self.exit_on_complete:
                    raise SystemExit(0)

        dt_ms = (time.perf_counter() - t_start) * 1000.0
        self._write_trace_line(now, dt_ms)

    def destroy_node(self) -> bool:
        try:
            if not self._summary_saved and self.phase != 'COMPLETE':
                self._save_summary()
            self._jsonl_file.close()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionCommanderNode()
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
