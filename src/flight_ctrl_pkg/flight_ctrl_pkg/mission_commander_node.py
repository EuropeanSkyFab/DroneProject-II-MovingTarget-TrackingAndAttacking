# -*- coding: utf-8 -*-
"""
mission_commander_node.py —— 课题二全流程自主战术指挥官与结构化可观测性评测裁判节点

功能：
  1. 模拟课题三战略指挥中心 (/strategy/cmd)，按真实物理态势事件驱动三机协同全任务状态转移：
       WAIT_READY -> TAKEOFF -> RESUME_PATROL -> TRACK -> INTERCEPT -> ATTACK -> RETURN -> COMPLETE
  2. 全链路订阅 /sim/target_gt、/vision/target_pose、/vision/diagnostics、/planning/diagnostics、
     /fleet/state 及各机 /drone_<id>/state，输出符合 JSON-Lines 规范的全链路追踪日志与量化验收报告。
"""

from __future__ import annotations

import json
import math
import os
import time
import uuid
from typing import Dict, List, Optional

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticStatus
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String


class MissionCommanderNode(Node):
    """事件驱动全流程多机协同战术指挥与量化评测节点。"""

    def __init__(self) -> None:
        super().__init__('mission_commander_node')

        self.declare_parameter('drone_ids', ['drone_1', 'drone_2', 'drone_3'])
        self.declare_parameter('patrol_duration_sec', 3.0)
        self.declare_parameter('track_duration_sec', 14.0)
        self.declare_parameter('intercept_duration_sec', 9.0)
        self.declare_parameter('attack_duration_sec', 22.0)
        self.declare_parameter('attack_mode', 'lead_only')  # 'lead_only' (1号主攻+2/3号立体观测掩护) | 'wolfpack'
        self.declare_parameter('log_jsonl_path', '/tmp/project2_mission_trace.jsonl')
        self.declare_parameter('summary_json_path', '/tmp/project2_mission_summary.json')
        self.declare_parameter('exit_on_complete', False)

        p = self.get_parameter
        self.drone_ids: List[str] = [str(d) for d in p('drone_ids').value]
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

        self.drone_states: Dict[str, Odometry] = {}
        self.fc_ready: Dict[str, bool] = {d: False for d in self.drone_ids}
        self.target_gt: Optional[Odometry] = None
        self.latest_vision: Optional[PoseStamped] = None
        self.latest_vision_recv: float = 0.0
        self.vision_diag: Dict[str, object] = {}
        self.plan_diag: Dict[str, object] = {}

        # 统计指标累加器
        self.vision_errors_m: List[float] = []
        self.vision_z_errors_m: List[float] = []
        self.min_inter_drone_dist: float = float('inf')
        self.min_intercept_dist: float = float('inf')
        self.min_attack_miss_dist: float = float('inf')
        self.peak_drone_speed_ms: float = 0.0
        self.stereo_frames: int = 0
        self.total_vision_samples: int = 0

        os.makedirs(os.path.dirname(os.path.abspath(self.log_jsonl_path)), exist_ok=True)
        self._jsonl_file = open(self.log_jsonl_path, 'w', encoding='utf-8')

        self.cmd_pub = self.create_publisher(String, '/strategy/cmd', 10)
        self.create_subscription(Odometry, '/sim/target_gt', self._on_gt, 20)
        self.create_subscription(PoseStamped, '/vision/target_pose', self._on_vision, 10)
        self.create_subscription(String, '/vision/diagnostics', self._on_vis_diag, 10)
        self.create_subscription(String, '/planning/diagnostics', self._on_plan_diag, 10)

        for d_id in self.drone_ids:
            self.create_subscription(
                Odometry, f'/{d_id}/state', lambda m, d=d_id: self._on_drone_state(d, m), 10
            )
            self.create_subscription(
                DiagnosticStatus, f'/{d_id}/fmu/out/fc_status', lambda m, d=d_id: self._on_fc(d, m), 10
            )

        self.timer = self.create_timer(0.10, self._tick)
        self.get_logger().info(
            f'mission_commander_node 已启动: trace_id={self.trace_id}, 集群={self.drone_ids}'
        )

    def _on_gt(self, msg: Odometry) -> None:
        self.target_gt = msg

    def _on_vision(self, msg: PoseStamped) -> None:
        self.latest_vision = msg
        self.latest_vision_recv = time.time()
        self.total_vision_samples += 1
        if str(self.vision_diag.get('mode', '')).startswith('stereo'):
            self.stereo_frames += 1
        if self.target_gt is not None:
            vp = msg.pose.position
            gp = self.target_gt.pose.pose.position
            err_3d = math.sqrt(
                (vp.x - gp.x) ** 2 + (vp.y - gp.y) ** 2 + (vp.z - gp.z) ** 2
            )
            self.vision_errors_m.append(err_3d)
            self.vision_z_errors_m.append(abs(float(vp.z) - float(gp.z)))

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

    def _on_drone_state(self, drone_id: str, msg: Odometry) -> None:
        self.drone_states[drone_id] = msg
        v = msg.twist.twist.linear
        spd = math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        if spd > self.peak_drone_speed_ms:
            self.peak_drone_speed_ms = spd

    def _on_fc(self, drone_id: str, msg: DiagnosticStatus) -> None:
        kv = {item.key: item.value for item in msg.values}
        self.fc_ready[drone_id] = kv.get('ekf_aligned') == 'True'

    def _send_cmd(self, drone_id: str, action: str) -> None:
        m = String()
        m.data = json.dumps({'drone_id': drone_id, 'action': action})
        self.cmd_pub.publish(m)
        self.get_logger().info(f'[战略指令下发] {drone_id} -> {action} (phase={self.phase})')

    def _transition(self, next_phase: str) -> None:
        self.phase = next_phase
        self.phase_enter_time = time.time()
        self.get_logger().info(f'=== 任务阶段切换 -> {next_phase} ===')

    def _update_safety_and_engagement_metrics(self) -> None:
        # 仅在飞行阶段统计空中最小安全间距
        airborne_pts = []
        for d in self.drone_ids:
            st = self.drone_states.get(d)
            if st is not None and float(st.pose.pose.position.z) > 1.5:
                p = st.pose.pose.position
                airborne_pts.append(np.array([float(p.x), float(p.y), float(p.z)]))

        for i in range(len(airborne_pts)):
            for j in range(i + 1, len(airborne_pts)):
                dist = float(np.linalg.norm(airborne_pts[i] - airborne_pts[j]))
                if dist < self.min_inter_drone_dist:
                    self.min_inter_drone_dist = dist

        if self.target_gt is not None and airborne_pts:
            gp = self.target_gt.pose.pose.position
            gt_vec = np.array([float(gp.x), float(gp.y), float(gp.z)])
            closest = min(float(np.linalg.norm(pt - gt_vec)) for pt in airborne_pts)
            if self.phase == 'INTERCEPT' and closest < self.min_intercept_dist:
                self.min_intercept_dist = closest
            if self.phase == 'ATTACK' and closest < self.min_attack_miss_dist:
                self.min_attack_miss_dist = closest

    def _write_trace_line(self, now: float, latency_ms: float) -> None:
        self.span_seq += 1
        rec = {
            'timestamp': round(now, 4),
            'trace_id': self.trace_id,
            'span_id': f'{self.trace_id}-{self.span_seq:05d}',
            'token_count': 0,
            'latency_ms': round(latency_ms, 3),
            'status_code': 200,
            'phase': self.phase,
            'drones': {
                d: {
                    'pos': [
                        round(float(self.drone_states[d].pose.pose.position.x), 3),
                        round(float(self.drone_states[d].pose.pose.position.y), 3),
                        round(float(self.drone_states[d].pose.pose.position.z), 3),
                    ],
                    'vel': [
                        round(float(self.drone_states[d].twist.twist.linear.x), 3),
                        round(float(self.drone_states[d].twist.twist.linear.y), 3),
                        round(float(self.drone_states[d].twist.twist.linear.z), 3),
                    ],
                }
                for d in self.drone_ids
                if d in self.drone_states
            },
            'target_gt': (
                [
                    round(float(self.target_gt.pose.pose.position.x), 3),
                    round(float(self.target_gt.pose.pose.position.y), 3),
                    round(float(self.target_gt.pose.pose.position.z), 3),
                ]
                if self.target_gt is not None
                else None
            ),
            'vision_pos': (
                [
                    round(float(self.latest_vision.pose.position.x), 3),
                    round(float(self.latest_vision.pose.position.y), 3),
                    round(float(self.latest_vision.pose.position.z), 3),
                ]
                if self.latest_vision is not None and (now - self.latest_vision_recv) < 0.5
                else None
            ),
            'vision_mode': self.vision_diag.get('mode', 'none'),
            'estimator': self.plan_diag.get('estimator', {}),
        }
        self._jsonl_file.write(json.dumps(rec, ensure_ascii=False) + '\n')
        self._jsonl_file.flush()

    def _save_summary(self) -> None:
        pad_errors = {}
        for idx, d in enumerate(self.drone_ids):
            st = self.drone_states.get(d)
            if st is not None:
                px = float(st.pose.pose.position.x)
                py = float(st.pose.pose.position.y)
                pad_errors[d] = round(math.hypot(px - 0.0, py - idx * 3.0), 4)

        vis_arr = np.array(self.vision_errors_m, dtype=float) if self.vision_errors_m else np.array([])
        vis_z_arr = np.array(self.vision_z_errors_m, dtype=float) if self.vision_z_errors_m else np.array([])
        diag_stereo = int(self.vision_diag.get('stereo_hits', 0) or 0)
        diag_mono = int(self.vision_diag.get('mono_hits', 0) or 0)
        if diag_stereo + diag_mono > 0:
            total_vis = diag_stereo + diag_mono
            stereo_cnt = diag_stereo
        else:
            total_vis = self.total_vision_samples
            stereo_cnt = self.stereo_frames
        summary = {
            'trace_id': self.trace_id,
            'drone_count': len(self.drone_ids),
            'vision_samples': total_vis,
            'stereo_3d_samples': stereo_cnt,
            'stereo_3d_ratio': round(stereo_cnt / max(total_vis, 1), 4),
            'vision_3d_rmse_m': round(float(np.sqrt(np.mean(vis_arr ** 2))), 4) if vis_arr.size else None,
            'vision_3d_median_err_m': round(float(np.median(vis_arr)), 4) if vis_arr.size else None,
            'vision_z_mae_m': round(float(np.mean(vis_z_arr)), 4) if vis_z_arr.size else None,
            'min_inter_drone_dist_m': (
                round(self.min_inter_drone_dist, 4) if math.isfinite(self.min_inter_drone_dist) else None
            ),
            'min_intercept_standoff_m': (
                round(self.min_intercept_dist, 4) if math.isfinite(self.min_intercept_dist) else None
            ),
            'attack_terminal_miss_m': (
                round(self.min_attack_miss_dist, 4) if math.isfinite(self.min_attack_miss_dist) else None
            ),
            'peak_drone_speed_ms': round(self.peak_drone_speed_ms, 3),
            'return_helipad_error_m': pad_errors,
            'completed': self.mission_completed,
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.summary_path)), exist_ok=True)
        with open(self.summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        self.get_logger().info(f'任务量化总结已写入 {self.summary_path}: {json.dumps(summary)}')

    def _tick(self) -> None:
        t_start = time.perf_counter()
        now = time.time()
        elapsed = now - self.phase_enter_time

        self._update_safety_and_engagement_metrics()

        if self.phase == 'WAIT_READY':
            all_have_state = all(d in self.drone_states for d in self.drone_ids)
            # 兼容真实 PX4 (需 ekf_aligned) 与纯运动学干跑 (等待 2.0s 后已有位姿即可)
            all_ekf = all(self.fc_ready.get(d, False) for d in self.drone_ids)
            if all_have_state and (all_ekf or elapsed > 3.0):
                self._transition('TAKEOFF')
                self._send_cmd('ALL', 'TAKEOFF')

        elif self.phase == 'TAKEOFF':
            all_climbed = all(
                float(self.drone_states[d].pose.pose.position.z) >= 7.8
                for d in self.drone_ids
                if d in self.drone_states
            )
            if all_climbed or elapsed > 18.0:
                if self.patrol_dur > 0.2:
                    self._transition('RESUME_PATROL')
                    self._send_cmd('ALL', 'RESUME_PATROL')
                else:
                    self._transition('TRACK')
                    self._send_cmd('ALL', 'TRACK')

        elif self.phase == 'RESUME_PATROL':
            if elapsed >= self.patrol_dur:
                self._transition('TRACK')
                self._send_cmd('ALL', 'TRACK')

        elif self.phase == 'TRACK':
            if elapsed >= self.track_dur:
                self._transition('INTERCEPT')
                self._send_cmd('ALL', 'INTERCEPT')

        elif self.phase == 'INTERCEPT':
            # 当主攻手实时逼近至 <=4.8m 有利攻击包线（且已稳定截击 >=3.5s）或满窗口时，果断切入末端致命打击
            striker_dist_now = float('inf')
            lead_id = self.drone_ids[0] if self.drone_ids else 'drone_1'
            if lead_id in self.drone_states and self.target_gt is not None:
                sp = self.drone_states[lead_id].pose.pose.position
                gp = self.target_gt.pose.pose.position
                striker_dist_now = math.sqrt(
                    (sp.x - gp.x) ** 2 + (sp.y - gp.y) ** 2 + (sp.z - gp.z) ** 2
                )
            in_strike_envelope = striker_dist_now <= 4.8 and elapsed >= 3.5
            if in_strike_envelope or elapsed >= self.intercept_dur:
                self._transition('ATTACK')
                if self.attack_mode == 'lead_only' and len(self.drone_ids) >= 2:
                    self._send_cmd(self.drone_ids[0], 'ATTACK')
                    for wing in self.drone_ids[1:]:
                        self._send_cmd(wing, 'TRACK')
                else:
                    self._send_cmd('ALL', 'ATTACK')

        elif self.phase == 'ATTACK':
            # 若已实现 <=0.70m 精确命中且攻击时间已满 2.5s，或达到总攻击窗口上限，安全脱离返航
            hit_achieved = self.min_attack_miss_dist <= 0.70 and elapsed >= 2.5
            if hit_achieved or elapsed >= self.attack_dur:
                self._transition('RETURN')
                # 切换到 INTERCEPT 高速返航档或直接 RETURN；使用 RETURN 分层归巢
                self._send_cmd('ALL', 'RETURN')

        elif self.phase == 'RETURN':
            all_landed = all(
                float(self.drone_states[d].pose.pose.position.z) < 0.38
                and math.hypot(
                    float(self.drone_states[d].twist.twist.linear.x),
                    float(self.drone_states[d].twist.twist.linear.y),
                )
                < 0.25
                for d in self.drone_ids
                if d in self.drone_states
            )
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
            if not self.mission_completed and self.phase != 'COMPLETE':
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
