#!/usr/bin/env python3
"""Run the full 4-package ROS 2 closed-loop 6-UAV vs 3-target mission to produce Contract #7 trace & summary."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

from diagnostic_msgs.msg import DiagnosticStatus
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String
import yaml


DRONE_IDS = ['drone_1', 'drone_2', 'drone_3', 'drone_4', 'drone_5', 'drone_6']


def _load_helipads(obstacles_yaml: Path) -> Dict[str, List[float]]:
    data = yaml.safe_load(obstacles_yaml.read_text(encoding='utf-8')) or {}
    pads: Dict[str, List[float]] = {}
    for hp in data.get('helipads') or []:
        did = str(hp.get('drone_id', '')).strip()
        if did:
            pads[did] = [float(hp.get('x', 0.0)), float(hp.get('y', 0.0)), float(hp.get('z', 0.05))]
    return pads


class SwarmDynamicsPlant(Node):
    """40 Hz closed-loop 6-UAV dynamics plant + 10 Hz onboard stereo vision triangulator."""

    def __init__(self, helipads: Dict[str, List[float]], drone_ids: List[str]) -> None:
        super().__init__('swarm_dynamics_plant')
        self.drone_ids = list(drone_ids)
        self.rng = random.Random(2026)
        self.pos: Dict[str, List[float]] = {}
        self.vel: Dict[str, List[float]] = {}
        self.sp_pos: Dict[str, Optional[List[float]]] = {}
        self.sp_ff: Dict[str, List[float]] = {}
        self.sp_ts: Dict[str, float] = {}
        self.active_gt: List[Dict] = []

        self.odom_pubs = {}
        self.bat_pubs = {}
        self.fc_pubs = {}
        self.vis_targets_pub = self.create_publisher(String, '/vision/targets', 10)
        self.vis_legacy_pub = self.create_publisher(PoseStamped, '/vision/target_pose', 10)
        self.vis_diag_pub = self.create_publisher(String, '/vision/diagnostics', 10)

        self.create_subscription(String, '/sim/targets_gt', self._on_targets_gt, 10)

        for idx, d in enumerate(self.drone_ids):
            hp = helipads.get(d, [-6.0, -5.0 + 2.0 * idx, 0.05])
            self.pos[d] = [hp[0], hp[1], hp[2]]
            self.vel[d] = [0.0, 0.0, 0.0]
            self.sp_pos[d] = None
            self.sp_ff[d] = [0.0, 0.0, 0.0]
            self.sp_ts[d] = 0.0

            self.create_subscription(
                PoseStamped, f'/{d}/fmu/in/trajectory_setpoint', lambda m, did=d: self._on_sp(did, m), 10
            )
            self.create_subscription(
                TwistStamped, f'/{d}/fmu/in/trajectory_ff', lambda m, did=d: self._on_ff(did, m), 10
            )
            self.odom_pubs[d] = self.create_publisher(Odometry, f'/{d}/state', 10)
            self.bat_pubs[d] = self.create_publisher(BatteryState, f'/{d}/fmu/out/battery_raw', 10)
            self.fc_pubs[d] = self.create_publisher(DiagnosticStatus, f'/{d}/fmu/out/fc_status', 10)

        self.dt = 0.025
        self.last_bat = 0.0
        self.last_vis = 0.0
        self.create_timer(self.dt, self._step)

    def _on_targets_gt(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            if isinstance(data, dict) and isinstance(data.get('targets'), list):
                self.active_gt = [t for t in data['targets'] if isinstance(t, dict) and t.get('status') == 'ACTIVE']
        except Exception:
            pass

    def _on_sp(self, did: str, msg: PoseStamped) -> None:
        p = msg.pose.position
        if math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.z):
            self.sp_pos[did] = [float(p.x), float(p.y), float(p.z)]
            self.sp_ts[did] = time.time()

    def _on_ff(self, did: str, msg: TwistStamped) -> None:
        v = msg.twist.linear
        if math.isfinite(v.x) and math.isfinite(v.y) and math.isfinite(v.z):
            self.sp_ff[did] = [float(v.x), float(v.y), float(v.z)]

    def _step(self) -> None:
        now = time.time()
        stamp = self.get_clock().now().to_msg()
        pub_bat = (now - self.last_bat) >= 0.25
        if pub_bat:
            self.last_bat = now
        pub_vis = (now - self.last_vis) >= 0.10
        if pub_vis:
            self.last_vis = now

        for idx, d in enumerate(self.drone_ids):
            p = self.pos[d]
            v = self.vel[d]
            sp = self.sp_pos[d]
            ff = self.sp_ff[d] if (now - self.sp_ts[d]) < 0.5 else [0.0, 0.0, 0.0]

            if sp is not None:
                kp = 2.6
                v_des = [
                    kp * (sp[0] - p[0]) + 0.65 * ff[0],
                    kp * (sp[1] - p[1]) + 0.65 * ff[1],
                    kp * (sp[2] - p[2]) + 0.65 * ff[2],
                ]
                spd = math.sqrt(v_des[0] ** 2 + v_des[1] ** 2 + v_des[2] ** 2)
                max_v = 11.5
                if spd > max_v:
                    s = max_v / spd
                    v_des = [v_des[0] * s, v_des[1] * s, v_des[2] * s]
                tau = 0.18
                alpha = min(1.0, self.dt / tau)
                v[0] += alpha * (v_des[0] - v[0])
                v[1] += alpha * (v_des[1] - v[1])
                v[2] += alpha * (v_des[2] - v[2])
                p[0] += v[0] * self.dt
                p[1] += v[1] * self.dt
                p[2] = max(0.05, p[2] + v[2] * self.dt)
                if sp[2] <= 0.25 and p[2] <= 0.16:
                    p[2] = 0.05
                    v[0] *= 0.4
                    v[1] *= 0.4
                    v[2] = 0.0

            odom = Odometry()
            odom.header.stamp = stamp
            odom.header.frame_id = 'map'
            odom.child_frame_id = f'{d}_base_link'
            odom.pose.pose.position.x = p[0]
            odom.pose.pose.position.y = p[1]
            odom.pose.pose.position.z = p[2]
            odom.pose.pose.orientation.w = 1.0
            odom.twist.twist.linear.x = v[0]
            odom.twist.twist.linear.y = v[1]
            odom.twist.twist.linear.z = v[2]
            self.odom_pubs[d].publish(odom)

            if pub_bat:
                bat = BatteryState()
                bat.header.stamp = stamp
                bat.voltage = 24.4
                bat.percentage = 0.90
                bat.present = True
                self.bat_pubs[d].publish(bat)

                fc = DiagnosticStatus()
                fc.level = b'\x00'
                fc.name = f'{d}_ekf2'
                fc.message = 'EKF2_READY'
                self.fc_pubs[d].publish(fc)

        if pub_vis and self.active_gt:
            airborne = any(self.pos[d][2] > 1.5 for d in self.drone_ids)
            if airborne:
                vis_items = []
                for gt in self.active_gt:
                    gx, gy, gz = float(gt['x']), float(gt['y']), float(gt['z'])
                    best_dist = min(
                        math.sqrt(
                            (self.pos[d][0] - gx) ** 2 + (self.pos[d][1] - gy) ** 2 + (self.pos[d][2] - gz) ** 2
                        )
                        for d in self.drone_ids
                    )
                    if best_dist <= 80.0:
                        sig = 0.05 + 0.0022 * best_dist
                        vx = gx + self.rng.gauss(0.0, sig)
                        vy = gy + self.rng.gauss(0.0, sig)
                        vz = gz + self.rng.gauss(0.0, 0.75 * sig)
                        vis_items.append(
                            {
                                'target_id': gt.get('target_id', 'red_moving_target'),
                                'class_id': gt.get('class_id', 'crimson_uav'),
                                'x': round(vx, 4),
                                'y': round(vy, 4),
                                'z': round(vz, 4),
                                'stereo': True,
                                'confidence': round(max(0.75, 0.98 - 0.002 * best_dist), 3),
                            }
                        )
                if vis_items:
                    diag = String()
                    diag.data = json.dumps(
                        {
                            'stamp': round(now, 4),
                            'mode': 'stereo_3d',
                            'detected_count': len(vis_items),
                            'stereo_3d_count': len(vis_items),
                        }
                    )
                    self.vis_diag_pub.publish(diag)

                    vt_msg = String()
                    vt_msg.data = json.dumps(
                        {
                            'stamp': round(now, 4),
                            'frame_id': 'map',
                            'mode': 'stereo_3d',
                            'targets': vis_items,
                        }
                    )
                    self.vis_targets_pub.publish(vt_msg)

                    leg = PoseStamped()
                    leg.header.stamp = stamp
                    leg.header.frame_id = 'map'
                    leg.pose.position.x = float(vis_items[0]['x'])
                    leg.pose.position.y = float(vis_items[0]['y'])
                    leg.pose.position.z = float(vis_items[0]['z'])
                    leg.pose.orientation.w = 1.0
                    self.vis_legacy_pub.publish(leg)


def main() -> int:
    parser = argparse.ArgumentParser(description='Execute full closed-loop 6v3 ROS 2 counter-swarm mission.')
    parser.add_argument(
        '--obstacles',
        type=Path,
        default=Path('/home/hkw/flying/ros2_ws/src/flight_ctrl_pkg/config/obstacles.yaml'),
    )
    parser.add_argument('--trace', type=Path, default=Path('/tmp/project2_mission_trace.jsonl'))
    parser.add_argument('--summary', type=Path, default=Path('/tmp/project2_mission_summary.json'))
    parser.add_argument('--timeout', type=float, default=110.0)
    args = parser.parse_args()

    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.trace.unlink(missing_ok=True)
    args.summary.unlink(missing_ok=True)

    helipads = _load_helipads(args.obstacles)
    drone_json = json.dumps(DRONE_IDS)
    obs_str = str(args.obstacles.resolve())

    procs: List[subprocess.Popen] = []
    cmds = [
        [
            'ros2',
            'run',
            'flight_ctrl_pkg',
            'target_sim_node',
            '--ros-args',
            '-p',
            'sim_backend:=kinematic',
            '-p',
            'enable_multi_target:=true',
            '-p',
            f'obstacles_yaml_path:={obs_str}',
            '-p',
            'speed_scale:=0.85',
            '-p',
            'hit_radius_m:=1.45',
        ],
        [
            'ros2',
            'run',
            'planning_pkg',
            'planning_node',
            '--ros-args',
            '-p',
            f'drone_ids:={drone_json}',
            '-p',
            f'obstacles_yaml_path:={obs_str}',
            '-p',
            'confirm_m:=2',
            '-p',
            'confirm_n:=4',
            '-p',
            'v_intercept:=10.8',
            '-p',
            'v_attack:=12.2',
        ],
        [
            'ros2',
            'run',
            'center_node_pkg',
            'center_node',
            '--ros-args',
            '-p',
            f'swarm_ids:={drone_json}',
            '-p',
            f'obstacles_yaml_path:={obs_str}',
            '-p',
            'takeoff_alt:=9.2',
            '-p',
            'cbf_safe_dist:=2.3',
        ],
        [
            'ros2',
            'run',
            'flight_ctrl_pkg',
            'mission_commander_node',
            '--ros-args',
            '-p',
            f'drone_ids:={drone_json}',
            '-p',
            f'obstacles_yaml_path:={obs_str}',
            '-p',
            f'log_jsonl_path:={args.trace}',
            '-p',
            f'summary_json_path:={args.summary}',
            '-p',
            'patrol_duration_sec:=1.2',
            '-p',
            'track_duration_sec:=2.2',
            '-p',
            'intercept_duration_sec:=3.0',
            '-p',
            'attack_duration_sec:=45.0',
            '-p',
            'exit_on_complete:=true',
        ],
    ]

    log_dir = Path('/tmp/project2_mission_logs')
    log_dir.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    plant = SwarmDynamicsPlant(helipads, DRONE_IDS)

    try:
        for idx, cmd in enumerate(cmds):
            lf = (log_dir / f'node_{idx}_{cmd[3]}.log').open('w', encoding='utf-8')
            p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            procs.append(p)

        t0 = time.monotonic()
        completed_summary = None
        while (time.monotonic() - t0) < args.timeout:
            rclpy.spin_once(plant, timeout_sec=0.025)
            if args.summary.is_file() and args.summary.stat().st_size > 10:
                try:
                    data = json.loads(args.summary.read_text(encoding='utf-8'))
                    if data.get('completed') is True:
                        completed_summary = data
                        break
                except Exception:
                    pass

        if completed_summary is None:
            print(f'ERROR [closed_loop_mission]: Timed out after {args.timeout}s waiting for completion.', file=sys.stderr)
            return 1

        print(json.dumps(completed_summary, indent=2, ensure_ascii=False))
        return 0
    finally:
        plant.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for p in procs:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    p.terminate()
        for p in procs:
            try:
                p.wait(timeout=3.0)
            except Exception:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except Exception:
                    pass


if __name__ == '__main__':
    raise SystemExit(main())
