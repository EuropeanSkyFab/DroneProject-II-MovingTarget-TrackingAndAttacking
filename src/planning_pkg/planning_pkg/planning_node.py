# -*- coding: utf-8 -*-
"""
planning_node.py —— 多目标 IMM 跟踪、动态子群 WTA、避障绕飞与 3D APN 制导主节点

兼容并扩展《接口规范》及 PROJECT.md § Interface Contracts #2–#5：
  订阅：
    - /vision/targets            (std_msgs/String JSON, 多目标视觉三维定位)
    - /external/targets_detected (std_msgs/String JSON, 多目标外部雷达引导)
    - /engagement/kill_event     (std_msgs/String JSON, 打击击毁事件触发即时重分配)
    - /fleet/state               (std_msgs/String JSON, 集群电量 SoC 等状态)
    - /vision/target_pose        (geometry_msgs/PoseStamped, 向后兼容单目标视觉)
    - /external/target_detected  (geometry_msgs/PoseStamped, 向后兼容单目标雷达)
    - /strategy/cmd              (std_msgs/String JSON, 支持动作切换与 neutralized_targets)
    - /drone_<id>/state          (nav_msgs/Odometry, 己方各机 ENU 位姿速度，支持动态发现)
  发布：
    - /planning/target_pose      (std_msgs/String JSON, 20 Hz 每机指令含 pos/vel/acc/target_id/imm_probs/pred_path)
    - /planning/diagnostics      (std_msgs/String JSON, 2 Hz 航迹库、WTA 映射与重分配遥测)
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from planning_pkg.estimator import IMMTrackBank, ManeuveringTargetEstimator, TrackObservation
from planning_pkg.guidance import CooperativeGuidancePlanner, DynamicsProfile, TargetTrackState
from planning_pkg.obstacles import ObstacleMap

_VALID_ACTIONS = {
    "TAKEOFF",
    "HOVER",
    "RETURN",
    "TRACK",
    "INTERCEPT",
    "ATTACK",
    "RESUME_PATROL",
}

_DRONE_STATE_TOPIC_RE = re.compile(r"^/(drone_\d+)/state$")


class PlanningNode(Node):
    """多源融合多目标 IMM 状态估计与集群协同制导规划节点。"""

    def __init__(self) -> None:
        super().__init__("planning_node")

        # -------------------- 参数声明（100% 保留原有 14 个参数名与默认值） --------------------
        self.declare_parameter("drone_ids", ["drone_1", "drone_2", "drone_3"])
        self.declare_parameter("plan_hz", 20.0)
        self.declare_parameter("v_cruise", 5.2)
        self.declare_parameter("v_intercept", 10.5)
        self.declare_parameter("v_attack", 12.2)
        self.declare_parameter("a_max", 10.5)
        self.declare_parameter("min_alt", 3.2)
        self.declare_parameter("max_alt", 35.0)
        self.declare_parameter("track_radius", 6.0)
        self.declare_parameter("track_alt_offset", 3.5)
        self.declare_parameter("max_coast_sec", 3.0)
        self.declare_parameter("sigma_vision", 0.36)
        self.declare_parameter("sigma_external", 0.55)
        self.declare_parameter("max_xy_dist", 950.0)
        self.declare_parameter("auto_discover_drones", True)
        self.declare_parameter("obstacles_yaml_path", "")
        self.declare_parameter("confirm_m", 3)
        self.declare_parameter("confirm_n", 5)

        p = self.get_parameter
        raw_drones = p("drone_ids").value
        if isinstance(raw_drones, str):
            try:
                parsed_d = json.loads(raw_drones)
                self.drone_ids: List[str] = [str(d) for d in parsed_d]
            except Exception:
                self.drone_ids = [s.strip() for s in raw_drones.split(",") if s.strip()]
        else:
            self.drone_ids = [str(d) for d in raw_drones]

        self.plan_hz: float = max(2.0, float(p("plan_hz").value))
        self.max_xy_dist: float = float(p("max_xy_dist").value)
        self.auto_discover_drones: bool = bool(p("auto_discover_drones").value)

        profile = DynamicsProfile(
            v_cruise=float(p("v_cruise").value),
            v_intercept=float(p("v_intercept").value),
            v_attack=float(p("v_attack").value),
            a_max=float(p("a_max").value),
            min_alt=float(p("min_alt").value),
            max_alt=float(p("max_alt").value),
            track_radius=float(p("track_radius").value),
            track_alt_offset=float(p("track_alt_offset").value),
        )

        obstacles_yaml_param = str(p("obstacles_yaml_path").value)
        self.obstacle_map, self.obstacle_source = ObstacleMap.load_default(
            yaml_path_param=obstacles_yaml_param,
            drone_radius=profile.drone_radius,
        )
        if self.obstacle_map.helipads:
            xs = [float(hp["x"]) for hp in self.obstacle_map.helipads]
            ys = [float(hp["y"]) for hp in self.obstacle_map.helipads]
            self._defense_center = np.array([sum(xs) / len(xs), sum(ys) / len(ys), 0.0], dtype=float)
        else:
            self._defense_center = np.array([0.0, 3.0, 0.0], dtype=float)

        sigma_vis = float(p("sigma_vision").value)
        sigma_ext = float(p("sigma_external").value)
        max_coast = float(p("max_coast_sec").value)
        confirm_m = int(p("confirm_m").value)
        confirm_n = int(p("confirm_n").value)

        self.estimator = ManeuveringTargetEstimator(
            sigma_vision=sigma_vis,
            sigma_external=sigma_ext,
            max_coast_sec=max_coast,
        )
        self.track_bank = IMMTrackBank(
            sigma_vision=sigma_vis,
            sigma_external=sigma_ext,
            max_coast_sec=max_coast,
            confirm_m=confirm_m,
            confirm_n=confirm_n,
        )
        self.planner = CooperativeGuidancePlanner(profile=profile, obstacle_map=self.obstacle_map)

        # -------------------- 内部集群状态缓存 --------------------
        self.drone_modes: Dict[str, str] = {d: "GROUND" for d in self.drone_ids}
        self.drone_kinematics: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.drone_batteries: Dict[str, float] = {}
        self._subscribed_drones: Set[str] = set()
        self._last_diag_sec: float = 0.0
        self._last_multi_vision_sec: float = -1e9
        self._last_multi_external_sec: float = -1e9
        self._estimator_mirrored_only: bool = False
        self._mirrored_update_count: int = 0
        self._mirrored_last_update_time: float = -1e9

        # -------------------- 订阅器（多目标 + 击毁 + 集群状态 + 兼容单目标） --------------------
        self.create_subscription(String, "/vision/targets", self._vision_targets_cb, 10)
        self.create_subscription(String, "/external/targets_detected", self._external_targets_cb, 10)
        self.create_subscription(String, "/engagement/kill_event", self._kill_event_cb, 10)
        self.create_subscription(String, "/fleet/state", self._fleet_state_cb, 10)
        self.create_subscription(PoseStamped, "/vision/target_pose", self._vision_cb, 10)
        self.create_subscription(PoseStamped, "/external/target_detected", self._external_cb, 10)
        self.create_subscription(String, "/strategy/cmd", self._strategy_cb, 10)

        for d_id in self.drone_ids:
            self._ensure_drone_subscription(d_id)

        # -------------------- 发布器（接口 C + 诊断） --------------------
        self.plan_pub = self.create_publisher(String, "/planning/target_pose", 20)
        self.diag_pub = self.create_publisher(String, "/planning/diagnostics", 10)

        self.timer = self.create_timer(1.0 / self.plan_hz, self._plan_tick)
        if self.auto_discover_drones:
            self._discover_timer = self.create_timer(1.0, self._discover_drones)

        self.get_logger().info(
            f"planning_node 已启动({self.plan_hz:.0f}Hz): 集群={self.drone_ids}, "
            f"V_max={profile.v_attack:.1f}m/s, R_track={profile.track_radius:.1f}m, "
            f"obstacles={len(self.obstacle_map.primitives)} ({self.obstacle_source})"
        )

    def _ensure_drone_subscription(self, d_id: str) -> None:
        if d_id not in self.drone_ids:
            self.drone_ids.append(d_id)
        if d_id not in self.drone_modes:
            self.drone_modes[d_id] = "GROUND"
        if d_id not in self._subscribed_drones:
            self._subscribed_drones.add(d_id)
            self.create_subscription(
                Odometry,
                f"/{d_id}/state",
                self._make_state_cb(d_id),
                10,
            )

    def _discover_drones(self) -> None:
        try:
            topics = self.get_topic_names_and_types()
        except Exception:
            return
        for topic_name, topic_types in topics:
            match = _DRONE_STATE_TOPIC_RE.match(topic_name)
            if not match:
                continue
            if any("Odometry" in t for t in topic_types):
                d_id = match.group(1)
                if d_id not in self._subscribed_drones:
                    self._ensure_drone_subscription(d_id)
                    self.get_logger().info(f"[规划] 自动发现并接入无人机: {d_id}")

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _extract_stamp_sec(self, msg: PoseStamped, fallback_now: float) -> float:
        s = int(msg.header.stamp.sec)
        ns = int(msg.header.stamp.nanosec)
        if s == 0 and ns == 0:
            return fallback_now
        t = s + ns * 1e-9
        if not math.isfinite(t) or t < (fallback_now - 60.0) or t > (fallback_now + 0.5):
            return fallback_now
        return t

    def _extract_json_stamp(self, data: Dict[str, Any], fallback_now: float) -> float:
        raw_stamp = data.get("stamp", None)
        if raw_stamp is None:
            return fallback_now
        try:
            t = float(raw_stamp)
        except (TypeError, ValueError):
            return fallback_now
        if not math.isfinite(t) or t <= 0.0 or t < (fallback_now - 60.0) or t > (fallback_now + 0.5):
            return fallback_now
        return t

    def _valid_xyz(self, x: float, y: float, z: float) -> bool:
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return False
        if abs(x) > self.max_xy_dist or abs(y) > self.max_xy_dist or z < -0.5 or z > 55.0:
            return False
        return True

    def _sanitize_target_pose(self, msg: PoseStamped) -> Tuple[bool, Tuple[float, float, float]]:
        pos = msg.pose.position
        x, y, z = float(pos.x), float(pos.y), float(pos.z)
        if not self._valid_xyz(x, y, z):
            return False, (0.0, 0.0, 0.0)
        return True, (x, y, z)

    def _parse_multi_targets(
        self,
        data: Dict[str, Any],
        is_vision: bool,
    ) -> List[TrackObservation]:
        obs_list: List[TrackObservation] = []
        raw_list = data.get("targets", [])
        if not isinstance(raw_list, list):
            return obs_list
        for item in raw_list:
            if not isinstance(item, dict):
                continue
            try:
                x = float(item.get("x", float("nan")))
                y = float(item.get("y", float("nan")))
                z = float(item.get("z", float("nan")))
            except (TypeError, ValueError):
                continue
            if not self._valid_xyz(x, y, z):
                continue
            # 仅对明确的语义 target_id（如外部引导/仿真器ID）绑定 external_id，避免视觉瞬态 det_id 误杀
            raw_tid = item.get("target_id", "")
            tid = str(raw_tid).strip() if raw_tid else ""
            cid = str(item.get("class_id", "")).strip()
            conf = float(item.get("conf", 1.0)) if item.get("conf") is not None else 1.0
            stereo = bool(item.get("stereo", True)) if is_vision else True
            obs_list.append(
                TrackObservation(
                    pos_enu=(x, y, z),
                    target_id=tid,
                    class_id=cid,
                    conf=conf,
                    stereo=stereo,
                )
            )
        return obs_list

    def _vision_targets_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(data, dict):
            return
        now_s = self._now_sec()
        ts = self._extract_json_stamp(data, now_s)
        if now_s - ts > 0.55:
            return
        self._last_multi_vision_sec = now_s
        obs_list = self._parse_multi_targets(data, is_vision=True)
        self.track_bank.update_batch(obs_list, ts, source="vision")

    def _external_targets_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(data, dict):
            return
        now_s = self._now_sec()
        ts = self._extract_json_stamp(data, now_s)
        if now_s - ts > 1.0:
            return
        self._last_multi_external_sec = now_s
        obs_list = self._parse_multi_targets(data, is_vision=False)
        self.track_bank.update_batch(obs_list, ts, source="external")

    def _vision_cb(self, msg: PoseStamped) -> None:
        m = copy.deepcopy(msg)
        ok, xyz = self._sanitize_target_pose(m)
        if not ok:
            self.get_logger().warn("[规划] 拒绝非法视觉坐标输入", throttle_duration_sec=2.0)
            return
        now_s = self._now_sec()
        ts = self._extract_stamp_sec(m, now_s)
        if now_s - ts > 0.55:
            return
        if now_s - self._last_multi_vision_sec <= 0.6:
            return
        self.estimator.update(xyz, ts, source="vision")
        self._estimator_mirrored_only = False

    def _external_cb(self, msg: PoseStamped) -> None:
        m = copy.deepcopy(msg)
        ok, xyz = self._sanitize_target_pose(m)
        if not ok:
            self.get_logger().warn("[规划] 拒绝非法外部雷达坐标输入", throttle_duration_sec=2.0)
            return
        now_s = self._now_sec()
        ts = self._extract_stamp_sec(m, now_s)
        if now_s - ts > 1.0:
            return
        if now_s - self._last_multi_external_sec <= 1.2:
            return
        self.estimator.update(xyz, ts, source="external")
        self._estimator_mirrored_only = False

    def _refresh_estimator_ownership(self) -> None:
        if self._estimator_mirrored_only:
            direct_override = (
                int(self.estimator.update_count) != self._mirrored_update_count
                or float(self.estimator.last_update_time) != self._mirrored_last_update_time
            )
            if direct_override and self.estimator.initialized:
                self._estimator_mirrored_only = False

    def _kill_event_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().error(f"[规划] 解析 /engagement/kill_event 失败: {exc}")
            return
        if not isinstance(data, dict):
            return
        now_s = self._now_sec()
        raw_tid = str(data.get("target_id", "")).strip()
        striker_id = str(data.get("striker_id", "")).strip()

        pos_enu: Optional[Tuple[float, float, float]] = None
        if "x" in data and "y" in data and "z" in data:
            try:
                px, py, pz = float(data["x"]), float(data["y"]), float(data["z"])
                if math.isfinite(px) and math.isfinite(py) and math.isfinite(pz):
                    pos_enu = (px, py, pz)
            except (TypeError, ValueError):
                pos_enu = None

        extra_tids: List[str] = []
        if striker_id and striker_id in self.planner.last_wta_assignments:
            extra_tids.append(self.planner.last_wta_assignments[striker_id])

        killed = self.track_bank.mark_killed(
            target_id=raw_tid or None,
            pos_enu=pos_enu,
            now_sec=now_s,
            extra_track_ids=extra_tids if (pos_enu is None or not raw_tid) else None,
        )
        if not killed and extra_tids:
            killed = self.track_bank.mark_killed(
                target_id=raw_tid or None,
                pos_enu=pos_enu,
                now_sec=now_s,
                extra_track_ids=extra_tids,
            )
        self.planner.notify_target_killed(killed, external_target_id=raw_tid)

        self._refresh_estimator_ownership()
        if self._estimator_mirrored_only:
            if not self.track_bank.get_confirmed_tracks(now_s):
                self.estimator.initialized = False
                self._estimator_mirrored_only = False
        elif self.estimator.initialized:
            if raw_tid == "legacy_target" or (
                pos_enu is not None
                and float(np.linalg.norm(self.estimator.x[0:3] - np.array(pos_enu, dtype=float))) <= 3.5
            ):
                self.estimator.initialized = False

        self.get_logger().info(
            f"[规划] 接收击毁事件 target={raw_tid}, striker={striker_id}, cleared={killed}"
        )

    def _fleet_state_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(data, dict):
            return
        drones_info = data.get("drones", {})
        if not isinstance(drones_info, dict):
            return
        for d_id, info in drones_info.items():
            if not isinstance(info, dict):
                continue
            if "battery_pct" in info:
                try:
                    val = float(info["battery_pct"])
                    if math.isfinite(val):
                        soc = val / 100.0 if val > 1.5 else val
                        self.drone_batteries[str(d_id)] = max(0.05, min(1.0, soc))
                except (TypeError, ValueError):
                    continue

    def _make_state_cb(self, drone_id: str):
        def _cb(msg: Odometry) -> None:
            m = copy.deepcopy(msg)
            p = m.pose.pose.position
            v = m.twist.twist.linear
            pos = np.array([float(p.x), float(p.y), float(p.z)], dtype=float)
            vel = np.array([float(v.x), float(v.y), float(v.z)], dtype=float)
            if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(vel))):
                return
            self.drone_kinematics[drone_id] = (pos, vel)
            if self.drone_modes.get(drone_id) == "RETURN":
                if pos[2] < 0.32 and float(np.linalg.norm(vel)) < 0.22:
                    self.drone_modes[drone_id] = "GROUND"

        return _cb

    def _strategy_cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().error(f"[规划] 解析 /strategy/cmd 失败: {exc}")
            return
        if not isinstance(data, dict):
            return

        now_s = self._now_sec()
        # 处理战略层面下发的已清除目标列表
        neutralized = data.get("neutralized_targets", [])
        if isinstance(neutralized, list) and neutralized:
            legacy_killed = False
            for nt in neutralized:
                if isinstance(nt, str) and nt.strip():
                    tid = nt.strip()
                    if tid == "legacy_target":
                        legacy_killed = True
                    killed = self.track_bank.mark_killed(target_id=tid, now_sec=now_s)
                    self.planner.notify_target_killed(killed, external_target_id=tid)
                elif isinstance(nt, dict):
                    tid = str(nt.get("target_id", "")).strip()
                    trk_id = str(nt.get("track_id", "")).strip()
                    effective_tid = tid or trk_id
                    if effective_tid == "legacy_target":
                        legacy_killed = True
                    pos_enu = None
                    if "x" in nt and "y" in nt and "z" in nt:
                        try:
                            px, py, pz = float(nt["x"]), float(nt["y"]), float(nt["z"])
                            if math.isfinite(px) and math.isfinite(py) and math.isfinite(pz):
                                pos_enu = (px, py, pz)
                        except (TypeError, ValueError):
                            pos_enu = None
                    killed = self.track_bank.mark_killed(
                        target_id=tid or None,
                        pos_enu=pos_enu,
                        now_sec=now_s,
                        extra_track_ids=[trk_id] if trk_id else None,
                    )
                    self.planner.notify_target_killed(killed, external_target_id=effective_tid)

            self._refresh_estimator_ownership()
            if self._estimator_mirrored_only:
                if not self.track_bank.get_confirmed_tracks(now_s):
                    self.estimator.initialized = False
                    self._estimator_mirrored_only = False
            elif self.estimator.initialized and legacy_killed:
                self.estimator.initialized = False

        if "action" not in data and "neutralized_targets" in data:
            return

        raw_id = str(data.get("drone_id", "drone_1"))
        action = str(data.get("action", "HOVER")).upper().strip()
        if action not in _VALID_ACTIONS:
            self.get_logger().warn(f"[规划] 忽略未知战术动作: {action}")
            return

        targets = self.drone_ids if raw_id.upper() == "ALL" else [raw_id]
        for d_id in targets:
            if d_id not in self.drone_modes:
                continue
            cur_mode = self.drone_modes[d_id]
            airborne_now = (
                d_id in self.drone_kinematics and float(self.drone_kinematics[d_id][0][2]) > 1.2
            )
            if cur_mode == "GROUND" and not airborne_now:
                if action != "TAKEOFF":
                    continue
            elif action == "TAKEOFF" and cur_mode != "GROUND":
                continue
            self.drone_modes[d_id] = action
            self.get_logger().info(f"[规划] {d_id} 战术状态切换 -> {action}")

    def _plan_tick(self) -> None:
        now_s = self._now_sec()
        self.track_bank.step_maintenance(now_s)
        confirmed_tracks = self.track_bank.get_confirmed_tracks(now_s)

        if confirmed_tracks:
            # 多目标 IMM + 动态子群 WTA 模式
            target_states: List[TargetTrackState] = []
            for trk in confirmed_tracks:
                t_pos, t_vel, t_acc = trk.get_kinematics_at(now_s)
                target_states.append(
                    TargetTrackState(
                        target_id=trk.track_id,
                        pos=t_pos,
                        vel=t_vel,
                        acc=t_acc,
                        predict_fn=lambda tau, t=trk: t.predict_pv_ahead(tau),
                        imm_probs=trk.imm_probs,
                        pred_path=trk.predict_path(),
                        class_id=trk.class_id,
                    )
                )
            self.estimator.sync_from_track(confirmed_tracks[0])
            self._estimator_mirrored_only = True
            self._mirrored_update_count = int(self.estimator.update_count)
            self._mirrored_last_update_time = float(self.estimator.last_update_time)
            cmds = self.planner.compute_multi_target_swarm_targets(
                now_sec=now_s,
                swarm_ids=self.drone_ids,
                drone_modes=self.drone_modes,
                drone_states=self.drone_kinematics,
                targets=target_states,
                drone_batteries=self.drone_batteries,
                defense_center=self._defense_center,
            )
            target_valid = True
            t_pos_legacy = target_states[0].pos
        else:
            self._refresh_estimator_ownership()
            if self._estimator_mirrored_only:
                self.estimator.initialized = False
                self._estimator_mirrored_only = False
                legacy_valid = False
            else:
                legacy_valid = self.estimator.is_valid(now_s)

            if legacy_valid:
                # 单目标 / 直接注入 self.estimator 向后兼容模式
                self.estimator.propagate_to(now_s)
                t_pos, t_vel, t_acc = self.estimator.get_kinematics_at(now_s)
                cmds = self.planner.compute_swarm_targets(
                    now_sec=now_s,
                    swarm_ids=self.drone_ids,
                    drone_modes=self.drone_modes,
                    drone_states=self.drone_kinematics,
                    target_valid=True,
                    target_pos=t_pos,
                    target_vel=t_vel,
                    target_acc=t_acc,
                    predict_fn=lambda tau: self.estimator.predict_pv_ahead(tau),
                )
                probs_legacy = {k: round(v, 4) for k, v in self.estimator.imm_probs.items()}
                path_legacy = self.estimator.predict_path()
                for payload in cmds.values():
                    if payload.get("target_id", "") == "legacy_target":
                        payload["imm_probs"] = probs_legacy
                        payload["pred_path"] = path_legacy
                target_valid = True
                t_pos_legacy = t_pos
            else:
                cmds = self.planner.compute_multi_target_swarm_targets(
                    now_sec=now_s,
                    swarm_ids=self.drone_ids,
                    drone_modes=self.drone_modes,
                    drone_states=self.drone_kinematics,
                    targets=[],
                    drone_batteries=self.drone_batteries,
                    defense_center=self._defense_center,
                )
                target_valid = False
                t_pos_legacy = np.zeros(3, dtype=float)

        pub_stamp = self._now_sec()
        for d_id, payload in cmds.items():
            x = max(-self.max_xy_dist, min(self.max_xy_dist, float(payload["x"])))
            y = max(-self.max_xy_dist, min(self.max_xy_dist, float(payload["y"])))
            z = float(payload["z"])
            out_json: Dict[str, Any] = {
                "drone_id": d_id,
                "stamp": pub_stamp,
                "x": round(x, 4),
                "y": round(y, 4),
                "z": round(z, 4),
                "vx": round(float(payload.get("vx", 0.0)), 4),
                "vy": round(float(payload.get("vy", 0.0)), 4),
                "vz": round(float(payload.get("vz", 0.0)), 4),
                "ax": round(float(payload.get("ax", 0.0)), 4),
                "ay": round(float(payload.get("ay", 0.0)), 4),
                "az": round(float(payload.get("az", 0.0)), 4),
                "action": str(payload["action"]),
                "t_go": payload.get("t_go", 0.0),
                "target_id": str(payload.get("target_id", "")),
            }
            if "focus_x" in payload and "focus_y" in payload and "focus_z" in payload:
                out_json["focus_x"] = round(float(payload["focus_x"]), 4)
                out_json["focus_y"] = round(float(payload["focus_y"]), 4)
                out_json["focus_z"] = round(float(payload["focus_z"]), 4)
            elif target_valid:
                out_json["focus_x"] = round(float(t_pos_legacy[0]), 4)
                out_json["focus_y"] = round(float(t_pos_legacy[1]), 4)
                out_json["focus_z"] = round(float(t_pos_legacy[2]), 4)

            if "imm_probs" in payload:
                out_json["imm_probs"] = payload["imm_probs"]
            if "pred_path" in payload:
                out_json["pred_path"] = payload["pred_path"]

            out_msg = String()
            out_msg.data = json.dumps(out_json)
            self.plan_pub.publish(out_msg)

        if now_s - self._last_diag_sec >= 0.5:
            self._last_diag_sec = now_s
            diag = {
                "stamp": round(now_s, 3),
                "modes": self.drone_modes,
                "estimator": self.estimator.diagnostics(now_s),
                "tracks": self.track_bank.diagnostics(now_s),
                "wta_assignments": dict(self.planner.last_wta_assignments),
                "neutralized_targets": sorted(self.planner.neutralized_targets),
                "re_tasking_events": self.planner.re_tasking_events,
                "obstacle_count": len(self.obstacle_map.primitives),
                "active_plans": {
                    k: {
                        "t_go": v.get("t_go"),
                        "reachable": v.get("reachable"),
                        "target_id": v.get("target_id", ""),
                    }
                    for k, v in cmds.items()
                },
            }
            d_msg = String()
            d_msg.data = json.dumps(diag)
            self.diag_pub.publish(d_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlanningNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
