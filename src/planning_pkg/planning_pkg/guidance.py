# -*- coding: utf-8 -*-
"""
guidance.py —— 多无人机动态子群 WTA 分配、广义 K 机协同合围/钳形封堵与 3D APN 制导

全部输入与输出严格采用全局 ENU 坐标系（米、米/秒）。
零 SciPy 依赖，使用纯 NumPy O(N^3) 匈牙利算法 (planning_pkg.estimator.hungarian_assignment)。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from planning_pkg.estimator import hungarian_assignment
from planning_pkg.obstacles import ObstacleMap


@dataclass
class DynamicsProfile:
    """己方无人机物理动力学包线参数（默认对齐实测阶跃标定基准）。"""

    v_cruise: float = 5.2
    v_intercept: float = 10.5
    v_attack: float = 12.2
    a_max: float = 9.5
    min_alt: float = 3.2
    max_alt: float = 35.0
    track_radius: float = 6.0
    track_alt_offset: float = 3.5
    apn_nav_gain: float = 3.8
    drone_radius: float = 0.45


@dataclass
class TargetTrackState:
    """标准化多目标跟踪态输入，用于动态子群 WTA 与协同制导规划。"""

    target_id: str
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    predict_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]]
    imm_probs: Dict[str, float] = field(default_factory=lambda: {"CV": 0.34, "CTRV": 0.33, "JINK": 0.33})
    pred_path: List[List[float]] = field(default_factory=list)
    priority: float = 1.0
    class_id: str = ""


def _clamp_vec_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    nrm = float(np.linalg.norm(vec))
    if nrm > max_norm and nrm > 1e-9:
        return vec * (max_norm / nrm)
    return vec


def _prioritize_normal_braking(
    a_raw: np.ndarray,
    priority_normals: Sequence[Tuple[np.ndarray, float]],
    a_max: float,
) -> np.ndarray:
    """在总加速度上限 a_max 内优先满足法向制动/脱困需求，剩余预算分配给切向与垂向机动。"""
    if not priority_normals:
        return _clamp_vec_norm(a_raw, a_max)

    a_n_vec = np.zeros(3, dtype=float)
    min_dot = 1.0
    if len(priority_normals) == 1:
        n_k, req_k = priority_normals[0]
        a_n_mag = min(a_max, max(float(np.dot(a_raw, n_k)), req_k))
        a_n_vec = a_n_mag * n_k
    else:
        min_dot = min(
            float(np.dot(priority_normals[i][0], priority_normals[j][0]))
            for i in range(len(priority_normals))
            for j in range(i + 1, len(priority_normals))
        )
        sorted_pn = sorted(priority_normals, key=lambda x: x[1])
        n_iters = 4 if min_dot >= -0.7 else 1
        for _ in range(n_iters):
            for n_k, req_k in sorted_pn:
                cur_n = float(np.dot(a_n_vec, n_k))
                if cur_n < req_k:
                    a_n_vec = a_n_vec + (req_k - cur_n) * n_k
            a_n_vec = _clamp_vec_norm(a_n_vec, a_max)

    n_norm = float(np.linalg.norm(a_n_vec))
    if n_norm <= 1e-6:
        return _clamp_vec_norm(a_raw, a_max)

    u_n = a_n_vec / n_norm
    a_tan_raw = a_raw - float(np.dot(a_raw, u_n)) * u_n
    if len(priority_normals) >= 2 and min_dot >= -0.7:
        for n_k, req_k in priority_normals:
            if float(np.dot(a_n_vec, n_k)) <= req_k + 0.5:
                tn = float(np.dot(a_tan_raw, n_k))
                if tn < 0.0:
                    a_tan_raw = a_tan_raw - tn * n_k
        a_tan_raw = a_tan_raw - float(np.dot(a_tan_raw, u_n)) * u_n

    a_rem = math.sqrt(max(0.0, a_max ** 2 - n_norm ** 2))
    return a_n_vec + _clamp_vec_norm(a_tan_raw, a_rem)


def _obstacle_surface_info(
    obstacle_map: ObstacleMap,
    p: np.ndarray,
) -> List[Tuple[float, np.ndarray, float, float]]:
    """计算点 p 到各三维相关障碍物/禁飞区膨胀边界的有符号水平距离、外法向及高度区间。"""
    x, y, z = float(p[0]), float(p[1]), float(p[2])
    out: List[Tuple[float, np.ndarray, float, float]] = []
    for obs in obstacle_map.primitives:
        z_lo, z_hi = obstacle_map._effective_z_interval(obs, 0.0)
        if z < z_lo - 0.35 or z > z_hi + 0.40:
            continue
        m = obstacle_map._effective_margin(obs, 0.0)
        dx = x - obs.cx
        dy = y - obs.cy
        if obs.obs_type in ("cylinder", "nfz"):
            r_eff = obs.radius + m
            dist = math.hypot(dx, dy)
            d_surf = dist - r_eff
            if dist > 1e-6:
                n_out = np.array([dx / dist, dy / dist, 0.0], dtype=float)
            else:
                n_out = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            if abs(obs.yaw) > 1e-6:
                cy = math.cos(-obs.yaw)
                sy = math.sin(-obs.yaw)
                lx = dx * cy - dy * sy
                ly = dx * sy + dy * cy
            else:
                lx, ly = dx, dy
            hx = 0.5 * obs.sx + m
            hy = 0.5 * obs.sy + m
            qx = abs(lx) - hx
            qy = abs(ly) - hy
            if qx > 0.0 or qy > 0.0:
                ox = max(qx, 0.0)
                oy = max(qy, 0.0)
                d_surf = math.hypot(ox, oy)
                nlx = math.copysign(ox, lx) if ox > 0.0 else 0.0
                nly = math.copysign(oy, ly) if oy > 0.0 else 0.0
                nrm = math.hypot(nlx, nly)
                nlx, nly = nlx / max(nrm, 1e-9), nly / max(nrm, 1e-9)
            else:
                if qx > qy:
                    d_surf = qx
                    nlx, nly = (1.0 if lx >= 0.0 else -1.0), 0.0
                else:
                    d_surf = qy
                    nlx, nly = 0.0, (1.0 if ly >= 0.0 else -1.0)
            if abs(obs.yaw) > 1e-6:
                wy_c = math.cos(obs.yaw)
                wy_s = math.sin(obs.yaw)
                nx = nlx * wy_c - nly * wy_s
                ny = nlx * wy_s + nly * wy_c
            else:
                nx, ny = nlx, nly
            n_out = np.array([nx, ny, 0.0], dtype=float)
        out.append((d_surf, n_out, z_lo, z_hi))
    return out


def solve_intercept_time(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    ego_speed_cap: float,
    a_max: float,
    predict_target_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]],
    t_max: float = 8.0,
) -> Tuple[float, bool]:
    """
    求解考虑初始速度转向惯性滞后的曲线 Apollonius 最早可达拦截时间 t_go^*:
        || p_t(tau) - (p_i + v_i * tau_inertia(tau)) || <= V_ego * tau

    返回：
        (t_go, reachable): 若在 [0.04, t_max] 内存在可达解则 reachable=True；
                           否则返回可达差距最小的最佳追击时刻与 reachable=False。
    """
    v_cap = max(1.0, float(ego_speed_cap))
    amax = max(1.0, float(a_max))
    p_i = np.asarray(ego_pos, dtype=float)
    v_i = np.asarray(ego_vel, dtype=float)

    p_t0, _ = predict_target_fn(0.0)
    d0 = float(np.linalg.norm(p_t0 - p_i))
    if d0 < 0.25:
        return 0.05, True

    def miss_gap(tau: float) -> float:
        p_t, _ = predict_target_fn(tau)
        tau_lag = min(tau * 0.35, v_cap / (2.2 * amax))
        eff_origin = p_i + v_i * tau_lag
        req_dist = float(np.linalg.norm(p_t - eff_origin))
        reach_dist = v_cap * max(0.0, tau - 0.5 * tau_lag)
        return req_dist - reach_dist

    n_grid = 32
    taus = np.linspace(0.05, float(t_max), n_grid)
    best_tau = float(taus[0])
    prev_tau = float(taus[0])
    prev_gap = miss_gap(prev_tau)
    if prev_gap <= 0.0:
        return prev_tau, True
    best_gap = prev_gap

    bracket: Optional[Tuple[float, float]] = None
    for tau_val in taus[1:]:
        t_cur = float(tau_val)
        g_cur = miss_gap(t_cur)
        if g_cur < best_gap:
            best_gap = g_cur
            best_tau = t_cur
        if g_cur <= 0.0:
            bracket = (prev_tau, t_cur)
            break
        prev_tau, prev_gap = t_cur, g_cur

    if bracket is None:
        return best_tau, False

    lo, hi = bracket
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        if miss_gap(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return hi, True


def _segments_cross_2d(
    a1: Tuple[float, float],
    a2: Tuple[float, float],
    b1: Tuple[float, float],
    b2: Tuple[float, float],
) -> bool:
    def _ccw(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    d1 = _ccw(b1, b2, a1)
    d2 = _ccw(b1, b2, a2)
    d3 = _ccw(a1, a2, b1)
    d4 = _ccw(a1, a2, b2)
    return ((d1 > 1e-9 and d2 < -1e-9) or (d1 < -1e-9 and d2 > 1e-9)) and (
        (d3 > 1e-9 and d4 < -1e-9) or (d3 < -1e-9 and d4 > 1e-9)
    )


def assign_encirclement_slots(
    drone_ids: Sequence[str],
    drone_positions: Dict[str, np.ndarray],
    center_xy: Tuple[float, float],
    radius: float,
    base_phase_rad: float,
    prev_assignment: Optional[Dict[str, int]] = None,
) -> Dict[str, Tuple[float, float, int]]:
    """
    为 K 架处于 TRACK 模式的无人机分配均匀相位环绕槽位（如三机 120°、六机 60° 锁相合围），
    采用纯 NumPy O(K^3) 匈牙利算法求解 1-Wasserstein (欧氏距离) 最小总机动代价，
    并附带防交叉校验的有界滞回防止槽位跳变。
    """
    active = [d for d in drone_ids if d in drone_positions]
    k_cnt = len(active)
    if k_cnt == 0:
        return {}

    cx, cy = float(center_xy[0]), float(center_xy[1])
    slots: List[Tuple[float, float]] = []
    for idx in range(k_cnt):
        ang = base_phase_rad + (2.0 * math.pi * idx) / k_cnt
        slots.append((cx + radius * math.cos(ang), cy + radius * math.sin(ang)))

    base_cost = np.zeros((k_cnt, k_cnt), dtype=float)
    for d_idx, d_id in enumerate(active):
        pos = drone_positions[d_id]
        px, py = float(pos[0]), float(pos[1])
        for slot_idx, (sx, sy) in enumerate(slots):
            base_cost[d_idx, slot_idx] = math.hypot(px - sx, py - sy)

    def _solve(hyst_bonus: float) -> Dict[str, Tuple[float, float, int]]:
        cost = base_cost.copy()
        if prev_assignment is not None and hyst_bonus > 0.0:
            for d_idx, d_id in enumerate(active):
                prev_slot = prev_assignment.get(d_id)
                if prev_slot is not None and 0 <= prev_slot < k_cnt:
                    cost[d_idx, prev_slot] -= hyst_bonus
        row_ind, col_ind, _ = hungarian_assignment(cost)
        res: Dict[str, Tuple[float, float, int]] = {}
        for r_i, c_j in zip(row_ind.tolist(), col_ind.tolist()):
            sx, sy = slots[c_j]
            res[active[r_i]] = (sx, sy, c_j)
        return res

    out = _solve(0.25 if prev_assignment is not None else 0.0)
    if prev_assignment is not None and k_cnt >= 2:
        segs = [
            ((float(drone_positions[d][0]), float(drone_positions[d][1])), (out[d][0], out[d][1]))
            for d in active
        ]
        crossed = any(
            _segments_cross_2d(segs[i][0], segs[i][1], segs[j][0], segs[j][1])
            for i in range(k_cnt)
            for j in range(i + 1, k_cnt)
        )
        if crossed:
            out = _solve(0.0)
    return out


def compute_target_threat_weights(
    targets: Sequence[TargetTrackState],
    defense_center: Optional[np.ndarray] = None,
) -> List[float]:
    """计算每个活跃目标的综合威胁权重 W_j（距离防御中心近度、向心速度、机动概率与优先级）。"""
    p_def = np.array([0.0, 3.0, 0.0], dtype=float) if defense_center is None else np.asarray(defense_center, dtype=float)
    weights: List[float] = []
    for tgt in targets:
        d_vec = p_def - tgt.pos
        d_def = float(np.linalg.norm(d_vec))
        u_in = d_vec / max(d_def, 1e-3)
        v_in = max(0.0, float(np.dot(tgt.vel, u_in)))
        mu_jink = float(tgt.imm_probs.get("JINK", 0.0))
        mu_ctrv = float(tgt.imm_probs.get("CTRV", 0.0))
        w_j = (
            (40.0 / (d_def + 8.0) + 0.35 * v_in)
            * (1.0 + 0.35 * mu_jink + 0.15 * mu_ctrv)
            * max(0.1, float(tgt.priority))
        )
        weights.append(float(w_j))
    return weights


def compute_wta_assignments(
    drone_ids: Sequence[str],
    drone_states: Dict[str, Tuple[np.ndarray, np.ndarray]],
    targets: Sequence[TargetTrackState],
    profile: DynamicsProfile,
    defense_center: Optional[np.ndarray] = None,
    drone_batteries: Optional[Dict[str, float]] = None,
    prev_assignments: Optional[Dict[str, str]] = None,
    obstacle_map: Optional[ObstacleMap] = None,
) -> Dict[str, str]:
    """
    基于纯 NumPy O(N^3) 匈牙利算法的动态子群武器-目标分配 (WTA)：
      - 综合曲线 Apollonius t_go、接近角、剩余电量 SoC、障碍物视线遮挡与防抖滞回奖励；
      - N >= M 时按威胁权重 W_j 均衡分群，保证每个活跃目标至少分配 1 架己方机 (K_j >= 1)；
      - N < M 时优先锁定威胁/拦截代价最优的 N 个目标立即接战。
    """
    active_drones = [d for d in drone_ids if d in drone_states]
    n_act = len(active_drones)
    m_tgt = len(targets)
    if n_act == 0 or m_tgt == 0:
        return {}

    threat_w = compute_target_threat_weights(targets, defense_center=defense_center)
    cost = np.zeros((n_act, m_tgt), dtype=float)

    for i, d_id in enumerate(active_drones):
        p_i, v_i = drone_states[d_id]
        v_i_norm = float(np.linalg.norm(v_i))
        soc = 1.0 if drone_batteries is None else max(0.05, min(1.0, float(drone_batteries.get(d_id, 1.0))))
        prev_tid = prev_assignments.get(d_id) if prev_assignments is not None else None

        for j, tgt in enumerate(targets):
            t_go_ij, reach = solve_intercept_time(
                p_i, v_i, profile.v_intercept, profile.a_max, tgt.predict_fn, t_max=5.0
            )
            c_tgo = float(t_go_ij) if reach else 8.0 + 1.5 * float(t_go_ij)

            los = tgt.pos - p_i
            d_los = float(np.linalg.norm(los))
            u_los = los / max(d_los, 1e-3)
            v_i_hat = v_i / v_i_norm if v_i_norm > 0.4 else u_los
            v_t_norm = float(np.linalg.norm(tgt.vel))
            v_t_hat = tgt.vel / v_t_norm if v_t_norm > 0.4 else np.zeros(3, dtype=float)
            c_aspect = 0.8 * (
                0.65 * 0.5 * (1.0 - float(np.dot(v_i_hat, u_los)))
                + 0.35 * max(0.0, float(np.dot(u_los, v_t_hat)))
            )

            c_batt = 2.0 * (1.0 - soc) * (1.0 + 0.5 * float(t_go_ij))
            c_obs = 0.75 if (obstacle_map is not None and not obstacle_map.segment_clear_3d(p_i, tgt.pos)) else 0.0
            c_hyst = -1.2 if prev_tid == tgt.target_id else 0.0

            cost[i, j] = c_tgo + c_aspect + c_batt + c_obs + c_hyst

    assignments: Dict[str, str] = {}
    if n_act >= m_tgt:
        base_k = n_act // m_tgt
        rem = n_act % m_tgt
        order = sorted(range(m_tgt), key=lambda idx: (-threat_w[idx], targets[idx].target_id))
        pack_sizes = [base_k] * m_tgt
        for idx in order[:rem]:
            pack_sizes[idx] += 1

        col_to_tgt: List[int] = []
        cols: List[np.ndarray] = []
        for j in range(m_tgt):
            for _ in range(pack_sizes[j]):
                cols.append(cost[:, j : j + 1])
                col_to_tgt.append(j)
        expanded = np.hstack(cols)
        row_ind, col_ind, _ = hungarian_assignment(expanded)
        for r_i, c_j in zip(row_ind.tolist(), col_ind.tolist()):
            assignments[active_drones[r_i]] = targets[col_to_tgt[c_j]].target_id
    else:
        max_w = max(max(threat_w), 1e-6)
        cost_biased = cost.copy()
        for j in range(m_tgt):
            cost_biased[:, j] -= 1.2 * (threat_w[j] / max_w)
        row_ind, col_ind, _ = hungarian_assignment(cost_biased)
        for r_i, c_j in zip(row_ind.tolist(), col_ind.tolist()):
            assignments[active_drones[r_i]] = targets[c_j].target_id

    return assignments


class CooperativeGuidancePlanner:
    """多机战术协同轨迹、动态子群 WTA、障碍物绕飞与 3D APN 制导规划器。"""

    def __init__(
        self,
        profile: Optional[DynamicsProfile] = None,
        obstacle_map: Optional[ObstacleMap] = None,
    ) -> None:
        self.cfg = profile if profile is not None else DynamicsProfile()
        self.obstacle_map: ObstacleMap = obstacle_map if obstacle_map is not None else ObstacleMap.empty(self.cfg.drone_radius)
        self._prev_track_slots: Dict[str, int] = {}
        self._track_slots_by_target: Dict[str, Dict[str, int]] = {}
        self._orbit_phase0: float = 0.0
        self._last_phase_time: Optional[float] = None
        self._active_striker_id: Optional[str] = None
        self._striker_lock_until: float = 0.0
        self._striker_locks: Dict[str, Tuple[str, float]] = {}
        self._prev_wta: Dict[str, str] = {}
        self.last_wta_assignments: Dict[str, str] = {}
        self.re_tasking_events: int = 0
        self.neutralized_targets: Set[str] = set()

    def _clamp_z(self, z_val: float, is_attack: bool = False) -> float:
        z_floor = self.cfg.min_alt if not is_attack else max(3.05, self.cfg.min_alt - 0.15)
        return max(z_floor, min(self.cfg.max_alt, float(z_val)))

    def notify_target_killed(
        self,
        killed_track_ids: Sequence[str],
        external_target_id: str = "",
    ) -> None:
        """响应击毁事件：清除死目标主攻锁定与滞回绑定，触发即时重分配计数。"""
        all_killed = set(killed_track_ids)
        if external_target_id:
            all_killed.add(str(external_target_id))
        if not all_killed:
            return
        self.neutralized_targets.update(all_killed)

        re_tasked = False
        for k_id in all_killed:
            self._striker_locks.pop(k_id, None)
            self._track_slots_by_target.pop(k_id, None)
        for d_id, tid in list(self._prev_wta.items()):
            if tid in all_killed:
                self._prev_wta.pop(d_id, None)
                re_tasked = True
        if self._active_striker_id is not None and not all_killed.isdisjoint({"legacy_target", external_target_id}):
            self._active_striker_id = None
            self._striker_lock_until = 0.0
        if re_tasked:
            self.re_tasking_events += 1

    def _apply_obstacle_detour_and_ff(
        self,
        drone_id: str,
        p_i: np.ndarray,
        v_i: np.ndarray,
        goal: np.ndarray,
        v_cap: float,
        is_attack: bool = False,
        apply_z_clamp: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        g_safe = goal.copy()
        if apply_z_clamp:
            g_safe[2] = self._clamp_z(g_safe[2], is_attack=is_attack)

        detour_active = False
        wp = g_safe
        if self.obstacle_map.primitives:
            wp, u_tan, detour_active, _ = self.obstacle_map.compute_detour_waypoint(
                p_i, g_safe, max_alt=self.cfg.max_alt, drone_id=drone_id
            )
            if apply_z_clamp:
                wp[2] = self._clamp_z(wp[2], is_attack=is_attack)

        diff = wp - p_i
        dist = float(np.linalg.norm(diff))
        if detour_active and dist > 1e-4:
            spd = min(v_cap, max(2.8, dist * 1.8))
            v_cmd = (diff / dist) * spd
        else:
            v_cmd = _clamp_vec_norm(diff * 2.0, v_cap)
        a_cmd = _clamp_vec_norm((v_cmd - v_i) * 3.0, self.cfg.a_max)
        return wp, v_cmd, a_cmd, detour_active

    def _compute_apn_strike(
        self,
        drone_id: str,
        p_i: np.ndarray,
        v_i: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_acc: np.ndarray,
        predict_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]],
        imm_probs: Optional[Dict[str, float]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool]:
        """3D 增广比例导引 (APN / ZEM-ZEV) 末端打击制导律，输出一致的 pos/vel/acc 前馈。"""
        t_go_raw, reachable = solve_intercept_time(
            p_i, v_i, self.cfg.v_attack, self.cfg.a_max, predict_fn, t_max=5.0
        )

        probs = imm_probs if imm_probs is not None else {"CV": 0.2, "CTRV": 0.4, "JINK": 0.4}
        w_acc = (
            0.25 * float(probs.get("CV", 0.0))
            + 0.85 * float(probs.get("CTRV", 0.0))
            + 1.0 * float(probs.get("JINK", 0.0))
        )
        w_acc = max(0.25, min(1.0, w_acc))
        a_t_eff = _clamp_vec_norm(np.asarray(target_acc, dtype=float) * w_acc, 12.0)

        r = target_pos - p_i
        R = float(np.linalg.norm(r))
        r_hat = r / max(R, 1e-3)
        v_r = target_vel - v_i
        V_c = -float(np.dot(v_r, r_hat))

        if V_c > 1.5:
            t_go = max(0.08, min(4.5, min(float(t_go_raw), R / V_c)))
        else:
            t_go = max(0.08, min(4.5, float(t_go_raw)))

        v_i_spd = float(np.linalg.norm(v_i))
        v_i_hat = v_i / v_i_spd if v_i_spd > 0.3 else r_hat

        if V_c > 0.5:
            zem = r + v_r * t_go + 0.5 * a_t_eff * (t_go ** 2)
            zem_perp = zem - float(np.dot(zem, r_hat)) * r_hat
            a_lat = (self.cfg.apn_nav_gain / (max(t_go, 0.12) ** 2)) * zem_perp
            a_lat_c = _clamp_vec_norm(a_lat, self.cfg.a_max)
            a_rem = math.sqrt(max(0.0, self.cfg.a_max ** 2 - float(np.dot(a_lat_c, a_lat_c))))
            cos_r = float(np.dot(v_i_hat, r_hat))
            if R < 10.0 and cos_r < 0.70:
                v_long_des = self.cfg.v_attack * max(0.55, cos_r)
            else:
                v_long_des = self.cfg.v_attack
            a_long_mag = max(-a_rem, min(a_rem, 2.8 * (v_long_des - float(np.dot(v_i, r_hat)))))
            a_cmd = a_lat_c + a_long_mag * r_hat
        else:
            p_lead, _ = predict_fn(min(t_go, 1.2))
            d_lead = p_lead - p_i
            u_lead = d_lead / max(float(np.linalg.norm(d_lead)), 1e-3)
            cos_align = float(np.dot(v_i_hat, u_lead))
            if cos_align < 0.5:
                v_des_mag = max(4.2, self.cfg.v_attack * max(0.30, 0.55 + 0.45 * cos_align))
            else:
                v_des_mag = self.cfg.v_attack
            a_cmd = 3.2 * (v_des_mag * u_lead - v_i)

        a_cmd[2] = max(-5.0, min(5.0, float(a_cmd[2])))
        a_cmd = _clamp_vec_norm(a_cmd, self.cfg.a_max)

        v_cmd = _clamp_vec_norm(v_i + a_cmd * 0.35, self.cfg.v_attack)
        tau_pos = max(0.14, min(0.22, t_go))
        p_cmd = p_i + v_cmd * tau_pos + 0.5 * a_cmd * (tau_pos ** 2)

        if R < 2.5:
            p_hit, _ = predict_fn(min(t_go, 0.16))
            p_through = p_hit + 0.25 * r_hat
            blend = max(0.0, min(1.0, (2.5 - R) / 1.5))
            p_cmd = (1.0 - blend) * p_cmd + blend * p_through

        p_cmd[2] = self._clamp_z(float(p_cmd[2]), is_attack=True)

        # 若去往当前打击预测点、高速前瞻轨迹或当前惯性速度方向存在障碍威胁，切入平滑绕飞与制动距离屏障
        if self.obstacle_map.primitives:
            target_safe = target_pos.copy()
            target_safe[2] = self._clamp_z(float(target_safe[2]), is_attack=True)
            target_los_clear = self.obstacle_map.segment_clear_3d(p_i, target_safe)

            p_check = target_safe if R < 4.0 else predict_fn(min(t_go, 0.85))[0]
            p_check_safe = p_check.copy()
            p_check_safe[2] = self._clamp_z(float(p_check_safe[2]), is_attack=True)

            v_spd = float(np.linalg.norm(v_cmd))
            look_dist_cmd = min(max(R, 2.0), max(float(np.linalg.norm(p_cmd - p_i)), v_spd * 0.95))
            p_ahead_cmd = p_i + (v_cmd / max(v_spd, 1e-6)) * look_dist_cmd if v_spd > 1e-3 else p_cmd
            p_ahead_cmd[2] = self._clamp_z(float(p_ahead_cmd[2]), is_attack=True)

            stop_dist = (v_i_spd ** 2) / (2.0 * max(1.0, 0.80 * self.cfg.a_max)) + 1.2
            look_dist_vel = min(max(R, 2.0), max(3.5, stop_dist))
            p_ahead_vel = p_i + v_i_hat * look_dist_vel if v_i_spd > 0.5 else p_cmd
            p_ahead_vel[2] = self._clamp_z(float(p_ahead_vel[2]), is_attack=True)

            safe_pad = 0.20 if v_i_spd > 8.0 else 0.0
            check_blocked = not self.obstacle_map.segment_clear_3d(p_i, p_check_safe, extra_margin=safe_pad)
            cmd_blocked = not self.obstacle_map.segment_clear_3d(p_i, p_cmd, extra_margin=safe_pad)
            ahead_cmd_blocked = not self.obstacle_map.segment_clear_3d(p_i, p_ahead_cmd, extra_margin=safe_pad)
            ahead_vel_blocked = not self.obstacle_map.segment_clear_3d(p_i, p_ahead_vel, extra_margin=safe_pad)

            near_obs = _obstacle_surface_info(self.obstacle_map, p_i)
            z_cur = float(p_i[2])
            d_pad = 0.35
            a_brake = 0.48 * self.cfg.a_max
            d_range = max(6.5, (self.cfg.v_attack ** 2) / (2.0 * max(1.0, a_brake)) + 1.5)
            barrier_active = False
            for d_surf, n_out, z_lo, z_hi in near_obs:
                if z_cur > z_hi:
                    if d_surf < 0.2 and z_cur < z_hi + 0.35:
                        barrier_active = True
                        break
                    continue
                if z_cur < z_lo:
                    if d_surf < 0.2 and z_cur > z_lo - 0.35:
                        barrier_active = True
                        break
                    continue
                v_in = -float(np.dot(v_i, n_out))
                d_stop_n = (max(0.0, v_in) ** 2) / (2.0 * max(1.0, a_brake))
                if d_surf < d_stop_n + d_pad + 0.35:
                    barrier_active = True
                    break

            need_detour = check_blocked or (not target_los_clear and (cmd_blocked or ahead_cmd_blocked))
            if need_detour or cmd_blocked or ahead_cmd_blocked or ahead_vel_blocked or barrier_active:
                detour_active = False
                if need_detour:
                    if check_blocked and not self.obstacle_map.segment_clear_3d(p_i, p_check_safe):
                        detour_goal = p_check_safe
                    elif not target_los_clear:
                        detour_goal = target_safe
                    else:
                        detour_goal = p_check_safe

                    wp, u_tan, detour_active, _ = self.obstacle_map.compute_detour_waypoint(
                        p_i, detour_goal, max_alt=self.cfg.max_alt, drone_id=drone_id
                    )
                    if detour_active:
                        wp[2] = self._clamp_z(float(wp[2]), is_attack=True)
                        cos_turn = float(np.dot(v_i_hat, u_tan))
                        if cos_turn < 0.75:
                            v_det_mag = max(5.8, self.cfg.v_attack * max(0.45, cos_turn + 0.15))
                        else:
                            v_det_mag = self.cfg.v_attack
                        v_des = u_tan * v_det_mag
                    else:
                        v_des = v_cmd.copy()
                else:
                    v_des = v_cmd.copy()

                a_repulse = np.zeros(3, dtype=float)
                barrier_engaged = False
                priority_normals: List[Tuple[np.ndarray, float]] = []
                for d_surf, n_out, z_lo, z_hi in sorted(near_obs, key=lambda it: -it[0]):
                    if z_cur > z_hi:
                        if d_surf < 0.2 and z_cur < z_hi + 0.35 and float(v_des[2]) < -1e-9:
                            v_des[2] = max(0.0, float(v_des[2]))
                            barrier_engaged = True
                        continue
                    if z_cur < z_lo:
                        if d_surf < 0.2 and z_cur > z_lo - 0.35 and float(v_des[2]) > 1e-9:
                            v_des[2] = min(0.0, float(v_des[2]))
                            barrier_engaged = True
                        continue
                    if d_surf > d_range:
                        continue
                    v_in_max = math.sqrt(2.0 * a_brake * max(0.0, d_surf - d_pad))
                    v_n_des = float(np.dot(v_des, n_out))
                    if v_n_des < -v_in_max:
                        v_n_target = -v_in_max if d_surf >= d_pad else min(3.2, (d_pad - d_surf) * 4.5)
                        v_des = v_des + (v_n_target - v_n_des) * n_out
                        barrier_engaged = True
                    elif d_surf < d_pad and v_n_des < (d_pad - d_surf) * 3.0:
                        v_n_target = min(3.0, (d_pad - d_surf) * 3.0)
                        v_des = v_des + (v_n_target - v_n_des) * n_out
                        barrier_engaged = True

                    v_n_cur = float(np.dot(v_i, n_out))
                    if v_n_cur < -v_in_max:
                        excess = (-v_n_cur) - v_in_max
                        a_rep = a_brake + excess * 6.0
                        a_repulse = a_repulse + a_rep * n_out
                        priority_normals.append((n_out, min(self.cfg.a_max, a_rep)))
                        barrier_engaged = True
                    elif d_surf < d_pad:
                        v_exit_req = min(3.0, (d_pad - d_surf) * 3.5)
                        a_rep = max(0.0, (v_exit_req - v_n_cur) * 4.5) + (d_pad - d_surf) * 7.5
                        if a_rep > 1e-6:
                            a_repulse = a_repulse + a_rep * n_out
                            priority_normals.append((n_out, min(self.cfg.a_max, a_rep)))
                        barrier_engaged = True

                v_des = _clamp_vec_norm(v_des, self.cfg.v_attack)
                if detour_active or barrier_engaged:
                    a_raw = (v_des - v_i) * 3.5 + a_repulse
                    a_cmd = _prioritize_normal_braking(a_raw, priority_normals, self.cfg.a_max)
                    v_cmd_step = _clamp_vec_norm(v_i + a_cmd * 0.30, self.cfg.v_attack)
                    v_cmd = _clamp_vec_norm(0.5 * v_cmd_step + 0.5 * v_des, self.cfg.v_attack)
                    tau_det = 0.20
                    p_cmd = p_i + v_cmd * tau_det + 0.5 * a_cmd * (tau_det ** 2)
                    p_cmd[2] = self._clamp_z(float(p_cmd[2]), is_attack=True)
                    if not self.obstacle_map.point_clear_3d(p_cmd):
                        p_cmd = self.obstacle_map.project_outside(p_cmd, margin_pad=0.35, max_alt=self.cfg.max_alt)
                        p_cmd[2] = self._clamp_z(float(p_cmd[2]), is_attack=True)

        return p_cmd, v_cmd, a_cmd, float(t_go), reachable

    def _plan_single_pack(
        self,
        now_sec: float,
        pack_ids: Sequence[str],
        drone_modes: Dict[str, str],
        drone_states: Dict[str, Tuple[np.ndarray, np.ndarray]],
        target_id: str,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_acc: np.ndarray,
        predict_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]],
        imm_probs: Optional[Dict[str, float]] = None,
        pred_path: Optional[List[List[float]]] = None,
        use_legacy_slot_cache: bool = False,
    ) -> Dict[str, Dict[str, object]]:
        """对分配给同一目标的一组己方无人机执行广义 K 机 TRACK / INTERCEPT / ATTACK 规划。"""
        results: Dict[str, Dict[str, object]] = {}
        probs_out = dict(imm_probs) if imm_probs is not None else {"CV": 0.34, "CTRV": 0.33, "JINK": 0.33}
        path_out = list(pred_path) if pred_path is not None else []

        def _attach_common(d_id: str, payload: Dict[str, object]) -> None:
            payload["target_id"] = target_id
            payload["focus_x"] = float(target_pos[0])
            payload["focus_y"] = float(target_pos[1])
            payload["focus_z"] = float(target_pos[2])
            payload["imm_probs"] = probs_out
            payload["pred_path"] = path_out
            results[d_id] = payload

        # 1. TRACK：广义 K 机匈牙利匀相合围 + 多层高度隔离
        track_ids = [d for d in pack_ids if drone_modes.get(d) == "TRACK" and d in drone_states]
        if track_ids:
            p_lead, _ = predict_fn(0.40)
            drone_pos_map = {d: drone_states[d][0] for d in track_ids}
            slot_cache = (
                self._prev_track_slots
                if use_legacy_slot_cache
                else self._track_slots_by_target.setdefault(target_id, {})
            )
            assigned = assign_encirclement_slots(
                track_ids,
                drone_pos_map,
                (float(p_lead[0]), float(p_lead[1])),
                radius=self.cfg.track_radius,
                base_phase_rad=self._orbit_phase0,
                prev_assignment=slot_cache,
            )
            k_cnt = len(track_ids)
            for d_id, (sx, sy, slot_idx) in assigned.items():
                slot_cache[d_id] = slot_idx
                if k_cnt == 3:
                    layer_dz = (0.0, 0.85, -0.65)[slot_idx % 3]
                elif k_cnt == 1:
                    layer_dz = 0.0
                else:
                    n_layers = min(k_cnt, 5)
                    layer_dz = 0.85 * ((slot_idx % n_layers) - 0.5 * (n_layers - 1))
                tz = self._clamp_z(float(p_lead[2]) + self.cfg.track_alt_offset + layer_dz, is_attack=False)
                p_i, v_i = drone_states[d_id]
                wp, v_cmd, a_cmd, _ = self._apply_obstacle_detour_and_ff(
                    d_id, p_i, v_i, np.array([sx, sy, tz], dtype=float), self.cfg.v_intercept, is_attack=False
                )
                _attach_common(
                    d_id,
                    {
                        "x": float(wp[0]),
                        "y": float(wp[1]),
                        "z": float(wp[2]),
                        "vx": float(v_cmd[0]),
                        "vy": float(v_cmd[1]),
                        "vz": float(v_cmd[2]),
                        "ax": float(a_cmd[0]),
                        "ay": float(a_cmd[1]),
                        "az": float(a_cmd[2]),
                        "action": "TRACK",
                        "t_go": 0.40,
                        "reachable": True,
                    },
                )

        # 2. INTERCEPT：广义 K 机无交叉多翼钳形封堵（1-Wasserstein 欧氏距离 + 共享预测时域匈牙利指派）
        intercept_ids = [d for d in pack_ids if drone_modes.get(d) == "INTERCEPT" and d in drone_states]
        if intercept_ids:
            k_int = len(intercept_ids)
            tgo_int_map: Dict[str, Tuple[float, bool]] = {}
            for d_id in intercept_ids:
                p_i, v_i = drone_states[d_id]
                t_go_raw, reachable = solve_intercept_time(
                    p_i, v_i, self.cfg.v_intercept, self.cfg.a_max, predict_fn, t_max=2.5
                )
                tgo_int_map[d_id] = (min(float(t_go_raw), 0.85), reachable)

            t_ref = max(0.25, min(0.85, float(np.median([v[0] for v in tgo_int_map.values()]))))
            p_ref_int, v_ref_int = predict_fn(t_ref)
            vh_r = math.hypot(float(v_ref_int[0]), float(v_ref_int[1]))
            if vh_r > 0.4:
                ux_r, uy_r = float(v_ref_int[0]) / vh_r, float(v_ref_int[1]) / vh_r
            else:
                d_ref = p_ref_int[0:2] - drone_states[intercept_ids[0]][0][0:2]
                dn_r = max(float(np.linalg.norm(d_ref)), 1e-6)
                ux_r, uy_r = float(d_ref[0]) / dn_r, float(d_ref[1]) / dn_r
            nx_r, ny_r = -uy_r, ux_r

            # 构造 K_int 个广义楔形/钳形槽位
            flank_offsets: List[Tuple[float, float]] = [(0.8, 0.0)]
            for k in range(1, k_int):
                m_tier = (k - 1) // 2
                sgn = 1.0 if (k % 2 == 1) else -1.0
                lat = 5.2 - 0.6 * min(m_tier, 2)
                axial = 0.5 - 2.4 * float(m_tier)
                flank_offsets.append((axial, sgn * lat))

            flank_pts = [
                (
                    float(p_ref_int[0]) + ax * ux_r + lt * nx_r,
                    float(p_ref_int[1]) + ax * uy_r + lt * ny_r,
                )
                for (ax, lt) in flank_offsets
            ]
            cost_int = np.zeros((k_int, k_int), dtype=float)
            for i_d, d_id in enumerate(intercept_ids):
                px_d, py_d = float(drone_states[d_id][0][0]), float(drone_states[d_id][0][1])
                for j_s, (fx, fy) in enumerate(flank_pts):
                    cost_int[i_d, j_s] = math.hypot(px_d - fx, py_d - fy)
            row_i, col_i, _ = hungarian_assignment(cost_int)
            slot_for_drone = {intercept_ids[r]: c for r, c in zip(row_i.tolist(), col_i.tolist())}

            n_layers = min(k_int, 5)
            for d_id in intercept_ids:
                slot_idx = slot_for_drone.get(d_id, 0)
                p_i, v_i = drone_states[d_id]
                t_go, reachable = tgo_int_map[d_id]
                ix, iy = flank_pts[slot_idx]

                dist_now = float(np.linalg.norm(target_pos - p_i))
                if dist_now < 2.4:
                    ix = 0.55 * ix + 0.45 * float(p_i[0])
                    iy = 0.55 * iy + 0.45 * float(p_i[1])

                iz = self._clamp_z(float(p_ref_int[2]) + 1.8 + 0.9 * (slot_idx % n_layers), is_attack=False)
                wp, v_cmd, a_cmd, _ = self._apply_obstacle_detour_and_ff(
                    d_id, p_i, v_i, np.array([ix, iy, iz], dtype=float), self.cfg.v_intercept, is_attack=False
                )
                _attach_common(
                    d_id,
                    {
                        "x": float(wp[0]),
                        "y": float(wp[1]),
                        "z": float(wp[2]),
                        "vx": float(v_cmd[0]),
                        "vy": float(v_cmd[1]),
                        "vz": float(v_cmd[2]),
                        "ax": float(a_cmd[0]),
                        "ay": float(a_cmd[1]),
                        "az": float(a_cmd[2]),
                        "action": "INTERCEPT",
                        "t_go": round(float(t_go), 3),
                        "reachable": reachable,
                    },
                )

        # 3. ATTACK：主攻手 Min-t_go 推举 (4s 锁定) + 3D APN/ZEM-ZEV 末端打击 + K_j-1 广义环绕掩护
        attack_ids = [d for d in pack_ids if drone_modes.get(d) == "ATTACK" and d in drone_states]
        if attack_ids:
            tgo_map: Dict[str, Tuple[float, bool]] = {}
            for d_id in attack_ids:
                p_i, v_i = drone_states[d_id]
                tgo_map[d_id] = solve_intercept_time(
                    p_i, v_i, self.cfg.v_attack, self.cfg.a_max, predict_fn, t_max=5.0
                )

            if use_legacy_slot_cache:
                if self._active_striker_id in attack_ids and float(now_sec) < self._striker_lock_until:
                    striker_id = self._active_striker_id
                else:
                    striker_id = min(attack_ids, key=lambda d: tgo_map[d][0])
                    self._active_striker_id = striker_id
                    self._striker_lock_until = float(now_sec) + 4.0
            else:
                lock_entry = self._striker_locks.get(target_id)
                if lock_entry is not None and lock_entry[0] in attack_ids and float(now_sec) < lock_entry[1]:
                    striker_id = lock_entry[0]
                else:
                    striker_id = min(attack_ids, key=lambda d: tgo_map[d][0])
                    self._striker_locks[target_id] = (striker_id, float(now_sec) + 4.0)

            p_s, v_s = drone_states[striker_id]
            p_cmd, v_cmd, a_cmd, t_eff, reach_s = self._compute_apn_strike(
                striker_id,
                p_s,
                v_s,
                target_pos,
                target_vel,
                target_acc,
                predict_fn,
                imm_probs=probs_out,
            )
            _attach_common(
                striker_id,
                {
                    "x": float(p_cmd[0]),
                    "y": float(p_cmd[1]),
                    "z": float(p_cmd[2]),
                    "vx": float(v_cmd[0]),
                    "vy": float(v_cmd[1]),
                    "vz": float(v_cmd[2]),
                    "ax": float(a_cmd[0]),
                    "ay": float(a_cmd[1]),
                    "az": float(a_cmd[2]),
                    "action": "ATTACK",
                    "t_go": round(float(t_eff), 3),
                    "reachable": reach_s,
                },
            )

            wing_ids = [d for d in attack_ids if d != striker_id]
            if wing_ids:
                p_cov, _ = predict_fn(0.5)
                k_w = len(wing_ids)
                ring_pts = [
                    (
                        float(p_cov[0]) + 6.2 * math.cos(self._orbit_phase0 + (2.0 * math.pi * idx) / k_w),
                        float(p_cov[1]) + 6.2 * math.sin(self._orbit_phase0 + (2.0 * math.pi * idx) / k_w),
                    )
                    for idx in range(k_w)
                ]
                cost_w = np.zeros((k_w, k_w), dtype=float)
                for i_w, d_id in enumerate(wing_ids):
                    px_w, py_w = float(drone_states[d_id][0][0]), float(drone_states[d_id][0][1])
                    for j_s, (rx, ry) in enumerate(ring_pts):
                        cost_w[i_w, j_s] = math.hypot(px_w - rx, py_w - ry)
                row_w, col_w, _ = hungarian_assignment(cost_w)
                n_w_layers = min(k_w, 4)
                for r_w, c_w in zip(row_w.tolist(), col_w.tolist()):
                    d_id = wing_ids[r_w]
                    wx, wy = ring_pts[c_w]
                    wz = self._clamp_z(float(target_pos[2]) + 2.8 + 0.85 * (c_w % n_w_layers), is_attack=False)
                    p_w, v_w = drone_states[d_id]
                    wp, v_cmd_w, a_cmd_w, _ = self._apply_obstacle_detour_and_ff(
                        d_id, p_w, v_w, np.array([wx, wy, wz], dtype=float), self.cfg.v_intercept, is_attack=False
                    )
                    _attach_common(
                        d_id,
                        {
                            "x": float(wp[0]),
                            "y": float(wp[1]),
                            "z": float(wp[2]),
                            "vx": float(v_cmd_w[0]),
                            "vy": float(v_cmd_w[1]),
                            "vz": float(v_cmd_w[2]),
                            "ax": float(a_cmd_w[0]),
                            "ay": float(a_cmd_w[1]),
                            "az": float(a_cmd_w[2]),
                            "action": "ATTACK",
                            "t_go": round(float(tgo_map[d_id][0]), 3),
                            "reachable": tgo_map[d_id][1],
                        },
                    )

        return results

    def _advance_orbit_phase(self, now_sec: float) -> None:
        if self._last_phase_time is None:
            self._last_phase_time = float(now_sec)
        dt_phase = max(0.0, min(0.25, float(now_sec) - self._last_phase_time))
        self._last_phase_time = float(now_sec)
        self._orbit_phase0 = (self._orbit_phase0 + 0.22 * dt_phase) % (2.0 * math.pi)

    def _plan_patrol_and_return(
        self,
        now_sec: float,
        swarm_ids: Sequence[str],
        drone_modes: Dict[str, str],
        drone_states: Dict[str, Tuple[np.ndarray, np.ndarray]],
        include_return: bool = False,
    ) -> Dict[str, Dict[str, object]]:
        results: Dict[str, Dict[str, object]] = {}
        patrol_ids = [d for d in swarm_ids if drone_modes.get(d) == "RESUME_PATROL" and d in drone_states]
        for idx, d_id in enumerate(patrol_ids):
            slot_rank = list(swarm_ids).index(d_id) if d_id in swarm_ids else idx
            px, py, pz = self._compute_patrol_waypoint(now_sec, slot_rank, len(swarm_ids))
            p_i, v_i = drone_states[d_id]
            wp, v_cmd, a_cmd, _ = self._apply_obstacle_detour_and_ff(
                d_id, p_i, v_i, np.array([px, py, pz], dtype=float), self.cfg.v_cruise, is_attack=False
            )
            results[d_id] = {
                "x": float(wp[0]),
                "y": float(wp[1]),
                "z": float(wp[2]),
                "vx": float(v_cmd[0]),
                "vy": float(v_cmd[1]),
                "vz": float(v_cmd[2]),
                "ax": float(a_cmd[0]),
                "ay": float(a_cmd[1]),
                "az": float(a_cmd[2]),
                "action": "TRACK",
                "t_go": 0.0,
                "reachable": True,
                "target_id": "",
            }

        if include_return:
            return_ids = [d for d in swarm_ids if drone_modes.get(d) == "RETURN" and d in drone_states]
            for idx, d_id in enumerate(return_ids):
                slot_rank = list(swarm_ids).index(d_id) if d_id in swarm_ids else idx
                hx, hy, hz = self.obstacle_map.get_helipad(d_id, index_fallback=slot_rank)
                p_i, v_i = drone_states[d_id]
                dist_xy = math.hypot(hx - float(p_i[0]), hy - float(p_i[1]))
                if dist_xy > 1.5:
                    goal_ret = np.array([hx, hy, max(self.cfg.min_alt + 0.8, hz + 3.8)], dtype=float)
                    wp, v_cmd, a_cmd, _ = self._apply_obstacle_detour_and_ff(
                        d_id, p_i, v_i, goal_ret, self.cfg.v_cruise, is_attack=False
                    )
                else:
                    wp = np.array([hx, hy, max(0.0, hz)], dtype=float)
                    v_cmd = np.array([0.0, 0.0, max(-1.5, (hz - float(p_i[2])) * 1.2)], dtype=float)
                    a_cmd = np.zeros(3, dtype=float)
                results[d_id] = {
                    "x": float(wp[0]),
                    "y": float(wp[1]),
                    "z": float(wp[2]),
                    "vx": float(v_cmd[0]),
                    "vy": float(v_cmd[1]),
                    "vz": float(v_cmd[2]),
                    "ax": float(a_cmd[0]),
                    "ay": float(a_cmd[1]),
                    "az": float(a_cmd[2]),
                    "action": "RETURN",
                    "t_go": 0.0,
                    "reachable": True,
                    "target_id": "",
                }
        return results

    def compute_swarm_targets(
        self,
        now_sec: float,
        swarm_ids: Sequence[str],
        drone_modes: Dict[str, str],
        drone_states: Dict[str, Tuple[np.ndarray, np.ndarray]],
        target_valid: bool,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
        target_acc: np.ndarray,
        predict_fn: Callable[[float], Tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Dict[str, object]]:
        """
        向后兼容的单目标集群制导规划入口（同时包含速度/加速度前馈与 3D APN / 避障支持）。
        """
        self._advance_orbit_phase(now_sec)
        results = self._plan_patrol_and_return(now_sec, swarm_ids, drone_modes, drone_states, include_return=False)
        if not target_valid:
            return results

        pack_res = self._plan_single_pack(
            now_sec=now_sec,
            pack_ids=swarm_ids,
            drone_modes=drone_modes,
            drone_states=drone_states,
            target_id="legacy_target",
            target_pos=np.asarray(target_pos, dtype=float),
            target_vel=np.asarray(target_vel, dtype=float),
            target_acc=np.asarray(target_acc, dtype=float),
            predict_fn=predict_fn,
            use_legacy_slot_cache=True,
        )
        results.update(pack_res)
        return results

    def compute_multi_target_swarm_targets(
        self,
        now_sec: float,
        swarm_ids: Sequence[str],
        drone_modes: Dict[str, str],
        drone_states: Dict[str, Tuple[np.ndarray, np.ndarray]],
        targets: Sequence[TargetTrackState],
        drone_batteries: Optional[Dict[str, float]] = None,
        defense_center: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, object]]:
        """多目标动态子群 WTA 分配 + 广义 K_j 机协同合围/钳形封堵 + 3D APN 制导 + 障碍绕飞主入口。"""
        self._advance_orbit_phase(now_sec)
        results = self._plan_patrol_and_return(now_sec, swarm_ids, drone_modes, drone_states, include_return=True)

        engage_modes = ("TRACK", "INTERCEPT", "ATTACK")
        active_ids = [d for d in swarm_ids if drone_modes.get(d) in engage_modes and d in drone_states]
        if not active_ids:
            self.last_wta_assignments = {}
            return results

        active_targets = [t for t in targets if t.target_id not in self.neutralized_targets]
        if not active_targets:
            # 全部目标已被击毁/暂未发现目标：保持安全避障悬停/巡逻盘旋
            self.last_wta_assignments = {}
            for idx, d_id in enumerate(active_ids):
                slot_rank = list(swarm_ids).index(d_id) if d_id in swarm_ids else idx
                px, py, pz = self._compute_patrol_waypoint(now_sec, slot_rank, len(swarm_ids))
                p_i, v_i = drone_states[d_id]
                wp, v_cmd, a_cmd, _ = self._apply_obstacle_detour_and_ff(
                    d_id, p_i, v_i, np.array([px, py, pz], dtype=float), self.cfg.v_cruise, is_attack=False
                )
                results[d_id] = {
                    "x": float(wp[0]),
                    "y": float(wp[1]),
                    "z": float(wp[2]),
                    "vx": float(v_cmd[0]),
                    "vy": float(v_cmd[1]),
                    "vz": float(v_cmd[2]),
                    "ax": float(a_cmd[0]),
                    "ay": float(a_cmd[1]),
                    "az": float(a_cmd[2]),
                    "action": "TRACK",
                    "t_go": 0.0,
                    "reachable": True,
                    "target_id": "",
                }
            return results

        wta_map = compute_wta_assignments(
            drone_ids=active_ids,
            drone_states=drone_states,
            targets=active_targets,
            profile=self.cfg,
            defense_center=defense_center,
            drone_batteries=drone_batteries,
            prev_assignments=self._prev_wta,
            obstacle_map=self.obstacle_map if self.obstacle_map.primitives else None,
        )
        for d_id, tid in wta_map.items():
            prev_tid = self._prev_wta.get(d_id)
            if prev_tid is not None and prev_tid != tid:
                self.re_tasking_events += 1
        self._prev_wta = dict(wta_map)
        self.last_wta_assignments = dict(wta_map)

        target_by_id = {t.target_id: t for t in active_targets}
        packs: Dict[str, List[str]] = {}
        for d_id, tid in wta_map.items():
            packs.setdefault(tid, []).append(d_id)

        for tid, pack_drones in packs.items():
            tgt = target_by_id[tid]
            pack_res = self._plan_single_pack(
                now_sec=now_sec,
                pack_ids=pack_drones,
                drone_modes=drone_modes,
                drone_states=drone_states,
                target_id=tgt.target_id,
                target_pos=tgt.pos,
                target_vel=tgt.vel,
                target_acc=tgt.acc,
                predict_fn=tgt.predict_fn,
                imm_probs=tgt.imm_probs,
                pred_path=tgt.pred_path,
                use_legacy_slot_cache=False,
            )
            results.update(pack_res)

        # 若 N < M 不会出现未分配的 active_ids；若防御性出现未分配，令其安全盘旋
        for idx, d_id in enumerate(active_ids):
            if d_id not in results:
                slot_rank = list(swarm_ids).index(d_id) if d_id in swarm_ids else idx
                px, py, pz = self._compute_patrol_waypoint(now_sec, slot_rank, len(swarm_ids))
                p_i, v_i = drone_states[d_id]
                wp, v_cmd, a_cmd, _ = self._apply_obstacle_detour_and_ff(
                    d_id, p_i, v_i, np.array([px, py, pz], dtype=float), self.cfg.v_cruise, is_attack=False
                )
                results[d_id] = {
                    "x": float(wp[0]),
                    "y": float(wp[1]),
                    "z": float(wp[2]),
                    "vx": float(v_cmd[0]),
                    "vy": float(v_cmd[1]),
                    "vz": float(v_cmd[2]),
                    "ax": float(a_cmd[0]),
                    "ay": float(a_cmd[1]),
                    "az": float(a_cmd[2]),
                    "action": "TRACK",
                    "t_go": 0.0,
                    "reachable": True,
                    "target_id": "",
                }

        return results

    def _compute_patrol_waypoint(self, now_sec: float, slot_rank: int, total_drones: int) -> Tuple[float, float, float]:
        """生成与多机起降坪一一对应的平行无交叉横向扫描巡逻航线（适配任意 N 架规模）。"""
        omega = 0.22
        hx, hy, _ = self.obstacle_map.get_helipad(f"drone_{slot_rank + 1}", index_fallback=slot_rank)
        base_y = hy if self.obstacle_map.helipads else float(slot_rank) * 3.5
        px = hx + 16.0 + 9.0 * math.sin(omega * float(now_sec))
        py = base_y + 1.2 * math.sin(omega * float(now_sec))
        layer = slot_rank % max(1, min(total_drones, 4))
        pz = self._clamp_z(9.0 + 0.8 * layer, is_attack=False)
        return float(px), float(py), float(pz)
