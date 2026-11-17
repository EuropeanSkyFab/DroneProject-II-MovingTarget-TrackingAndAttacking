# -*- coding: utf-8 -*-
"""
guidance.py —— 多无人机协同最优拦截、三维增广比例导引 (3D APN) 与动态合围制导律

全部输入与输出严格采用全局 ENU 坐标系（米、米/秒）。
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


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
        # 初始速度惯性滑行等效时间常数 tau_lag = min(tau * 0.4, v_cap / (2*a_max))
        tau_lag = min(tau * 0.35, v_cap / (2.2 * amax))
        eff_origin = p_i + v_i * tau_lag
        req_dist = float(np.linalg.norm(p_t - eff_origin))
        # 起步加速损耗修正
        reach_dist = v_cap * max(0.0, tau - 0.5 * tau_lag)
        return req_dist - reach_dist

    # 粗网格扫描寻找首个符号过零区间（保证捕获最小正根而非远端伪根）
    n_grid = 32
    taus = np.linspace(0.05, float(t_max), n_grid)
    best_tau = float(taus[0])
    best_gap = float("inf")
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


def assign_encirclement_slots(
    drone_ids: Sequence[str],
    drone_positions: Dict[str, np.ndarray],
    center_xy: Tuple[float, float],
    radius: float,
    base_phase_rad: float,
    prev_assignment: Optional[Dict[str, int]] = None,
) -> Dict[str, Tuple[float, float, int]]:
    """
    为 K 架处于 TRACK 模式的无人机分配均匀相位环绕槽位（如三机 120° 锁相合围），
    通过精确全排列最小化总水平机动代价并附带滞回项防止槽位跳变。
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

    best_perm: Tuple[int, ...] = tuple(range(k_cnt))
    best_cost = float("inf")

    for perm in itertools.permutations(range(k_cnt)):
        cost = 0.0
        for d_idx, slot_idx in enumerate(perm):
            d_id = active[d_idx]
            pos = drone_positions[d_id]
            sx, sy = slots[slot_idx]
            cost += (float(pos[0]) - sx) ** 2 + (float(pos[1]) - sy) ** 2
            # 滞回偏好：维持上一拍槽位可获 2.5 m^2 等效奖励，杜绝临界对称线高频互换
            if prev_assignment is not None and prev_assignment.get(d_id) == slot_idx:
                cost -= 2.5
        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    out: Dict[str, Tuple[float, float, int]] = {}
    for d_idx, slot_idx in enumerate(best_perm):
        sx, sy = slots[slot_idx]
        out[active[d_idx]] = (sx, sy, slot_idx)
    return out


class CooperativeGuidancePlanner:
    """多机战术协同轨迹与制导规划器（支持 TRACK / INTERCEPT / ATTACK / RESUME_PATROL）。"""

    def __init__(self, profile: Optional[DynamicsProfile] = None) -> None:
        self.cfg = profile if profile is not None else DynamicsProfile()
        self._prev_track_slots: Dict[str, int] = {}
        self._orbit_phase0: float = 0.0
        self._last_phase_time: Optional[float] = None
        self._active_striker_id: Optional[str] = None
        self._striker_lock_until: float = 0.0

    def _clamp_z(self, z_val: float, is_attack: bool = False) -> float:
        z_floor = self.cfg.min_alt if not is_attack else max(3.05, self.cfg.min_alt - 0.15)
        return max(z_floor, min(self.cfg.max_alt, float(z_val)))

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
        为集群中所有需要规划输出的无人机计算当前拍的最优期望三维目标点（ENU）。

        返回字典：drone_id -> {"x": float, "y": float, "z": float, "action": str, "t_go": float, "reachable": bool}
        """
        if self._last_phase_time is None:
            self._last_phase_time = float(now_sec)
        dt_phase = max(0.0, min(0.25, float(now_sec) - self._last_phase_time))
        self._last_phase_time = float(now_sec)
        # 协同合围环绕角速度 0.22 rad/s (~12.6 deg/s)，保证动态全方位立体视角覆盖
        self._orbit_phase0 = (self._orbit_phase0 + 0.22 * dt_phase) % (2.0 * math.pi)

        results: Dict[str, Dict[str, object]] = {}

        # 1. 处理 RESUME_PATROL（即使尚未发现目标也能协同巡逻搜索）
        patrol_ids = [d for d in swarm_ids if drone_modes.get(d) == "RESUME_PATROL" and d in drone_states]
        for idx, d_id in enumerate(patrol_ids):
            slot_rank = swarm_ids.index(d_id) if d_id in swarm_ids else idx
            px, py, pz = self._compute_patrol_waypoint(now_sec, slot_rank, len(swarm_ids))
            results[d_id] = {
                "x": px,
                "y": py,
                "z": pz,
                "action": "TRACK",
                "t_go": 0.0,
                "reachable": True,
            }

        if not target_valid:
            return results

        # 2. 处理 TRACK 模式：120° 锁相合围 + 35° 前斜相机最佳俯视高度保持
        track_ids = [d for d in swarm_ids if drone_modes.get(d) == "TRACK" and d in drone_states]
        if track_ids:
            p_lead, _ = predict_fn(0.40)
            drone_pos_map = {d: drone_states[d][0] for d in track_ids}
            assigned = assign_encirclement_slots(
                track_ids,
                drone_pos_map,
                (float(p_lead[0]), float(p_lead[1])),
                radius=self.cfg.track_radius,
                base_phase_rad=self._orbit_phase0,
                prev_assignment=self._prev_track_slots,
            )
            for d_id, (sx, sy, slot_idx) in assigned.items():
                self._prev_track_slots[d_id] = slot_idx
                # 高度分层 (+0.0 / +0.85 / -0.65 m) 兼顾完美视场与充足防撞裕度
                layer_dz = (0.0, 0.85, -0.65)[slot_idx % 3]
                tz = self._clamp_z(float(p_lead[2]) + self.cfg.track_alt_offset + layer_dz, is_attack=False)
                results[d_id] = {
                    "x": float(sx),
                    "y": float(sy),
                    "z": float(tz),
                    "action": "TRACK",
                    "t_go": 0.40,
                    "reachable": True,
                }

        # 3. 处理 INTERCEPT 模式：Apollonius 最早可达拦截 + 无交叉最优左右双翼钳形封堵
        intercept_ids = [d for d in swarm_ids if drone_modes.get(d) == "INTERCEPT" and d in drone_states]
        wing_flank_sign: Dict[str, float] = {}
        if len(intercept_ids) >= 3:
            # 为 2 架侧翼机动态匹配不交叉的左(+1)/右(-1)侧翼槽位，消除目标掉头时双机对穿
            p_ref_int, v_ref_int = predict_fn(0.8)
            vh_r = math.hypot(float(v_ref_int[0]), float(v_ref_int[1]))
            ux_r, uy_r = (float(v_ref_int[0]) / vh_r, float(v_ref_int[1]) / vh_r) if vh_r > 0.4 else (1.0, 0.0)
            nx_r, ny_r = -uy_r, ux_r
            flank_l = np.array([float(p_ref_int[0]) + 5.2 * nx_r, float(p_ref_int[1]) + 5.2 * ny_r])
            flank_r = np.array([float(p_ref_int[0]) - 5.2 * nx_r, float(p_ref_int[1]) - 5.2 * ny_r])
            w1, w2 = intercept_ids[1], intercept_ids[2]
            p_w1, p_w2 = drone_states[w1][0][0:2], drone_states[w2][0][0:2]
            c_direct = float(np.sum((p_w1 - flank_l) ** 2) + np.sum((p_w2 - flank_r) ** 2))
            c_swap = float(np.sum((p_w1 - flank_r) ** 2) + np.sum((p_w2 - flank_l) ** 2))
            if c_direct <= c_swap:
                wing_flank_sign[w1], wing_flank_sign[w2] = 1.0, -1.0
            else:
                wing_flank_sign[w1], wing_flank_sign[w2] = -1.0, 1.0

        for rank, d_id in enumerate(intercept_ids):
            p_i, v_i = drone_states[d_id]
            t_go_raw, reachable = solve_intercept_time(
                p_i, v_i, self.cfg.v_intercept, self.cfg.a_max, predict_fn, t_max=2.5
            )
            t_go = min(float(t_go_raw), 0.85)
            p_int, v_int = predict_fn(t_go)
            vh = math.hypot(float(v_int[0]), float(v_int[1]))
            if vh > 0.5:
                ux, uy = float(v_int[0]) / vh, float(v_int[1]) / vh
            else:
                diff_xy = p_int[0:2] - p_i[0:2]
                dn = max(float(np.linalg.norm(diff_xy)), 1e-6)
                ux, uy = float(diff_xy[0]) / dn, float(diff_xy[1]) / dn
            nx, ny = -uy, ux

            if not reachable:
                ix = float(p_int[0]) - 2.2 * ux
                iy = float(p_int[1]) - 2.2 * uy
            else:
                if rank == 0:
                    ix = float(p_int[0]) + 0.8 * ux
                    iy = float(p_int[1]) + 0.8 * uy
                else:
                    sgn = wing_flank_sign.get(d_id, 1.0 if rank % 2 == 1 else -1.0)
                    ix = float(p_int[0]) + 0.5 * ux + sgn * 5.2 * nx
                    iy = float(p_int[1]) + 0.5 * uy + sgn * 5.2 * ny

            dist_now = float(np.linalg.norm(target_pos - p_i))
            if dist_now < 2.4:
                ix = 0.55 * ix + 0.45 * float(p_i[0])
                iy = 0.55 * iy + 0.45 * float(p_i[1])

            iz = self._clamp_z(float(p_int[2]) + 1.8 + 1.3 * (rank % 3), is_attack=False)
            results[d_id] = {
                "x": ix,
                "y": iy,
                "z": iz,
                "action": "INTERCEPT",
                "t_go": round(float(t_go), 3),
                "reachable": reachable,
            }

        # 4. 处理 ATTACK 模式：动态最短可达时间 (Min-t_go) 主攻手优选 + 远距 Apollonius 切弦 / 末端零滞后命中 + 双机高空立体照射掩护
        attack_ids = [d for d in swarm_ids if drone_modes.get(d) == "ATTACK" and d in drone_states]
        if attack_ids:
            tgo_map: Dict[str, Tuple[float, bool]] = {}
            for d_id in attack_ids:
                p_i, v_i = drone_states[d_id]
                tgo_map[d_id] = solve_intercept_time(
                    p_i, v_i, self.cfg.v_attack, self.cfg.a_max, predict_fn, t_max=5.0
                )

            # 带承诺锁定的最优主攻手动态推举：优先选迎头最短 t_go 无人机执行致命一击，避免尾追绕弯
            if (
                self._active_striker_id in attack_ids
                and float(now_sec) < self._striker_lock_until
            ):
                striker_id = self._active_striker_id
            else:
                striker_id = min(attack_ids, key=lambda d: tgo_map[d][0])
                self._active_striker_id = striker_id
                self._striker_lock_until = float(now_sec) + 4.5

            wing_rank = 0
            for d_id in attack_ids:
                p_i, v_i = drone_states[d_id]
                t_go_raw, reachable = tgo_map[d_id]
                r_vec = target_pos - p_i
                r_norm = float(np.linalg.norm(r_vec))

                if d_id == striker_id:
                    # 受限超前窗口 [0.16s, 0.72s]：杜绝靶机拐点大跨度反跑，近距精确抵消飞控闭环滞后直穿靶心
                    t_far = min(float(t_go_raw), 0.72)
                    blend = min(1.0, max(0.0, (r_norm - 1.2) / 5.5))
                    t_eff = 0.16 + blend * max(0.0, t_far - 0.16)
                    p_hit, _ = predict_fn(t_eff)

                    z_aim = float(target_pos[2]) + max(
                        -0.25, min(0.25, float(p_hit[2]) - float(target_pos[2]))
                    )
                    strike_pt = np.array([float(p_hit[0]), float(p_hit[1]), z_aim], dtype=float)

                    # 极近距 (<4.5m) 沿接近矢量追加 0.85m 穿透余量防止到达前减速
                    close_dir = strike_pt - p_i
                    close_d = float(np.linalg.norm(close_dir))
                    if 0.05 < close_d < 4.5:
                        strike_pt += (close_dir / close_d) * 0.85

                    sz = self._clamp_z(float(strike_pt[2]), is_attack=True)
                    results[d_id] = {
                        "x": float(strike_pt[0]),
                        "y": float(strike_pt[1]),
                        "z": sz,
                        "action": "ATTACK",
                        "t_go": round(float(t_eff), 3),
                        "reachable": reachable,
                    }
                else:
                    wing_rank += 1
                    p_cov, _ = predict_fn(0.5)
                    wing_ang = self._orbit_phase0 + math.pi * (0.65 if wing_rank == 1 else -0.65)
                    wx = float(p_cov[0]) + 6.2 * math.cos(wing_ang)
                    wy = float(p_cov[1]) + 6.2 * math.sin(wing_ang)
                    wz = self._clamp_z(float(target_pos[2]) + 2.8 + 1.0 * wing_rank, is_attack=False)
                    results[d_id] = {
                        "x": wx,
                        "y": wy,
                        "z": wz,
                        "action": "ATTACK",
                        "t_go": round(float(t_go_raw), 3),
                        "reachable": reachable,
                    }

        return results

    def _compute_patrol_waypoint(self, now_sec: float, slot_rank: int, total_drones: int) -> Tuple[float, float, float]:
        """生成与 H1/H2/H3 停机坪一一对应的平行无交叉横向扫描巡逻航线。"""
        _ = total_drones
        omega = 0.22
        px = 16.0 + 9.0 * math.sin(omega * float(now_sec))
        py = float(slot_rank) * 3.5 + 1.2 * math.sin(omega * float(now_sec))
        pz = self._clamp_z(9.0 + 0.8 * slot_rank, is_attack=False)
        return float(px), float(py), float(pz)
