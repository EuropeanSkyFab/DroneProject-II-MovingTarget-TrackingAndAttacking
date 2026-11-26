# -*- coding: utf-8 -*-
"""
estimator.py —— 纯 NumPy 匈牙利分配器、三模型 IMM 滤波器与多目标航迹库

坐标系约定：全局 ENU（东-北-天，米 / 秒），不做任何 NED 转换。

核心能力：
  1. hungarian_assignment: 零 SciPy 依赖的 O(N^3) Kuhn-Munkres / Jonker-Volgenant
     矩形最优指派求解器（支持 +inf / max_cost 门控与负数代价）。
  2. IMMTrack: 异构三模型交互式多模型滤波器（6D CV、7D 笛卡尔 3D-CTRV EKF、
     9D Singer-CA "JINK"），自适应 Δt 马尔可夫转移矩阵、无偏角速度混合与多步航迹预测。
  3. IMMTrackBank: 两阶段匈牙利 GNN 关联、马氏+欧氏双门控、重复起批抑制、
     M-of-N (3/5) 航迹确认、遮挡滑行推演与击毁终结管理。
  4. ManeuveringTargetEstimator: 向后兼容的单目标封装接口（委托给内部 IMMTrack）。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Deque, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np


_MODEL_NAMES: Tuple[str, str, str] = ("CV", "CTRV", "JINK")
_GENERATOR_LAMBDA = np.array(
    [
        [-0.50, 0.28, 0.22],
        [0.20, -0.45, 0.25],
        [0.25, 0.35, -0.60],
    ],
    dtype=float,
)


def hungarian_assignment(
    cost_matrix: Union[np.ndarray, Sequence[Sequence[float]]],
    max_cost: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Pure-NumPy O(N^3) Kuhn-Munkres rectangular assignment solver.

    Maximizes valid matched cardinality first, then minimizes total cost over
    valid (finite and <= max_cost) edges.

    Returns:
        (row_ind, col_ind, total_cost)
    """
    cost = np.asarray(cost_matrix, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost_matrix must be 2-dimensional")
    if np.isnan(cost).any():
        raise ValueError("cost_matrix must not contain NaN")
    if np.any(cost == -np.inf):
        raise ValueError("cost_matrix must not contain -inf")

    n_rows, n_cols = cost.shape
    if n_rows == 0 or n_cols == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), 0.0

    valid = np.isfinite(cost)
    if max_cost is not None:
        valid &= cost <= float(max_cost)
    if not np.any(valid):
        return np.empty(0, dtype=int), np.empty(0, dtype=int), 0.0

    transposed = n_rows > n_cols
    if transposed:
        c_work = cost.T
        v_work = valid.T
    else:
        c_work = cost
        v_work = valid

    r, c = c_work.shape
    valid_vals = c_work[v_work]
    f_min = float(np.min(valid_vals))
    f_max = float(np.max(valid_vals))
    delta_span = max(0.0, f_max - f_min)
    infeasible_penalty = float(r) * delta_span + 100.0

    shifted = np.where(v_work, c_work - f_min, infeasible_penalty)
    a_rows: List[List[float]] = [[0.0] * (c + 1)]
    for row in shifted.tolist():
        a_rows.append([0.0] + row)

    inf_sentinel = float("inf")
    u = [0.0] * (r + 1)
    v = [0.0] * (c + 1)
    p = [0] * (c + 1)
    way = [0] * (c + 1)

    for i in range(1, r + 1):
        p[0] = i
        j0 = 0
        minv = [inf_sentinel] * (c + 1)
        used = [False] * (c + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta_step = inf_sentinel
            j1 = 0
            row_i0 = a_rows[i0]
            u_i0 = u[i0]
            for j in range(1, c + 1):
                if not used[j]:
                    cur = row_i0[j] - u_i0 - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta_step:
                        delta_step = minv[j]
                        j1 = j
            for j in range(c + 1):
                if used[j]:
                    u[p[j]] += delta_step
                    v[j] -= delta_step
                else:
                    minv[j] -= delta_step
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    pairs: List[Tuple[int, int]] = []
    for j in range(1, c + 1):
        i_match = p[j]
        if i_match > 0:
            ri = i_match - 1
            cj = j - 1
            if v_work[ri, cj]:
                if transposed:
                    pairs.append((cj, ri))
                else:
                    pairs.append((ri, cj))

    if not pairs:
        return np.empty(0, dtype=int), np.empty(0, dtype=int), 0.0

    pairs.sort(key=lambda rc: (rc[0], rc[1]))
    row_ind = np.array([rc[0] for rc in pairs], dtype=int)
    col_ind = np.array([rc[1] for rc in pairs], dtype=int)
    total_cost = float(np.sum(cost[row_ind, col_ind]))
    return row_ind, col_ind, total_cost


def _singer_blocks_param(dt: float, tau_m: float, sigma_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """构造九维三轴 Singer 离散状态转移矩阵 F(dt) 与过程噪声协方差 Q(dt)。"""
    dt = max(0.0, min(float(dt), 5.0))
    if dt < 1e-6:
        return np.eye(9, dtype=float), np.zeros((9, 9), dtype=float)

    alpha = 1.0 / max(0.05, float(tau_m))
    adt = alpha * dt
    eadt = math.exp(-adt)
    e2adt = math.exp(-2.0 * adt)

    f_pa = (adt - 1.0 + eadt) / (alpha * alpha)
    f_va = (1.0 - eadt) / alpha
    f_aa = eadt

    F = np.eye(9, dtype=float)
    for i in range(3):
        F[i, i + 3] = dt
        F[i, i + 6] = f_pa
        F[i + 3, i + 6] = f_va
        F[i + 6, i + 6] = f_aa

    s2 = 2.0 * alpha * (float(sigma_m) ** 2)
    a2 = alpha * alpha
    a3 = a2 * alpha
    a4 = a2 * a2
    a5 = a4 * alpha

    q11 = (1.0 - e2adt + 2.0 * adt + (2.0 / 3.0) * (adt ** 3) - 2.0 * (adt ** 2) - 4.0 * adt * eadt) / (2.0 * a5)
    q12 = (e2adt + 1.0 - 2.0 * eadt + 2.0 * adt * eadt - 2.0 * adt + adt * adt) / (2.0 * a4)
    q13 = (1.0 - e2adt - 2.0 * adt * eadt) / (2.0 * a3)
    q22 = (4.0 * eadt - 3.0 - e2adt + 2.0 * adt) / (2.0 * a3)
    q23 = (e2adt + 1.0 - 2.0 * eadt) / (2.0 * a2)
    q33 = (1.0 - e2adt) / (2.0 * alpha)

    Q = np.zeros((9, 9), dtype=float)
    axis_scale = (1.0, 1.0, 0.45)
    for i, sc in enumerate(axis_scale):
        w = s2 * sc
        Q[i, i] = w * max(q11, 1e-9)
        Q[i, i + 3] = Q[i + 3, i] = w * q12
        Q[i, i + 6] = Q[i + 6, i] = w * q13
        Q[i + 3, i + 3] = w * max(q22, 1e-8)
        Q[i + 3, i + 6] = Q[i + 6, i + 3] = w * q23
        Q[i + 6, i + 6] = w * max(q33, 1e-7)

    return F, Q


def _cv_blocks(dt: float, qa_xy: float = 0.08, qa_z: float = 0.06) -> Tuple[np.ndarray, np.ndarray]:
    """构造 6D 匀速模型状态转移矩阵与过程噪声。"""
    dt = max(0.0, min(float(dt), 5.0))
    F = np.eye(6, dtype=float)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    Q = np.zeros((6, 6), dtype=float)
    if dt < 1e-6:
        return F, Q
    dt2 = dt * dt
    dt3 = 0.5 * dt2 * dt
    dt4 = 0.25 * dt2 * dt2
    for i, qa in enumerate((qa_xy, qa_xy, qa_z)):
        q2 = qa * qa
        Q[i, i] = q2 * max(dt4, 1e-9)
        Q[i, i + 3] = Q[i + 3, i] = q2 * dt3
        Q[i + 3, i + 3] = q2 * max(dt2, 1e-8)
    return F, Q


def _ctrv_step_and_jacobian(
    x_ctrv: np.ndarray,
    dt: float,
    max_turn_rate: float,
    tau_omega: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """7D 笛卡尔 3D-CTRV 精确非线性递推、解析雅可比矩阵 F1 与过程噪声 Q1。"""
    dt = max(0.0, min(float(dt), 5.0))
    px, py, pz, vx, vy, vz, omega = [float(v) for v in x_ctrv]
    omega = max(-max_turn_rate, min(max_turn_rate, omega))
    if dt < 1e-6:
        return x_ctrv.copy(), np.eye(7, dtype=float), np.zeros((7, 7), dtype=float)

    th = omega * dt
    sin_th = math.sin(th)
    cos_th = math.cos(th)
    decay_w = math.exp(-dt / max(1.0, tau_omega))

    if abs(th) > 1e-3:
        A = sin_th / omega
        B = (cos_th - 1.0) / omega
        C = (1.0 - cos_th) / omega
        inv_w2 = 1.0 / (omega * omega)
        dA_dw = (omega * dt * cos_th - sin_th) * inv_w2
        dB_dw = (-omega * dt * sin_th - (cos_th - 1.0)) * inv_w2
        dC_dw = -dB_dw
    else:
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        w2 = omega * omega
        A = dt - w2 * dt3 / 6.0
        B = -0.5 * omega * dt2 + (w2 * omega) * dt4 / 24.0
        C = -B
        dA_dw = -omega * dt3 / 3.0
        dB_dw = -0.5 * dt2 + w2 * dt4 / 8.0
        dC_dw = -dB_dw

    x_next = np.array(
        [
            px + A * vx + B * vy,
            py + C * vx + A * vy,
            pz + vz * dt,
            cos_th * vx - sin_th * vy,
            sin_th * vx + cos_th * vy,
            vz,
            decay_w * omega,
        ],
        dtype=float,
    )

    F = np.eye(7, dtype=float)
    F[0, 3] = A
    F[0, 4] = B
    F[0, 6] = dA_dw * vx + dB_dw * vy
    F[1, 3] = C
    F[1, 4] = A
    F[1, 6] = dC_dw * vx + dA_dw * vy
    F[2, 5] = dt
    F[3, 3] = cos_th
    F[3, 4] = -sin_th
    F[3, 6] = -dt * (sin_th * vx + cos_th * vy)
    F[4, 3] = sin_th
    F[4, 4] = cos_th
    F[4, 6] = dt * (cos_th * vx - sin_th * vy)
    F[6, 6] = decay_w

    Q = np.zeros((7, 7), dtype=float)
    dt2 = dt * dt
    dt3 = 0.5 * dt2 * dt
    dt4 = 0.25 * dt2 * dt2
    qv_xy = 0.14
    qv_z = 0.12
    for i, qv in enumerate((qv_xy, qv_xy, qv_z)):
        q2 = qv * qv
        Q[i, i] = q2 * max(dt4, 1e-9)
        Q[i, i + 3] = Q[i + 3, i] = q2 * dt3
        Q[i + 3, i + 3] = q2 * max(dt2, 1e-8)
    q_omega = 0.35
    Q[6, 6] = (q_omega * q_omega) * dt + 1e-5
    return x_next, F, Q


def _transition_matrix(dt: float) -> np.ndarray:
    """基于连续时间生成矩阵计算 Δt 自适应 3x3 马尔可夫转移矩阵 Π(Δt)。"""
    dt_c = max(1e-4, min(1.0, float(dt)))
    raw = np.eye(3, dtype=float) + _GENERATOR_LAMBDA * dt_c
    raw = np.clip(raw, 1e-4, 1.0)
    row_sums = np.sum(raw, axis=1, keepdims=True)
    return raw / np.maximum(row_sums, 1e-12)


class IMMTrack:
    """异构三模型交互式多模型 (IMM: CV / 3D-CTRV / Singer-CA) 单目标航迹滤波器。"""

    def __init__(
        self,
        track_id: str = "trk_0",
        external_id: str = "",
        class_id: str = "",
        tau_maneuver: float = 1.35,
        sigma_maneuver: float = 3.6,
        sigma_vision: float = 0.36,
        sigma_external: float = 0.55,
        mahalanobis_gate: float = 18.0,
        max_consecutive_rejects: int = 5,
        max_coast_sec: float = 3.0,
        max_target_speed: float = 25.0,
        max_turn_rate: float = 1.2,
        confirm_n: int = 5,
    ) -> None:
        assert tau_maneuver > 0.05, "tau_maneuver must be positive"
        assert sigma_maneuver > 0.0, "sigma_maneuver must be positive"
        assert sigma_vision > 0.0 and sigma_external > 0.0, "measurement noise must be positive"

        self.track_id: str = str(track_id)
        self.external_id: str = str(external_id)
        self.class_id: str = str(class_id)

        self.tau_m: float = float(tau_maneuver)
        self.sigma_m: float = float(sigma_maneuver)
        self.tau_jink: float = min(self.tau_m, 1.1)
        self.sigma_jink: float = max(self.sigma_m, 4.5)
        self.sigma_vis: float = float(sigma_vision)
        self.sigma_ext: float = float(sigma_external)
        self.gate_thresh: float = float(mahalanobis_gate)
        self.max_rejects: int = int(max_consecutive_rejects)
        self.max_coast_sec: float = float(max_coast_sec)
        self.max_speed: float = float(max_target_speed)
        self.max_turn_rate: float = float(max_turn_rate)

        self.H: np.ndarray = np.zeros((3, 9), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self._H_cv: np.ndarray = np.eye(3, 6, dtype=float)
        self._H_ctrv: np.ndarray = np.eye(3, 7, dtype=float)
        self._H_jink: np.ndarray = np.eye(3, 9, dtype=float)

        self.x: np.ndarray = np.zeros(9, dtype=float)
        self.P: np.ndarray = np.eye(9, dtype=float) * 25.0
        self.mu: np.ndarray = np.array([0.45, 0.35, 0.20], dtype=float)

        self.x_cv: np.ndarray = np.zeros(6, dtype=float)
        self.P_cv: np.ndarray = np.eye(6, dtype=float) * 25.0
        self.x_ctrv: np.ndarray = np.zeros(7, dtype=float)
        self.P_ctrv: np.ndarray = np.eye(7, dtype=float) * 25.0
        self.x_jink: np.ndarray = np.zeros(9, dtype=float)
        self.P_jink: np.ndarray = np.eye(9, dtype=float) * 25.0

        self.initialized: bool = False
        self.status: str = "TENTATIVE"
        self.last_update_time: float = 0.0
        self.last_state_time: float = 0.0
        self.last_vision_time: float = -1e9
        self.last_source: str = "none"
        self.turn_rate_est: float = 0.0
        self.reject_count: int = 0
        self.update_count: int = 0
        self.last_mahalanobis_sq: float = 0.0
        self.hit_history: Deque[bool] = deque(maxlen=max(1, int(confirm_n)))
        self.consecutive_misses: int = 0
        self.last_hit_check_time: float = -1e9

    @property
    def imm_probs(self) -> Dict[str, float]:
        """返回三模型当前后验概率字典。"""
        return {
            "CV": float(self.mu[0]),
            "CTRV": float(self.mu[1]),
            "JINK": float(self.mu[2]),
        }

    def reset(self, pos_enu: Tuple[float, float, float], timestamp: float, source: str = "init") -> None:
        """以初始三维位置重置三模型 IMM 滤波状态与协方差。"""
        px, py, pz = float(pos_enu[0]), float(pos_enu[1]), float(pos_enu[2])
        ts = float(timestamp)

        self.mu = np.array([0.45, 0.35, 0.20], dtype=float)
        self.x_cv = np.array([px, py, pz, 0.0, 0.0, 0.0], dtype=float)
        self.P_cv = np.diag([1.0, 1.0, 1.0, 9.0, 9.0, 4.0]).astype(float)

        self.x_ctrv = np.array([px, py, pz, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.P_ctrv = np.diag([1.0, 1.0, 1.0, 9.0, 9.0, 4.0, 0.8]).astype(float)

        self.x_jink = np.array([px, py, pz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.P_jink = np.diag([1.0, 1.0, 1.0, 9.0, 9.0, 4.0, 12.0, 12.0, 6.0]).astype(float)

        self.initialized = True
        self.last_update_time = ts
        self.last_state_time = ts
        if source == "vision":
            self.last_vision_time = ts
        self.last_source = source
        self.turn_rate_est = 0.0
        self.reject_count = 0
        self.update_count = 1
        self.last_mahalanobis_sq = 0.0
        self.consecutive_misses = 0
        self.last_hit_check_time = ts
        self._fuse_state()

    def _clamp_vec_kinematics(self, x_vec: np.ndarray, has_acc: bool = False, has_omega: bool = False) -> None:
        vel = x_vec[3:6]
        spd = float(np.linalg.norm(vel))
        if spd > self.max_speed and spd > 1e-9:
            x_vec[3:6] = vel * (self.max_speed / spd)
        if has_acc:
            acc = x_vec[6:9]
            acc_norm = float(np.linalg.norm(acc))
            if acc_norm > 15.0 and acc_norm > 1e-9:
                x_vec[6:9] = acc * (15.0 / acc_norm)
        if has_omega:
            x_vec[6] = max(-self.max_turn_rate, min(self.max_turn_rate, float(x_vec[6])))

    def _jink_omega(self) -> float:
        vx, vy = float(self.x_jink[3]), float(self.x_jink[4])
        ax, ay = float(self.x_jink[6]), float(self.x_jink[7])
        vh2 = vx * vx + vy * vy
        if vh2 > 0.64:
            w = (vx * ay - vy * ax) / vh2
            return max(-self.max_turn_rate, min(self.max_turn_rate, w))
        return float(self.x_ctrv[6])

    def _ctrv_acc(self) -> np.ndarray:
        w = float(self.x_ctrv[6])
        return np.array([-w * float(self.x_ctrv[4]), w * float(self.x_ctrv[3]), 0.0], dtype=float)

    def _mix_models(self, dt: float) -> Tuple[np.ndarray, Tuple[np.ndarray, ...], Tuple[np.ndarray, ...]]:
        pi = _transition_matrix(dt)
        c_bar = self.mu @ pi
        c_bar = np.maximum(c_bar, 1e-12)
        mu_cond = (self.mu[:, None] * pi) / c_bar[None, :]

        x_pvs = [self.x_cv[0:6], self.x_ctrv[0:6], self.x_jink[0:6]]
        P_pvs = [self.P_cv[0:6, 0:6], self.P_ctrv[0:6, 0:6], self.P_jink[0:6, 0:6]]

        x0_pvs: List[np.ndarray] = []
        P0_pvs: List[np.ndarray] = []
        for j in range(3):
            xm = np.zeros(6, dtype=float)
            for i in range(3):
                xm += float(mu_cond[i, j]) * x_pvs[i]
            Pm = np.zeros((6, 6), dtype=float)
            for i in range(3):
                dx = x_pvs[i] - xm
                Pm += float(mu_cond[i, j]) * (P_pvs[i] + np.outer(dx, dx))
            x0_pvs.append(xm)
            P0_pvs.append(0.5 * (Pm + Pm.T))

        # CV 6D
        x0_cv = x0_pvs[0]
        P0_cv = P0_pvs[0]

        # CTRV 7D: 无偏角速度混合（避免 CV 零角速度拉低转弯率）
        w_ctrv = float(self.x_ctrv[6])
        w_jink = self._jink_omega()
        w_srcs = [w_ctrv, w_ctrv, w_jink]
        w_vars = [0.50, float(self.P_ctrv[6, 6]), 0.45]
        w0 = float(mu_cond[0, 1] + mu_cond[1, 1]) * w_ctrv + float(mu_cond[2, 1]) * w_jink
        w0 = max(-self.max_turn_rate, min(self.max_turn_rate, w0))
        P0_ww = 0.0
        for i in range(3):
            dw = w_srcs[i] - w0
            P0_ww += float(mu_cond[i, 1]) * (w_vars[i] + dw * dw)
        P0_ww = max(0.04, P0_ww)
        P0_pvw = float(mu_cond[1, 1]) * self.P_ctrv[0:6, 6:7]

        x0_ctrv = np.concatenate([x0_pvs[1], np.array([w0], dtype=float)])
        P0_ctrv = np.zeros((7, 7), dtype=float)
        P0_ctrv[0:6, 0:6] = P0_pvs[1]
        P0_ctrv[0:6, 6:7] = P0_pvw
        P0_ctrv[6:7, 0:6] = P0_pvw.T
        P0_ctrv[6, 6] = P0_ww
        P0_ctrv = 0.5 * (P0_ctrv + P0_ctrv.T)

        # JINK 9D: 加速度混合
        a_srcs = [np.zeros(3, dtype=float), self._ctrv_acc(), self.x_jink[6:9]]
        P_a_prior = np.diag([4.0, 4.0, 1.5]).astype(float)
        P_a_srcs = [P_a_prior, P_a_prior, self.P_jink[6:9, 6:9]]
        a0 = np.zeros(3, dtype=float)
        for i in range(3):
            a0 += float(mu_cond[i, 2]) * a_srcs[i]
        P0_aa = np.zeros((3, 3), dtype=float)
        for i in range(3):
            da = a_srcs[i] - a0
            P0_aa += float(mu_cond[i, 2]) * (P_a_srcs[i] + np.outer(da, da))
        P0_pva = float(mu_cond[2, 2]) * self.P_jink[0:6, 6:9]

        x0_jink = np.concatenate([x0_pvs[2], a0])
        P0_jink = np.zeros((9, 9), dtype=float)
        P0_jink[0:6, 0:6] = P0_pvs[2]
        P0_jink[0:6, 6:9] = P0_pva
        P0_jink[6:9, 0:6] = P0_pva.T
        P0_jink[6:9, 6:9] = P0_aa
        P0_jink = 0.5 * (P0_jink + P0_jink.T)

        return c_bar, (x0_cv, x0_ctrv, x0_jink), (P0_cv, P0_ctrv, P0_jink)

    def _predict_models(
        self,
        dt: float,
        x0s: Tuple[np.ndarray, np.ndarray, np.ndarray],
        P0s: Tuple[np.ndarray, np.ndarray, np.ndarray],
        coast_damping: bool = False,
    ) -> None:
        # Model 0: CV
        F0, Q0 = _cv_blocks(dt)
        self.x_cv = F0 @ x0s[0]
        self._clamp_vec_kinematics(self.x_cv)
        self.P_cv = F0 @ P0s[0] @ F0.T + Q0
        self.P_cv = 0.5 * (self.P_cv + self.P_cv.T)

        # Model 1: CTRV
        tau_w = 3.5 if coast_damping else 30.0
        x1, F1, Q1 = _ctrv_step_and_jacobian(x0s[1], dt, self.max_turn_rate, tau_omega=tau_w)
        self.x_ctrv = x1
        self._clamp_vec_kinematics(self.x_ctrv, has_omega=True)
        self.P_ctrv = F1 @ P0s[1] @ F1.T + Q1
        self.P_ctrv = 0.5 * (self.P_ctrv + self.P_ctrv.T)

        # Model 2: JINK (Singer-CA)
        F2, Q2 = _singer_blocks_param(dt, self.tau_jink, self.sigma_jink)
        self.x_jink = F2 @ x0s[2]
        self._clamp_vec_kinematics(self.x_jink, has_acc=True)
        self.P_jink = F2 @ P0s[2] @ F2.T + Q2
        self.P_jink = 0.5 * (self.P_jink + self.P_jink.T)

        if coast_damping:
            for Pm in (self.P_cv, self.P_ctrv, self.P_jink):
                for ax_i in range(3):
                    if Pm[ax_i, ax_i] > 100.0:
                        scale = math.sqrt(100.0 / float(Pm[ax_i, ax_i]))
                        Pm[ax_i, :] *= scale
                        Pm[:, ax_i] *= scale
                Pm[:] = 0.5 * (Pm + Pm.T)

    def _fuse_state(self) -> None:
        mu0, mu1, mu2 = float(self.mu[0]), float(self.mu[1]), float(self.mu[2])
        x_pvs = (self.x_cv[0:6], self.x_ctrv[0:6], self.x_jink[0:6])
        P_pvs = (self.P_cv[0:6, 0:6], self.P_ctrv[0:6, 0:6], self.P_jink[0:6, 0:6])

        a_srcs = (np.zeros(3, dtype=float), self._ctrv_acc(), self.x_jink[6:9])
        P_a_prior = np.diag([4.0, 4.0, 1.5]).astype(float)
        P_a_srcs = (P_a_prior, P_a_prior, self.P_jink[6:9, 6:9])

        x9_models = (
            np.concatenate([x_pvs[0], a_srcs[0]]),
            np.concatenate([x_pvs[1], a_srcs[1]]),
            np.concatenate([x_pvs[2], a_srcs[2]]),
        )
        x_fused = mu0 * x9_models[0] + mu1 * x9_models[1] + mu2 * x9_models[2]
        self._clamp_vec_kinematics(x_fused, has_acc=True)

        P_fused = np.zeros((9, 9), dtype=float)
        weights = (mu0, mu1, mu2)
        for j in range(3):
            Pj = np.zeros((9, 9), dtype=float)
            Pj[0:6, 0:6] = P_pvs[j]
            Pj[6:9, 6:9] = P_a_srcs[j]
            if j == 2:
                Pj[0:6, 6:9] = self.P_jink[0:6, 6:9]
                Pj[6:9, 0:6] = self.P_jink[6:9, 0:6]
            dx = x9_models[j] - x_fused
            P_fused += weights[j] * (Pj + np.outer(dx, dx))
        P_fused = 0.5 * (P_fused + P_fused.T)
        for d in range(9):
            if P_fused[d, d] < 1e-9:
                P_fused[d, d] = 1e-9

        self.x = x_fused
        self.P = P_fused
        w_est = mu1 * float(self.x_ctrv[6]) + mu2 * self._jink_omega()
        self.turn_rate_est = max(-self.max_turn_rate, min(self.max_turn_rate, w_est))

    def propagate_to(self, timestamp: float, coast_damping: bool = False) -> None:
        """将内部三模型 IMM 状态与协方差前向传播至指定时间戳。"""
        ts = float(timestamp)
        if not self.initialized or not math.isfinite(ts):
            return
        dt = ts - self.last_state_time
        if dt <= 1e-5:
            return
        age = ts - self.last_update_time
        damp = coast_damping or (age > 0.35)
        dt_rem = dt
        while dt_rem > 1e-5:
            step = min(dt_rem, 0.5)
            c_bar, x0s, P0s = self._mix_models(step)
            self.mu = c_bar / max(float(np.sum(c_bar)), 1e-12)
            self._predict_models(step, x0s, P0s, coast_damping=damp)
            dt_rem -= step
        self._fuse_state()
        self.last_state_time = ts

    def _effective_sigma(self, ts: float, source: str, stereo: bool, conf: float) -> float:
        if source == "vision":
            c_raw = float(conf)
            c_val = 0.2 if not math.isfinite(c_raw) else max(0.2, min(1.0, c_raw))
            scale = 0.85 if stereo else 1.8
            return self.sigma_vis * scale / math.sqrt(c_val)
        vis_recent = (max(ts, self.last_state_time) - self.last_vision_time) < 0.6
        return self.sigma_ext * (2.5 if vis_recent else 1.0)

    def compute_mahalanobis_sq(
        self,
        pos_enu: Tuple[float, float, float],
        timestamp: float,
        source: str = "vision",
        stereo: bool = True,
        conf: float = 1.0,
    ) -> Tuple[float, np.ndarray]:
        """计算当前融合状态对观测的马氏距离平方与补偿后观测向量。"""
        px, py, pz = float(pos_enu[0]), float(pos_enu[1]), float(pos_enu[2])
        ts = float(timestamp)
        if ts >= self.last_state_time:
            z_eff = np.array([px, py, pz], dtype=float)
        else:
            lag = self.last_state_time - ts
            z_eff = np.array([px, py, pz], dtype=float) + self.x[3:6] * lag + 0.5 * self.x[6:9] * (lag ** 2)

        sigma = self._effective_sigma(ts, source, stereo, conf)
        R = np.diag([sigma * sigma, sigma * sigma, 1.35 * sigma * sigma]).astype(float)
        S = self.P[0:3, 0:3] + R
        y = z_eff - self.x[0:3]
        try:
            d2 = float(y.T @ np.linalg.solve(S, y))
        except np.linalg.LinAlgError:
            d2 = float("inf")
        return d2, z_eff

    def update(
        self,
        pos_enu: Tuple[float, float, float],
        timestamp: float,
        source: str = "vision",
        stereo: bool = True,
        conf: float = 1.0,
        allow_reset_on_rejects: bool = True,
        skip_gate: bool = False,
    ) -> bool:
        """融合一帧三维位置观测，执行三模型 IMM 交互、预测、更新与模式概率融合。"""
        px, py, pz = float(pos_enu[0]), float(pos_enu[1]), float(pos_enu[2])
        ts = float(timestamp)
        conf_val = float(conf)
        if not (
            math.isfinite(px)
            and math.isfinite(py)
            and math.isfinite(pz)
            and math.isfinite(ts)
            and math.isfinite(conf_val)
        ):
            return False

        if not self.initialized or (ts - self.last_update_time) > self.max_coast_sec * 1.5:
            self.reset((px, py, pz), ts, source=source)
            return True

        if ts < self.last_state_time - 0.45:
            return False

        dt_pred = max(0.0, ts - self.last_state_time)
        if dt_pred > 1e-5:
            c_bar, x0s, P0s = self._mix_models(dt_pred)
            self._predict_models(dt_pred, x0s, P0s, coast_damping=False)
            self.mu = c_bar / max(float(np.sum(c_bar)), 1e-12)
            self._fuse_state()
            self.last_state_time = ts
            z_eff = np.array([px, py, pz], dtype=float)
        else:
            lag = self.last_state_time - ts
            z_eff = np.array([px, py, pz], dtype=float) + self.x[3:6] * lag + 0.5 * self.x[6:9] * (lag ** 2)

        sigma = self._effective_sigma(ts, source, stereo, conf)
        R = np.diag([sigma * sigma, sigma * sigma, 1.35 * sigma * sigma]).astype(float)

        # 融合态马氏门控
        S_fused = self.P[0:3, 0:3] + R
        y_fused = z_eff - self.x[0:3]
        try:
            d_m2 = float(y_fused.T @ np.linalg.solve(S_fused, y_fused))
        except np.linalg.LinAlgError:
            return False

        self.last_mahalanobis_sq = d_m2
        if not skip_gate and d_m2 > self.gate_thresh:
            self.reject_count += 1
            if allow_reset_on_rejects and self.reject_count >= self.max_rejects:
                self.reset((px, py, pz), max(ts, self.last_state_time), source=source)
                return True
            return False

        self.reject_count = 0

        # 对三个异构模型分别执行观测更新与对数似然计算
        log_likes = np.zeros(3, dtype=float)
        models_x = [self.x_cv, self.x_ctrv, self.x_jink]
        models_P = [self.P_cv, self.P_ctrv, self.P_jink]
        models_H = [self._H_cv, self._H_ctrv, self._H_jink]
        dims = (6, 7, 9)

        for j in range(3):
            xj = models_x[j]
            Pj = models_P[j]
            Hj = models_H[j]
            dim_j = dims[j]
            yj = z_eff - xj[0:3]
            Sj = Pj[0:3, 0:3] + R
            try:
                Sj_inv = np.linalg.inv(Sj)
                sign, logdet = np.linalg.slogdet(Sj)
                if sign <= 0.0 or not math.isfinite(logdet):
                    log_likes[j] = -1e6
                    continue
            except np.linalg.LinAlgError:
                log_likes[j] = -1e6
                continue

            d2_j = float(yj.T @ Sj_inv @ yj)
            log_likes[j] = -0.5 * (3.0 * math.log(2.0 * math.pi) + float(logdet) + d2_j)

            Kj = Pj[:, 0:3] @ Sj_inv
            xj_upd = xj + Kj @ yj
            self._clamp_vec_kinematics(xj_upd, has_acc=(j == 2), has_omega=(j == 1))
            I_KH = np.eye(dim_j, dtype=float) - Kj @ Hj
            Pj_upd = I_KH @ Pj @ I_KH.T + Kj @ R @ Kj.T
            Pj_upd = 0.5 * (Pj_upd + Pj_upd.T)
            for d_i in range(dim_j):
                if Pj_upd[d_i, d_i] < 1e-9:
                    Pj_upd[d_i, d_i] = 1e-9

            if j == 0:
                self.x_cv, self.P_cv = xj_upd, Pj_upd
            elif j == 1:
                self.x_ctrv, self.P_ctrv = xj_upd, Pj_upd
            else:
                self.x_jink, self.P_jink = xj_upd, Pj_upd

        # Log-sum-exp 模式概率更新 + 下界平滑保护
        max_ll = float(np.max(log_likes))
        lik_rel = np.exp(np.clip(log_likes - max_ll, -50.0, 0.0))
        mu_unnorm = lik_rel * self.mu
        s_mu = float(np.sum(mu_unnorm))
        if s_mu > 1e-15:
            mu_new = mu_unnorm / s_mu
        else:
            mu_new = self.mu.copy()
        mu_new = np.maximum(mu_new, 0.005)
        self.mu = mu_new / float(np.sum(mu_new))

        self._fuse_state()
        now_mark = max(ts, self.last_state_time)
        self.last_update_time = now_mark
        if source == "vision":
            self.last_vision_time = now_mark
        self.last_source = source
        self.update_count += 1
        self.consecutive_misses = 0
        return True

    def is_valid(self, current_time: float) -> bool:
        """判断当前目标状态是否在有效保鲜与外推窗口内。"""
        ct = float(current_time)
        if not math.isfinite(ct) or not self.initialized or self.status in ("KILLED", "DELETED"):
            return False
        return (ct - self.last_update_time) <= self.max_coast_sec

    def get_kinematics_at(self, query_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取在 query_time 时刻的目标估计位置、速度、加速度（不修改内部滤波状态）。"""
        qt = float(query_time)
        if not math.isfinite(qt):
            return self.x[0:3].copy(), self.x[3:6].copy(), self.x[6:9].copy()
        dt = max(0.0, qt - self.last_state_time)
        if dt < 1e-4:
            return self.x[0:3].copy(), self.x[3:6].copy(), self.x[6:9].copy()
        p, v = self.predict_pv_ahead(dt)
        alpha = 1.0 / self.tau_m
        a = self.x[6:9] * math.exp(-alpha * min(dt, 3.0))
        return p, v, a

    def _predict_single_state_pv(self, s: np.ndarray, tau: float) -> Tuple[np.ndarray, np.ndarray]:
        px, py, pz = float(s[0]), float(s[1]), float(s[2])
        vx, vy, vz = float(s[3]), float(s[4]), float(s[5])
        ax, ay, az = float(s[6]), float(s[7]), float(s[8])

        vh = math.hypot(vx, vy)
        omega_eff = self.turn_rate_est * math.exp(-0.22 * tau)
        if vh > 0.9 and abs(omega_eff) > 0.035:
            th = omega_eff * tau
            sin_th = math.sin(th)
            cos_th = math.cos(th)
            dx = vx * (sin_th / omega_eff) + vy * ((cos_th - 1.0) / omega_eff)
            dy = vx * ((1.0 - cos_th) / omega_eff) + vy * (sin_th / omega_eff)
            nvx = vx * cos_th - vy * sin_th
            nvy = vx * sin_th + vy * cos_th
            a_tang = (vx * ax + vy * ay) / vh
            tau_a = min(tau, 1.5)
            ux, uy = nvx / max(math.hypot(nvx, nvy), 1e-6), nvy / max(math.hypot(nvx, nvy), 1e-6)
            dx += 0.5 * a_tang * (tau_a ** 2) * ux
            dy += 0.5 * a_tang * (tau_a ** 2) * uy
            spd_new = max(0.0, min(self.max_speed, vh + a_tang * tau_a))
            nvx, nvy = ux * spd_new, uy * spd_new
        else:
            alpha = 1.0 / self.tau_m
            eadt = math.exp(-alpha * tau)
            f_pa = (alpha * tau - 1.0 + eadt) / (alpha * alpha)
            f_va = (1.0 - eadt) / alpha
            dx = vx * tau + ax * f_pa
            dy = vy * tau + ay * f_pa
            nvx = vx + ax * f_va
            nvy = vy + ay * f_va

        alpha_z = 1.0 / self.tau_m
        tau_z = min(tau, 2.0)
        eadt_z = math.exp(-alpha_z * tau_z)
        f_pa_z = (alpha_z * tau_z - 1.0 + eadt_z) / (alpha_z * alpha_z)
        f_va_z = (1.0 - eadt_z) / alpha_z
        nz = max(0.2, pz + vz * tau_z + az * f_pa_z)
        nvz = vz * math.exp(-0.35 * tau) + az * f_va_z * 0.5
        pos_pred = np.array([px + dx, py + dy, nz], dtype=float)
        vel_pred = np.array([nvx, nvy, nvz], dtype=float)
        v_norm = float(np.linalg.norm(vel_pred))
        if v_norm > self.max_speed:
            vel_pred *= self.max_speed / v_norm
        return pos_pred, vel_pred

    def predict_pv_ahead(
        self,
        dt_ahead: float,
        from_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """基于三模型 IMM 概率加权的多模型混合运动学外推，预测未来 dt_ahead 秒后的位置与速度。"""
        dt_val = float(dt_ahead)
        if not math.isfinite(dt_val):
            if from_state is not None:
                return np.array(from_state[0:3], dtype=float), np.array(from_state[3:6], dtype=float)
            return self.x[0:3].copy(), self.x[3:6].copy()
        tau = max(0.0, min(dt_val, 8.0))
        if from_state is not None:
            if tau < 1e-5:
                return np.array(from_state[0:3], dtype=float), np.array(from_state[3:6], dtype=float)
            return self._predict_single_state_pv(from_state, tau)

        if tau < 1e-5:
            return self.x[0:3].copy(), self.x[3:6].copy()

        # Model 0: CV rollout
        p0 = self.x_cv[0:3] + self.x_cv[3:6] * tau
        p0 = np.array([p0[0], p0[1], max(0.2, float(p0[2]))], dtype=float)
        v0 = self.x_cv[3:6].copy()

        # Model 1: CTRV rollout (gentle turn-rate damping)
        px1, py1, pz1, vx1, vy1, vz1, w1 = [float(v) for v in self.x_ctrv]
        w_eff = w1 * math.exp(-0.10 * tau)
        if abs(w_eff) > 1e-3:
            th = w_eff * tau
            s_th = math.sin(th)
            c_th = math.cos(th)
            A = s_th / w_eff
            B = (c_th - 1.0) / w_eff
            C = (1.0 - c_th) / w_eff
            p1_x = px1 + A * vx1 + B * vy1
            p1_y = py1 + C * vx1 + A * vy1
            v1_x = c_th * vx1 - s_th * vy1
            v1_y = s_th * vx1 + c_th * vy1
        else:
            p1_x = px1 + vx1 * tau
            p1_y = py1 + vy1 * tau
            v1_x, v1_y = vx1, vy1
        tau_z1 = min(tau, 2.0)
        p1_z = max(0.2, pz1 + vz1 * tau_z1)
        v1_z = vz1 * math.exp(-0.25 * tau)
        p1 = np.array([p1_x, p1_y, p1_z], dtype=float)
        v1 = np.array([v1_x, v1_y, v1_z], dtype=float)

        # Model 2: JINK (Singer-CA rollout)
        tau_j = min(tau, 3.0)
        F2, _ = _singer_blocks_param(tau_j, self.tau_jink, self.sigma_jink)
        x2_j = F2 @ self.x_jink
        if tau > 3.0:
            rem = tau - 3.0
            x2_j[0:3] = x2_j[0:3] + x2_j[3:6] * rem
        p2 = np.array([x2_j[0], x2_j[1], max(0.2, float(x2_j[2]))], dtype=float)
        v2 = np.array([x2_j[3], x2_j[4], float(x2_j[5]) * math.exp(-0.30 * tau)], dtype=float)

        mu0, mu1, mu2 = float(self.mu[0]), float(self.mu[1]), float(self.mu[2])
        pos_pred = mu0 * p0 + mu1 * p1 + mu2 * p2
        pos_pred[2] = max(0.2, float(pos_pred[2]))
        vel_pred = mu0 * v0 + mu1 * v1 + mu2 * v2
        v_norm = float(np.linalg.norm(vel_pred))
        if v_norm > self.max_speed and v_norm > 1e-9:
            vel_pred *= self.max_speed / v_norm
        return pos_pred, vel_pred

    def predict_path(
        self,
        horizons_sec: Sequence[float] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    ) -> List[List[float]]:
        """返回多时间步预测轨迹 [[x, y, z], ...]。"""
        path: List[List[float]] = []
        for tau in horizons_sec:
            p, _ = self.predict_pv_ahead(float(tau))
            path.append([round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)])
        return path

    def diagnostics(self, current_time: float) -> Dict[str, object]:
        """输出结构化诊断字典供可观测性追踪。"""
        ct = float(current_time)
        age = (ct - self.last_update_time) if (self.initialized and math.isfinite(ct)) else -1.0
        pos_std = float(math.sqrt(max(0.0, float(np.trace(self.P[0:3, 0:3])) / 3.0)))
        return {
            "track_id": self.track_id,
            "external_id": self.external_id,
            "class_id": self.class_id,
            "status": self.status,
            "initialized": self.initialized,
            "valid": self.is_valid(current_time),
            "source": self.last_source,
            "age_sec": round(age, 3),
            "pos_enu": [round(float(v), 3) for v in self.x[0:3]],
            "vel_enu": [round(float(v), 3) for v in self.x[3:6]],
            "acc_enu": [round(float(v), 3) for v in self.x[6:9]],
            "turn_rate_rads": round(float(self.turn_rate_est), 4),
            "pos_rmse_std_m": round(pos_std, 3),
            "mahalanobis_sq": round(float(self.last_mahalanobis_sq), 3),
            "updates": self.update_count,
            "imm_probs": {k: round(v, 4) for k, v in self.imm_probs.items()},
        }


@dataclass
class TrackObservation:
    """标准化多目标观测输入结构（全局 ENU）。"""

    pos_enu: Tuple[float, float, float]
    target_id: str = ""
    class_id: str = ""
    conf: float = 1.0
    stereo: bool = True
    timestamp: Optional[float] = None


def _coerce_observation(obs: Any) -> Optional[TrackObservation]:
    if isinstance(obs, TrackObservation):
        try:
            px, py, pz = float(obs.pos_enu[0]), float(obs.pos_enu[1]), float(obs.pos_enu[2])
            conf = float(obs.conf)
            ts_val = float(obs.timestamp) if obs.timestamp is not None else None
        except (TypeError, ValueError, IndexError):
            return None
        if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz) and math.isfinite(conf)):
            return None
        if ts_val is not None and not math.isfinite(ts_val):
            return None
        return TrackObservation(
            pos_enu=(px, py, pz),
            target_id=str(obs.target_id),
            class_id=str(obs.class_id),
            conf=conf,
            stereo=bool(obs.stereo),
            timestamp=ts_val,
        )
    if isinstance(obs, dict):
        try:
            if "pos_enu" in obs:
                raw_p = obs["pos_enu"]
                px, py, pz = float(raw_p[0]), float(raw_p[1]), float(raw_p[2])
            else:
                px = float(obs.get("x", float("nan")))
                py = float(obs.get("y", float("nan")))
                pz = float(obs.get("z", float("nan")))
            raw_conf = obs.get("conf", 1.0)
            conf = 1.0 if raw_conf is None else float(raw_conf)
            ts = obs.get("stamp", obs.get("timestamp", None))
            ts_val = float(ts) if ts is not None else None
        except (TypeError, ValueError, IndexError):
            return None
        if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz) and math.isfinite(conf)):
            return None
        if ts_val is not None and not math.isfinite(ts_val):
            return None
        tid = str(obs.get("target_id", ""))
        cid = str(obs.get("class_id", ""))
        stereo = bool(obs.get("stereo", True))
        return TrackObservation(
            pos_enu=(px, py, pz),
            target_id=tid,
            class_id=cid,
            conf=conf,
            stereo=stereo,
            timestamp=ts_val,
        )
    return None


class IMMTrackBank:
    """多目标三模型 IMM 航迹库：两阶段匈牙利 GNN 关联、M/N 确认、遮挡滑行与击毁终结。"""

    def __init__(
        self,
        tau_maneuver: float = 1.35,
        sigma_maneuver: float = 3.6,
        sigma_vision: float = 0.36,
        sigma_external: float = 0.55,
        mahalanobis_gate: float = 18.0,
        max_coast_sec: float = 3.0,
        max_target_speed: float = 25.0,
        max_turn_rate: float = 1.2,
        confirm_m: int = 3,
        confirm_n: int = 5,
        tentative_timeout_sec: float = 1.5,
        coast_enter_sec: float = 0.35,
        min_birth_sep_m: float = 2.0,
        kill_radius_m: float = 3.5,
        kill_suppression_sec: float = 2.5,
    ) -> None:
        self.tau_m = float(tau_maneuver)
        self.sigma_m = float(sigma_maneuver)
        self.sigma_vis = float(sigma_vision)
        self.sigma_ext = float(sigma_external)
        self.gate_thresh = float(mahalanobis_gate)
        self.max_coast_sec = float(max_coast_sec)
        self.max_speed = float(max_target_speed)
        self.max_turn_rate = float(max_turn_rate)

        self.confirm_m = max(1, int(confirm_m))
        self.confirm_n = max(self.confirm_m, int(confirm_n))
        self.tentative_timeout_sec = float(tentative_timeout_sec)
        self.coast_enter_sec = float(coast_enter_sec)
        self.min_birth_sep_m = float(min_birth_sep_m)
        self.kill_radius_m = float(kill_radius_m)
        self.kill_suppression_sec = float(kill_suppression_sec)

        self.tracks: Dict[str, IMMTrack] = {}
        self._next_seq: int = 1
        self.killed_external_ids: Set[str] = set()
        self.recently_killed_zones: List[Tuple[np.ndarray, float]] = []

    def _spawn_track(self, obs: TrackObservation, ts: float, source: str) -> IMMTrack:
        tid = f"trk_{self._next_seq}"
        self._next_seq += 1
        trk = IMMTrack(
            track_id=tid,
            external_id=obs.target_id,
            class_id=obs.class_id,
            tau_maneuver=self.tau_m,
            sigma_maneuver=self.sigma_m,
            sigma_vision=self.sigma_vis,
            sigma_external=self.sigma_ext,
            mahalanobis_gate=self.gate_thresh,
            max_consecutive_rejects=5,
            max_coast_sec=self.max_coast_sec,
            max_target_speed=self.max_speed,
            max_turn_rate=self.max_turn_rate,
            confirm_n=self.confirm_n,
        )
        trk.reset(obs.pos_enu, ts, source=source)
        trk.hit_history.append(True)
        trk.last_hit_check_time = ts
        if self.confirm_m <= 1:
            trk.status = "CONFIRMED"
        else:
            trk.status = "TENTATIVE"
        self.tracks[tid] = trk
        return trk

    def _is_in_killed_zone(self, pos: Tuple[float, float, float], ts: float) -> bool:
        p = np.array(pos, dtype=float)
        for kz_pos, kz_time in self.recently_killed_zones:
            if ts - kz_time <= self.kill_suppression_sec:
                if float(np.linalg.norm(p - kz_pos)) <= self.kill_radius_m:
                    return True
        return False

    def _build_cost_matrix(
        self,
        trk_list: List[IMMTrack],
        obs_list: List[TrackObservation],
        obs_indices: List[int],
        ts: float,
        source: str,
    ) -> np.ndarray:
        n_t = len(trk_list)
        n_o = len(obs_indices)
        cost = np.full((n_t, n_o), np.inf, dtype=float)
        for i, trk in enumerate(trk_list):
            age = max(0.0, ts - trk.last_update_time)
            euclid_max = max(12.0, float(np.linalg.norm(trk.x[3:6])) * age + 8.0)
            for j_col, obs_idx in enumerate(obs_indices):
                obs = obs_list[obs_idx]
                obs_ts = ts if obs.timestamp is None else float(obs.timestamp)
                if not math.isfinite(obs_ts) or not math.isfinite(obs.conf) or trk.last_state_time - obs_ts > 0.45:
                    continue
                if obs.target_id and obs.target_id in self.killed_external_ids:
                    continue
                d2, z_eff = trk.compute_mahalanobis_sq(
                    obs.pos_enu, obs_ts, source=source, stereo=obs.stereo, conf=obs.conf
                )
                euclid = float(np.linalg.norm(z_eff - trk.x[0:3]))
                if not math.isfinite(d2) or d2 > self.gate_thresh or not math.isfinite(euclid) or euclid > euclid_max:
                    continue
                class_pen = 0.0
                if trk.class_id and obs.class_id and trk.class_id != obs.class_id:
                    class_pen = 4.0
                cost[i, j_col] = d2 + class_pen
        return cost

    def update_batch(
        self,
        observations: Sequence[Any],
        timestamp: float,
        source: str = "vision",
    ) -> List[str]:
        """对一批多目标观测执行两阶段匈牙利 GNN 关联、航迹更新与起批。"""
        ts = float(timestamp)
        if not math.isfinite(ts):
            return []

        self.recently_killed_zones = [
            (kp, kt) for (kp, kt) in self.recently_killed_zones if ts - kt <= self.kill_suppression_sec
        ]

        parsed: List[TrackObservation] = []
        for raw in observations:
            coerced = _coerce_observation(raw)
            if coerced is None:
                continue
            if not math.isfinite(coerced.conf):
                continue
            obs_ts = ts if coerced.timestamp is None else float(coerced.timestamp)
            if not math.isfinite(obs_ts) or ts - obs_ts > 0.45:
                continue
            if coerced.target_id and coerced.target_id in self.killed_external_ids:
                continue
            parsed.append(coerced)

        # 1. 所有现存存活航迹前向推演至 ts
        for trk in self.tracks.values():
            if trk.status not in ("KILLED", "DELETED") and ts > trk.last_state_time:
                trk.propagate_to(ts)

        confirmed_list = [t for t in self.tracks.values() if t.status in ("CONFIRMED", "COASTING")]
        tentative_list = [t for t in self.tracks.values() if t.status == "TENTATIVE"]

        unmatched_obs_indices = list(range(len(parsed)))
        matched_track_ids: Set[str] = set()

        # 2. Stage 1: 已确认 / 滑行航迹优先匈牙利关联
        if confirmed_list and unmatched_obs_indices:
            cost_c = self._build_cost_matrix(confirmed_list, parsed, unmatched_obs_indices, ts, source)
            row_c, col_c, _ = hungarian_assignment(cost_c)
            matched_obs_local: Set[int] = set()
            for r_i, c_j in zip(row_c.tolist(), col_c.tolist()):
                trk = confirmed_list[r_i]
                obs = parsed[unmatched_obs_indices[c_j]]
                obs_ts = ts if obs.timestamp is None else float(obs.timestamp)
                ok = trk.update(
                    obs.pos_enu,
                    obs_ts,
                    source=source,
                    stereo=obs.stereo,
                    conf=obs.conf,
                    allow_reset_on_rejects=False,
                    skip_gate=True,
                )
                if ok:
                    if obs.target_id and not trk.external_id:
                        trk.external_id = obs.target_id
                    if obs.class_id and not trk.class_id:
                        trk.class_id = obs.class_id
                    trk.status = "CONFIRMED"
                    matched_track_ids.add(trk.track_id)
                    matched_obs_local.add(c_j)
            unmatched_obs_indices = [
                idx for pos_i, idx in enumerate(unmatched_obs_indices) if pos_i not in matched_obs_local
            ]

        # 3. Stage 2: 暂定航迹匈牙利关联
        if tentative_list and unmatched_obs_indices:
            cost_t = self._build_cost_matrix(tentative_list, parsed, unmatched_obs_indices, ts, source)
            row_t, col_t, _ = hungarian_assignment(cost_t)
            matched_obs_local_t: Set[int] = set()
            for r_i, c_j in zip(row_t.tolist(), col_t.tolist()):
                trk = tentative_list[r_i]
                obs = parsed[unmatched_obs_indices[c_j]]
                obs_ts = ts if obs.timestamp is None else float(obs.timestamp)
                ok = trk.update(
                    obs.pos_enu,
                    obs_ts,
                    source=source,
                    stereo=obs.stereo,
                    conf=obs.conf,
                    allow_reset_on_rejects=False,
                    skip_gate=True,
                )
                if ok:
                    if obs.target_id and not trk.external_id:
                        trk.external_id = obs.target_id
                    if obs.class_id and not trk.class_id:
                        trk.class_id = obs.class_id
                    trk.hit_history.append(True)
                    trk.last_hit_check_time = ts
                    if sum(1 for h in trk.hit_history if h) >= self.confirm_m:
                        trk.status = "CONFIRMED"
                    matched_track_ids.add(trk.track_id)
                    matched_obs_local_t.add(c_j)
            unmatched_obs_indices = [
                idx for pos_i, idx in enumerate(unmatched_obs_indices) if pos_i not in matched_obs_local_t
            ]

        # 对未命中的暂定航迹记录 miss (M-of-N 窗口)
        for trk in tentative_list:
            if trk.track_id not in matched_track_ids and (ts - trk.last_hit_check_time) > 0.04:
                trk.hit_history.append(False)
                trk.consecutive_misses += 1
                trk.last_hit_check_time = ts

        # 4. 剩余未关联观测起批 (重复起批抑制 + 击毁区抑制)
        for obs_idx in unmatched_obs_indices:
            obs = parsed[obs_idx]
            obs_ts = ts if obs.timestamp is None else float(obs.timestamp)
            if obs.target_id and obs.target_id in self.killed_external_ids:
                continue
            if self._is_in_killed_zone(obs.pos_enu, ts):
                continue
            obs_p = np.array(obs.pos_enu, dtype=float)
            too_close = False
            for trk in self.tracks.values():
                if trk.status in ("KILLED", "DELETED"):
                    continue
                if float(np.linalg.norm(obs_p - trk.x[0:3])) < self.min_birth_sep_m:
                    too_close = True
                    break
            if too_close:
                continue
            new_trk = self._spawn_track(obs, obs_ts, source=source)
            matched_track_ids.add(new_trk.track_id)

        self.step_maintenance(ts)
        return sorted(matched_track_ids)

    def step_maintenance(self, current_time: float) -> None:
        """执行滑行推演、遮挡降级与超时航迹清理。"""
        now = float(current_time)
        if not math.isfinite(now):
            return
        self.recently_killed_zones = [
            (kp, kt) for (kp, kt) in self.recently_killed_zones if now - kt <= self.kill_suppression_sec
        ]
        to_delete: List[str] = []
        for tid, trk in self.tracks.items():
            if trk.status in ("KILLED", "DELETED"):
                to_delete.append(tid)
                continue
            if now > trk.last_state_time:
                trk.propagate_to(now, coast_damping=True)
            age = now - trk.last_update_time
            if trk.status == "TENTATIVE":
                hits = sum(1 for h in trk.hit_history if h)
                if age > self.tentative_timeout_sec or (
                    len(trk.hit_history) >= self.confirm_n
                    and hits < self.confirm_m
                    and trk.consecutive_misses >= 3
                ):
                    trk.status = "DELETED"
                    to_delete.append(tid)
            else:
                if age > self.max_coast_sec:
                    trk.status = "DELETED"
                    to_delete.append(tid)
                elif age > self.coast_enter_sec:
                    trk.status = "COASTING"
                else:
                    trk.status = "CONFIRMED"

        for tid in to_delete:
            self.tracks.pop(tid, None)

    def mark_killed(
        self,
        target_id: Optional[str] = None,
        pos_enu: Optional[Tuple[float, float, float]] = None,
        now_sec: Optional[float] = None,
        extra_track_ids: Optional[Sequence[str]] = None,
    ) -> List[str]:
        """依据 target_id、空间近邻或指派航迹 ID 终结目标并抑制原地重生。"""
        if now_sec is not None and math.isfinite(float(now_sec)):
            ts = float(now_sec)
        else:
            finite_times = [
                t.last_state_time
                for t in self.tracks.values()
                if math.isfinite(t.last_state_time)
            ]
            finite_times.extend(
                kt for (_, kt) in self.recently_killed_zones if math.isfinite(kt)
            )
            ts = max(finite_times) if finite_times else 0.0
        tid_str = str(target_id).strip() if target_id else ""
        if tid_str:
            self.killed_external_ids.add(tid_str)

        pos_arr: Optional[np.ndarray] = None
        if pos_enu is not None:
            px, py, pz = float(pos_enu[0]), float(pos_enu[1]), float(pos_enu[2])
            if math.isfinite(px) and math.isfinite(py) and math.isfinite(pz):
                pos_arr = np.array([px, py, pz], dtype=float)
                self.recently_killed_zones.append((pos_arr, ts))

        extra_set = set(extra_track_ids) if extra_track_ids else set()
        killed_tracks: List[str] = []
        for key, trk in list(self.tracks.items()):
            match_id = bool(tid_str and (trk.track_id == tid_str or trk.external_id == tid_str))
            match_extra = trk.track_id in extra_set
            pos_id_compatible = (not tid_str) or (not trk.external_id) or (trk.external_id == tid_str)
            match_pos = (
                pos_arr is not None
                and pos_id_compatible
                and float(np.linalg.norm(trk.x[0:3] - pos_arr)) <= self.kill_radius_m
            )
            if match_id or match_extra or match_pos:
                if trk.external_id:
                    self.killed_external_ids.add(trk.external_id)
                self.recently_killed_zones.append((trk.x[0:3].copy(), ts))
                trk.status = "KILLED"
                self.tracks.pop(key, None)
                killed_tracks.append(key)
        return killed_tracks

    def get_confirmed_tracks(self, current_time: Optional[float] = None) -> List[IMMTrack]:
        """获取所有处于 CONFIRMED / COASTING 且在滑行保鲜期内的航迹。"""
        out: List[IMMTrack] = []
        for trk in self.tracks.values():
            if trk.status not in ("CONFIRMED", "COASTING"):
                continue
            if current_time is not None:
                ct = float(current_time)
                if not math.isfinite(ct) or not trk.is_valid(ct):
                    continue
            out.append(trk)
        out.sort(key=lambda t: t.track_id)
        return out

    def get_active_tracks(self, include_tentative: bool = False) -> List[IMMTrack]:
        """获取当前所有活跃航迹。"""
        valid_states = ("CONFIRMED", "COASTING", "TENTATIVE") if include_tentative else ("CONFIRMED", "COASTING")
        out = [t for t in self.tracks.values() if t.status in valid_states]
        out.sort(key=lambda t: t.track_id)
        return out

    def diagnostics(self, current_time: float) -> List[Dict[str, object]]:
        """返回所有活跃航迹的结构化诊断列表。"""
        return [t.diagnostics(current_time) for t in self.get_active_tracks(include_tentative=True)]


class ManeuveringTargetEstimator:
    """九维机动目标状态估计与曲率轨迹预测器（内部委托给三模型 IMMTrack，100% 向后兼容）。"""

    def __init__(
        self,
        tau_maneuver: float = 1.35,
        sigma_maneuver: float = 3.6,
        sigma_vision: float = 0.36,
        sigma_external: float = 0.55,
        mahalanobis_gate: float = 18.0,
        max_consecutive_rejects: int = 5,
        max_coast_sec: float = 3.0,
        max_target_speed: float = 25.0,
        max_turn_rate: float = 1.2,
    ) -> None:
        self._imm = IMMTrack(
            track_id="primary",
            tau_maneuver=tau_maneuver,
            sigma_maneuver=sigma_maneuver,
            sigma_vision=sigma_vision,
            sigma_external=sigma_external,
            mahalanobis_gate=mahalanobis_gate,
            max_consecutive_rejects=max_consecutive_rejects,
            max_coast_sec=max_coast_sec,
            max_target_speed=max_target_speed,
            max_turn_rate=max_turn_rate,
        )
        self._imm.status = "CONFIRMED"
        self.tau_m: float = self._imm.tau_m
        self.sigma_m: float = self._imm.sigma_m
        self.sigma_vis: float = self._imm.sigma_vis
        self.sigma_ext: float = self._imm.sigma_ext
        self.gate_thresh: float = self._imm.gate_thresh
        self.max_rejects: int = self._imm.max_rejects
        self.max_coast_sec: float = self._imm.max_coast_sec
        self.max_speed: float = self._imm.max_speed
        self.max_turn_rate: float = self._imm.max_turn_rate
        self.H: np.ndarray = self._imm.H
        self._sync_from_imm()

    def _sync_from_imm(self) -> None:
        self.x: np.ndarray = self._imm.x
        self.P: np.ndarray = self._imm.P
        self.last_update_time: float = self._imm.last_update_time
        self.last_state_time: float = self._imm.last_state_time
        self.last_vision_time: float = self._imm.last_vision_time
        self.last_source: str = self._imm.last_source
        self.turn_rate_est: float = self._imm.turn_rate_est
        self.reject_count: int = self._imm.reject_count
        self.update_count: int = self._imm.update_count
        self.last_mahalanobis_sq: float = self._imm.last_mahalanobis_sq

    @property
    def initialized(self) -> bool:
        """返回底层 IMM 跟踪器初始化状态（可读写以支持单目标击毁清零）。"""
        return self._imm.initialized

    @initialized.setter
    def initialized(self, value: bool) -> None:
        self._imm.initialized = bool(value)

    @property
    def imm_probs(self) -> Dict[str, float]:
        """返回三模型 IMM 概率分布。"""
        return self._imm.imm_probs

    def reset(self, pos_enu: Tuple[float, float, float], timestamp: float, source: str = "init") -> None:
        """以初始三维位置重置滤波器状态与协方差。"""
        self._imm.reset(pos_enu, timestamp, source=source)
        self._imm.status = "CONFIRMED"
        self._sync_from_imm()

    def _singer_blocks(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        return _singer_blocks_param(dt, self.tau_m, self.sigma_m)

    def propagate_to(self, timestamp: float) -> None:
        """将内部状态与协方差前向传播至指定时间戳。"""
        self._imm.propagate_to(timestamp)
        self._sync_from_imm()

    def _clamp_kinematics(self) -> None:
        self._imm._clamp_vec_kinematics(self._imm.x, has_acc=True)
        self._sync_from_imm()

    def update(
        self,
        pos_enu: Tuple[float, float, float],
        timestamp: float,
        source: str = "vision",
    ) -> bool:
        """融合一帧三维位置观测。"""
        ok = self._imm.update(
            pos_enu,
            timestamp,
            source=source,
            stereo=True,
            conf=1.0,
            allow_reset_on_rejects=True,
            skip_gate=False,
        )
        self._imm.status = "CONFIRMED"
        self._sync_from_imm()
        return ok

    def is_valid(self, current_time: float) -> bool:
        """判断当前目标状态是否在有效保鲜与外推窗口内。"""
        return self._imm.is_valid(current_time)

    def get_kinematics_at(self, query_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取在 query_time 时刻的目标估计位置、速度、加速度（不修改内部滤波状态）。"""
        return self._imm.get_kinematics_at(query_time)

    def predict_pv_ahead(
        self,
        dt_ahead: float,
        from_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """预测未来 dt_ahead 秒后的三维位置与速度。"""
        return self._imm.predict_pv_ahead(dt_ahead, from_state=from_state)

    def predict_path(
        self,
        horizons_sec: Sequence[float] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0),
    ) -> List[List[float]]:
        """返回多步未来位置预测序列。"""
        return self._imm.predict_path(horizons_sec=horizons_sec)

    def sync_from_track(self, track: IMMTrack) -> None:
        """从多目标航迹库中的主目标同步状态供向后兼容诊断。"""
        self._imm.x = track.x.copy()
        self._imm.P = track.P.copy()
        self._imm.mu = track.mu.copy()
        self._imm.x_cv = track.x_cv.copy()
        self._imm.P_cv = track.P_cv.copy()
        self._imm.x_ctrv = track.x_ctrv.copy()
        self._imm.P_ctrv = track.P_ctrv.copy()
        self._imm.x_jink = track.x_jink.copy()
        self._imm.P_jink = track.P_jink.copy()
        self._imm.initialized = track.initialized
        self._imm.status = track.status
        self._imm.last_update_time = track.last_update_time
        self._imm.last_state_time = track.last_state_time
        self._imm.last_vision_time = track.last_vision_time
        self._imm.last_source = track.last_source
        self._imm.turn_rate_est = track.turn_rate_est
        self._imm.reject_count = track.reject_count
        self._imm.update_count = track.update_count
        self._imm.last_mahalanobis_sq = track.last_mahalanobis_sq
        self._sync_from_imm()

    def diagnostics(self, current_time: float) -> Dict[str, object]:
        """输出结构化诊断字典供可观测性追踪。"""
        d = self._imm.diagnostics(current_time)
        return {
            "initialized": d["initialized"],
            "valid": d["valid"],
            "source": d["source"],
            "age_sec": d["age_sec"],
            "pos_enu": d["pos_enu"],
            "vel_enu": d["vel_enu"],
            "acc_enu": d["acc_enu"],
            "turn_rate_rads": d["turn_rate_rads"],
            "pos_rmse_std_m": d["pos_rmse_std_m"],
            "mahalanobis_sq": d["mahalanobis_sq"],
            "updates": d["updates"],
            "imm_probs": d["imm_probs"],
        }
