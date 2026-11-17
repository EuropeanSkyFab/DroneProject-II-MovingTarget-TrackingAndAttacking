# -*- coding: utf-8 -*-
"""
estimator.py —— 机动目标九维 Singer-CA / CTRV 多源融合状态估计与曲率轨迹预测器

坐标系约定：全局 ENU（东-北-天，米 / 秒），不做任何 NED 转换。

数学模型：
  状态向量 x = [px, py, pz, vx, vy, vz, ax, ay, az]^T in R^9
  运动模型：三轴解耦离散时间 Singer 相关加速度模型（机动时间常数 tau_m），
            耦合水平面瞬时转弯角速度估计 omega = (vx*ay - vy*ax)/(vx^2 + vy^2) 用于曲率外推。
  观测融合：支持异构异步传感器（高精度机载视觉 vision 与外部雷达引导 external），
            采用 Joseph 对称正定协方差更新与马氏距离 chi^2 野值门控。
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


class ManeuveringTargetEstimator:
    """九维机动目标卡尔曼状态估计与曲率轨迹预测器。"""

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
        assert tau_maneuver > 0.05, "tau_maneuver must be positive"
        assert sigma_maneuver > 0.0, "sigma_maneuver must be positive"
        assert sigma_vision > 0.0 and sigma_external > 0.0, "measurement noise must be positive"

        self.tau_m: float = float(tau_maneuver)
        self.sigma_m: float = float(sigma_maneuver)
        self.sigma_vis: float = float(sigma_vision)
        self.sigma_ext: float = float(sigma_external)
        self.gate_thresh: float = float(mahalanobis_gate)
        self.max_rejects: int = int(max_consecutive_rejects)
        self.max_coast_sec: float = float(max_coast_sec)
        self.max_speed: float = float(max_target_speed)
        self.max_turn_rate: float = float(max_turn_rate)

        self.x: np.ndarray = np.zeros(9, dtype=float)
        self.P: np.ndarray = np.eye(9, dtype=float) * 25.0
        self.H: np.ndarray = np.zeros((3, 9), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.initialized: bool = False
        self.last_update_time: float = 0.0
        self.last_state_time: float = 0.0
        self.last_vision_time: float = -1e9
        self.last_source: str = "none"
        self.turn_rate_est: float = 0.0
        self.reject_count: int = 0
        self.update_count: int = 0
        self.last_mahalanobis_sq: float = 0.0

    def reset(self, pos_enu: Tuple[float, float, float], timestamp: float, source: str = "init") -> None:
        """以初始三维位置重置滤波器状态与协方差。"""
        px, py, pz = float(pos_enu[0]), float(pos_enu[1]), float(pos_enu[2])
        self.x = np.array([px, py, pz, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.P = np.diag([1.0, 1.0, 1.0, 9.0, 9.0, 4.0, 12.0, 12.0, 6.0]).astype(float)
        self.initialized = True
        self.last_update_time = float(timestamp)
        self.last_state_time = float(timestamp)
        if source == "vision":
            self.last_vision_time = float(timestamp)
        self.last_source = source
        self.turn_rate_est = 0.0
        self.reject_count = 0
        self.update_count = 1
        self.last_mahalanobis_sq = 0.0

    def _singer_blocks(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """构造九维三轴 Singer 离散状态转移矩阵 F(dt) 与过程噪声协方差 Q(dt)。"""
        dt = max(0.0, min(float(dt), 5.0))
        if dt < 1e-6:
            return np.eye(9, dtype=float), np.zeros((9, 9), dtype=float)

        alpha = 1.0 / self.tau_m
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

        # 精确 Singer 过程噪声分量
        s2 = 2.0 * alpha * (self.sigma_m ** 2)
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
        # Z 轴机动强度通常低于水平面，赋予 0.45 权重抑制高度抖动
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

    def propagate_to(self, timestamp: float) -> None:
        """将内部状态与协方差前向传播至指定时间戳。"""
        if not self.initialized:
            return
        dt = float(timestamp) - self.last_state_time
        if dt <= 1e-5:
            return
        F, Q = self._singer_blocks(dt)
        self.x = F @ self.x
        self._clamp_kinematics()
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        self.last_state_time = float(timestamp)

    def _clamp_kinematics(self) -> None:
        """施加物理速度与加速度上界保护，防止极端测量脉冲引发数值发散。"""
        vel = self.x[3:6]
        spd = float(np.linalg.norm(vel))
        if spd > self.max_speed:
            self.x[3:6] = vel * (self.max_speed / spd)
        acc = self.x[6:9]
        acc_norm = float(np.linalg.norm(acc))
        max_acc = 15.0
        if acc_norm > max_acc:
            self.x[6:9] = acc * (max_acc / acc_norm)

    def update(
        self,
        pos_enu: Tuple[float, float, float],
        timestamp: float,
        source: str = "vision",
    ) -> bool:
        """
        融合一帧三维位置观测。

        返回：
          True 表示观测被接纳并更新状态；False 表示非法值、严重乱序或被马氏门控拒绝。
        """
        px, py, pz = float(pos_enu[0]), float(pos_enu[1]), float(pos_enu[2])
        ts = float(timestamp)
        if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz) and math.isfinite(ts)):
            return False

        if not self.initialized or (ts - self.last_update_time) > self.max_coast_sec * 1.5:
            self.reset((px, py, pz), ts, source=source)
            return True

        # 丢弃严重过期的乱序包 (> 0.45s)
        if ts < self.last_state_time - 0.45:
            return False

        if ts > self.last_state_time:
            self.propagate_to(ts)
            z_eff = np.array([px, py, pz], dtype=float)
        else:
            # 轻微延迟包 (0 < lag <= 0.45s)：使用速度一阶时间对齐补偿观测到当前滤波时刻
            lag = self.last_state_time - ts
            z_eff = np.array([px, py, pz], dtype=float) + self.x[3:6] * lag + 0.5 * self.x[6:9] * (lag ** 2)

        # 异构信源自适应噪声：当高精度视觉锁定时，降低粗糙雷达权重防止拉偏
        if source == "vision":
            sigma = self.sigma_vis
        else:
            vis_recent = (max(ts, self.last_state_time) - self.last_vision_time) < 0.6
            sigma = self.sigma_ext * (2.5 if vis_recent else 1.0)

        R = np.eye(3, dtype=float) * (sigma ** 2)
        R[2, 2] *= 1.35  # 高度方向适当放宽噪声容差

        y = z_eff - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + R
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return False

        d_m2 = float(y.T @ S_inv @ y)
        self.last_mahalanobis_sq = d_m2

        if d_m2 > self.gate_thresh:
            self.reject_count += 1
            if self.reject_count >= self.max_rejects:
                # 连续多帧大幅跳变：判定为目标真实机动瞬移或新目标捕获，重置跟踪器
                self.reset((px, py, pz), max(ts, self.last_state_time), source=source)
                return True
            return False

        self.reject_count = 0
        K = self.P @ self.H.T @ S_inv
        self.x = self.x + K @ y
        self._clamp_kinematics()

        # Joseph 对称正定协方差更新公式
        I_KH = np.eye(9, dtype=float) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        # 更新水平瞬时转弯角速度估计 omega = (vx*ay - vy*ax) / v_h^2
        vx, vy = float(self.x[3]), float(self.x[4])
        ax, ay = float(self.x[6]), float(self.x[7])
        vh_sq = vx * vx + vy * vy
        if vh_sq > 0.64:
            omega_raw = (vx * ay - vy * ax) / vh_sq
            omega_raw = max(-self.max_turn_rate, min(self.max_turn_rate, omega_raw))
            self.turn_rate_est = 0.72 * self.turn_rate_est + 0.28 * omega_raw
        else:
            self.turn_rate_est *= 0.85

        now_mark = max(ts, self.last_state_time)
        self.last_update_time = now_mark
        if source == "vision":
            self.last_vision_time = now_mark
        self.last_source = source
        self.update_count += 1
        return True

    def is_valid(self, current_time: float) -> bool:
        """判断当前目标状态是否在有效保鲜与外推窗口内。"""
        if not self.initialized:
            return False
        return (float(current_time) - self.last_update_time) <= self.max_coast_sec

    def get_kinematics_at(self, query_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取在 query_time 时刻的目标估计位置、速度、加速度（不修改内部滤波状态）。"""
        dt = max(0.0, float(query_time) - self.last_state_time)
        if dt < 1e-4:
            return self.x[0:3].copy(), self.x[3:6].copy(), self.x[6:9].copy()
        p, v = self.predict_pv_ahead(dt, from_state=self.x)
        alpha = 1.0 / self.tau_m
        a = self.x[6:9] * math.exp(-alpha * min(dt, 3.0))
        return p, v, a

    def predict_pv_ahead(
        self,
        dt_ahead: float,
        from_state: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        融合 CTRV 恒定转弯率圆弧积分与垂直 Singer 阻尼外推，预测未来 dt_ahead 秒后的三维位置与速度。
        """
        s = self.x if from_state is None else from_state
        tau = max(0.0, min(float(dt_ahead), 8.0))
        px, py, pz = float(s[0]), float(s[1]), float(s[2])
        vx, vy, vz = float(s[3]), float(s[4]), float(s[5])
        ax, ay, az = float(s[6]), float(s[7]), float(s[8])

        if tau < 1e-5:
            return np.array([px, py, pz], dtype=float), np.array([vx, vy, vz], dtype=float)

        vh = math.hypot(vx, vy)
        omega = self.turn_rate_est
        # 长时间外推时平滑衰减角速度，防止过卷成死圈
        omega_eff = omega * math.exp(-0.22 * tau)

        if vh > 0.9 and abs(omega_eff) > 0.035:
            # 精确 CTRV 水平圆弧积分 + 切向线加速度修正
            th = omega_eff * tau
            sin_th = math.sin(th)
            cos_th = math.cos(th)
            dx = vx * (sin_th / omega_eff) + vy * ((cos_th - 1.0) / omega_eff)
            dy = vx * ((1.0 - cos_th) / omega_eff) + vy * (sin_th / omega_eff)
            nvx = vx * cos_th - vy * sin_th
            nvy = vx * sin_th + vy * cos_th

            # 叠加沿速度切向的受限加减速分量
            a_tang = (vx * ax + vy * ay) / vh
            tau_a = min(tau, 1.5)
            ux, uy = nvx / max(math.hypot(nvx, nvy), 1e-6), nvy / max(math.hypot(nvx, nvy), 1e-6)
            dx += 0.5 * a_tang * (tau_a ** 2) * ux
            dy += 0.5 * a_tang * (tau_a ** 2) * uy
            spd_new = max(0.0, min(self.max_speed, vh + a_tang * tau_a))
            nvx, nvy = ux * spd_new, uy * spd_new
        else:
            # 直线 Singer 相关加速度解析积分
            alpha = 1.0 / self.tau_m
            eadt = math.exp(-alpha * tau)
            f_pa = (alpha * tau - 1.0 + eadt) / (alpha * alpha)
            f_va = (1.0 - eadt) / alpha
            dx = vx * tau + ax * f_pa
            dy = vy * tau + ay * f_pa
            nvx = vx + ax * f_va
            nvy = vy + ay * f_va

        # Z 轴带软阻尼边界外推（避免长时间外推预测目标钻入地下）
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

    def diagnostics(self, current_time: float) -> Dict[str, object]:
        """输出结构化诊断字典供可观测性追踪。"""
        age = float(current_time) - self.last_update_time if self.initialized else -1.0
        pos_std = float(math.sqrt(max(0.0, float(np.trace(self.P[0:3, 0:3])) / 3.0)))
        return {
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
        }
