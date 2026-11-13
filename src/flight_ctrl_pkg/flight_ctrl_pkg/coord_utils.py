# -*- coding: utf-8 -*-
"""
坐标系与姿态转换工具模块（课题二飞控桥接专用）

全系统约定（严格遵守《课题二内部接口规范 V3.0》）：
  - 上层模块（中枢、视觉、规划）统一使用全局 ENU（东-北-天，Z 向上为正），机体 FLU（前-左-上）
  - PX4 飞控底层使用局部 NED（北-东-地，Z 向下为正），机体 FRD（前-右-下）
  - 全部 ENU <-> NED 转换仅在此层完成
"""

import math
from typing import Sequence, Tuple


def wrap_pi(angle: float) -> float:
    """将角度规范化到 [-pi, pi]。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def enu_to_ned_pos(
    x_east: float,
    y_north: float,
    z_up: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> Tuple[float, float, float]:
    """全局 ENU 位置 -> PX4 局部 NED 位置（扣除多机出生点世界偏移）。"""
    local_east = float(x_east) - float(offset_x)
    local_north = float(y_north) - float(offset_y)
    down = -float(z_up)
    return local_north, local_east, down


def ned_to_enu_pos(
    north: float,
    east: float,
    down: float,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> Tuple[float, float, float]:
    """PX4 局部 NED 位置 -> 全局 ENU 位置（加上多机出生点世界偏移）。"""
    world_east = float(east) + float(offset_x)
    world_north = float(north) + float(offset_y)
    up = -float(down)
    return world_east, world_north, up


def ned_to_enu_vel(vn: float, ve: float, vd: float) -> Tuple[float, float, float]:
    """NED 速度 -> ENU 速度。"""
    return float(ve), float(vn), -float(vd)


def yaw_enu_to_ned(yaw_enu_rad: float) -> float:
    """ENU 偏航角（东=0, 北=+pi/2）-> NED 偏航角（北=0, 东=+pi/2）。"""
    return wrap_pi(math.pi / 2.0 - float(yaw_enu_rad))


def yaw_ned_to_enu(yaw_ned_rad: float) -> float:
    """NED 偏航角（北=0, 东=+pi/2）-> ENU 偏航角（东=0, 北=+pi/2）。"""
    return wrap_pi(math.pi / 2.0 - float(yaw_ned_rad))


def _quat_wxyz_to_matrix(w: float, x: float, y: float, z: float) -> Tuple[Tuple[float, float, float], ...]:
    """Hamilton 四元数 (w, x, y, z) -> 3x3 旋转矩阵。"""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    w, x, y, z = w / n, x / n, y / n, z / n
    return (
        (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
        (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
        (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    )


def _matrix_to_quat_xyzw(m: Sequence[Sequence[float]]) -> Tuple[float, float, float, float]:
    """3x3 正交旋转矩阵 -> ROS 标准四元数 (x, y, z, w)。"""
    m00, m01, m02 = m[0]
    m10, m11, m12 = m[1]
    m20, m21, m22 = m[2]
    trace = m00 + m11 + m22

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def quat_frd_ned_to_flu_enu(q_wxyz: Sequence[float]) -> Tuple[float, float, float, float]:
    """
    将 PX4 VehicleAttitude.q (w, x, y, z)（机体 FRD 相对世界 NED）
    转换为 ROS Odometry 四元数 (x, y, z, w)（机体 FLU 相对世界 ENU）。

    推导：R_{ENU <- FLU} = C_{ENU <- NED} @ R_{NED <- FRD} @ C_{FRD <- FLU}
    其中：
      C_{ENU <- NED} = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
      C_{FRD <- FLU} = diag(1, -1, -1)
    """
    if len(q_wxyz) != 4:
        raise ValueError(f"Expected 4-element quaternion, got {len(q_wxyz)}")
    r = _quat_wxyz_to_matrix(
        float(q_wxyz[0]), float(q_wxyz[1]), float(q_wxyz[2]), float(q_wxyz[3])
    )
    # 先左乘 C_{ENU <- NED}（交换0/1行，第2行取反），再右乘 diag(1, -1, -1)（第1/2列取反）
    r_enu_flu = (
        (r[1][0], -r[1][1], -r[1][2]),
        (r[0][0], -r[0][1], -r[0][2]),
        (-r[2][0], r[2][1], r[2][2]),
    )
    return _matrix_to_quat_xyzw(r_enu_flu)


def clamp_step_3d(
    cur_n: float,
    cur_e: float,
    cur_d: float,
    tgt_n: float,
    tgt_e: float,
    tgt_d: float,
    max_step: float,
) -> Tuple[float, float, float]:
    """基于三维向量距离的比例平滑步长限幅（NED 空间）。"""
    assert max_step > 0.0, "max_step must be positive"
    dn = float(tgt_n) - float(cur_n)
    de = float(tgt_e) - float(cur_e)
    dd = float(tgt_d) - float(cur_d)
    dist = math.sqrt(dn * dn + de * de + dd * dd)
    if dist <= max_step or dist < 1e-9:
        return float(tgt_n), float(tgt_e), float(tgt_d)
    scale = max_step / dist
    return cur_n + dn * scale, cur_e + de * scale, cur_d + dd * scale


def slew_yaw_ned(cur_yaw: float, tgt_yaw: float, max_delta: float) -> float:
    """带最大步长限幅的最短路径偏航角更新（NED 弧度）。"""
    assert max_delta >= 0.0, "max_delta must be non-negative"
    diff = wrap_pi(float(tgt_yaw) - float(cur_yaw))
    if abs(diff) <= max_delta:
        return wrap_pi(float(tgt_yaw))
    return wrap_pi(float(cur_yaw) + math.copysign(max_delta, diff))


def compute_shadow_and_ff(
    shadow_ned: Tuple[float, float, float],
    real_ned: Tuple[float, float, float],
    tgt_ned: Tuple[float, float, float],
    tgt_vel_ned: Tuple[float, float, float],
    max_step: float,
    max_tether: float,
    max_speed: float,
    gain_p: float,
    deadband_m: float,
    max_vz_up: float,
    max_vz_dn: float,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    结合防积分抗饱和拴绳（anti-windup tether）与速度前馈，计算平滑位置/速度设定点（NED）。

    返回: ((sn, se, sd), (vn, ve, vd))
    """
    sn, se, sd = shadow_ned
    rn, re, rd = real_ned
    tn, te, td = tgt_ned

    # 1. Anti-windup tether：防止影子位置脱离开实际飞机太远造成拐弯滞后
    t_dn = float(sn) - float(rn)
    t_de = float(se) - float(re)
    t_dd = float(sd) - float(rd)
    t_dist = math.sqrt(t_dn * t_dn + t_de * t_de + t_dd * t_dd)
    if t_dist > max_tether and t_dist > 1e-9:
        sc = max_tether / t_dist
        sn = rn + t_dn * sc
        se = re + t_de * sc
        sd = rd + t_dd * sc

    # 2. 位置设定点按 3D 矢量步长平滑推进
    new_shadow = clamp_step_3d(sn, se, sd, tn, te, td, max_step)

    # 3. 速度前馈：目标估计速度 + 朝向目标误差的比例闭合速度
    edn = float(tn) - float(rn)
    ede = float(te) - float(re)
    edd = float(td) - float(rd)
    edist = math.sqrt(edn * edn + ede * ede + edd * edd)

    if edist <= deadband_m:
        vn_err, ve_err, vd_err = 0.0, 0.0, 0.0
    else:
        v_mag = min(float(max_speed), edist * float(gain_p))
        inv = v_mag / edist
        vn_err, ve_err, vd_err = edn * inv, ede * inv, edd * inv

    vn = float(tgt_vel_ned[0]) + vn_err
    ve = float(tgt_vel_ned[1]) + ve_err
    vd = float(tgt_vel_ned[2]) + vd_err

    # 总速度矢量限幅 + 垂直速率独立限幅（NED 中向上为负、向下为正）
    vd = max(-float(max_vz_up), min(float(max_vz_dn), vd))
    h_spd = math.hypot(vn, ve)
    if h_spd > max_speed and h_spd > 1e-9:
        h_sc = max_speed / h_spd
        vn *= h_sc
        ve *= h_sc

    return new_shadow, (vn, ve, vd)


def compute_yawspeed_ned(
    cur_yaw: float,
    tgt_yaw: float,
    tau: float = 0.25,
    max_yaw_rate_rad: float = math.pi,
) -> float:
    """基于最小弧度差计算平滑偏航角速度前馈（NED 弧度/秒）。"""
    assert tau > 0.0, "tau must be positive"
    assert max_yaw_rate_rad >= 0.0, "max_yaw_rate_rad must be non-negative"
    err = wrap_pi(float(tgt_yaw) - float(cur_yaw))
    rate = err / float(tau)
    return max(-float(max_yaw_rate_rad), min(float(max_yaw_rate_rad), rate))


def compute_gimbal_pitch_ned(
    drone_z_enu: float,
    target_z_enu: float,
    horiz_dist_m: float,
    min_pitch_deg: float = -55.0,
    max_pitch_deg: float = 15.0,
) -> float:
    """计算由无人机指向空中目标的三维俯仰角（度，向下为负）。"""
    dz = float(target_z_enu) - float(drone_z_enu)
    hd = max(0.5, float(horiz_dist_m))
    pitch_deg = math.degrees(math.atan2(dz, hd))
    return max(float(min_pitch_deg), min(float(max_pitch_deg), pitch_deg))


def compute_shadow_ff_accel(
    shadow_ned: Tuple[float, float, float],
    real_ned: Tuple[float, float, float],
    tgt_ned: Tuple[float, float, float],
    tgt_vel_ned: Tuple[float, float, float],
    prev_vel_cmd_ned: Tuple[float, float, float],
    prev_acc_cmd_ned: Tuple[float, float, float],
    dt: float,
    max_step: float,
    max_tether: float,
    max_speed: float,
    gain_p: float,
    deadband_m: float,
    max_vz_up: float,
    max_vz_dn: float,
    max_acc_hor: float = 8.0,
    max_acc_vert: float = 5.0,
    acc_alpha: float = 0.28,
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    """
    计算 3-DOF 位置 + 速度 + 平滑加速度前馈设定点（NED），匹配 PX4 PositionControl 级联前馈律。

    返回: ((sn, se, sd), (vn, ve, vd), (an, ae, ad))
    """
    new_shadow, vel_cmd = compute_shadow_and_ff(
        shadow_ned,
        real_ned,
        tgt_ned,
        tgt_vel_ned,
        max_step=max_step,
        max_tether=max_tether,
        max_speed=max_speed,
        gain_p=gain_p,
        deadband_m=deadband_m,
        max_vz_up=max_vz_up,
        max_vz_dn=max_vz_dn,
    )
    dt_safe = max(1e-3, float(dt))
    raw_an = (vel_cmd[0] - float(prev_vel_cmd_ned[0])) / dt_safe
    raw_ae = (vel_cmd[1] - float(prev_vel_cmd_ned[1])) / dt_safe
    raw_ad = (vel_cmd[2] - float(prev_vel_cmd_ned[2])) / dt_safe

    an = (1.0 - acc_alpha) * float(prev_acc_cmd_ned[0]) + acc_alpha * raw_an
    ae = (1.0 - acc_alpha) * float(prev_acc_cmd_ned[1]) + acc_alpha * raw_ae
    ad = (1.0 - acc_alpha) * float(prev_acc_cmd_ned[2]) + acc_alpha * raw_ad

    ah = math.hypot(an, ae)
    if ah > max_acc_hor and ah > 1e-9:
        sc = float(max_acc_hor) / ah
        an *= sc
        ae *= sc
    ad = max(-float(max_acc_vert), min(float(max_acc_vert), ad))
    return new_shadow, vel_cmd, (an, ae, ad)
