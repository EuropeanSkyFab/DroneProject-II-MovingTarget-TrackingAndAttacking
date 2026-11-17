# -*- coding: utf-8 -*-
"""
geo_utils.py —— 机载视觉三维几何反投影与多机立体射线三角测量库

坐标系约定（严格遵守《课题二内部接口规范 V3.0》）：
  - 世界系：全局 ENU（东-北-天，Z 轴向上为正），frame_id = "world"
  - 严禁做 ENU <-> NED 转换（NED 转换由飞控桥接统一负责）
  - 支持两种三维定位机制：
      1) 多机异构视角三维异面射线最小二乘三角测量 (triangulate_rays)：
         当 >=2 架无人机在同步时间窗内捕获目标且视差角充足时，无需假定目标高度即可直接解算空中机动靶真实三维坐标 (x, y, z)。
      2) 单目针孔射线-参考高度面求交 (pixel_to_world)：
         兼容单机工作与地面/定高靶标，支持融合相机相对机体安装杆臂偏移 cam_offset_body。
"""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np


def quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """四元数 (x, y, z, w) -> 3x3 右手正交旋转矩阵。"""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return np.eye(3, dtype=float)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """ZYX 欧拉角（弧度）-> 旋转矩阵 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return Rz @ Ry @ Rx


def camera_mount_matrix(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    """
    相机安装外参：相机光学坐标系 (x右, y下, z沿光轴前) -> 机体 FLU 坐标系 (x前, y左, z上) 的旋转矩阵 R_bc。
    默认 (0,0,0) 对应标准垂直朝下相机；x500_down_cam 前斜下视 35° 对应 [0.95993109, 0.0, -1.57079633]。
    """
    base = np.array([
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
    ], dtype=float)
    if abs(roll) < 1e-12 and abs(pitch) < 1e-12 and abs(yaw) < 1e-12:
        return base
    return euler_to_matrix(roll, pitch, yaw) @ base


def pixel_to_ray_world(
    u: float,
    v: float,
    camera_info_k: Sequence[float],
    drone_position: Tuple[float, float, float],
    drone_quaternion: Tuple[float, float, float, float],
    R_bc: np.ndarray,
    cam_offset_body: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    将像素坐标 (u, v) 转换为全局 ENU 世界系下的归一化三维观测射线 (origin_enu, unit_dir_enu)。
    """
    fx, fy, cx, cy = float(camera_info_k[0]), float(camera_info_k[4]), float(camera_info_k[2]), float(camera_info_k[5])
    if fx <= 1e-9 or fy <= 1e-9:
        return None
    if not (math.isfinite(u) and math.isfinite(v)):
        return None

    d_cam = np.array([(float(u) - cx) / fx, (float(v) - cy) / fy, 1.0], dtype=float)
    R_wb = quat_to_matrix(*drone_quaternion)
    d_world = R_wb @ (R_bc @ d_cam)
    norm_d = float(np.linalg.norm(d_world))
    if norm_d <= 1e-9 or not np.all(np.isfinite(d_world)):
        return None

    unit_dir = d_world / norm_d
    offset_b = np.asarray(cam_offset_body, dtype=float)
    origin = np.asarray(drone_position, dtype=float) + (R_wb @ offset_b)
    if not np.all(np.isfinite(origin)):
        return None
    return origin, unit_dir


def pixel_to_world(
    u: float,
    v: float,
    camera_info_k: Sequence[float],
    drone_position: Tuple[float, float, float],
    drone_quaternion: Tuple[float, float, float, float],
    R_bc: np.ndarray,
    ground_z: float = 0.0,
    cam_offset_body: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Optional[Tuple[float, float, float]]:
    """
    将像素坐标反投影到全局 ENU 世界系（与水平参考高度面 z = ground_z 求交）。
    保持原有调用签名 100% 兼容。
    """
    ray = pixel_to_ray_world(
        u, v, camera_info_k, drone_position, drone_quaternion, R_bc, cam_offset_body=cam_offset_body
    )
    if ray is None:
        return None
    p0, d_unit = ray
    dz = float(d_unit[2])
    if abs(dz) < 1e-6:
        return None

    t = (float(ground_z) - float(p0[2])) / dz
    if t <= 0.0:
        return None

    p = p0 + t * d_unit
    if not np.all(np.isfinite(p)):
        return None
    return float(p[0]), float(p[1]), float(p[2])


def triangulate_rays(
    rays: Sequence[Tuple[np.ndarray, np.ndarray]],
    min_parallax_deg: float = 2.5,
    max_skew_rms_m: float = 1.6,
    min_depth_m: float = 0.4,
    max_depth_m: float = 180.0,
    min_baseline_m: float = 0.0,
) -> Optional[Tuple[float, float, float, float]]:
    """
    多机异面视线三维最小二乘立体三角测量：
        min_p  sum_{i=1}^M || (I - d_i d_i^T) (p - o_i) ||_2^2

    参数：
        rays: 列表 [(origin_i, unit_dir_i), ...]，至少包含 2 条来自不同视角的单位方向射线
        min_parallax_deg: 最小允许双机视线夹角（度），低于此阈值视为共线退化并返回 None
        max_skew_rms_m: 最大允许异面射线到交点的均方根正交残差（米）

    返回：
        (px, py, pz, rms_error_m) 或 None（退化/发散/位于相机后方时安全弃权）。
    """
    m_cnt = len(rays)
    if m_cnt < 2:
        return None

    # 1. 校验最大两两视线夹角（基线视差角）
    max_angle_rad = 0.0
    unit_rays = []
    for o_i, d_i in rays:
        o_arr = np.asarray(o_i, dtype=float)
        d_arr = np.asarray(d_i, dtype=float)
        dn = float(np.linalg.norm(d_arr))
        if dn < 1e-9:
            return None
        unit_rays.append((o_arr, d_arr / dn))

    max_baseline = 0.0
    for i in range(m_cnt):
        for j in range(i + 1, m_cnt):
            b_ij = float(np.linalg.norm(unit_rays[i][0] - unit_rays[j][0]))
            if b_ij > max_baseline:
                max_baseline = b_ij
            cos_val = max(-1.0, min(1.0, float(np.dot(unit_rays[i][1], unit_rays[j][1]))))
            ang = math.acos(cos_val)
            if ang > max_angle_rad:
                max_angle_rad = ang

    if max_baseline < float(min_baseline_m):
        return None
    if math.degrees(max_angle_rad) < float(min_parallax_deg):
        return None

    # 2. 组装正规方程 A p = b
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    eye3 = np.eye(3, dtype=float)
    for o_arr, d_unit in unit_rays:
        proj = eye3 - np.outer(d_unit, d_unit)
        A += proj
        b += proj @ o_arr

    try:
        cond = float(np.linalg.cond(A))
        if not math.isfinite(cond) or cond > 1e5:
            return None
        p_est = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None

    if not np.all(np.isfinite(p_est)):
        return None

    # 3. 正深度校验 (Cheirality check) 与异面残差审计
    sq_err_sum = 0.0
    for o_arr, d_unit in unit_rays:
        rel = p_est - o_arr
        depth = float(np.dot(rel, d_unit))
        if depth < min_depth_m or depth > max_depth_m:
            return None
        perp = rel - depth * d_unit
        sq_err_sum += float(np.dot(perp, perp))

    rms_err = math.sqrt(sq_err_sum / m_cnt)
    if rms_err > float(max_skew_rms_m):
        return None

    return float(p_est[0]), float(p_est[1]), float(p_est[2]), float(rms_err)
