# -*- coding: utf-8 -*-
"""
geo_utils.py —— 机载视觉三维几何反投影、多目标异面射线簇三角测量与 3D 障碍物视线遮挡 (LoS) 检验库

坐标系约定（严格遵守《课题二内部接口规范 V3.0》及 PROJECT.md）：
  - 世界系：全局 ENU（东-北-天，Z 轴向上为正），frame_id = "world"
  - 严禁做 ENU <-> NED 转换（NED 转换由飞控桥接统一负责）
  - 支持能力：
      1) 单目针孔射线与参考高度面求交 (pixel_to_world) 及射线生成 (pixel_to_ray_world)
      2) 单目标多机异面射线最小二乘立体三角测量 (triangulate_rays)
      3) 3D 障碍物加载 (load_obstacles_from_yaml / load_obstacles_from_json) 与
         精确线段-圆柱体/AABB/旋转 OBB 视线遮挡判定 (segment_intersects_obstacle / has_line_of_sight)
      4) 多目标全配对极线中点 -> 空间-类别中心点聚类 -> 每机单射线去重与 IRLS 离群剔除 ->
         全局射线互斥集合打包 (triangulate_multi_target)，彻底消除共面交叉鬼影 (Ghost Triangulation)。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np


_GENERIC_CLASSES = frozenset({'', 'generic', 'uav', 'target', 'unknown'})


@dataclass(frozen=True)
class RayObservation:
    """单架无人机观测到的单目标三维世界射线。"""

    drone_id: str
    origin: np.ndarray
    direction: np.ndarray
    class_id: str = 'crimson_uav'
    conf: float = 1.0
    timestamp: float = 0.0
    det_id: int = 0
    pixel_uv: Optional[Tuple[float, float]] = None
    area_px: float = 0.0


@dataclass(frozen=True)
class Obstacle3D:
    """全局 ENU 坐标系下的物理不透明 3D 障碍物（用于光学视线遮挡检验）。"""

    obs_id: str
    obs_type: str  # 'cylinder' | 'box'
    cx: float
    cy: float
    z_min: float
    z_max: float
    radius: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    yaw: float = 0.0
    safety_margin_m: float = 0.0


@dataclass(frozen=True)
class TriangulatedTarget:
    """三维定位输出目标（可直接序列化为 /vision/targets JSON 契约）。"""

    det_id: int
    class_id: str
    x: float
    y: float
    z: float
    conf: float
    stereo: bool
    rays_used: int
    parallax_deg: float
    skew_rms_m: float
    drone_ids: Tuple[str, ...] = ()
    ray_indices: Tuple[int, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'det_id': int(self.det_id),
            'class_id': str(self.class_id),
            'x': float(self.x),
            'y': float(self.y),
            'z': float(self.z),
            'conf': float(self.conf),
            'stereo': bool(self.stereo),
            'rays_used': int(self.rays_used),
            'parallax_deg': float(self.parallax_deg),
            'skew_rms_m': float(self.skew_rms_m),
        }


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

    返回：
        (px, py, pz, rms_error_m) 或 None（退化/发散/位于相机后方时安全弃权）。
    """
    m_cnt = len(rays)
    if m_cnt < 2:
        return None

    unit_rays = []
    for o_i, d_i in rays:
        o_arr = np.asarray(o_i, dtype=float)
        d_arr = np.asarray(d_i, dtype=float)
        dn = float(np.linalg.norm(d_arr))
        if dn < 1e-9:
            return None
        unit_rays.append((o_arr, d_arr / dn))

    upper_deg = 180.0 - float(min_parallax_deg)
    max_baseline = 0.0
    has_valid_parallax = False
    for i in range(m_cnt):
        for j in range(i + 1, m_cnt):
            b_ij = float(np.linalg.norm(unit_rays[i][0] - unit_rays[j][0]))
            if b_ij > max_baseline:
                max_baseline = b_ij
            cos_val = max(-1.0, min(1.0, float(np.dot(unit_rays[i][1], unit_rays[j][1]))))
            ang_deg = math.degrees(math.acos(cos_val))
            if float(min_parallax_deg) <= ang_deg <= upper_deg:
                has_valid_parallax = True

    if max_baseline < float(min_baseline_m):
        return None
    if not has_valid_parallax:
        return None

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


def _parse_obstacles_payload(data: Any) -> Tuple[List[Obstacle3D], Optional[Dict[str, float]]]:
    """解析 obstacles.yaml / JSON 负载，忽略虚拟禁飞区 no_fly_zones（非物理遮挡）。"""
    if data is None:
        return [], None

    arena_bounds: Optional[Dict[str, float]] = None
    raw_obstacles: Any = []

    if isinstance(data, dict):
        if isinstance(data.get('arena_bounds'), dict):
            arena_bounds = {}
            for k, v in data['arena_bounds'].items():
                try:
                    arena_bounds[str(k)] = float(v)
                except (TypeError, ValueError):
                    pass
            if not arena_bounds:
                arena_bounds = None
        raw_obstacles = data.get('obstacles') or []
    elif isinstance(data, list):
        raw_obstacles = data
    else:
        return [], None

    obstacles: List[Obstacle3D] = []
    if not isinstance(raw_obstacles, list):
        return [], arena_bounds

    for idx, item in enumerate(raw_obstacles):
        if not isinstance(item, dict):
            continue
        obs_type = str(item.get('type', item.get('obs_type', ''))).strip().lower()
        if obs_type not in ('cylinder', 'box'):
            continue
        obs_id = str(item.get('id', item.get('obs_id', f'obs_{idx}')))
        cx = float(item.get('cx', item.get('x', 0.0)))
        cy = float(item.get('cy', item.get('y', 0.0)))
        if 'z_min' in item and 'z_max' in item:
            z_lo = float(item['z_min'])
            z_hi = float(item['z_max'])
        else:
            cz = float(item.get('cz', item.get('z', 0.0)))
            sz = float(item.get('sz', item.get('height', 10.0)))
            z_lo = cz - 0.5 * sz
            z_hi = cz + 0.5 * sz
        if z_hi < z_lo:
            z_lo, z_hi = z_hi, z_lo
        obstacles.append(
            Obstacle3D(
                obs_id=obs_id,
                obs_type=obs_type,
                cx=cx,
                cy=cy,
                z_min=z_lo,
                z_max=z_hi,
                radius=float(item.get('radius', 0.0)),
                sx=float(item.get('sx', 0.0)),
                sy=float(item.get('sy', 0.0)),
                yaw=float(item.get('yaw', 0.0)),
                safety_margin_m=float(item.get('safety_margin_m', 0.0)),
            )
        )
    return obstacles, arena_bounds


def load_obstacles_from_yaml(
    yaml_path: Optional[str],
) -> Tuple[List[Obstacle3D], Optional[Dict[str, float]]]:
    """从 YAML 文件加载物理 3D 障碍物与场地边界；文件缺失或为空时安全返回 ([], None)。"""
    if not yaml_path or not isinstance(yaml_path, str) or not os.path.isfile(yaml_path):
        return [], None
    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
    except Exception:
        return [], None
    return _parse_obstacles_payload(data)


def load_obstacles_from_json(
    json_str: Optional[str],
) -> Tuple[List[Obstacle3D], Optional[Dict[str, float]]]:
    """从 JSON 字符串加载物理 3D 障碍物与场地边界；为空或无效时安全返回 ([], None)。"""
    if not json_str or not str(json_str).strip():
        return [], None
    try:
        data = json.loads(json_str)
    except Exception:
        return [], None
    return _parse_obstacles_payload(data)


def _segment_intersects_cylinder(
    o: np.ndarray,
    v: np.ndarray,
    s_max: float,
    obs: Obstacle3D,
    margin_m: float,
) -> bool:
    r = float(obs.radius) + float(margin_m)
    if r <= 0.0:
        return False

    z_lo_bound = min(obs.z_min, obs.z_max)
    z_hi_bound = max(obs.z_min, obs.z_max)
    s_lo = 0.0
    s_hi = s_max

    vz = float(v[2])
    oz = float(o[2])
    if abs(vz) < 1e-12:
        if oz < z_lo_bound or oz > z_hi_bound:
            return False
    else:
        t1 = (z_lo_bound - oz) / vz
        t2 = (z_hi_bound - oz) / vz
        if t1 > t2:
            t1, t2 = t2, t1
        s_lo = max(s_lo, t1)
        s_hi = min(s_hi, t2)
        if s_lo > s_hi:
            return False

    dx = float(o[0] - obs.cx)
    dy = float(o[1] - obs.cy)
    vx = float(v[0])
    vy = float(v[1])
    a = vx * vx + vy * vy
    c = dx * dx + dy * dy - r * r
    if a < 1e-12:
        return c <= 0.0

    b = dx * vx + dy * vy
    disc = b * b - a * c
    if disc < 0.0:
        return False
    sqrt_disc = math.sqrt(disc)
    r1 = (-b - sqrt_disc) / a
    r2 = (-b + sqrt_disc) / a
    if r1 > r2:
        r1, r2 = r2, r1
    return max(s_lo, r1) <= min(s_hi, r2)


def _segment_intersects_box(
    o: np.ndarray,
    v: np.ndarray,
    s_max: float,
    obs: Obstacle3D,
    margin_m: float,
) -> bool:
    hx = 0.5 * float(obs.sx) + float(margin_m)
    hy = 0.5 * float(obs.sy) + float(margin_m)
    if hx <= 0.0 or hy <= 0.0:
        return False

    ox_rel = float(o[0] - obs.cx)
    oy_rel = float(o[1] - obs.cy)
    vx_rel = float(v[0])
    vy_rel = float(v[1])

    if abs(obs.yaw) > 1e-9:
        c = math.cos(-obs.yaw)
        s = math.sin(-obs.yaw)
        ox_loc = c * ox_rel - s * oy_rel
        oy_loc = s * ox_rel + c * oy_rel
        vx_loc = c * vx_rel - s * vy_rel
        vy_loc = s * vx_rel + c * vy_rel
    else:
        ox_loc, oy_loc = ox_rel, oy_rel
        vx_loc, vy_loc = vx_rel, vy_rel

    z_lo_bound = min(obs.z_min, obs.z_max)
    z_hi_bound = max(obs.z_min, obs.z_max)

    s_lo = 0.0
    s_hi = s_max
    for origin_k, dir_k, min_k, max_k in (
        (ox_loc, vx_loc, -hx, hx),
        (oy_loc, vy_loc, -hy, hy),
        (float(o[2]), float(v[2]), z_lo_bound, z_hi_bound),
    ):
        if abs(dir_k) < 1e-12:
            if origin_k < min_k or origin_k > max_k:
                return False
        else:
            t1 = (min_k - origin_k) / dir_k
            t2 = (max_k - origin_k) / dir_k
            if t1 > t2:
                t1, t2 = t2, t1
            s_lo = max(s_lo, t1)
            s_hi = min(s_hi, t2)
            if s_lo > s_hi:
                return False
    return s_lo <= s_hi


def segment_intersects_obstacle(
    p0: Sequence[float],
    p1: Sequence[float],
    obs: Obstacle3D,
    margin_m: float = 0.0,
    end_clearance_m: float = 0.05,
) -> bool:
    """精确检验 3D 线段 [p0, p1] 是否与圆柱体或 AABB/旋转 OBB 障碍物相交。"""
    o = np.asarray(p0, dtype=float)
    q = np.asarray(p1, dtype=float)
    v = q - o
    seg_len = float(np.linalg.norm(v))
    if not (np.all(np.isfinite(o)) and np.all(np.isfinite(q))):
        return False
    if seg_len <= 1e-9:
        s_max = 0.0
    elif seg_len <= float(end_clearance_m):
        s_max = 1.0
    else:
        s_max = 1.0 - float(end_clearance_m) / seg_len

    if obs.obs_type == 'cylinder':
        return _segment_intersects_cylinder(o, v, s_max, obs, margin_m)
    if obs.obs_type == 'box':
        return _segment_intersects_box(o, v, s_max, obs, margin_m)
    return False


def has_line_of_sight(
    p0: Sequence[float],
    p1: Sequence[float],
    obstacles: Sequence[Obstacle3D],
    margin_m: float = 0.0,
    end_clearance_m: float = 0.05,
) -> bool:
    """检查两点 p0 <-> p1 之间是否存在无障碍物遮挡的光学直视视线 (LoS)（含快速 AABB 预剔除）。"""
    if not obstacles:
        return True
    x0, y0, z0 = float(p0[0]), float(p0[1]), float(p0[2])
    x1, y1, z1 = float(p1[0]), float(p1[1]), float(p1[2])
    seg_x_lo = x0 if x0 < x1 else x1
    seg_x_hi = x1 if x0 < x1 else x0
    seg_y_lo = y0 if y0 < y1 else y1
    seg_y_hi = y1 if y0 < y1 else y0
    seg_z_lo = z0 if z0 < z1 else z1
    seg_z_hi = z1 if z0 < z1 else z0
    m = float(margin_m)

    for obs in obstacles:
        z_lo = obs.z_min if obs.z_min <= obs.z_max else obs.z_max
        z_hi = obs.z_max if obs.z_min <= obs.z_max else obs.z_min
        if seg_z_hi < z_lo or seg_z_lo > z_hi:
            continue
        if obs.obs_type == 'cylinder':
            r = obs.radius + m
            if seg_x_hi < obs.cx - r or seg_x_lo > obs.cx + r or seg_y_hi < obs.cy - r or seg_y_lo > obs.cy + r:
                continue
        elif obs.obs_type == 'box':
            half_extent = 0.5 * (obs.sx + obs.sy) + m if abs(obs.yaw) > 1e-9 else 0.0
            hx = half_extent if half_extent > 0.0 else (0.5 * obs.sx + m)
            hy = half_extent if half_extent > 0.0 else (0.5 * obs.sy + m)
            if seg_x_hi < obs.cx - hx or seg_x_lo > obs.cx + hx or seg_y_hi < obs.cy - hy or seg_y_lo > obs.cy + hy:
                continue
        if segment_intersects_obstacle(
            p0, p1, obs, margin_m=margin_m, end_clearance_m=end_clearance_m
        ):
            return False
    return True


def _classes_compatible(c1: str, c2: str) -> bool:
    if c1 == c2:
        return True
    return c1 in _GENERIC_CLASSES or c2 in _GENERIC_CLASSES


def _merge_class_id(c1: str, c2: str) -> str:
    if c1 not in _GENERIC_CLASSES:
        return c1
    if c2 not in _GENERIC_CLASSES:
        return c2
    return c1 or c2 or 'crimson_uav'


def _point_in_bounds(
    p: Sequence[float],
    arena_bounds: Optional[Dict[str, float]],
    bounds_margin_m: float,
) -> bool:
    if not arena_bounds:
        return True
    x, y = float(p[0]), float(p[1])
    if 'x_min' in arena_bounds and x < float(arena_bounds['x_min']) - bounds_margin_m:
        return False
    if 'x_max' in arena_bounds and x > float(arena_bounds['x_max']) + bounds_margin_m:
        return False
    if 'y_min' in arena_bounds and y < float(arena_bounds['y_min']) - bounds_margin_m:
        return False
    if 'y_max' in arena_bounds and y > float(arena_bounds['y_max']) + bounds_margin_m:
        return False
    return True


def _solve_3x3_sym_psd(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """闭式求解 3x3 对称半正定线性方程组 A p = b，含解析条件数认证。"""
    a00 = float(A[0, 0])
    a01 = float(A[0, 1])
    a02 = float(A[0, 2])
    a11 = float(A[1, 1])
    a12 = float(A[1, 2])
    a22 = float(A[2, 2])
    tr = a00 + a11 + a22
    if not math.isfinite(tr) or tr <= 1e-12:
        return None
    c00 = a11 * a22 - a12 * a12
    c01 = a02 * a12 - a01 * a22
    c02 = a01 * a12 - a02 * a11
    det = a00 * c00 + a01 * c01 + a02 * c02
    if not math.isfinite(det) or det <= 1e-18:
        return None
    if det >= 2.5e-6 * (tr * tr * tr):
        c11 = a00 * a22 - a02 * a02
        c12 = a01 * a02 - a00 * a12
        c22 = a00 * a11 - a01 * a01
        b0 = float(b[0])
        b1 = float(b[1])
        b2 = float(b[2])
        inv_det = 1.0 / det
        p_est = np.array(
            [
                (c00 * b0 + c01 * b1 + c02 * b2) * inv_det,
                (c01 * b0 + c11 * b1 + c12 * b2) * inv_det,
                (c02 * b0 + c12 * b1 + c22 * b2) * inv_det,
            ],
            dtype=float,
        )
        return p_est if np.all(np.isfinite(p_est)) else None
    try:
        eigs = np.linalg.eigvalsh(A)
        if (
            eigs[0] <= 1e-12
            or not math.isfinite(float(eigs[2] / eigs[0]))
            or (eigs[2] / eigs[0]) > 1e5
        ):
            return None
        p_est = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return p_est if np.all(np.isfinite(p_est)) else None


def _unconstrained_ls_point(unit_rays: Sequence[Tuple[np.ndarray, np.ndarray]]) -> Optional[np.ndarray]:
    A = np.zeros((3, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    eye3 = np.eye(3, dtype=float)
    for o_arr, d_unit in unit_rays:
        proj = eye3 - np.outer(d_unit, d_unit)
        A += proj
        b += proj @ o_arr
    try:
        cond = float(np.linalg.cond(A))
        if not math.isfinite(cond) or cond > 1e6:
            return None
        p_est = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    return p_est if np.all(np.isfinite(p_est)) else None


def _cluster_parallax_stats(
    unit_rays: Sequence[Tuple[np.ndarray, np.ndarray]],
    min_parallax_deg: float,
) -> Tuple[float, float, bool]:
    """返回 (best_valid_parallax_deg, max_baseline_m, has_valid_pair)。"""
    n = len(unit_rays)
    best_ang = 0.0
    max_baseline = 0.0
    has_valid = False
    upper_deg = 180.0 - float(min_parallax_deg)
    for i in range(n):
        for j in range(i + 1, n):
            b_ij = float(np.linalg.norm(unit_rays[i][0] - unit_rays[j][0]))
            if b_ij > max_baseline:
                max_baseline = b_ij
            cos_val = max(-1.0, min(1.0, float(np.dot(unit_rays[i][1], unit_rays[j][1]))))
            ang_deg = math.degrees(math.acos(cos_val))
            if float(min_parallax_deg) <= ang_deg <= upper_deg:
                has_valid = True
                if ang_deg > best_ang:
                    best_ang = ang_deg
    return best_ang, max_baseline, has_valid


def triangulate_multi_target(
    rays: Sequence[RayObservation],
    *,
    obstacles: Sequence[Obstacle3D] = (),
    arena_bounds: Optional[Dict[str, float]] = None,
    triangulation_sync_sec: float = 0.18,
    min_parallax_deg: float = 8.0,
    max_skew_rms_m: float = 0.55,
    min_depth_m: float = 0.4,
    max_depth_m: float = 180.0,
    min_baseline_m: float = 3.4,
    min_alt_m: float = 1.5,
    max_alt_m: float = 35.0,
    cluster_radius_m: float = 1.5,
    min_cluster_views: int = 2,
    los_margin_m: float = 0.0,
    los_end_clearance_m: float = 0.05,
    bounds_margin_m: float = 5.0,
) -> List[TriangulatedTarget]:
    """
    多目标多机异面视线三维三角测量（零鬼影流水线）：
      1) 全配对极线中点生成（跨机、类别兼容、时间同步、基线/双向视差/正深度/2*skew/高程/边界/LoS 门控）
      2) 空间-类别局部密度中心点聚类与支持射线收集（每架无人机最多贡献 1 条射线）
      3) IRLS 离群射线剔除、几何重解与精化点二次 LoS 校验
      4) 全局射线互斥贪心集合打包（确保每条射线最多归属于 1 个高支持度真实目标，彻底剔除交叉鬼影）
    """
    if len(rays) < max(2, int(min_cluster_views)):
        return []

    norm_rays: List[Tuple[int, RayObservation, np.ndarray, np.ndarray]] = []
    for idx, r in enumerate(rays):
        o = np.asarray(r.origin, dtype=float)
        d = np.asarray(r.direction, dtype=float)
        if o.shape != (3,) or d.shape != (3,):
            continue
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(d))):
            continue
        dn = float(np.linalg.norm(d))
        if dn < 1e-9:
            continue
        norm_rays.append((idx, r, o, d / dn))

    r_cnt = len(norm_rays)
    if r_cnt < max(2, int(min_cluster_views)):
        return []

    min_views = max(2, int(min_cluster_views))
    upper_ang_deg = 180.0 - float(min_parallax_deg)
    max_pair_skew = 2.0 * float(max_skew_rms_m)

    # -------------------- Stage 1: Vectorized All-Pairs Epipolar Midpoints --------------------
    origins_mat = np.array([item[2] for item in norm_rays], dtype=float)
    dirs_mat = np.array([item[3] for item in norm_rays], dtype=float)
    timestamps_arr = np.array([float(item[1].timestamp) for item in norm_rays], dtype=float)
    confs_arr = np.array([float(item[1].conf) for item in norm_rays], dtype=float)
    drone_arr = [item[1].drone_id for item in norm_rays]
    cls_arr = [item[1].class_id for item in norm_rays]

    drone_map: Dict[str, int] = {}
    drone_codes_list: List[int] = []
    for d_id in drone_arr:
        if d_id not in drone_map:
            drone_map[d_id] = len(drone_map)
        drone_codes_list.append(drone_map[d_id])
    drone_codes = np.array(drone_codes_list, dtype=np.int32)

    cls_map_r: Dict[str, int] = {}
    cls_codes_list: List[int] = []
    for c_id in cls_arr:
        if c_id in _GENERIC_CLASSES:
            cls_codes_list.append(-1)
        else:
            if c_id not in cls_map_r:
                cls_map_r[c_id] = len(cls_map_r)
            cls_codes_list.append(cls_map_r[c_id])
    cls_codes_r = np.array(cls_codes_list, dtype=np.int32)

    eye3 = np.eye(3, dtype=float)
    P_mats = eye3[None, :, :] - dirs_mat[:, :, None] * dirs_mat[:, None, :]
    Po_vecs = np.einsum('rij,rj->ri', P_mats, origins_mat)

    iu, ju = np.triu_indices(r_cnt, k=1)
    cos_min_upper = math.cos(math.radians(float(min_parallax_deg)))
    cos_max_lower = math.cos(math.radians(upper_ang_deg))

    w0_mat = origins_mat[iu] - origins_mat[ju]
    base_sq = np.einsum('ij,ij->i', w0_mat, w0_mat)
    dot_vec = np.clip(np.einsum('ij,ij->i', dirs_mat[iu], dirs_mat[ju]), -1.0, 1.0)
    denom_vec = 1.0 - dot_vec * dot_vec
    ang_deg_vec = np.degrees(np.arccos(dot_vec))
    base_vec = np.sqrt(base_sq)

    ang_mat = np.zeros((r_cnt, r_cnt), dtype=float)
    base_mat = np.zeros((r_cnt, r_cnt), dtype=float)
    ang_mat[iu, ju] = ang_deg_vec
    ang_mat[ju, iu] = ang_deg_vec
    base_mat[iu, ju] = base_vec
    base_mat[ju, iu] = base_vec
    valid_par_mat = (ang_mat >= float(min_parallax_deg)) & (ang_mat <= upper_ang_deg)

    pair_mask = (
        (drone_codes[iu] != drone_codes[ju])
        & ((cls_codes_r[iu] == cls_codes_r[ju]) | (cls_codes_r[iu] < 0) | (cls_codes_r[ju] < 0))
        & (base_sq >= float(min_baseline_m) ** 2)
        & (dot_vec <= cos_min_upper)
        & (dot_vec >= cos_max_lower)
        & (denom_vec > 1e-9)
    )
    if triangulation_sync_sec > 0.0:
        ti = timestamps_arr[iu]
        tj = timestamps_arr[ju]
        sync_bad = (ti > 0.0) & (tj > 0.0) & (np.abs(ti - tj) > float(triangulation_sync_sec))
        pair_mask &= ~sync_bad

    valid_idx = np.where(pair_mask)[0]
    if valid_idx.size == 0:
        return []

    iu_v = iu[valid_idx]
    ju_v = ju[valid_idx]
    w0_v = w0_mat[valid_idx]
    dot_v = dot_vec[valid_idx]
    denom_v = denom_vec[valid_idx]
    d1_v = np.einsum('ij,ij->i', dirs_mat[iu_v], w0_v)
    d2_v = np.einsum('ij,ij->i', dirs_mat[ju_v], w0_v)
    ta_v = (dot_v * d2_v - d1_v) / denom_v
    tb_v = (d2_v - dot_v * d1_v) / denom_v

    pa_v = origins_mat[iu_v] + ta_v[:, None] * dirs_mat[iu_v]
    pb_v = origins_mat[ju_v] + tb_v[:, None] * dirs_mat[ju_v]
    skew_v = np.linalg.norm(pa_v - pb_v, axis=1)
    mid_v = 0.5 * (pa_v + pb_v)

    geom_mask = (
        (ta_v >= min_depth_m)
        & (ta_v <= max_depth_m)
        & (tb_v >= min_depth_m)
        & (tb_v <= max_depth_m)
        & (skew_v <= max_pair_skew)
        & (mid_v[:, 2] >= min_alt_m)
        & (mid_v[:, 2] <= max_alt_m)
    )
    if arena_bounds:
        if 'x_min' in arena_bounds:
            geom_mask &= mid_v[:, 0] >= float(arena_bounds['x_min']) - bounds_margin_m
        if 'x_max' in arena_bounds:
            geom_mask &= mid_v[:, 0] <= float(arena_bounds['x_max']) + bounds_margin_m
        if 'y_min' in arena_bounds:
            geom_mask &= mid_v[:, 1] >= float(arena_bounds['y_min']) - bounds_margin_m
        if 'y_max' in arena_bounds:
            geom_mask &= mid_v[:, 1] <= float(arena_bounds['y_max']) + bounds_margin_m

    surv_idx = np.where(geom_mask)[0]
    if surv_idx.size == 0:
        return []

    # (i_pos, j_pos, midpoint, skew, parallax_deg, baseline, merged_cls, pair_conf)
    hypotheses: List[Tuple[int, int, np.ndarray, float, float, float, str, float]] = []
    for k_s in surv_idx:
        a = int(iu_v[k_s])
        b = int(ju_v[k_s])
        orig_pair = int(valid_idx[k_s])
        merged_cls = _merge_class_id(cls_arr[a], cls_arr[b])
        pair_conf = 0.5 * (float(confs_arr[a]) + float(confs_arr[b]))
        hypotheses.append((
            a,
            b,
            mid_v[k_s],
            float(skew_v[k_s]),
            float(ang_deg_vec[orig_pair]),
            float(base_vec[orig_pair]),
            merged_cls,
            pair_conf,
        ))

    if not hypotheses:
        return []

    # -------------------- Stage 2: Spatial-Class Medoid Seeding --------------------
    mids = np.array([h[2] for h in hypotheses], dtype=float)
    skews = np.array([h[3] for h in hypotheses], dtype=float)
    confs = np.array([h[7] for h in hypotheses], dtype=float)
    inv_skew_scale = 1.0 / max(float(max_skew_rms_m), 1e-6)
    base_weights = np.exp(-skews * inv_skew_scale) * np.clip(confs, 0.1, 1.0)

    h_cnt = len(hypotheses)
    mids_sq = np.einsum('ij,ij->i', mids, mids)
    dist_sq_mat = np.maximum(0.0, mids_sq[:, None] + mids_sq[None, :] - 2.0 * (mids @ mids.T))
    adj_mat = dist_sq_mat <= float(cluster_radius_m) ** 2

    h_classes = [h[6] for h in hypotheses]
    unique_cls = list(set(h_classes))
    if len(unique_cls) > 1:
        cls_map = {c: idx for idx, c in enumerate(unique_cls)}
        cls_codes = np.array([cls_map[c] for c in h_classes], dtype=np.int32)
        is_gen = np.array([c in _GENERIC_CLASSES for c in h_classes], dtype=bool)
        adj_mat &= (cls_codes[:, None] == cls_codes[None, :]) | is_gen[:, None] | is_gen[None, :]

    density_scores = adj_mat.astype(float) @ base_weights

    seed_order = sorted(
        range(h_cnt),
        key=lambda idx_h: (float(density_scores[idx_h]), -float(skews[idx_h])),
        reverse=True,
    )

    support_radius = max(1.2, 2.2 * float(max_skew_rms_m), float(cluster_radius_m))
    support_sq = support_radius * support_radius
    dedup_radius = min(0.8, 0.6 * float(cluster_radius_m))

    max_skew_f = float(max_skew_rms_m)
    inv_max_skew = 1.0 / max(max_skew_f, 1e-6)
    min_base_f = float(min_baseline_m)
    min_depth_f = float(min_depth_m)
    max_depth_f = float(max_depth_m)

    def _score_candidate(k_views: int, mean_conf: float, parallax_deg: float, rms_err: float) -> float:
        quality = (
            1.0 * mean_conf
            + 0.3 * min(parallax_deg / 35.0, 1.0)
            + 0.2 * max(0.0, 1.0 - rms_err * inv_max_skew)
        )
        return 10.0 * float(k_views) - 6.0 + quality

    def _solve_and_prune(
        pos_indices: List[int],
    ) -> Optional[Tuple[float, Tuple[int, ...], float, float, float, float, float, str, float]]:
        cur = list(pos_indices)
        while len(cur) >= min_views:
            k_cur = len(cur)
            max_base = 0.0
            best_ang = 0.0
            has_valid_pair = False
            for i in range(k_cur):
                pi = cur[i]
                for j in range(i + 1, k_cur):
                    pj = cur[j]
                    b_ij = float(base_mat[pi, pj])
                    if b_ij > max_base:
                        max_base = b_ij
                    if valid_par_mat[pi, pj]:
                        has_valid_pair = True
                        a_ij = float(ang_mat[pi, pj])
                        if a_ij > best_ang:
                            best_ang = a_ij
            if not has_valid_pair or max_base < min_base_f:
                return None

            A = P_mats[cur].sum(axis=0)
            b = Po_vecs[cur].sum(axis=0)
            p_est = _solve_3x3_sym_psd(A, b)
            if p_est is None:
                return None

            rels = p_est[None, :] - origins_mat[cur]
            depths = np.einsum('ij,ij->i', rels, dirs_mat[cur])
            perps = rels - depths[:, None] * dirs_mat[cur]
            perp_sq = np.einsum('ij,ij->i', perps, perps)
            errs = np.sqrt(np.maximum(0.0, perp_sq))

            bad_depth = (depths < min_depth_f) | (depths > max_depth_f)
            if np.any(bad_depth):
                if k_cur <= min_views:
                    return None
                eff_errs = np.where(bad_depth, 1e6 + np.abs(depths), errs)
                cur.pop(int(np.argmax(eff_errs)))
                continue

            rms_err = math.sqrt(float(np.mean(perp_sq)))
            if k_cur == 2:
                max_ray_err_cap = max_skew_f
            elif k_cur == 3:
                max_ray_err_cap = 1.35 * max_skew_f
            else:
                inlier_med = float(np.median(errs))
                max_ray_err_cap = min(1.35 * max_skew_f, max(0.75 * max_skew_f, 4.5 * inlier_med))

            if rms_err > max_skew_f or float(errs.max()) > max_ray_err_cap:
                if k_cur <= min_views:
                    return None
                cur.pop(int(np.argmax(errs)))
                continue

            tx, ty, tz = float(p_est[0]), float(p_est[1]), float(p_est[2])
            if tz < min_alt_m or tz > max_alt_m or not _point_in_bounds((tx, ty, tz), arena_bounds, bounds_margin_m):
                return None

            if obstacles:
                p_tuple = (tx, ty, tz)
                occluded_positions = [
                    k_i
                    for k_i, pos in enumerate(cur)
                    if not has_line_of_sight(
                        origins_mat[pos],
                        p_tuple,
                        obstacles,
                        margin_m=los_margin_m,
                        end_clearance_m=los_end_clearance_m,
                    )
                ]
                if occluded_positions:
                    if k_cur - len(occluded_positions) < min_views:
                        return None
                    for k_i in reversed(occluded_positions):
                        cur.pop(k_i)
                    continue

            class_weights: Dict[str, float] = {}
            conf_sum = 0.0
            for pos in cur:
                r_obj = norm_rays[pos][1]
                conf_sum += float(r_obj.conf)
                if r_obj.class_id not in _GENERIC_CLASSES:
                    class_weights[r_obj.class_id] = class_weights.get(r_obj.class_id, 0.0) + float(r_obj.conf)
            if class_weights:
                final_cls = max(class_weights.items(), key=lambda kv: kv[1])[0]
            else:
                final_cls = norm_rays[cur[0]][1].class_id or 'crimson_uav'

            mean_conf = conf_sum / float(k_cur)
            view_bonus = min(0.15, 0.04 * max(0, k_cur - 2))
            skew_penalty = 0.20 * min(1.0, rms_err * inv_max_skew)
            final_conf = float(min(0.99, max(0.1, mean_conf + view_bonus - skew_penalty)))
            score = _score_candidate(k_cur, mean_conf, best_ang, rms_err)

            return (
                score,
                tuple(cur),
                tx,
                ty,
                tz,
                rms_err,
                best_ang,
                final_cls,
                final_conf,
            )

        return None

    def _collect_rays_around(
        center_pt: np.ndarray,
        ref_cls: str,
        radius_sq: float,
        ref_ts: float = 0.0,
    ) -> List[int]:
        rel_all = center_pt[None, :] - origins_mat
        depths_all = np.einsum('ij,ij->i', rel_all, dirs_mat)
        perp_all = rel_all - depths_all[:, None] * dirs_mat
        orth_sq_all = np.einsum('ij,ij->i', perp_all, perp_all)
        cand_mask = (depths_all >= min_depth_f) & (depths_all <= max_depth_f) & (orth_sq_all <= radius_sq)
        if triangulation_sync_sec > 0.0 and ref_ts > 0.0:
            cand_mask &= ~((timestamps_arr > 0.0) & (np.abs(timestamps_arr - ref_ts) > float(triangulation_sync_sec)))
        cand_positions = np.where(cand_mask)[0]
        if cand_positions.size < min_views:
            return []

        per_drone_best: Dict[str, Tuple[float, float, int]] = {}
        for pos in cand_positions:
            pos_i = int(pos)
            r_obj = norm_rays[pos_i][1]
            if not _classes_compatible(ref_cls, r_obj.class_id):
                continue
            if obstacles and not has_line_of_sight(
                origins_mat[pos_i], center_pt, obstacles, margin_m=los_margin_m, end_clearance_m=los_end_clearance_m
            ):
                continue
            orth_dist = math.sqrt(float(orth_sq_all[pos_i]))
            prev = per_drone_best.get(r_obj.drone_id)
            cand_key = (orth_dist, -float(r_obj.conf), pos_i)
            if prev is None or cand_key < prev:
                per_drone_best[r_obj.drone_id] = cand_key

        if len(per_drone_best) < min_views:
            return []
        return [v[2] for v in sorted(per_drone_best.values(), key=lambda x: x[0])]

    # -------------------- Stage 3: Cluster Expansion, Center Refinement & Memoization --------------------
    candidates: List[Tuple[float, Tuple[int, ...], float, float, float, float, float, str, float]] = []
    seen_centers: List[Tuple[np.ndarray, str]] = []
    tested_clusters: Set[Tuple[int, ...]] = set()
    covered_pairs: Set[Tuple[int, int]] = set()
    ray_has_strong = np.zeros(r_cnt, dtype=bool)
    h_ia = np.array([h[0] for h in hypotheses], dtype=np.int32)
    h_ib = np.array([h[1] for h in hypotheses], dtype=np.int32)

    for h_idx in seed_order:
        ia, ib, seed_mid, _, _, _, seed_cls, _ = hypotheses[h_idx]
        p_key = (ia, ib) if ia < ib else (ib, ia)
        if p_key in covered_pairs:
            continue
        if ray_has_strong[ia] and ray_has_strong[ib]:
            nb_idx = np.where(adj_mat[h_idx])[0]
            if np.all(ray_has_strong[h_ia[nb_idx]] & ray_has_strong[h_ib[nb_idx]]):
                continue

        skip_seed = False
        for c_pos, c_cls in seen_centers:
            if (
                float(np.linalg.norm(seed_mid - c_pos)) <= 0.6 * cluster_radius_m
                and _classes_compatible(seed_cls, c_cls)
            ):
                skip_seed = True
                break
        if skip_seed:
            continue

        ti_s = float(timestamps_arr[ia])
        tj_s = float(timestamps_arr[ib])
        if ti_s > 0.0 and tj_s > 0.0:
            seed_ts = 0.5 * (ti_s + tj_s)
        else:
            seed_ts = max(ti_s, tj_s, 0.0)

        cluster_pos = _collect_rays_around(seed_mid, seed_cls, support_sq, ref_ts=seed_ts)
        if not cluster_pos:
            continue

        c_key = tuple(sorted(cluster_pos))
        if c_key in tested_clusters:
            continue
        tested_clusters.add(c_key)

        solved = _solve_and_prune(cluster_pos)
        if solved is None:
            continue

        # 1-step center refinement around converged p_est
        p_refined = np.array([solved[2], solved[3], solved[4]], dtype=float)
        pos_ts_vals = [float(timestamps_arr[p]) for p in solved[1] if timestamps_arr[p] > 0.0]
        ref_ts = (sum(pos_ts_vals) / float(len(pos_ts_vals))) if pos_ts_vals else seed_ts
        refined_pos = _collect_rays_around(p_refined, solved[7], support_sq, ref_ts=ref_ts)
        if refined_pos:
            r_key = tuple(sorted(refined_pos))
            if r_key != c_key and r_key not in tested_clusters:
                tested_clusters.add(r_key)
                solved_ref = _solve_and_prune(refined_pos)
                if solved_ref is not None and solved_ref[0] >= solved[0]:
                    solved = solved_ref

        score, used_pos, tx, ty, tz, skew_rms, parallax_deg, final_cls, final_conf = solved
        center_arr = np.array([tx, ty, tz], dtype=float)
        k_views = len(used_pos)

        if k_views >= 3 and skew_rms <= 0.45 * max_skew_f:
            for p_u in used_pos:
                ray_has_strong[p_u] = True
            for i_u in range(k_views):
                pu_i = used_pos[i_u]
                for j_u in range(i_u + 1, k_views):
                    pu_j = used_pos[j_u]
                    covered_pairs.add((pu_i, pu_j) if pu_i < pu_j else (pu_j, pu_i))

        merged = False
        for c_i, existing in enumerate(candidates):
            ex_score, _, ex_x, ex_y, ex_z, _, _, ex_cls, _ = existing
            if (
                math.sqrt((tx - ex_x) ** 2 + (ty - ex_y) ** 2 + (tz - ex_z) ** 2) <= dedup_radius
                and _classes_compatible(final_cls, ex_cls)
            ):
                if score > ex_score:
                    candidates[c_i] = solved
                merged = True
                break
        if not merged:
            candidates.append(solved)
            seen_centers.append((center_arr, final_cls))

    if not candidates:
        return []

    # -------------------- Stage 4: Parsimonious Maximum-Weight Set Packing --------------------
    ray_to_cands: Dict[int, List[int]] = {}
    for c_idx, cand in enumerate(candidates):
        for p_u in cand[1]:
            ray_to_cands.setdefault(p_u, []).append(c_idx)

    visited = [False] * len(candidates)
    chosen_cands: List[Tuple[float, Tuple[int, ...], float, float, float, float, float, str, float]] = []

    for start_idx in range(len(candidates)):
        if visited[start_idx]:
            continue
        comp: List[int] = []
        stack = [start_idx]
        visited[start_idx] = True
        while stack:
            cur_c = stack.pop()
            comp.append(cur_c)
            for p_u in candidates[cur_c][1]:
                for nb_c in ray_to_cands.get(p_u, ()):
                    if not visited[nb_c]:
                        visited[nb_c] = True
                        stack.append(nb_c)

        if len(comp) == 1:
            chosen_cands.append(candidates[comp[0]])
            continue

        comp_cands = [candidates[idx] for idx in comp]
        existing_keys = {tuple(sorted(c[1])) for c in comp_cands}
        extra_cands: List[Tuple[float, Tuple[int, ...], float, float, float, float, float, str, float]] = []
        base_len = len(comp_cands)
        for i in range(base_len):
            set_i = set(comp_cands[i][1])
            for j in range(base_len):
                if i == j:
                    continue
                set_j = set(comp_cands[j][1])
                if not (set_i & set_j):
                    continue
                diff_pos = sorted(set_i - set_j)
                if len(diff_pos) < min_views:
                    continue
                d_key = tuple(diff_pos)
                if d_key in existing_keys:
                    continue
                existing_keys.add(d_key)
                re_s = _solve_and_prune(diff_pos)
                if re_s is not None:
                    r_key = tuple(sorted(re_s[1]))
                    if r_key not in existing_keys:
                        existing_keys.add(r_key)
                        extra_cands.append(re_s)
        comp_cands.extend(extra_cands)
        comp_cands.sort(key=lambda c: c[0], reverse=True)

        m_len = len(comp_cands)
        if m_len > 32:
            chosen_cands.extend(comp_cands)
            continue

        local_bit: Dict[int, int] = {}
        masks: List[int] = []
        scores_l: List[float] = []
        for cand in comp_cands:
            m_bits = 0
            for p_u in cand[1]:
                bit = local_bit.get(p_u)
                if bit is None:
                    bit = len(local_bit)
                    local_bit[p_u] = bit
                m_bits |= (1 << bit)
            masks.append(m_bits)
            scores_l.append(cand[0])

        suffix_sum = [0.0] * (m_len + 1)
        for idx_s in range(m_len - 1, -1, -1):
            suffix_sum[idx_s] = suffix_sum[idx_s + 1] + max(0.0, scores_l[idx_s])

        # Seed best_score with greedy non-overlapping pick
        best_score = 0.0
        best_pick: List[int] = []
        g_mask = 0
        for idx_g in range(m_len):
            if (g_mask & masks[idx_g]) == 0:
                g_mask |= masks[idx_g]
                best_score += scores_l[idx_g]
                best_pick.append(idx_g)

        node_budget = [200000]

        def _bnb(idx_b: int, used_m: int, cur_s: float, cur_p: List[int]) -> None:
            nonlocal best_score, best_pick
            if node_budget[0] <= 0:
                return
            node_budget[0] -= 1
            if cur_s + suffix_sum[idx_b] <= best_score + 1e-12:
                return
            if idx_b == m_len:
                if cur_s > best_score:
                    best_score = cur_s
                    best_pick = list(cur_p)
                return
            if (used_m & masks[idx_b]) == 0:
                cur_p.append(idx_b)
                _bnb(idx_b + 1, used_m | masks[idx_b], cur_s + scores_l[idx_b], cur_p)
                cur_p.pop()
            _bnb(idx_b + 1, used_m, cur_s, cur_p)

        _bnb(0, 0, 0.0, [])
        for pick_i in best_pick:
            chosen_cands.append(comp_cands[pick_i])

    chosen_cands.sort(key=lambda c: c[0], reverse=True)
    claimed_orig_indices: Set[int] = set()
    accepted: List[Tuple[float, TriangulatedTarget]] = []

    for score, used_pos, tx, ty, tz, skew_rms, parallax_deg, final_cls, final_conf in chosen_cands:
        avail_pos = [p for p in used_pos if norm_rays[p][0] not in claimed_orig_indices]
        if len(avail_pos) < min_views:
            continue

        if len(avail_pos) < len(used_pos):
            re_solved = _solve_and_prune(avail_pos)
            if re_solved is None:
                continue
            score, used_pos, tx, ty, tz, skew_rms, parallax_deg, final_cls, final_conf = re_solved
            avail_pos = list(used_pos)
            if len(avail_pos) < min_views:
                continue

        dup = False
        for _, ex_tgt in accepted:
            if (
                math.sqrt((tx - ex_tgt.x) ** 2 + (ty - ex_tgt.y) ** 2 + (tz - ex_tgt.z) ** 2) <= dedup_radius
                and _classes_compatible(final_cls, ex_tgt.class_id)
            ):
                dup = True
                break
        if dup:
            continue

        orig_ids = tuple(norm_rays[p][0] for p in avail_pos)
        drone_tuple = tuple(norm_rays[p][1].drone_id for p in avail_pos)
        claimed_orig_indices.update(orig_ids)

        accepted.append(
            (
                score,
                TriangulatedTarget(
                    det_id=len(accepted),
                    class_id=final_cls,
                    x=float(tx),
                    y=float(ty),
                    z=float(tz),
                    conf=float(final_conf),
                    stereo=True,
                    rays_used=len(avail_pos),
                    parallax_deg=float(parallax_deg),
                    skew_rms_m=float(skew_rms),
                    drone_ids=drone_tuple,
                    ray_indices=orig_ids,
                ),
            )
        )

    accepted.sort(key=lambda item: item[0], reverse=True)
    results: List[TriangulatedTarget] = []
    for det_idx, (_, tgt) in enumerate(accepted):
        results.append(
            TriangulatedTarget(
                det_id=det_idx,
                class_id=tgt.class_id,
                x=tgt.x,
                y=tgt.y,
                z=tgt.z,
                conf=tgt.conf,
                stereo=tgt.stereo,
                rays_used=tgt.rays_used,
                parallax_deg=tgt.parallax_deg,
                skew_rms_m=tgt.skew_rms_m,
                drone_ids=tgt.drone_ids,
                ray_indices=tgt.ray_indices,
            )
        )
    return results
