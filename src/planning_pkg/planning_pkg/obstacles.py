# -*- coding: utf-8 -*-
"""
obstacles.py —— 三维障碍物/禁飞区地图解析、精确碰撞检测与高度分层可见图避障规划

坐标系约定：全局 ENU（东-北-天，米），对齐 PROJECT.md § Interface Contracts #2。
零 SciPy 依赖，兼容 Python 3.8 / NumPy 1.17+ 与 Python 3.11 / NumPy 1.26+。
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml


@dataclass
class ArenaBounds:
    """赛场三维安全边界（ENU）。"""

    x_min: float = -100.0
    x_max: float = 150.0
    y_min: float = -100.0
    y_max: float = 100.0
    z_min: float = 0.0
    z_max: float = 50.0


@dataclass
class ObstaclePrimitive:
    """三维障碍物或禁飞区基元（支持圆柱 cylinder / 旋转长方体 box / 禁飞区 nfz）。"""

    obs_id: str
    obs_type: str  # "cylinder", "box", "nfz"
    cx: float
    cy: float
    z_min: float
    z_max: float
    radius: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    yaw: float = 0.0
    safety_margin_m: float = 1.2


class ObstacleMap:
    """三维障碍物地图与高度分层缓存可见图 A* + 越顶选择绕飞规划器。"""

    def __init__(
        self,
        bounds: Optional[ArenaBounds] = None,
        helipads: Optional[Sequence[Dict[str, Any]]] = None,
        primitives: Optional[Sequence[ObstaclePrimitive]] = None,
        drone_radius: float = 0.45,
    ) -> None:
        self.bounds: ArenaBounds = bounds if bounds is not None else ArenaBounds()
        self.helipads: List[Dict[str, Any]] = list(helipads) if helipads is not None else []
        self.primitives: List[ObstaclePrimitive] = list(primitives) if primitives is not None else []
        self.drone_radius: float = max(0.1, float(drone_radius))
        self._vis_cache: Dict[int, Tuple[List[Tuple[float, float]], List[List[Tuple[int, float]]]]] = {}
        self._side_memory: Dict[str, int] = {}

    @classmethod
    def empty(cls, drone_radius: float = 0.45) -> "ObstacleMap":
        """构造无障碍物默认空地图。"""
        return cls(bounds=ArenaBounds(), helipads=[], primitives=[], drone_radius=drone_radius)

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]], drone_radius: float = 0.45) -> "ObstacleMap":
        """从 YAML 解析后的字典构造 ObstacleMap。"""
        if not cfg:
            return cls.empty(drone_radius=drone_radius)

        b_raw = cfg.get("arena_bounds", {}) or {}
        bounds = ArenaBounds(
            x_min=float(b_raw.get("x_min", -100.0)),
            x_max=float(b_raw.get("x_max", 150.0)),
            y_min=float(b_raw.get("y_min", -100.0)),
            y_max=float(b_raw.get("y_max", 100.0)),
            z_min=float(b_raw.get("z_min", 0.0)),
            z_max=float(b_raw.get("z_max", 50.0)),
        )

        helipads: List[Dict[str, Any]] = []
        for idx, hp in enumerate(cfg.get("helipads", []) or []):
            if not isinstance(hp, dict):
                continue
            helipads.append(
                {
                    "id": str(hp.get("id", f"H{idx + 1}")),
                    "drone_id": str(hp.get("drone_id", f"drone_{idx + 1}")),
                    "x": float(hp.get("x", 0.0)),
                    "y": float(hp.get("y", idx * 3.0)),
                    "z": float(hp.get("z", 0.0)),
                    "yaw": float(hp.get("yaw", 0.0)),
                }
            )

        primitives: List[ObstaclePrimitive] = []
        for idx, obs in enumerate(cfg.get("obstacles", []) or []):
            if not isinstance(obs, dict):
                continue
            otype = str(obs.get("type", "cylinder")).lower().strip()
            if otype not in ("cylinder", "box"):
                otype = "cylinder"
            primitives.append(
                ObstaclePrimitive(
                    obs_id=str(obs.get("id", f"obs_{idx}")),
                    obs_type=otype,
                    cx=float(obs.get("cx", 0.0)),
                    cy=float(obs.get("cy", 0.0)),
                    z_min=float(obs.get("z_min", 0.0)),
                    z_max=float(obs.get("z_max", 15.0)),
                    radius=float(obs.get("radius", 2.0)),
                    sx=float(obs.get("sx", 4.0)),
                    sy=float(obs.get("sy", 4.0)),
                    yaw=float(obs.get("yaw", 0.0)),
                    safety_margin_m=float(obs.get("safety_margin_m", 1.2)),
                )
            )

        for idx, nfz in enumerate(cfg.get("no_fly_zones", []) or []):
            if not isinstance(nfz, dict):
                continue
            primitives.append(
                ObstaclePrimitive(
                    obs_id=str(nfz.get("id", f"nfz_{idx}")),
                    obs_type="nfz",
                    cx=float(nfz.get("cx", 0.0)),
                    cy=float(nfz.get("cy", 0.0)),
                    z_min=float(nfz.get("z_min", 0.0)),
                    z_max=float(nfz.get("z_max", bounds.z_max)),
                    radius=float(nfz.get("radius", 3.0)),
                    safety_margin_m=float(nfz.get("safety_margin_m", 1.5)),
                )
            )

        return cls(bounds=bounds, helipads=helipads, primitives=primitives, drone_radius=drone_radius)

    @classmethod
    def from_yaml(cls, path: str, drone_radius: float = 0.45) -> "ObstacleMap":
        """从 YAML 文件路径加载 ObstacleMap。"""
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cls.from_dict(cfg if isinstance(cfg, dict) else {}, drone_radius=drone_radius)

    @classmethod
    def load_default(
        cls,
        yaml_path_param: str = "",
        drone_radius: float = 0.45,
    ) -> Tuple["ObstacleMap", str]:
        """按 ROS 参数 -> ament share -> 源码相对路径 -> 空地图 顺序容错加载障碍物配置。"""
        candidates: List[str] = []
        if yaml_path_param and str(yaml_path_param).strip():
            candidates.append(str(yaml_path_param).strip())

        try:
            from ament_index_python.packages import get_package_share_directory

            share_dir = get_package_share_directory("flight_ctrl_pkg")
            candidates.append(os.path.join(share_dir, "config", "obstacles.yaml"))
        except Exception:
            pass

        here = os.path.abspath(__file__)
        cur = os.path.dirname(here)
        for _ in range(10):
            candidates.append(os.path.join(cur, "flight_ctrl_pkg", "config", "obstacles.yaml"))
            candidates.append(os.path.join(cur, "src", "flight_ctrl_pkg", "config", "obstacles.yaml"))
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent

        for path in candidates:
            if path and os.path.isfile(path):
                try:
                    return cls.from_yaml(path, drone_radius=drone_radius), path
                except Exception:
                    continue
        return cls.empty(drone_radius=drone_radius), "empty_fallback"

    def get_helipad(self, drone_id: str, index_fallback: int = 0) -> Tuple[float, float, float]:
        """查询指定无人机的起降点 ENU 坐标，缺省按槽位索引回退。"""
        for hp in self.helipads:
            if hp.get("drone_id") == drone_id:
                return float(hp["x"]), float(hp["y"]), float(hp["z"])
        if 0 <= index_fallback < len(self.helipads):
            hp = self.helipads[index_fallback]
            return float(hp["x"]), float(hp["y"]), float(hp["z"])
        return 0.0, float(index_fallback) * 3.0, 0.0

    def _effective_z_interval(self, obs: ObstaclePrimitive, extra_margin: float = 0.0) -> Tuple[float, float]:
        m = obs.safety_margin_m + self.drone_radius + extra_margin
        z_lo = obs.z_min - 0.35
        z_hi = obs.z_max + (0.5 if obs.obs_type == "nfz" else min(m, 1.0))
        return z_lo, z_hi

    def _effective_margin(self, obs: ObstaclePrimitive, extra_margin: float = 0.0) -> float:
        return obs.safety_margin_m + self.drone_radius + extra_margin

    def _point_in_obs_2d(self, x: float, y: float, obs: ObstaclePrimitive, extra_margin: float = 0.0) -> bool:
        m = self._effective_margin(obs, extra_margin)
        dx = x - obs.cx
        dy = y - obs.cy
        if obs.obs_type in ("cylinder", "nfz"):
            r_eff = obs.radius + m
            return (dx * dx + dy * dy) <= (r_eff * r_eff)
        if abs(obs.yaw) > 1e-6:
            cy = math.cos(-obs.yaw)
            sy = math.sin(-obs.yaw)
            lx = dx * cy - dy * sy
            ly = dx * sy + dy * cy
        else:
            lx, ly = dx, dy
        hx = 0.5 * obs.sx + m
        hy = 0.5 * obs.sy + m
        return (-hx <= lx <= hx) and (-hy <= ly <= hy)

    def _segment_hits_obs_2d(
        self,
        p0_xy: Tuple[float, float],
        p1_xy: Tuple[float, float],
        obs: ObstaclePrimitive,
        t_lo: float = 0.0,
        t_hi: float = 1.0,
        extra_margin: float = 0.0,
    ) -> bool:
        m = self._effective_margin(obs, extra_margin)
        x0, y0 = p0_xy
        x1, y1 = p1_xy
        dx = x1 - x0
        dy = y1 - y0

        if obs.obs_type in ("cylinder", "nfz"):
            r_eff = obs.radius + m
            fx = x0 - obs.cx
            fy = y0 - obs.cy
            a = dx * dx + dy * dy
            c = fx * fx + fy * fy - r_eff * r_eff
            if a < 1e-12:
                return c <= 0.0
            b = 2.0 * (fx * dx + fy * dy)
            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                return False
            sq = math.sqrt(disc)
            tc0 = (-b - sq) / (2.0 * a)
            tc1 = (-b + sq) / (2.0 * a)
            return max(t_lo, tc0) <= min(t_hi, tc1) + 1e-9

        # Box (supports optional yaw)
        ox0 = x0 - obs.cx
        oy0 = y0 - obs.cy
        if abs(obs.yaw) > 1e-6:
            cy = math.cos(-obs.yaw)
            sy = math.sin(-obs.yaw)
            lx0 = ox0 * cy - oy0 * sy
            ly0 = ox0 * sy + oy0 * cy
            ldx = dx * cy - dy * sy
            ldy = dx * sy + dy * cy
        else:
            lx0, ly0 = ox0, oy0
            ldx, ldy = dx, dy

        hx = 0.5 * obs.sx + m
        hy = 0.5 * obs.sy + m
        t0, t1 = t_lo, t_hi
        for l0, ld, he in ((lx0, ldx, hx), (ly0, ldy, hy)):
            if abs(ld) < 1e-12:
                if l0 < -he or l0 > he:
                    return False
            else:
                ta = (-he - l0) / ld
                tb = (he - l0) / ld
                t0 = max(t0, min(ta, tb))
                t1 = min(t1, max(ta, tb))
                if t0 > t1 + 1e-9:
                    return False
        return t0 <= t1 + 1e-9

    def _segment_hits_obs_3d(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        obs: ObstaclePrimitive,
        extra_margin: float = 0.0,
    ) -> bool:
        z_lo, z_hi = self._effective_z_interval(obs, extra_margin)
        dz = float(p1[2]) - float(p0[2])
        if abs(dz) < 1e-9:
            if float(p0[2]) < z_lo or float(p0[2]) > z_hi:
                return False
            t0, t1 = 0.0, 1.0
        else:
            tza = (z_lo - float(p0[2])) / dz
            tzb = (z_hi - float(p0[2])) / dz
            t0 = max(0.0, min(tza, tzb))
            t1 = min(1.0, max(tza, tzb))
            if t0 > t1 + 1e-9:
                return False
        return self._segment_hits_obs_2d(
            (float(p0[0]), float(p0[1])),
            (float(p1[0]), float(p1[1])),
            obs,
            t_lo=t0,
            t_hi=t1,
            extra_margin=extra_margin,
        )

    def point_clear_3d(self, pt: np.ndarray, extra_margin: float = 0.0) -> bool:
        """检测三维点是否位于所有膨胀障碍物及禁飞区外部。"""
        x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return False
        for obs in self.primitives:
            z_lo, z_hi = self._effective_z_interval(obs, extra_margin)
            if z_lo <= z <= z_hi and self._point_in_obs_2d(x, y, obs, extra_margin):
                return False
        return True

    def segment_clear_3d(self, p0: np.ndarray, p1: np.ndarray, extra_margin: float = 0.0) -> bool:
        """检测三维直线段是否完全无碰撞。"""
        a = np.asarray(p0, dtype=float)
        b = np.asarray(p1, dtype=float)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            return False
        for obs in self.primitives:
            if self._segment_hits_obs_3d(a, b, obs, extra_margin=extra_margin):
                return False
        return True

    def project_outside(
        self,
        pt: np.ndarray,
        margin_pad: float = 0.30,
        max_alt: Optional[float] = None,
    ) -> np.ndarray:
        """若点位于障碍物/禁飞区内部，沿赛场边界内最近水平或垂直安全边界向外投影。"""
        out = np.asarray(pt, dtype=float).copy()
        x_lo = self.bounds.x_min + 0.5 if (self.bounds.x_max - self.bounds.x_min) > 1.2 else self.bounds.x_min
        x_hi = self.bounds.x_max - 0.5 if (self.bounds.x_max - self.bounds.x_min) > 1.2 else self.bounds.x_max
        y_lo = self.bounds.y_min + 0.5 if (self.bounds.y_max - self.bounds.y_min) > 1.2 else self.bounds.y_min
        y_hi = self.bounds.y_max - 0.5 if (self.bounds.y_max - self.bounds.y_min) > 1.2 else self.bounds.y_max
        z_ceil_bounds = self.bounds.z_max - 0.5 if (self.bounds.z_max - self.bounds.z_min) > 1.2 else self.bounds.z_max
        z_ceil = min(z_ceil_bounds, float(max_alt)) if max_alt is not None else z_ceil_bounds

        if not np.all(np.isfinite(out)):
            out = np.array(
                [0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi), max(self.bounds.z_min, min(z_ceil, 5.0))],
                dtype=float,
            )

        out[0] = max(x_lo, min(x_hi, float(out[0])))
        out[1] = max(y_lo, min(y_hi, float(out[1])))
        out[2] = max(self.bounds.z_min, min(self.bounds.z_max, float(out[2])))
        orig = out.copy()

        def _in_bounds(px: float, py: float) -> bool:
            return (x_lo <= px <= x_hi) and (y_lo <= py <= y_hi)

        def _dist_orig(cand: np.ndarray) -> float:
            return (
                math.hypot(float(cand[0]) - float(orig[0]), float(cand[1]) - float(orig[1]))
                + abs(float(cand[2]) - float(orig[2]))
            )

        def _prim_horiz_cands(obs: ObstaclePrimitive, cur: np.ndarray, allow_fast_single: bool) -> List[np.ndarray]:
            m = self._effective_margin(obs) + margin_pad
            pts: List[np.ndarray] = []
            if obs.obs_type in ("cylinder", "nfz"):
                dx = float(cur[0]) - obs.cx
                dy = float(cur[1]) - obs.cy
                d = math.hypot(dx, dy)
                ux, uy = (dx / d, dy / d) if d > 1e-5 else (1.0, 0.0)
                r_out = obs.radius + m
                wx, wy = obs.cx + ux * r_out, obs.cy + uy * r_out
                if _in_bounds(wx, wy):
                    p_rad = np.array([wx, wy, float(cur[2])], dtype=float)
                    if allow_fast_single and self.point_clear_3d(p_rad):
                        return [p_rad]
                    pts.append(p_rad)
                for k in range(24):
                    ang = (2.0 * math.pi * k) / 24.0
                    ax = obs.cx + r_out * math.cos(ang)
                    ay = obs.cy + r_out * math.sin(ang)
                    if _in_bounds(ax, ay):
                        pts.append(np.array([ax, ay, float(cur[2])], dtype=float))
            else:
                dx = float(cur[0]) - obs.cx
                dy = float(cur[1]) - obs.cy
                cy = math.cos(-obs.yaw)
                sy = math.sin(-obs.yaw)
                lx = dx * cy - dy * sy
                ly = dx * sy + dy * cy
                hx = 0.5 * obs.sx + m
                hy = 0.5 * obs.sy + m
                wy_c = math.cos(obs.yaw)
                wy_s = math.sin(obs.yaw)
                face_local: List[Tuple[float, float, float]] = [
                    (max(0.0, hx - lx), hx, ly),
                    (max(0.0, hx + lx), -hx, ly),
                    (max(0.0, hy - ly), lx, hy),
                    (max(0.0, hy + ly), lx, -hy),
                ]
                face_local.sort(key=lambda item: item[0])
                if allow_fast_single:
                    _, fx_l, fy_l = face_local[0]
                    fx_w = obs.cx + fx_l * wy_c - fy_l * wy_s
                    fy_w = obs.cy + fx_l * wy_s + fy_l * wy_c
                    if _in_bounds(fx_w, fy_w):
                        p_face = np.array([fx_w, fy_w, float(cur[2])], dtype=float)
                        if self.point_clear_3d(p_face):
                            return [p_face]
                local_pts: List[Tuple[float, float]] = [(fx_l, fy_l) for _, fx_l, fy_l in face_local]
                for frac in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0):
                    local_pts.append((hx, frac * hy))
                    local_pts.append((-hx, frac * hy))
                    local_pts.append((frac * hx, hy))
                    local_pts.append((frac * hx, -hy))
                for lx_out, ly_out in local_pts:
                    wx = obs.cx + lx_out * wy_c - ly_out * wy_s
                    wy = obs.cy + lx_out * wy_s + ly_out * wy_c
                    if _in_bounds(wx, wy):
                        pts.append(np.array([wx, wy, float(cur[2])], dtype=float))
            return pts

        any_hit = False
        max_iters = min(12, max(4, len(self.primitives) + 1))
        for iter_idx in range(max_iters):
            moved = False
            for obs in self.primitives:
                z_lo, z_hi = self._effective_z_interval(obs)
                if not (z_lo <= out[2] <= z_hi):
                    continue
                if not self._point_in_obs_2d(float(out[0]), float(out[1]), obs):
                    continue
                any_hit = True
                allow_fast = (not moved) and (iter_idx == 0)
                cands: List[Tuple[int, int, int, float, np.ndarray]] = []
                for p_h in _prim_horiz_cands(obs, out, allow_fast_single=allow_fast):
                    is_clr = self.point_clear_3d(p_h)
                    cands.append((0 if is_clr else 1, 0, 0, _dist_orig(p_h), p_h))

                z_top = z_hi + margin_pad
                if z_top <= z_ceil:
                    p_v = np.array([float(out[0]), float(out[1]), z_top], dtype=float)
                    is_clr_v = self.point_clear_3d(p_v)
                    is_nfz_v = 1 if obs.obs_type == "nfz" else 0
                    alt_pen_if_blocked = 0 if is_clr_v else 1
                    cands.append((0 if is_clr_v else 1, is_nfz_v, alt_pen_if_blocked, _dist_orig(p_v), p_v))

                if cands:
                    cands.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
                    out = cands[0][4]
                    moved = True
            if not moved:
                break

        if (not any_hit) or self.point_clear_3d(out):
            return out

        # Stage 2: 全局确定性回退（应对多基元深度重叠导致的局部边界互锁）
        fb_cands: List[Tuple[int, float, np.ndarray]] = []
        for obs in self.primitives:
            z_lo, z_hi = self._effective_z_interval(obs)
            if z_lo <= orig[2] <= z_hi:
                for p_h in _prim_horiz_cands(obs, orig, allow_fast_single=False):
                    if self.point_clear_3d(p_h):
                        fb_cands.append((0, _dist_orig(p_h), p_h))
            z_top = z_hi + margin_pad
            if z_lo <= orig[2] <= z_top and z_top <= z_ceil:
                for bx, by in ((orig[0], orig[1]), (out[0], out[1])):
                    p_v = np.array([float(bx), float(by), z_top], dtype=float)
                    if self.point_clear_3d(p_v):
                        fb_cands.append((1 if obs.obs_type == "nfz" else 0, _dist_orig(p_v), p_v))

        max_span = math.hypot(x_hi - x_lo, y_hi - y_lo)
        r_probe = 1.0
        while r_probe <= max_span + 1.0:
            for k in range(24):
                ang = (2.0 * math.pi * k) / 24.0
                px = float(orig[0]) + r_probe * math.cos(ang)
                py = float(orig[1]) + r_probe * math.sin(ang)
                if _in_bounds(px, py):
                    p_r = np.array([px, py, float(orig[2])], dtype=float)
                    if self.point_clear_3d(p_r):
                        fb_cands.append((0, _dist_orig(p_r), p_r))
            if any(item[0] == 0 for item in fb_cands):
                break
            r_probe = max(r_probe + 1.0, r_probe * 1.45)

        for cx_b, cy_b in (
            (x_lo, y_lo),
            (x_lo, y_hi),
            (x_hi, y_lo),
            (x_hi, y_hi),
            (x_lo, float(orig[1])),
            (x_hi, float(orig[1])),
            (float(orig[0]), y_lo),
            (float(orig[0]), y_hi),
        ):
            p_b = np.array([cx_b, cy_b, float(orig[2])], dtype=float)
            if self.point_clear_3d(p_b):
                fb_cands.append((0, _dist_orig(p_b), p_b))

        if fb_cands:
            fb_cands.sort(key=lambda item: (item[0], item[1]))
            return fb_cands[0][2]

        if max_alt is not None and z_ceil_bounds > z_ceil:
            for obs in self.primitives:
                z_lo, z_hi = self._effective_z_interval(obs)
                z_top = z_hi + margin_pad
                if z_lo <= orig[2] <= z_top <= z_ceil_bounds:
                    p_v = np.array([float(orig[0]), float(orig[1]), z_top], dtype=float)
                    if self.point_clear_3d(p_v):
                        return p_v

        return out

    def _active_mask(self, z_min_band: float, z_max_band: float) -> int:
        mask = 0
        for idx, obs in enumerate(self.primitives):
            z_lo, z_hi = self._effective_z_interval(obs)
            if not (z_max_band < z_lo or z_min_band > z_hi):
                mask |= 1 << idx
        return mask

    def _get_visibility_graph(
        self, mask: int
    ) -> Tuple[List[Tuple[float, float]], List[List[Tuple[int, float]]]]:
        if mask in self._vis_cache:
            return self._vis_cache[mask]

        active_obs = [obs for idx, obs in enumerate(self.primitives) if (mask >> idx) & 1]
        verts: List[Tuple[float, float]] = []
        pad = 0.40
        for obs in active_obs:
            m = self._effective_margin(obs) + pad
            if obs.obs_type in ("cylinder", "nfz"):
                r_poly = (obs.radius + m) / math.cos(math.pi / 8.0)
                for k in range(8):
                    ang = (2.0 * math.pi * k) / 8.0
                    vx = obs.cx + r_poly * math.cos(ang)
                    vy = obs.cy + r_poly * math.sin(ang)
                    verts.append((vx, vy))
            else:
                hx = 0.5 * obs.sx + m
                hy = 0.5 * obs.sy + m
                c_y = math.cos(obs.yaw)
                s_y = math.sin(obs.yaw)
                for sx_sign in (-1.0, 1.0):
                    for sy_sign in (-1.0, 1.0):
                        lx = sx_sign * hx
                        ly = sy_sign * hy
                        vx = obs.cx + lx * c_y - ly * s_y
                        vy = obs.cy + lx * s_y + ly * c_y
                        verts.append((vx, vy))

        # 过滤位于赛场边界外或其他活跃障碍物内的顶点
        filtered: List[Tuple[float, float]] = []
        for vx, vy in verts:
            if not (
                self.bounds.x_min + 0.4 <= vx <= self.bounds.x_max - 0.4
                and self.bounds.y_min + 0.4 <= vy <= self.bounds.y_max - 0.4
            ):
                continue
            inside_any = False
            for obs in active_obs:
                if self._point_in_obs_2d(vx, vy, obs, extra_margin=0.05):
                    inside_any = True
                    break
            if not inside_any:
                filtered.append((vx, vy))

        n_v = len(filtered)
        adj: List[List[Tuple[int, float]]] = [[] for _ in range(n_v)]
        for i in range(n_v):
            pi = filtered[i]
            for j in range(i + 1, n_v):
                pj = filtered[j]
                clear = True
                for obs in active_obs:
                    if self._segment_hits_obs_2d(pi, pj, obs, extra_margin=0.0):
                        clear = False
                        break
                if clear:
                    dist = math.hypot(pj[0] - pi[0], pj[1] - pi[1])
                    adj[i].append((j, dist))
                    adj[j].append((i, dist))

        self._vis_cache[mask] = (filtered, adj)
        return filtered, adj

    def _plan_lateral_astar(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        drone_id: Optional[str] = None,
    ) -> Optional[List[np.ndarray]]:
        z_lo = min(float(start[2]), float(goal[2]))
        z_hi = max(float(start[2]), float(goal[2]))
        mask = self._active_mask(z_lo, z_hi)
        active_obs = [obs for idx, obs in enumerate(self.primitives) if (mask >> idx) & 1]

        verts, adj = self._get_visibility_graph(mask)
        n_v = len(verts)
        s_xy = (float(start[0]), float(start[1]))
        g_xy = (float(goal[0]), float(goal[1]))

        def seg_2d_clear(a_xy: Tuple[float, float], b_xy: Tuple[float, float]) -> bool:
            for obs in active_obs:
                if self._segment_hits_obs_2d(a_xy, b_xy, obs, extra_margin=0.0):
                    return False
            return True

        start_edges: List[Tuple[int, float]] = []
        goal_edges: Dict[int, float] = {}
        # 目标或起点方向侧向滞回（避免左右绕飞频繁切换）
        dir_xy = np.array([g_xy[0] - s_xy[0], g_xy[1] - s_xy[1]], dtype=float)
        dir_norm = float(np.linalg.norm(dir_xy))
        prev_side = self._side_memory.get(drone_id, 0) if drone_id else 0

        for idx, v_xy in enumerate(verts):
            if seg_2d_clear(s_xy, v_xy):
                d_s = math.hypot(v_xy[0] - s_xy[0], v_xy[1] - s_xy[1])
                if prev_side != 0 and dir_norm > 1.0:
                    cross_val = dir_xy[0] * (v_xy[1] - s_xy[1]) - dir_xy[1] * (v_xy[0] - s_xy[0])
                    side = 1 if cross_val >= 0.0 else -1
                    if side != prev_side:
                        d_s += 0.8
                start_edges.append((idx, d_s))
            if seg_2d_clear(v_xy, g_xy):
                goal_edges[idx] = math.hypot(g_xy[0] - v_xy[0], g_xy[1] - v_xy[1])

        s_node = n_v
        g_node = n_v + 1
        g_score: Dict[int, float] = {s_node: 0.0}
        came_from: Dict[int, int] = {}
        pq: List[Tuple[float, int]] = [(math.hypot(g_xy[0] - s_xy[0], g_xy[1] - s_xy[1]), s_node)]
        visited: set = set()

        def h_fn(node_idx: int) -> float:
            if node_idx == g_node:
                return 0.0
            px, py = s_xy if node_idx == s_node else verts[node_idx]
            return math.hypot(g_xy[0] - px, g_xy[1] - py)

        found = False
        while pq:
            _, cur = heapq.heappop(pq)
            if cur in visited:
                continue
            visited.add(cur)
            if cur == g_node:
                found = True
                break

            if cur == s_node:
                nbrs = start_edges
                if seg_2d_clear(s_xy, g_xy):
                    nbrs = nbrs + [(g_node, math.hypot(g_xy[0] - s_xy[0], g_xy[1] - s_xy[1]))]
            else:
                nbrs = list(adj[cur])
                if cur in goal_edges:
                    nbrs.append((g_node, goal_edges[cur]))

            cur_g = g_score[cur]
            for nxt, w in nbrs:
                if nxt in visited:
                    continue
                tentative_g = cur_g + w
                if tentative_g < g_score.get(nxt, float("inf")):
                    g_score[nxt] = tentative_g
                    came_from[nxt] = cur
                    heapq.heappush(pq, (tentative_g + h_fn(nxt), nxt))

        if not found:
            return None

        # 重建路径
        node_seq: List[int] = []
        cur = g_node
        while cur in came_from:
            node_seq.append(cur)
            cur = came_from[cur]
        node_seq.append(s_node)
        node_seq.reverse()

        xy_pts: List[Tuple[float, float]] = []
        for nid in node_seq:
            if nid == s_node:
                xy_pts.append(s_xy)
            elif nid == g_node:
                xy_pts.append(g_xy)
            else:
                xy_pts.append(verts[nid])

        # 按水平累积弧长线性插值 Z，随后执行三维 string-pulling 剪枝
        cum_s = [0.0]
        for k in range(1, len(xy_pts)):
            cum_s.append(cum_s[-1] + math.hypot(xy_pts[k][0] - xy_pts[k - 1][0], xy_pts[k][1] - xy_pts[k - 1][1]))
        total_s = max(cum_s[-1], 1e-6)
        raw_3d = [
            np.array(
                [
                    xy_pts[k][0],
                    xy_pts[k][1],
                    float(start[2]) + (float(goal[2]) - float(start[2])) * (cum_s[k] / total_s),
                ],
                dtype=float,
            )
            for k in range(len(xy_pts))
        ]

        # String-pulling 剪枝
        pulled: List[np.ndarray] = [raw_3d[0]]
        anchor = 0
        while anchor < len(raw_3d) - 1:
            furthest = anchor + 1
            for probe in range(len(raw_3d) - 1, anchor + 1, -1):
                if self.segment_clear_3d(raw_3d[anchor], raw_3d[probe]):
                    furthest = probe
                    break
            pulled.append(raw_3d[furthest])
            anchor = furthest

        if drone_id and len(pulled) >= 3 and dir_norm > 1.0:
            wp1 = pulled[1]
            cross_val = dir_xy[0] * (float(wp1[1]) - s_xy[1]) - dir_xy[1] * (float(wp1[0]) - s_xy[0])
            self._side_memory[drone_id] = 1 if cross_val >= 0.0 else -1

        return pulled

    def _plan_overflight(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        max_alt: float,
    ) -> Optional[List[np.ndarray]]:
        blocking = [obs for obs in self.primitives if self._segment_hits_obs_3d(start, goal, obs)]
        if not blocking:
            return [start.copy(), goal.copy()]
        z_over = max(obs.z_max + min(self._effective_margin(obs), 1.0) for obs in blocking) + 0.6
        z_cap = min(self.bounds.z_max - 0.8, float(max_alt) - 0.5)
        z_climb = max(z_over, float(start[2]), float(goal[2]))
        wp1 = np.array([float(start[0]), float(start[1]), z_climb], dtype=float)
        wp2 = np.array([float(goal[0]), float(goal[1]), z_climb], dtype=float)
        for _ in range(4):
            if z_climb > z_cap:
                return None
            wp1 = np.array([float(start[0]), float(start[1]), z_climb], dtype=float)
            wp2 = np.array([float(goal[0]), float(goal[1]), z_climb], dtype=float)
            extra = [obs for obs in self.primitives if self._segment_hits_obs_3d(wp1, wp2, obs)]
            if not extra:
                break
            z_next = max(obs.z_max + min(self._effective_margin(obs), 1.0) for obs in extra) + 0.6
            if z_next <= z_climb + 1e-4:
                return None
            z_climb = z_next
        else:
            return None

        raw = [start.copy(), wp1, wp2, goal.copy()]
        poly: List[np.ndarray] = [raw[0]]
        for pt in raw[1:]:
            if float(np.linalg.norm(pt - poly[-1])) > 1e-4:
                poly.append(pt)
        if len(poly) < 2:
            poly.append(goal.copy())
        for k in range(len(poly) - 1):
            if not self.segment_clear_3d(poly[k], poly[k + 1]):
                return None
        return poly

    def _string_pull_3d(self, pts: Sequence[np.ndarray]) -> List[np.ndarray]:
        if len(pts) <= 2:
            return [pt.copy() for pt in pts]
        pulled: List[np.ndarray] = [pts[0].copy()]
        anchor = 0
        while anchor < len(pts) - 1:
            furthest = anchor + 1
            for probe in range(len(pts) - 1, anchor + 1, -1):
                if self.segment_clear_3d(pts[anchor], pts[probe]):
                    furthest = probe
                    break
            pulled.append(pts[furthest].copy())
            anchor = furthest
        return pulled

    def _plan_stepped_lateral(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        max_alt: float,
        drone_id: Optional[str] = None,
    ) -> Optional[List[np.ndarray]]:
        z_cap = max(
            min(self.bounds.z_max - 0.8, float(max_alt) - 0.5),
            min(self.bounds.z_max - 0.1, max(float(start[2]), float(goal[2]))),
        )
        z_floor = self.bounds.z_min + 0.5
        cand_zs: List[float] = [
            max(float(start[2]), float(goal[2])),
            min(float(start[2]), float(goal[2])),
            0.5 * (float(start[2]) + float(goal[2])),
        ]
        for obs in self.primitives:
            if (
                self._point_in_obs_2d(float(start[0]), float(start[1]), obs)
                or self._point_in_obs_2d(float(goal[0]), float(goal[1]), obs)
                or self._segment_hits_obs_3d(start, goal, obs)
            ):
                _, z_hi_obs = self._effective_z_interval(obs)
                for off in (0.20, 0.6):
                    if z_hi_obs + off <= z_cap:
                        cand_zs.append(z_hi_obs + off)

        seen: List[float] = []
        best: Optional[List[np.ndarray]] = None
        best_cost = float("inf")
        for z_val in cand_zs:
            z_try = max(z_floor, min(z_cap, float(z_val)))
            if any(abs(z_try - z0) < 1e-3 for z0 in seen):
                continue
            seen.append(z_try)
            s_lvl = np.array([float(start[0]), float(start[1]), z_try], dtype=float)
            g_lvl = np.array([float(goal[0]), float(goal[1]), z_try], dtype=float)
            if not (self.point_clear_3d(s_lvl) and self.point_clear_3d(g_lvl)):
                continue
            if not (self.segment_clear_3d(start, s_lvl) and self.segment_clear_3d(g_lvl, goal)):
                continue
            mid = self._plan_lateral_astar(s_lvl, g_lvl, drone_id=drone_id)
            if mid is None:
                continue
            combined = [start.copy()] + mid + [goal.copy()]
            dedup: List[np.ndarray] = [combined[0]]
            for pt in combined[1:]:
                if float(np.linalg.norm(pt - dedup[-1])) > 1e-4:
                    dedup.append(pt)
            pulled = self._string_pull_3d(dedup)
            if all(self.segment_clear_3d(pulled[k], pulled[k + 1]) for k in range(len(pulled) - 1)):
                c = self._path_cost(pulled)
                if c < best_cost:
                    best_cost = c
                    best = pulled
        return best

    @staticmethod
    def _path_cost(poly: Sequence[np.ndarray]) -> float:
        cost = 0.0
        for k in range(1, len(poly)):
            d = poly[k] - poly[k - 1]
            horiz = math.hypot(float(d[0]), float(d[1]))
            dz = abs(float(d[2]))
            cost += horiz + 2.2 * dz
        return cost

    def plan_path_3d(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        max_alt: float = 35.0,
        drone_id: Optional[str] = None,
    ) -> List[np.ndarray]:
        """规划从 start 到 goal 的三维无碰撞折线路径 [start, ..., goal]。"""
        s = self.project_outside(np.asarray(start, dtype=float), max_alt=max_alt)
        g = self.project_outside(np.asarray(goal, dtype=float), max_alt=max_alt)
        if not self.primitives or self.segment_clear_3d(s, g):
            return [s, g]

        candidates: List[List[np.ndarray]] = []
        lat_poly = self._plan_lateral_astar(s, g, drone_id=drone_id)
        if lat_poly is not None and all(
            self.segment_clear_3d(lat_poly[k], lat_poly[k + 1]) for k in range(len(lat_poly) - 1)
        ):
            candidates.append(lat_poly)
        else:
            step_poly = self._plan_stepped_lateral(s, g, max_alt=max_alt, drone_id=drone_id)
            if step_poly is not None:
                candidates.append(step_poly)

        over_poly = self._plan_overflight(s, g, max_alt=max_alt)
        if over_poly is not None:
            candidates.append(over_poly)

        if candidates:
            return min(candidates, key=self._path_cost)
        return [s, g]

    def compute_detour_waypoint(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        max_alt: float = 35.0,
        drone_id: Optional[str] = None,
        min_lookahead_m: float = 2.5,
    ) -> Tuple[np.ndarray, np.ndarray, bool, List[np.ndarray]]:
        """计算当前拍无碰撞前瞻航路点、切向单位矢量与是否绕飞标志。"""
        s_raw = np.asarray(start, dtype=float)
        g_raw = np.asarray(goal, dtype=float)
        if not np.all(np.isfinite(s_raw)):
            s_raw = self.project_outside(s_raw, max_alt=max_alt)
        if not np.all(np.isfinite(g_raw)):
            g_raw = self.project_outside(g_raw, max_alt=max_alt)
        if not self.primitives:
            d = g_raw - s_raw
            dn = float(np.linalg.norm(d))
            u = d / dn if dn > 1e-6 else np.zeros(3, dtype=float)
            return g_raw, u, False, [s_raw, g_raw]

        s = self.project_outside(s_raw, max_alt=max_alt)
        g = self.project_outside(g_raw, max_alt=max_alt)
        start_projected = float(np.linalg.norm(s - s_raw)) > 0.05
        if self.segment_clear_3d(s, g):
            if start_projected and not self.segment_clear_3d(s_raw, g):
                d_exit = s - s_raw
                dn_exit = float(np.linalg.norm(d_exit))
                u_exit = d_exit / dn_exit if dn_exit > 1e-6 else np.zeros(3, dtype=float)
                return s, u_exit, True, [s_raw, s, g]
            d = g - s
            dn = float(np.linalg.norm(d))
            u = d / dn if dn > 1e-6 else np.zeros(3, dtype=float)
            # 若原起点就在障碍物外但终点被投影出障碍物，使用安全目标 g
            detour_needed = float(np.linalg.norm(g - g_raw)) > 0.05
            return g, u, detour_needed, [s, g]

        poly = self.plan_path_3d(s, g, max_alt=max_alt, drone_id=drone_id)
        if len(poly) <= 2:
            wp = poly[-1]
            if start_projected and not self.segment_clear_3d(s_raw, wp):
                d_exit = s - s_raw
                dn_exit = float(np.linalg.norm(d_exit))
                u_exit = d_exit / dn_exit if dn_exit > 1e-6 else np.zeros(3, dtype=float)
                return s, u_exit, True, [s_raw] + list(poly)
            d = wp - s
            dn = float(np.linalg.norm(d))
            u = d / dn if dn > 1e-6 else np.zeros(3, dtype=float)
            return wp, u, True, poly

        # 选择沿路径可视且距离 >= min_lookahead_m 的最远航路点
        chosen_wp = poly[1]
        for idx in range(len(poly) - 1, 0, -1):
            if self.segment_clear_3d(s, poly[idx]):
                chosen_wp = poly[idx]
                break

        seg_vec = chosen_wp - s
        seg_len = float(np.linalg.norm(seg_vec))
        # 若首个拐点极近，尝试沿下一段延伸以保持高速过弯不顿挫
        if seg_len < min_lookahead_m and len(poly) >= 3:
            for k in range(1, len(poly) - 1):
                if np.allclose(chosen_wp, poly[k], atol=1e-4):
                    nxt_vec = poly[k + 1] - poly[k]
                    nxt_len = float(np.linalg.norm(nxt_vec))
                    if nxt_len > 1e-4:
                        step_d = min(nxt_len, min_lookahead_m)
                        probe = poly[k] + (nxt_vec / nxt_len) * step_d
                        if self.segment_clear_3d(s, probe):
                            chosen_wp = probe
                            seg_vec = chosen_wp - s
                            seg_len = float(np.linalg.norm(seg_vec))
                    break

        if start_projected and not self.segment_clear_3d(s_raw, chosen_wp):
            d_exit = s - s_raw
            dn_exit = float(np.linalg.norm(d_exit))
            u_exit = d_exit / dn_exit if dn_exit > 1e-6 else np.zeros(3, dtype=float)
            return s, u_exit, True, [s_raw] + list(poly)

        u = seg_vec / seg_len if seg_len > 1e-6 else np.zeros(3, dtype=float)
        return chosen_wp, u, True, poly
