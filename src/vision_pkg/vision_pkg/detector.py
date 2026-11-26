# -*- coding: utf-8 -*-
"""
detector.py —— 抗强光、多特征谱系（Multi-Signature）机载视觉目标检测器

支持多目标、多涂装协同感知与完全向后兼容的检测接口：
  - ColorBlobDetector / MultiSignatureDetector：
      单次矢量化 OpenCV 4.11 + NumPy 流水线（HSV + CIE-LAB + 全局强光抑制 +
      局部 L* 对比度 + 外环 LAB ΔE 边界伪影剔除 + 可选帧间运动/热红外线索加权），
      同时支持 crimson_uav / amber_uav / slate_uav 多靶标并发检出。
  - YoloDetector：
      基于 ultralytics YOLOv8，未安装依赖时自动优雅降级。

兼容接口（100% 保持原签名与语义）：
  - detect(bgr_image) -> (u, v) | None
  - detect_all(bgr_image) -> [(u, v), ...]
  - detect_with_meta(bgr_image) -> (u, v, area_px, bbox_span_px) | None

多目标结构化接口：
  - detect_targets(bgr_image, *, prev_bgr=None, thermal_image=None, cue_mask=None) -> List[Detection]
  - detect_all_with_meta(...)  # detect_targets 别名
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


@dataclass(frozen=True)
class SignatureProfile:
    """单类无人机目标的视觉特征谱系配置。"""

    name: str
    hsv_ranges: Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...]
    lab_lower: Optional[Tuple[int, int, int]] = None
    lab_upper: Optional[Tuple[int, int, int]] = None
    min_red_excess: Optional[int] = None
    min_local_l_diff: Optional[int] = None
    min_ring_delta_e: float = 12.0
    min_area: float = 24.0
    max_area: float = 25000.0
    min_extent: float = 0.25
    max_aspect_ratio: float = 4.5
    min_compactness: float = 0.15
    close_kernel_size: int = 3
    priority: int = 0
    glare_sensitive: bool = True


@dataclass(frozen=True)
class Detection:
    """结构化单帧二维目标检测结果。"""

    u: float
    v: float
    area_px: float
    bbox_span_px: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    class_id: str
    conf: float
    aspect_ratio: float
    extent: float
    ring_delta_e: float
    cue_score: float = 0.0


RED_LEGACY_PROFILE = SignatureProfile(
    name='crimson_uav',
    hsv_ranges=(
        ((0, 110, 55), (12, 255, 255)),
        ((168, 110, 55), (180, 255, 255)),
    ),
    lab_lower=None,
    lab_upper=None,
    min_red_excess=35,
    min_local_l_diff=None,
    min_ring_delta_e=0.0,
    min_area=24.0,
    max_area=1e9,
    min_extent=0.0,
    max_aspect_ratio=100.0,
    min_compactness=0.0,
    close_kernel_size=5,
    priority=0,
    glare_sensitive=True,
)

DEFAULT_TARGET_PROFILES: Tuple[SignatureProfile, ...] = (
    SignatureProfile(
        name='crimson_uav',
        hsv_ranges=(
            ((0, 110, 55), (12, 255, 255)),
            ((168, 110, 55), (180, 255, 255)),
        ),
        lab_lower=(25, 145, 115),
        lab_upper=(255, 255, 255),
        min_red_excess=35,
        min_local_l_diff=None,
        min_ring_delta_e=14.0,
        min_area=24.0,
        max_area=25000.0,
        min_extent=0.25,
        max_aspect_ratio=4.5,
        min_compactness=0.15,
        close_kernel_size=1,
        priority=0,
        glare_sensitive=True,
    ),
    SignatureProfile(
        name='amber_uav',
        hsv_ranges=(
            ((10, 140, 80), (21, 255, 255)),
        ),
        # LAB a* >= 148 将橙红/琥珀色靶机与土黄集装箱 (a*=136) 及黄停机坪 (a*=130) 干净分离
        lab_lower=(50, 148, 145),
        lab_upper=(255, 255, 255),
        min_red_excess=None,
        min_local_l_diff=None,
        min_ring_delta_e=14.0,
        min_area=24.0,
        max_area=25000.0,
        min_extent=0.25,
        max_aspect_ratio=4.5,
        min_compactness=0.15,
        close_kernel_size=1,
        priority=1,
        glare_sensitive=True,
    ),
    SignatureProfile(
        name='slate_uav',
        hsv_ranges=(
            ((128, 70, 45), (165, 255, 255)),
            ((98, 66, 70), (118, 89, 117)),
        ),
        lab_lower=(35, 118, 0),
        lab_upper=(245, 255, 122),
        min_red_excess=None,
        min_local_l_diff=None,
        min_ring_delta_e=14.0,
        min_area=24.0,
        max_area=25000.0,
        min_extent=0.25,
        max_aspect_ratio=4.5,
        min_compactness=0.15,
        close_kernel_size=1,
        priority=2,
        glare_sensitive=True,
    ),
)


def _parse_triplet(val: Any) -> Tuple[int, int, int]:
    if not isinstance(val, (list, tuple)) or len(val) != 3:
        raise ValueError('triplet must be a 3-element sequence')
    out: List[int] = []
    for elem in val:
        if isinstance(elem, bool):
            raise TypeError('bool is not a valid numeric channel value')
        f = float(elem)
        if not math.isfinite(f):
            raise ValueError('non-finite channel value')
        out.append(int(f))
    return (out[0], out[1], out[2])


def _parse_finite_float(val: Any, default: float) -> float:
    if val is None:
        return default
    if isinstance(val, bool):
        raise TypeError('bool is not a valid float')
    f = float(val)
    if not math.isfinite(f):
        raise ValueError('non-finite float')
    return f


def _parse_opt_int(val: Any) -> Optional[int]:
    if val is None:
        return None
    if isinstance(val, bool):
        raise TypeError('bool is not a valid int')
    f = float(val)
    if not math.isfinite(f):
        raise ValueError('non-finite int')
    return int(f)


def profiles_from_json(json_str: Optional[str]) -> Optional[Tuple[SignatureProfile, ...]]:
    """从 JSON 字符串解析自定义 SignatureProfile 列表；空字符串或无效 JSON 返回 None。"""
    if not json_str or not str(json_str).strip():
        return None
    try:
        raw = json.loads(json_str)
    except Exception:
        return None
    if isinstance(raw, dict):
        raw = raw.get('profiles', [])
    if not isinstance(raw, list) or not raw:
        return None

    parsed: List[SignatureProfile] = []
    for item in raw:
        if not isinstance(item, dict) or 'name' not in item or 'hsv_ranges' not in item:
            continue
        try:
            raw_name = item['name']
            if not isinstance(raw_name, str) or not raw_name.strip():
                continue
            raw_hsv = item['hsv_ranges']
            if not isinstance(raw_hsv, (list, tuple)) or not raw_hsv:
                continue
            hsv_list: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = []
            for r in raw_hsv:
                if not isinstance(r, (list, tuple)) or len(r) != 2:
                    raise ValueError('invalid hsv range pair')
                hsv_list.append((_parse_triplet(r[0]), _parse_triplet(r[1])))
            hsv_ranges = tuple(hsv_list)

            lab_lo = item.get('lab_lower')
            lab_hi = item.get('lab_upper')
            lab_lower = _parse_triplet(lab_lo) if lab_lo is not None else None
            lab_upper = _parse_triplet(lab_hi) if lab_hi is not None else None

            min_area_v = _parse_finite_float(item.get('min_area'), 24.0)
            max_area_v = _parse_finite_float(item.get('max_area'), 25000.0)
            min_ext_v = _parse_finite_float(item.get('min_extent'), 0.25)
            max_asp_v = _parse_finite_float(item.get('max_aspect_ratio'), 4.5)
            min_comp_v = _parse_finite_float(item.get('min_compactness'), 0.15)
            min_ring_v = _parse_finite_float(item.get('min_ring_delta_e'), 12.0)

            close_k = _parse_opt_int(item.get('close_kernel_size', 1))
            prio = _parse_opt_int(item.get('priority', 0))
            parsed.append(
                SignatureProfile(
                    name=raw_name.strip(),
                    hsv_ranges=hsv_ranges,
                    lab_lower=lab_lower,
                    lab_upper=lab_upper,
                    min_red_excess=_parse_opt_int(item.get('min_red_excess')),
                    min_local_l_diff=_parse_opt_int(item.get('min_local_l_diff')),
                    min_ring_delta_e=min_ring_v,
                    min_area=min_area_v,
                    max_area=max_area_v,
                    min_extent=min_ext_v,
                    max_aspect_ratio=max_asp_v,
                    min_compactness=min_comp_v,
                    close_kernel_size=max(1, close_k if close_k is not None else 1),
                    priority=prio if prio is not None else 0,
                    glare_sensitive=bool(item.get('glare_sensitive', True)),
                )
            )
        except (TypeError, ValueError, IndexError, KeyError):
            continue
    return tuple(parsed) if parsed else None


def _normalize_bgr(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """校验并规整输入图像为连续 uint8 BGR (H>=4, W>=4, C==3)。"""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None
    if image.ndim == 2:
        if image.shape[0] < 4 or image.shape[1] < 4:
            return None
        arr = _to_uint8(image)
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if image.ndim != 3 or image.shape[0] < 4 or image.shape[1] < 4:
        return None
    ch = image.shape[2]
    arr = _to_uint8(image)
    if ch == 3:
        return np.ascontiguousarray(arr)
    if ch == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if ch == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    return None


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        max_v = float(np.nanmax(arr)) if arr.size > 0 else 0.0
        scale = 255.0 if max_v <= 1.0 + 1e-6 else 1.0
        return np.clip(np.nan_to_num(arr, nan=0.0) * scale, 0.0, 255.0).astype(np.uint8)
    return np.clip(arr, 0, 255).astype(np.uint8)


def _build_cue_map(
    bgr: np.ndarray,
    prev_bgr: Optional[np.ndarray] = None,
    thermal_image: Optional[np.ndarray] = None,
    cue_mask: Optional[np.ndarray] = None,
    cur_gray: Optional[np.ndarray] = None,
    prev_gray: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], bool]:
    """构建 [0, 255] uint8 线索加权图（运动帧差 / 热红外 / 外部掩膜），并检测全局自运动。"""
    h, w = bgr.shape[:2]
    cue_u8: Optional[np.ndarray] = None
    ego_motion = False
    ego_thresh_px = int(0.12 * h * w)

    if prev_gray is not None and prev_gray.shape[:2] == (h, w):
        g_cur = (
            cur_gray
            if (cur_gray is not None and cur_gray.shape[:2] == (h, w))
            else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        )
        diff = cv2.absdiff(g_cur, prev_gray)
        _, active = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
        nz = cv2.countNonZero(active)
        if nz > ego_thresh_px:
            ego_motion = True
        elif nz >= 16:
            m_u8 = cv2.convertScaleAbs(cv2.subtract(diff, 10), alpha=255.0 / 35.0)
            cue_u8 = m_u8 if cue_u8 is None else cv2.max(cue_u8, m_u8)
    elif prev_bgr is not None and prev_bgr is not bgr:
        prev_norm = _normalize_bgr(prev_bgr)
        if prev_norm is not None and prev_norm.shape[:2] == (h, w):
            g_cur = (
                cur_gray
                if (cur_gray is not None and cur_gray.shape[:2] == (h, w))
                else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            )
            gray_prev = cv2.cvtColor(prev_norm, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(g_cur, gray_prev)
            _, active = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
            nz = cv2.countNonZero(active)
            if nz > ego_thresh_px:
                ego_motion = True
            elif nz >= 16:
                m_u8 = cv2.convertScaleAbs(cv2.subtract(diff, 10), alpha=255.0 / 35.0)
                cue_u8 = m_u8 if cue_u8 is None else cv2.max(cue_u8, m_u8)

    if thermal_image is not None and isinstance(thermal_image, np.ndarray) and thermal_image.size > 0:
        if thermal_image.shape[:2] == (h, w):
            if thermal_image.ndim == 3:
                th = cv2.cvtColor(_to_uint8(thermal_image), cv2.COLOR_BGR2GRAY)
            elif thermal_image.dtype == np.uint16:
                th_min = float(np.min(thermal_image))
                th_max = float(np.max(thermal_image))
                if th_max > th_min + 1.0:
                    scale = 255.0 / (th_max - th_min)
                    th = np.clip(
                        (thermal_image.astype(np.float32) - th_min) * scale, 0, 255
                    ).astype(np.uint8)
                else:
                    th = np.zeros((h, w), dtype=np.uint8)
            else:
                th = _to_uint8(thermal_image)
            t_u8 = cv2.convertScaleAbs(cv2.subtract(th, 120), alpha=255.0 / 100.0)
            cue_u8 = t_u8 if cue_u8 is None else cv2.max(cue_u8, t_u8)

    if cue_mask is not None and isinstance(cue_mask, np.ndarray) and cue_mask.size > 0:
        if cue_mask.shape[:2] == (h, w):
            cm = cue_mask[:, :, 0] if cue_mask.ndim == 3 else cue_mask
            if cm.dtype == np.bool_:
                cm_u8 = cm.astype(np.uint8) * 255
            elif np.issubdtype(cm.dtype, np.floating):
                cm_u8 = np.clip(cm * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                cm_u8 = _to_uint8(cm)
            cue_u8 = cm_u8 if cue_u8 is None else cv2.max(cue_u8, cm_u8)

    return cue_u8, ego_motion


def _split_touching_roi(
    fg_roi: np.ndarray,
    x0: int,
    y0: int,
    area: float,
    bw: int,
    bh: int,
    min_part_area: float = 24.0,
) -> List[Tuple[float, float, float, int, int, int, int, np.ndarray]]:
    """
    对可疑粘连连通域执行 1px 零填充 ROI 局部距离变换，分离同色近邻/对角粘连目标。
    通过峰值谷值深度与分块最小面积检验，确保单一机翼细长目标不会被误拆。
    """
    if area < max(48.0, 2.0 * float(min_part_area)) or max(bw, bh) < 12 or min(bw, bh) < 5:
        return []

    fg_u8 = fg_roi.astype(np.uint8) * 255
    fg_pad = cv2.copyMakeBorder(fg_u8, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(fg_pad, cv2.DIST_L2, 3)[1:-1, 1:-1]
    max_d = float(dist.max())
    if max_d < 2.0:
        return []

    dilated = cv2.dilate(dist, np.ones((5, 5), np.uint8))
    peak_ys, peak_xs = np.where((dist >= 0.50 * max_d) & (dist >= dilated - 1e-5))
    if len(peak_xs) < 2:
        return []

    # 贪心非极大值抑制，保留间距 >= 6 px 的显著峰值
    order = np.argsort(dist[peak_ys, peak_xs])[::-1]
    peaks: List[Tuple[int, int, float]] = []
    for idx in order:
        px = int(peak_xs[idx])
        py = int(peak_ys[idx])
        pd = float(dist[py, px])
        if all((px - qx) ** 2 + (py - qy) ** 2 >= 36 for qx, qy, _ in peaks):
            peaks.append((px, py, pd))
        if len(peaks) >= 4:
            break

    if len(peaks) < 2:
        return []

    # 验证主峰与次峰连线间存在真实“颈部”谷值（避免把单架均匀长条机翼切两半）
    p1x, p1y, p1d = peaks[0]
    p2x, p2y, p2d = peaks[1]
    n_steps = max(int(math.hypot(p2x - p1x, p2y - p1y)) + 1, 5)
    xs = np.clip(np.round(np.linspace(p1x, p2x, n_steps)).astype(int), 0, fg_roi.shape[1] - 1)
    ys = np.clip(np.round(np.linspace(p1y, p2y, n_steps)).astype(int), 0, fg_roi.shape[0] - 1)
    min_valley = float(np.min(dist[ys, xs]))
    if min_valley >= 0.72 * min(p1d, p2d):
        return []

    # Voronoi 最近峰分割
    fg_y, fg_x = np.where(fg_roi)
    if fg_y.size == 0:
        return []
    dists_to_peaks = np.stack(
        [(fg_x - px) ** 2 + (fg_y - py) ** 2 for px, py, _ in peaks],
        axis=0,
    )
    assign = np.argmin(dists_to_peaks, axis=0)

    parts: List[Tuple[float, float, float, int, int, int, int, np.ndarray]] = []
    rh, rw = fg_roi.shape
    for k in range(len(peaks)):
        sel = assign == k
        if not np.any(sel):
            continue
        ky = fg_y[sel]
        kx = fg_x[sel]
        k_area = float(kx.size)
        if k_area < min_part_area:
            continue
        sub_mask = np.zeros((rh, rw), dtype=bool)
        sub_mask[ky, kx] = True
        min_x = int(kx.min())
        max_x = int(kx.max())
        min_y = int(ky.min())
        max_y = int(ky.max())
        parts.append((
            float(x0 + kx.mean()),
            float(y0 + ky.mean()),
            k_area,
            x0 + min_x,
            y0 + min_y,
            max_x - min_x + 1,
            max_y - min_y + 1,
            sub_mask,
        ))
    return parts if len(parts) >= 2 else []


class ColorBlobDetector:
    """
    抗强光多特征谱系色块检测器。

    默认构造参数与旧版 ColorBlobDetector 完全一致（仅激活红区高光抑制谱系），
    传入 `enable_multi_signature=True` 或 `profiles=DEFAULT_TARGET_PROFILES` 可同时
    检测 crimson_uav / amber_uav / slate_uav 多目标。
    """

    def __init__(
        self,
        hsv_lower1: Tuple[int, int, int] = (0, 110, 55),
        hsv_upper1: Tuple[int, int, int] = (12, 255, 255),
        hsv_lower2: Tuple[int, int, int] = (168, 110, 55),
        hsv_upper2: Tuple[int, int, int] = (180, 255, 255),
        min_area: float = 24.0,
        enable_glare_suppression: bool = True,
        profiles: Optional[Union[str, Sequence[SignatureProfile]]] = None,
        enable_multi_signature: bool = False,
        enable_touching_split: bool = True,
    ) -> None:
        self.lower1 = np.array(hsv_lower1, dtype=np.uint8)
        self.upper1 = np.array(hsv_upper1, dtype=np.uint8)
        self.lower2 = np.array(hsv_lower2, dtype=np.uint8)
        self.upper2 = np.array(hsv_upper2, dtype=np.uint8)
        self.min_area = float(min_area)
        self.enable_glare_suppression = bool(enable_glare_suppression)
        self.enable_touching_split = bool(enable_touching_split)

        if profiles is not None:
            if isinstance(profiles, str) and profiles.lower() == 'default':
                base_profiles = DEFAULT_TARGET_PROFILES
            elif isinstance(profiles, str):
                parsed = profiles_from_json(profiles)
                base_profiles = parsed if parsed is not None else DEFAULT_TARGET_PROFILES
            else:
                base_profiles = tuple(profiles)
            self.profiles: Tuple[SignatureProfile, ...] = tuple(
                sorted(
                    (
                        replace(p, min_area=max(p.min_area, self.min_area))
                        if self.min_area != 24.0
                        else p
                        for p in base_profiles
                    ),
                    key=lambda p: p.priority,
                )
            )
            self.multi_signature = len(self.profiles) > 1 or (
                len(self.profiles) == 1 and self.profiles[0] != RED_LEGACY_PROFILE
            )
        elif enable_multi_signature:
            self.profiles = tuple(
                sorted(
                    (
                        replace(p, min_area=max(p.min_area, self.min_area))
                        if self.min_area != 24.0
                        else p
                        for p in DEFAULT_TARGET_PROFILES
                    ),
                    key=lambda p: p.priority,
                )
            )
            self.multi_signature = True
        else:
            legacy_profile = replace(
                RED_LEGACY_PROFILE,
                hsv_ranges=(
                    (tuple(int(v) for v in hsv_lower1), tuple(int(v) for v in hsv_upper1)),
                    (tuple(int(v) for v in hsv_lower2), tuple(int(v) for v in hsv_upper2)),
                ),
                min_red_excess=35 if self.enable_glare_suppression else None,
                min_area=self.min_area,
                glare_sensitive=self.enable_glare_suppression,
            )
            self.profiles = (legacy_profile,)
            self.multi_signature = False

        self._kernel_small = np.ones((3, 3), np.uint8)
        self._kernel_close5 = np.ones((5, 5), np.uint8)
        self._glare_lower = np.array((0, 0, 205), dtype=np.uint8)
        self._glare_upper = np.array((180, 65, 255), dtype=np.uint8)
        self._kernel_cache: Dict[int, np.ndarray] = {
            3: self._kernel_small,
            5: self._kernel_close5,
        }
        # 预构建每个谱系的 uint8 阈值数组与属性，避免每帧重复构造 NumPy 数组
        self._compiled_profiles = tuple(
            (
                p,
                tuple(
                    (np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
                    for lo, hi in p.hsv_ranges
                ),
                np.array(p.lab_lower, dtype=np.uint8) if p.lab_lower is not None else None,
                np.array(p.lab_upper, dtype=np.uint8) if p.lab_upper is not None else None,
                bool(
                    p.glare_sensitive
                    and self.enable_glare_suppression
                    and any(lo[1] <= 65 and hi[2] >= 205 for lo, hi in p.hsv_ranges)
                ),
            )
            for p in self.profiles
        )
        self._need_lab = any(
            p.lab_lower is not None
            or p.lab_upper is not None
            or p.min_local_l_diff is not None
            or p.min_ring_delta_e > 0.0
            for p in self.profiles
        )
        self._need_local_l = any(p.min_local_l_diff is not None for p in self.profiles)

    def _get_kernel(self, size: int) -> np.ndarray:
        k = max(1, int(size))
        if k not in self._kernel_cache:
            self._kernel_cache[k] = np.ones((k, k), np.uint8)
        return self._kernel_cache[k]

    def _evaluate_candidate(
        self,
        profile: SignatureProfile,
        bgr_img: np.ndarray,
        lab_roi: Optional[Tuple[int, int, np.ndarray]],
        cue_map: Optional[np.ndarray],
        cx: float,
        cy: float,
        area: float,
        bx: int,
        by: int,
        bw: int,
        bh: int,
        fg_roi: np.ndarray,
        claimed_roi: Optional[np.ndarray],
        img_h: int,
        img_w: int,
        prev_gray_full: Optional[np.ndarray] = None,
        cur_gray_full: Optional[np.ndarray] = None,
    ) -> Optional[Detection]:
        if area < profile.min_area or area > profile.max_area:
            return None
        if bw <= 0 or bh <= 0:
            return None

        extent = float(area / float(bw * bh))
        aspect = float(max(bw, bh) / float(max(1, min(bw, bh))))
        if extent < profile.min_extent or aspect > profile.max_aspect_ratio:
            return None

        if profile.min_compactness > 0.0:
            fg_u8 = fg_roi.astype(np.uint8) * 255
            contours, _ = cv2.findContours(fg_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                perim = sum(float(cv2.arcLength(c, True)) for c in contours)
                if perim > 1e-6:
                    compactness = (4.0 * math.pi * float(area)) / (perim * perim)
                    if compactness < profile.min_compactness:
                        return None

        cue_score = 0.0
        if cue_map is not None:
            cue_roi = cue_map[by:by + bh, bx:bx + bw]
            if np.any(fg_roi):
                cue_score = float(np.clip(np.mean(cue_roi[fg_roi]) / 255.0, 0.0, 1.0))
        elif prev_gray_full is not None and np.any(fg_roi):
            pad_m = 4
            mx0 = max(0, bx - pad_m)
            my0 = max(0, by - pad_m)
            mx1 = min(img_w, bx + bw + pad_m)
            my1 = min(img_h, by + bh + pad_m)
            if cur_gray_full is not None:
                g_patch = cur_gray_full[my0:my1, mx0:mx1]
            else:
                g_patch = cv2.cvtColor(bgr_img[my0:my1, mx0:mx1], cv2.COLOR_BGR2GRAY)
            diff_patch = cv2.absdiff(g_patch, prev_gray_full[my0:my1, mx0:mx1])
            fg_p = np.zeros((my1 - my0, mx1 - mx0), dtype=bool)
            fg_p[by - my0:by - my0 + bh, bx - mx0:bx - mx0 + bw] = fg_roi
            bg_p = ~fg_p
            fg_mean_d = float(diff_patch[fg_p].mean())
            bg_mean_d = float(diff_patch[bg_p].mean()) if np.any(bg_p) else 0.0
            net_d = max(0.0, fg_mean_d - bg_mean_d)
            if net_d >= 12.0:
                cue_score = float(np.clip((net_d - 10.0) / 35.0, 0.0, 1.0))

        ring_de = 0.0
        if profile.min_ring_delta_e > 0.0:
            pad = max(4, min(10, max(bw, bh) // 2))
            rx0 = max(0, bx - pad)
            ry0 = max(0, by - pad)
            rx1 = min(img_w, bx + bw + pad)
            ry1 = min(img_h, by + bh + pad)
            lab_patch: Optional[np.ndarray] = None
            if lab_roi is not None:
                ox, oy, lab_arr = lab_roi
                if rx0 >= ox and ry0 >= oy and rx1 <= ox + lab_arr.shape[1] and ry1 <= oy + lab_arr.shape[0]:
                    lab_patch = lab_arr[ry0 - oy:ry1 - oy, rx0 - ox:rx1 - ox]
            if lab_patch is None:
                lab_patch = cv2.cvtColor(bgr_img[ry0:ry1, rx0:rx1], cv2.COLOR_BGR2LAB)

            fg_patch = np.zeros((ry1 - ry0, rx1 - rx0), dtype=bool)
            fg_patch[by - ry0:by - ry0 + bh, bx - rx0:bx - rx0 + bw] = fg_roi
            ring_mask = ~fg_patch
            if claimed_roi is not None:
                ring_mask &= (claimed_roi[ry0:ry1, rx0:rx1] == 0)

            fg_pixels = lab_patch[fg_patch]
            ring_pixels = lab_patch[ring_mask]
            if fg_pixels.size > 0 and ring_pixels.size > 0:
                fg_mean = np.mean(fg_pixels, axis=0, dtype=np.float32)
                n_ring = ring_pixels.shape[0]
                ring_med = np.partition(ring_pixels, n_ring >> 1, axis=0)[n_ring >> 1].astype(np.float32)
                ring_de = float(np.linalg.norm(fg_mean - ring_med))
                eff_ring_thr = profile.min_ring_delta_e
                if cue_score > 0.2:
                    eff_ring_thr = max(4.0, profile.min_ring_delta_e * (1.0 - 0.65 * cue_score))
                if ring_de < eff_ring_thr:
                    return None
            else:
                return None

        conf = (
            0.45
            + 0.25 * min(ring_de / 45.0, 1.0)
            + 0.15 * min(area / 120.0, 1.0)
            + 0.20 * cue_score
        )
        conf = float(min(0.99, max(0.05, conf)))

        return Detection(
            u=float(cx),
            v=float(cy),
            area_px=float(area),
            bbox_span_px=float(max(bw, bh)),
            bbox=(int(bx), int(by), int(bw), int(bh)),
            class_id=profile.name,
            conf=conf,
            aspect_ratio=aspect,
            extent=extent,
            ring_delta_e=ring_de,
            cue_score=cue_score,
        )

    def detect_targets(
        self,
        bgr_image: np.ndarray,
        *,
        prev_bgr: Optional[np.ndarray] = None,
        prev_gray: Optional[np.ndarray] = None,
        thermal_image: Optional[np.ndarray] = None,
        cue_mask: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """多目标、多谱系矢量化检测，返回按置信度与面积排序的 Detection 列表。"""
        bgr = _normalize_bgr(bgr_image)
        if bgr is None:
            return []

        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        cur_gray: Optional[np.ndarray] = None
        if prev_bgr is not None or prev_gray is not None:
            cur_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        cue_map: Optional[np.ndarray] = None
        ego_motion = False
        if prev_bgr is not None or thermal_image is not None or cue_mask is not None:
            cue_map, ego_motion = _build_cue_map(
                bgr,
                prev_bgr=prev_bgr,
                thermal_image=thermal_image,
                cue_mask=cue_mask,
                cur_gray=cur_gray,
                prev_gray=prev_gray,
            )
        elif prev_gray is not None and prev_gray.shape[:2] == (h, w) and cur_gray is not None:
            diff_full = cv2.absdiff(cur_gray, prev_gray)
            _, act_full = cv2.threshold(diff_full, 12, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(act_full) > int(0.12 * h * w):
                ego_motion = True

        prev_gray_full: Optional[np.ndarray] = (
            prev_gray
            if (not ego_motion and cue_map is None and prev_gray is not None and prev_gray.shape[:2] == (h, w))
            else None
        )

        claimed_mask: Optional[np.ndarray] = None
        detections: List[Detection] = []
        multi_p = len(self._compiled_profiles) > 1

        for profile, hsv_arrs, lab_lo_arr, lab_hi_arr, needs_glare_roi in self._compiled_profiles:
            if not hsv_arrs:
                continue

            lo0, hi0 = hsv_arrs[0]
            mask = cv2.inRange(hsv, lo0, hi0)
            for lo, hi in hsv_arrs[1:]:
                mask |= cv2.inRange(hsv, lo, hi)

            # 早期快速跳过空类（先用 tight boundingRect 裁出活跃区域，避免后续全图操作）
            rx0, ry0, rw0, rh0 = cv2.boundingRect(mask)
            if rw0 <= 0 or rh0 <= 0 or (rw0 * rh0) < int(profile.min_area):
                continue
            sub_mask0 = mask[ry0:ry0 + rh0, rx0:rx0 + rw0]

            if cv2.countNonZero(sub_mask0) < int(profile.min_area):
                continue

            if rw0 * rh0 > 4096 and profile.min_local_l_diff is None and profile.close_kernel_size <= 1:
                n_isl, isl_labels, isl_stats, _ = cv2.connectedComponentsWithStats(sub_mask0, connectivity=8)
                if n_isl <= 1:
                    continue
                isl_areas = isl_stats[1:, cv2.CC_STAT_AREA]
                isl_valid = np.where(isl_areas >= profile.min_area)[0] + 1
                if isl_valid.size == 0:
                    continue
                roi_boxes: List[Tuple[int, int, int, int, np.ndarray]] = []
                for i_lbl in isl_valid:
                    ix = int(isl_stats[i_lbl, cv2.CC_STAT_LEFT])
                    iy = int(isl_stats[i_lbl, cv2.CC_STAT_TOP])
                    iw = int(isl_stats[i_lbl, cv2.CC_STAT_WIDTH])
                    ih = int(isl_stats[i_lbl, cv2.CC_STAT_HEIGHT])
                    isl_patch = np.where(
                        isl_labels[iy:iy + ih, ix:ix + iw] == i_lbl,
                        np.uint8(255),
                        np.uint8(0),
                    )
                    roi_boxes.append((rx0 + ix, ry0 + iy, iw, ih, isl_patch))
            else:
                roi_boxes = [(rx0, ry0, rw0, rh0, sub_mask0)]

            for rx, ry, rw, rh, sub_mask in roi_boxes:
                lab_roi: Optional[Tuple[int, int, np.ndarray]] = None
                need_roi_lab = (lab_lo_arr is not None and lab_hi_arr is not None) or (
                    profile.min_local_l_diff is not None
                )
                if need_roi_lab:
                    pad_l = 18 if profile.min_local_l_diff is not None else 10
                    lx0 = max(0, rx - pad_l)
                    ly0 = max(0, ry - pad_l)
                    lx1 = min(w, rx + rw + pad_l)
                    ly1 = min(h, ry + rh + pad_l)
                    sub_lab_pad = cv2.cvtColor(bgr[ly0:ly1, lx0:lx1], cv2.COLOR_BGR2LAB)
                    lab_roi = (lx0, ly0, sub_lab_pad)
                    off_x = rx - lx0
                    off_y = ry - ly0
                    sub_lab = sub_lab_pad[off_y:off_y + rh, off_x:off_x + rw]
                    if lab_lo_arr is not None and lab_hi_arr is not None:
                        lab_mask = cv2.inRange(sub_lab, lab_lo_arr, lab_hi_arr)
                        sub_mask = cv2.bitwise_and(sub_mask, lab_mask)

                if profile.min_red_excess is not None and self.enable_glare_suppression:
                    b_ch, g_ch, r_ch = cv2.split(bgr[ry:ry + rh, rx:rx + rw])
                    sub_red_excess = cv2.subtract(r_ch, cv2.max(g_ch, b_ch))
                    _, chroma_mask = cv2.threshold(
                        sub_red_excess, int(profile.min_red_excess), 255, cv2.THRESH_BINARY
                    )
                    sub_mask = cv2.bitwise_and(sub_mask, chroma_mask)

                if needs_glare_roi:
                    sub_glare = cv2.bitwise_not(
                        cv2.inRange(hsv[ry:ry + rh, rx:rx + rw], self._glare_lower, self._glare_upper)
                    )
                    sub_mask = cv2.bitwise_and(sub_mask, sub_glare)

                if profile.min_local_l_diff is not None and lab_roi is not None:
                    l_ch_pad = lab_roi[2][:, :, 0]
                    l_blur_pad = cv2.boxFilter(l_ch_pad, -1, (37, 37), normalize=True)
                    l_bg_pad = cv2.absdiff(l_ch_pad, l_blur_pad)
                    off_x = rx - lab_roi[0]
                    off_y = ry - lab_roi[1]
                    sub_l_bg = l_bg_pad[off_y:off_y + rh, off_x:off_x + rw]
                    _, contrast_mask = cv2.threshold(
                        sub_l_bg, int(profile.min_local_l_diff), 255, cv2.THRESH_BINARY
                    )
                    sub_cue: Optional[np.ndarray] = (
                        cue_map[ry:ry + rh, rx:rx + rw] if cue_map is not None else None
                    )
                    if sub_cue is None and prev_gray_full is not None:
                        cur_sub_gray = (
                            cur_gray[ry:ry + rh, rx:rx + rw]
                            if cur_gray is not None
                            else cv2.cvtColor(bgr[ry:ry + rh, rx:rx + rw], cv2.COLOR_BGR2GRAY)
                        )
                        diff_sub = cv2.absdiff(cur_sub_gray, prev_gray_full[ry:ry + rh, rx:rx + rw])
                        _, act_sub = cv2.threshold(diff_sub, 12, 255, cv2.THRESH_BINARY)
                        if cv2.countNonZero(act_sub) >= 12:
                            sub_cue = cv2.convertScaleAbs(cv2.subtract(diff_sub, 10), alpha=255.0 / 35.0)
                    if sub_cue is not None:
                        relaxed_thr = max(4, int(profile.min_local_l_diff // 2))
                        _, relaxed_mask = cv2.threshold(sub_l_bg, relaxed_thr, 255, cv2.THRESH_BINARY)
                        _, cued_hi = cv2.threshold(sub_cue, 90, 255, cv2.THRESH_BINARY)
                        contrast_mask = cv2.bitwise_or(
                            contrast_mask,
                            cv2.bitwise_and(relaxed_mask, cued_hi),
                        )
                    sub_mask = cv2.bitwise_and(sub_mask, contrast_mask)

                if claimed_mask is not None:
                    sub_claimed = claimed_mask[ry:ry + rh, rx:rx + rw]
                    sub_mask = cv2.bitwise_and(sub_mask, cv2.bitwise_not(sub_claimed))

                if cv2.countNonZero(sub_mask) < int(profile.min_area):
                    continue

                # 进一步收紧 ROI 到最终二值掩膜外接矩形，大幅加速形态学与连通域分析
                rx2, ry2, rw2, rh2 = cv2.boundingRect(sub_mask)
                if rw2 <= 0 or rh2 <= 0:
                    continue
                roi_x = rx + rx2
                roi_y = ry + ry2
                roi_mask = sub_mask[ry2:ry2 + rh2, rx2:rx2 + rw2]

                open_kernel = self._kernel_close5 if profile.min_area >= 120.0 else self._kernel_small
                roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, open_kernel)
                if profile.close_kernel_size > 1:
                    roi_mask = cv2.morphologyEx(
                        roi_mask, cv2.MORPH_CLOSE, self._get_kernel(profile.close_kernel_size)
                    )

                if claimed_mask is not None:
                    roi_mask = cv2.bitwise_and(
                        roi_mask, cv2.bitwise_not(claimed_mask[roi_y:roi_y + rh2, roi_x:roi_x + rw2])
                    )

                if cv2.countNonZero(roi_mask) < int(profile.min_area):
                    continue

                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    roi_mask, connectivity=8
                )
                if num_labels <= 1:
                    continue

                areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float64)
                valid_indices = np.where((areas >= profile.min_area) & (areas <= profile.max_area))[0] + 1
                if valid_indices.size == 0:
                    continue

                for lbl in valid_indices:
                    lx = int(stats[lbl, cv2.CC_STAT_LEFT])
                    ly = int(stats[lbl, cv2.CC_STAT_TOP])
                    bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
                    bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
                    bx = roi_x + lx
                    by = roi_y + ly
                    area = float(stats[lbl, cv2.CC_STAT_AREA])
                    cx = float(roi_x + centroids[lbl, 0])
                    cy = float(roi_y + centroids[lbl, 1])

                    fg_roi = labels[ly:ly + bh, lx:lx + bw] == lbl

                    split_parts: List[Tuple[float, float, float, int, int, int, int, np.ndarray]] = []
                    if self.enable_touching_split and profile.close_kernel_size <= 3:
                        aspect_c = max(bw, bh) / float(max(1, min(bw, bh)))
                        extent_c = area / float(max(1, bw * bh))
                        if not (extent_c >= 0.85 and aspect_c <= 1.35):
                            split_parts = _split_touching_roi(
                                fg_roi, bx, by, area, bw, bh, min_part_area=profile.min_area
                            )

                    if split_parts:
                        for sx, sy, s_area, sbx, sby, sbw, sbh, s_fg in split_parts:
                            sub_roi = s_fg[sby - by:sby - by + sbh, sbx - bx:sbx - bx + sbw]
                            det = self._evaluate_candidate(
                                profile,
                                bgr,
                                lab_roi,
                                cue_map,
                                sx,
                                sy,
                                s_area,
                                sbx,
                                sby,
                                sbw,
                                sbh,
                                sub_roi,
                                claimed_mask,
                                h,
                                w,
                                prev_gray_full,
                                cur_gray,
                            )
                            if det is not None:
                                detections.append(det)
                                if multi_p:
                                    if claimed_mask is None:
                                        claimed_mask = np.zeros((h, w), dtype=np.uint8)
                                    claimed_mask[sby:sby + sbh, sbx:sbx + sbw][sub_roi] = 255
                    else:
                        det = self._evaluate_candidate(
                            profile,
                            bgr,
                            lab_roi,
                            cue_map,
                            cx,
                            cy,
                            area,
                            bx,
                            by,
                            bw,
                            bh,
                            fg_roi,
                            claimed_mask,
                            h,
                            w,
                            prev_gray_full,
                            cur_gray,
                        )
                        if det is not None:
                            detections.append(det)
                            if multi_p:
                                if claimed_mask is None:
                                    claimed_mask = np.zeros((h, w), dtype=np.uint8)
                                claimed_mask[by:by + bh, bx:bx + bw][fg_roi] = 255

        if self.multi_signature:
            detections.sort(key=lambda d: (d.conf, d.area_px), reverse=True)
        else:
            detections.sort(key=lambda d: d.area_px, reverse=True)
        return detections

    detect_all_with_meta = detect_targets

    def _extract_candidates(self, bgr_image: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """返回 [(u, v, area, bbox_max_span), ...]，按面积降序排列。"""
        dets = self.detect_targets(bgr_image)
        out = [(d.u, d.v, d.area_px, d.bbox_span_px) for d in dets]
        out.sort(key=lambda r: r[2], reverse=True)
        return out

    def detect_all(self, bgr_image: np.ndarray) -> List[Tuple[float, float]]:
        """返回所有激活谱系中面积超过 min_area 的目标质心列表 [(u, v), ...]。"""
        return [(d.u, d.v) for d in self.detect_targets(bgr_image)]

    def detect(self, bgr_image: np.ndarray) -> Optional[Tuple[float, float]]:
        """返回主目标质心 (u, v)。"""
        cands = self._extract_candidates(bgr_image)
        return (cands[0][0], cands[0][1]) if cands else None

    def detect_with_meta(self, bgr_image: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """返回主目标 (u, v, area_px, bbox_span_px)。"""
        cands = self._extract_candidates(bgr_image)
        return cands[0] if cands else None


class MultiSignatureDetector(ColorBlobDetector):
    """默认启用 crimson_uav + amber_uav + slate_uav 三类特征谱系的多目标检测器。"""

    def __init__(
        self,
        profiles: Optional[Union[str, Sequence[SignatureProfile]]] = None,
        min_area: float = 24.0,
        enable_glare_suppression: bool = True,
        enable_touching_split: bool = True,
    ) -> None:
        super().__init__(
            min_area=min_area,
            enable_glare_suppression=enable_glare_suppression,
            profiles=DEFAULT_TARGET_PROFILES if profiles is None else profiles,
            enable_multi_signature=True,
            enable_touching_split=enable_touching_split,
        )


class YoloDetector:
    """YOLOv8 检测后端：未安装 ultralytics 时优雅降级。"""

    def __init__(self, model_path: str = "yolov8n.pt", conf_thres: float = 0.4, target_class: int = -1) -> None:
        self.conf_thres = float(conf_thres)
        self.target_class = int(target_class)
        self._model = None
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
        except Exception:
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def _extract_hits(
        self, bgr_image: np.ndarray
    ) -> List[Tuple[float, float, float, float, float, int, Tuple[int, int, int, int]]]:
        if self._model is None or bgr_image is None or bgr_image.size == 0:
            return []
        results = self._model.predict(bgr_image, verbose=False, conf=self.conf_thres)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        boxes = results[0].boxes
        hits: List[Tuple[float, float, float, float, float, int, Tuple[int, int, int, int]]] = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if self.target_class >= 0 and cls_id != self.target_class:
                continue
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            bw = max(1.0, float(x2 - x1))
            bh = max(1.0, float(y2 - y1))
            hits.append((
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                conf,
                bw * bh,
                max(bw, bh),
                cls_id,
                (int(round(x1)), int(round(y1)), int(round(bw)), int(round(bh))),
            ))
        hits.sort(key=lambda h: h[2], reverse=True)
        return hits

    def detect_targets(
        self,
        bgr_image: np.ndarray,
        *,
        prev_bgr: Optional[np.ndarray] = None,
        prev_gray: Optional[np.ndarray] = None,
        thermal_image: Optional[np.ndarray] = None,
        cue_mask: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        del prev_bgr, prev_gray, thermal_image, cue_mask
        hits = self._extract_hits(bgr_image)
        names = getattr(self._model, 'names', {}) if self._model is not None else {}
        out: List[Detection] = []
        for u, v, conf, area, span, cls_id, bbox in hits:
            bx, by, bw, bh = bbox
            aspect = float(max(bw, bh) / float(max(1, min(bw, bh))))
            cls_name = str(names.get(cls_id, f'class_{cls_id}')) if isinstance(names, dict) else f'class_{cls_id}'
            out.append(
                Detection(
                    u=u,
                    v=v,
                    area_px=area,
                    bbox_span_px=span,
                    bbox=bbox,
                    class_id=cls_name,
                    conf=conf,
                    aspect_ratio=aspect,
                    extent=1.0,
                    ring_delta_e=0.0,
                    cue_score=0.0,
                )
            )
        return out

    detect_all_with_meta = detect_targets

    def detect_all(self, bgr_image: np.ndarray) -> List[Tuple[float, float]]:
        return [(u, v) for u, v, _, _, _, _, _ in self._extract_hits(bgr_image)]

    def detect(self, bgr_image: np.ndarray) -> Optional[Tuple[float, float]]:
        hits = self._extract_hits(bgr_image)
        return (hits[0][0], hits[0][1]) if hits else None

    def detect_with_meta(self, bgr_image: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        hits = self._extract_hits(bgr_image)
        if not hits:
            return None
        u, v, _, area, span, _, _ = hits[0]
        return u, v, area, span
