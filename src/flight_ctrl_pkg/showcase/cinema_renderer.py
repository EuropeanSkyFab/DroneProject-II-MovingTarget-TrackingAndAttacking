#!/usr/bin/env python3
"""Blockbuster 3D perspective cinema renderer with AR HUD, EO/IR thermal PiP, and 0.25x slow-mo."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

SLOWMO_FACTOR: float = 1.0 / 4.0
SLOWMO_TIME_SCALE: float = 0.25

DEFAULT_OUTPUT_DIR = Path(
    os.environ.get(
        'SHOWCASE_OUTPUT_DIR',
        '/home/hkw/.gemini/antigravity/brain/d204ee53-0b23-4a81-ac6e-d2bd544d5472',
    )
)
DEFAULT_TRACE_PATH = Path(os.environ.get('PROJECT2_TRACE_PATH', '/tmp/project2_mission_trace.jsonl'))
DEFAULT_SUMMARY_PATH = Path(
    os.environ.get('PROJECT2_SUMMARY_PATH', '/tmp/project2_mission_summary.json')
)


def _find_default_obstacles_yaml() -> Path:
    candidates = [
        Path(__file__).resolve().parent.parent / 'config' / 'obstacles.yaml',
        Path('/home/hkw/flying/ros2_ws/src/flight_ctrl_pkg/config/obstacles.yaml'),
    ]
    try:
        from ament_index_python.packages import get_package_share_directory

        candidates.append(Path(get_package_share_directory('flight_ctrl_pkg')) / 'config' / 'obstacles.yaml')
    except Exception:
        pass
    for p in candidates:
        if p.is_file():
            return p
    return candidates[0]


def _load_fonts() -> Dict[str, ImageFont.FreeTypeFont]:
    mono_candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
    ]
    sans_candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    mono_path = next((p for p in mono_candidates if Path(p).is_file()), None)
    sans_path = next((p for p in sans_candidates if Path(p).is_file()), mono_path)
    if sans_path is None:
        df = ImageFont.load_default()
        return {'title': df, 'badge': df, 'hud': df, 'small': df, 'micro': df}
    return {
        'title': ImageFont.truetype(sans_path, 22),
        'badge': ImageFont.truetype(sans_path, 16),
        'hud': ImageFont.truetype(mono_path or sans_path, 15),
        'small': ImageFont.truetype(mono_path or sans_path, 13),
        'micro': ImageFont.truetype(mono_path or sans_path, 11),
    }


def _is_finite_vec3(v: Any) -> bool:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        return False
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)) for x in v)


def load_and_validate_trace(
    trace_path: Path, summary_path: Optional[Path] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse and strictly validate JSONL trace and optional summary JSON before rendering."""
    if not trace_path.is_file():
        raise ValueError(f'Corrupt or missing trace file: {trace_path}')

    raw_text = trace_path.read_text(encoding='utf-8', errors='strict')
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError(f'Trace JSONL must contain at least 2 valid frames, got {len(lines)}: {trace_path}')

    records: List[Dict[str, Any]] = []
    prev_ts = -float('inf')
    for idx, ln in enumerate(lines, start=1):
        try:
            rec = json.loads(ln)
        except json.JSONDecodeError as exc:
            raise ValueError(f'Corrupt trace JSON parse failure at line {idx}: {exc}') from exc
        if not isinstance(rec, dict):
            raise ValueError(f'Corrupt trace line {idx}: expected JSON object')
        ts = rec.get('timestamp')
        if not isinstance(ts, (int, float)) or isinstance(ts, bool) or not math.isfinite(float(ts)):
            raise ValueError(f'Corrupt trace line {idx}: invalid timestamp {ts!r}')
        ts_f = float(ts)
        if ts_f < prev_ts - 1e-6:
            raise ValueError(f'Corrupt trace line {idx}: non-monotonic timestamp {ts_f} < {prev_ts}')
        prev_ts = ts_f

        drones = rec.get('drones')
        if not isinstance(drones, dict) or len(drones) == 0:
            raise ValueError(f'Corrupt trace line {idx}: drones field must be non-empty dict')
        for d_id, d_val in drones.items():
            if not isinstance(d_id, str) or not isinstance(d_val, dict):
                raise ValueError(f'Corrupt trace line {idx}: invalid drone entry {d_id!r}')
            if not _is_finite_vec3(d_val.get('pos')) or not _is_finite_vec3(d_val.get('vel')):
                raise ValueError(f'Corrupt trace line {idx}: non-finite pos/vel for {d_id}')

        records.append(rec)

    summary: Dict[str, Any] = {}
    if summary_path is not None and summary_path.exists():
        try:
            parsed_sum = json.loads(summary_path.read_text(encoding='utf-8'))
        except Exception as exc:
            raise ValueError(f'Corrupt summary JSON parse failure at {summary_path}: {exc}') from exc
        if not isinstance(parsed_sum, dict):
            raise ValueError(f'Corrupt summary JSON at {summary_path}: expected object')
        summary = parsed_sum

    return records, summary


def _load_arena_cfg(obstacles_path: Path) -> Dict[str, Any]:
    if not obstacles_path.is_file():
        return {'obstacles': [], 'no_fly_zones': [], 'helipads': {}, 'arena_bounds': {}}
    with obstacles_path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _build_camera_pose(
    eye: np.ndarray, target: np.ndarray, width: int, height: int, fov_deg: float = 58.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute world-to-camera R, t and intrinsic K matrix (ENU world -> OpenCV cam coords)."""
    fwd = target - eye
    dist = float(np.linalg.norm(fwd))
    if dist < 1e-4:
        fwd = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        fwd = fwd / dist
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(fwd, world_up)
    rn = float(np.linalg.norm(right))
    if rn < 1e-4:
        right = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        right = right / rn
    cam_down = np.cross(fwd, right)
    R = np.vstack([right, cam_down, fwd])  # rows = cam +X right, +Y down, +Z forward
    t = -R @ eye
    fx = (0.5 * width) / math.tan(math.radians(0.5 * fov_deg))
    fy = fx
    K = np.array([[fx, 0.0, 0.5 * width], [0.0, fy, 0.5 * height], [0.0, 0.0, 1.0]], dtype=np.float64)
    return R, t, K


def project_3d_to_2d(
    pts_world: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Project (N,3) world points to 2D image coordinates and return (uv, cam_z)."""
    pts = np.asarray(pts_world, dtype=np.float64).reshape(-1, 3)
    cam = (R @ pts.T).T + t.reshape(1, 3)
    zc = cam[:, 2]
    safe_z = np.where(zc > 0.2, zc, 0.2)
    u = K[0, 0] * (cam[:, 0] / safe_z) + K[0, 2]
    v = K[1, 1] * (cam[:, 1] / safe_z) + K[1, 2]
    return np.column_stack([u, v]), zc


def _clip_poly_near(cam_pts: np.ndarray, z_near: float = 0.4) -> Optional[np.ndarray]:
    """Clip a 3D camera-space polygon against z >= z_near plane."""
    n = len(cam_pts)
    if n < 3:
        return None
    out: List[np.ndarray] = []
    for i in range(n):
        a = cam_pts[i]
        b = cam_pts[(i + 1) % n]
        a_in = a[2] >= z_near
        b_in = b[2] >= z_near
        if a_in:
            out.append(a)
        if a_in != b_in:
            denom = b[2] - a[2]
            alpha = (z_near - a[2]) / denom if abs(denom) > 1e-9 else 0.0
            out.append(a + alpha * (b - a))
    if len(out) < 3:
        return None
    return np.asarray(out, dtype=np.float64)


def _clip_seg_near(a: np.ndarray, b: np.ndarray, z_near: float = 0.4) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    a_in = a[2] >= z_near
    b_in = b[2] >= z_near
    if not a_in and not b_in:
        return None
    if a_in and b_in:
        return a, b
    denom = b[2] - a[2]
    alpha = (z_near - a[2]) / denom if abs(denom) > 1e-9 else 0.0
    mid = a + alpha * (b - a)
    return (a, mid) if a_in else (mid, b)


def _build_sky_background(width: int, height: int) -> np.ndarray:
    """Create a 1920x1080 golden-hour atmospheric twilight-to-amber sky backdrop."""
    ys = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]

    # Multi-stop sunset palette (BGR)
    top = np.array([38.0, 18.0, 10.0], dtype=np.float32)
    mid = np.array([82.0, 48.0, 28.0], dtype=np.float32)
    horizon = np.array([72.0, 130.0, 224.0], dtype=np.float32)
    ground_haze = np.array([34.0, 32.0, 36.0], dtype=np.float32)

    img = np.zeros((height, width, 3), dtype=np.float32)
    m1 = ys <= 0.42
    t1 = np.clip(ys / 0.42, 0.0, 1.0)
    m2 = (ys > 0.42) & (ys <= 0.62)
    t2 = np.clip((ys - 0.42) / 0.20, 0.0, 1.0)
    m3 = ys > 0.62
    t3 = np.clip((ys - 0.62) / 0.38, 0.0, 1.0)

    img += np.where(m1[:, :, None], top * (1.0 - t1[:, :, None]) + mid * t1[:, :, None], 0.0)
    img += np.where(m2[:, :, None], mid * (1.0 - t2[:, :, None]) + horizon * t2[:, :, None], 0.0)
    img += np.where(m3[:, :, None], horizon * (1.0 - t3[:, :, None]) + ground_haze * t3[:, :, None], 0.0)

    # Solar glow halo
    sun_x, sun_y = 0.78, 0.34
    dx = (xs - sun_x) * (width / float(height))
    dy = ys - sun_y
    r2 = dx * dx + dy * dy
    halo = np.exp(-r2 * 18.0)[:, :, None] * np.array([95.0, 165.0, 255.0], dtype=np.float32)
    core = np.exp(-r2 * 450.0)[:, :, None] * np.array([210.0, 240.0, 255.0], dtype=np.float32)
    img = np.clip(img + halo + core, 0.0, 255.0).astype(np.uint8)
    return img


class ArenaMeshLibrary:
    """Precomputed 3D faces for architectural obstacles, NFZ volumes, and helipads."""

    def __init__(self, arena_cfg: Dict[str, Any]) -> None:
        self.sun_dir = np.array([0.62, -0.46, 0.63], dtype=np.float64)
        self.sun_dir /= np.linalg.norm(self.sun_dir)
        self.faces: List[Tuple[np.ndarray, np.ndarray, Tuple[int, int, int], bool]] = []
        self._build_meshes(arena_cfg)

    def _add_box(
        self, cx: float, cy: float, z0: float, z1: float, sx: float, sy: float, yaw: float, base_bgr: Tuple[int, int, int]
    ) -> None:
        hx, hy = 0.5 * sx, 0.5 * sy
        c, s = math.cos(yaw), math.sin(yaw)
        corners_local = np.array(
            [
                [-hx, -hy, z0],
                [hx, -hy, z0],
                [hx, hy, z0],
                [-hx, hy, z0],
                [-hx, -hy, z1],
                [hx, -hy, z1],
                [hx, hy, z1],
                [-hx, hy, z1],
            ],
            dtype=np.float64,
        )
        wx = cx + c * corners_local[:, 0] - s * corners_local[:, 1]
        wy = cy + s * corners_local[:, 0] + c * corners_local[:, 1]
        wz = corners_local[:, 2]
        verts = np.column_stack([wx, wy, wz])

        quad_indices = [
            (0, 1, 5, 4, (0, -1, 0)),
            (1, 2, 6, 5, (1, 0, 0)),
            (2, 3, 7, 6, (0, 1, 0)),
            (3, 0, 4, 7, (-1, 0, 0)),
            (4, 5, 6, 7, (0, 0, 1)),
        ]
        for i0, i1, i2, i3, nloc in quad_indices:
            nx = c * nloc[0] - s * nloc[1]
            ny = s * nloc[0] + c * nloc[1]
            nz = float(nloc[2])
            normal = np.array([nx, ny, nz], dtype=np.float64)
            poly = verts[[i0, i1, i2, i3]]
            self.faces.append((poly, normal, base_bgr, i0 == 4))

    def _add_cylinder(
        self, cx: float, cy: float, z0: float, z1: float, radius: float, segs: int, base_bgr: Tuple[int, int, int]
    ) -> None:
        ang = np.linspace(0.0, 2.0 * math.pi, segs + 1)
        bot = np.column_stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang), np.full(segs + 1, z0)])
        top = np.column_stack([cx + radius * np.cos(ang), cy + radius * np.sin(ang), np.full(segs + 1, z1)])
        for k in range(segs):
            mid_a = 0.5 * (ang[k] + ang[k + 1])
            normal = np.array([math.cos(mid_a), math.sin(mid_a), 0.0], dtype=np.float64)
            poly = np.array([bot[k], bot[k + 1], top[k + 1], top[k]], dtype=np.float64)
            self.faces.append((poly, normal, base_bgr, False))
        # Roof cap
        self.faces.append((top[:-1], np.array([0.0, 0.0, 1.0], dtype=np.float64), base_bgr, True))

    def _build_meshes(self, cfg: Dict[str, Any]) -> None:
        palette = [
            (72, 68, 62),
            (66, 72, 82),
            (84, 74, 66),
            (58, 64, 76),
            (78, 82, 88),
        ]
        for idx, obs in enumerate(cfg.get('obstacles') or []):
            cx = float(obs.get('cx', 0.0))
            cy = float(obs.get('cy', 0.0))
            z0 = float(obs.get('z_min', 0.0))
            z1 = float(obs.get('z_max', 12.0))
            col = palette[idx % len(palette)]
            if str(obs.get('type', 'box')).lower() == 'cylinder':
                self._add_cylinder(cx, cy, z0, z1, float(obs.get('radius', 2.0)), 14, col)
            else:
                self._add_box(
                    cx,
                    cy,
                    z0,
                    z1,
                    float(obs.get('sx', 3.0)),
                    float(obs.get('sy', 3.0)),
                    float(obs.get('yaw', 0.0)),
                    col,
                )


def _interpolate_trace_state(
    records: List[Dict[str, Any]], sim_t: float
) -> Dict[str, Any]:
    """Interpolate drone & target 3D poses seamlessly at arbitrary simulation time sim_t."""
    ts_list = [float(r['timestamp']) for r in records]
    idx = int(np.searchsorted(ts_list, sim_t, side='right')) - 1
    idx = max(0, min(idx, len(records) - 1))
    nxt = min(idx + 1, len(records) - 1)
    r0 = records[idx]
    r1 = records[nxt]
    t0 = float(r0['timestamp'])
    t1 = float(r1['timestamp'])
    alpha = 0.0 if t1 <= t0 + 1e-7 else max(0.0, min(1.0, (sim_t - t0) / (t1 - t0)))

    drones_out: Dict[str, Dict[str, np.ndarray]] = {}
    for did, d0 in r0.get('drones', {}).items():
        p0 = np.asarray(d0['pos'], dtype=np.float64)
        v0 = np.asarray(d0['vel'], dtype=np.float64)
        if did in r1.get('drones', {}):
            d1 = r1['drones'][did]
            p1 = np.asarray(d1['pos'], dtype=np.float64)
            v1 = np.asarray(d1['vel'], dtype=np.float64)
            p = (1.0 - alpha) * p0 + alpha * p1
            v = (1.0 - alpha) * v0 + alpha * v1
        else:
            p, v = p0, v0
        drones_out[did] = {'pos': p, 'vel': v}

    t1_map = {t['target_id']: t for t in (r1.get('targets_gt') or []) if isinstance(t, dict) and 'target_id' in t}
    targets_out: List[Dict[str, Any]] = []
    for t0_item in r0.get('targets_gt') or []:
        if not isinstance(t0_item, dict):
            continue
        tid = str(t0_item.get('target_id', 'unknown'))
        p0 = np.asarray(t0_item.get('pos', [0.0, 0.0, 5.0]), dtype=np.float64)
        t_mate = t1_map.get(tid)
        if t_mate is not None:
            p1 = np.asarray(t_mate.get('pos', p0), dtype=np.float64)
            p = (1.0 - alpha) * p0 + alpha * p1
            vel = (p1 - p0) / max(t1 - t0, 1e-3)
        else:
            p = p0
            vel = np.zeros(3, dtype=np.float64)
        targets_out.append(
            {
                'target_id': tid,
                'class_id': str(t0_item.get('class_id', tid)),
                'pos': p,
                'vel': vel,
                'status': str(t0_item.get('status', 'ACTIVE')),
            }
        )

    return {
        'sim_t': sim_t,
        'phase': str(r0.get('phase', 'TRACK')),
        'drones': drones_out,
        'targets': targets_out,
        'wta_assignments': r0.get('wta_assignments', {}) or {},
        'kills': r0.get('kills', []) or [],
        'estimator': r0.get('estimator', {}) or {},
        'vision_pos': r0.get('vision_pos'),
    }


def _extract_kill_events(records: List[Dict[str, Any]]) -> List[Tuple[float, str, np.ndarray]]:
    """Detect simulation timestamps and positions of target neutralization events."""
    seen: Dict[str, Tuple[float, str, np.ndarray]] = {}
    last_tgt_pos: Dict[str, np.ndarray] = {}
    for r in records:
        ts = float(r['timestamp'])
        cur_pos_map: Dict[str, np.ndarray] = {}
        for tgt in r.get('targets_gt') or []:
            if isinstance(tgt, dict) and tgt.get('target_id'):
                tid = str(tgt['target_id'])
                p = np.asarray(tgt.get('pos', [25.0, 0.0, 6.0]), dtype=np.float64)
                cur_pos_map[tid] = p
                last_tgt_pos[tid] = p
                if tgt.get('status') in ('NEUTRALIZED', 'KILLED', 'DESTROYED') and tid not in seen:
                    seen[tid] = (ts, tid, p)
        for k in r.get('kills') or []:
            if isinstance(k, dict) and k.get('target_id'):
                tid = str(k['target_id'])
                if tid not in seen:
                    raw_p = k.get('pos')
                    pos = (
                        np.asarray(raw_p, dtype=np.float64)
                        if raw_p is not None
                        else cur_pos_map.get(tid, last_tgt_pos.get(tid, np.array([25.0, 0.0, 6.0])))
                    )
                    seen[tid] = (float(k.get('stamp', k.get('timestamp', ts))), tid, pos)
    return sorted(seen.values(), key=lambda x: x[0])


def _build_timeline_schedule(
    records: List[Dict[str, Any]], kills: List[Tuple[float, str, np.ndarray]], duration_s: float, fps: int
) -> List[Tuple[float, int, str, float, Optional[Tuple[float, str, np.ndarray]]]]:
    """Map output frames to (sim_t, act_idx, act_name, time_scale, active_kill_event)."""
    t_start = float(records[0]['timestamp'])
    t_end = max(float(records[-1]['timestamp']), t_start + 1.0)
    span = t_end - t_start
    total_frames = max(int(round(duration_s * fps)), 300)

    # Identify first/mid kill window to inject 0.25x bullet-time slow-motion strike segment
    if kills:
        k_t, k_tid, k_pos = kills[0]
        # Place slow-mo dilation strictly inside the trace time range
        slo_center = max(t_start + 0.35 * span, min(t_end - 0.18 * span, k_t))
        slo_half = min(0.85, 0.06 * span)
        slo_a = max(t_start + 0.1 * span, slo_center - slo_half)
        slo_b = min(t_end - 0.08 * span, slo_center + slo_half)
    else:
        slo_center = t_start + 0.72 * span
        slo_a = t_start + 0.67 * span
        slo_b = t_start + 0.76 * span

    # Allocate video fraction across 5 dramatic acts:
    # Act 1 [0.00..0.18]: Scramble | Act 2 [0.18..0.38]: Stereo Track | Act 3 [0.38..0.60]: Wolfpack Intercept
    # Act 4 [0.60..0.82]: 0.25x Slow-Mo Strike | Act 5 [0.82..1.00]: BDA & Helipad RTB
    t_act1_end = min(slo_a, t_start + 0.18 * span)
    t_act2_end = min(slo_a, t_start + 0.42 * span)
    t_act3_end = slo_a

    schedule: List[Tuple[float, int, str, float, Optional[Tuple[float, str, np.ndarray]]]] = []
    for f in range(total_frames):
        u = f / float(max(total_frames - 1, 1))
        if u < 0.18:
            local = u / 0.18
            st = t_start + local * (t_act1_end - t_start)
            schedule.append((st, 1, 'ACT I // SCRAMBLE & CLIMB', 1.0, None))
        elif u < 0.38:
            local = (u - 0.18) / 0.20
            st = t_act1_end + local * (t_act2_end - t_act1_end)
            schedule.append((st, 2, 'ACT II // STEREO & IMM TRACK', 1.0, None))
        elif u < 0.60:
            local = (u - 0.38) / 0.22
            st = t_act2_end + local * (t_act3_end - t_act2_end)
            schedule.append((st, 3, 'ACT III // WOLFPACK ENCIRCLEMENT', 1.0, None))
        elif u < 0.82:
            local = (u - 0.60) / 0.22
            st = slo_a + local * max(slo_b - slo_a, 0.25)
            # Slow-motion dilated speed = 0.25 (1.0 / 4.0)
            k_evt = kills[0] if kills else (slo_center, 'crimson_uav', np.array([28.0, 2.0, 6.5]))
            schedule.append((st, 4, 'ACT IV // 0.25x SLOW-MO STRIKE', SLOWMO_FACTOR, k_evt))
        else:
            local = (u - 0.82) / 0.18
            st = slo_b + local * max(t_end - slo_b, 0.25)
            schedule.append((st, 5, 'ACT V // BDA & HELIPAD RTB', 1.0, None))
    return schedule


def _choreograph_camera(
    act_idx: int,
    u_total: float,
    state: Dict[str, Any],
    active_kill: Optional[Tuple[float, str, np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute sweeping Hollywood-style camera eye, look-at target, and FOV for each Act."""
    drones_pos = [d['pos'] for d in state['drones'].values()]
    fleet_ctr = np.mean(drones_pos, axis=0) if drones_pos else np.array([15.0, 0.0, 4.0])
    active_targets = [t for t in state['targets'] if t['status'] == 'ACTIVE']
    lead_tgt = active_targets[0]['pos'] if active_targets else (
        active_kill[2] if active_kill is not None else np.array([32.0, 0.0, 6.0])
    )

    if act_idx == 1:
        # Dramatic low-angle dolly sweep framing all 6 climbing UAVs over Helipads H1-H6
        prog = max(0.0, min(1.0, u_total / 0.18))
        ang = -0.75 + 0.55 * prog
        eye = fleet_ctr + np.array([-13.5 + 3.0 * math.cos(ang), -12.0 + 3.5 * math.sin(ang), 4.8 + 2.0 * prog])
        tgt = 0.72 * fleet_ctr + 0.28 * np.array([18.0, 2.0, 5.5])
        return eye, tgt, 60.0
    if act_idx == 2:
        # Elevated lateral tracking dolly showcasing canyon + targets
        prog = (u_total - 0.18) / 0.20
        focus = 0.55 * fleet_ctr + 0.45 * lead_tgt
        eye = focus + np.array([-15.0 + 6.0 * prog, -19.5 + 5.0 * prog, 10.5])
        return eye, focus + np.array([4.0, 0.0, 0.5]), 56.0
    if act_idx == 3:
        # High-speed chase sweep along canyon corridor
        prog = (u_total - 0.38) / 0.22
        focus = 0.45 * fleet_ctr + 0.55 * lead_tgt
        theta = -1.2 + 1.1 * prog
        eye = focus + np.array([19.0 * math.cos(theta), 19.0 * math.sin(theta), 8.2 - 1.5 * prog])
        return eye, focus, 58.0
    if act_idx == 4:
        # 0.25x Bullet-Time tight orbital rotation around strike point
        prog = (u_total - 0.60) / 0.22
        strike_pt = active_kill[2] if active_kill is not None else lead_tgt
        orbit_ang = -2.2 + 2.35 * prog
        eye = strike_pt + np.array([10.5 * math.cos(orbit_ang), 10.5 * math.sin(orbit_ang), 3.2 + 1.2 * math.sin(prog * math.pi)])
        return eye, strike_pt, 50.0
    # Act 5: Golden-hour panoramic pull-back overlooking returning fleet & helipads
    prog = (u_total - 0.82) / 0.18
    eye = np.array([-16.0 - 5.0 * prog, -18.0 - 4.0 * prog, 14.0 + 6.0 * prog])
    tgt = np.array([14.0, 0.0, 3.0])
    return eye, tgt, 62.0


def _draw_quadrotor_3d(
    img: np.ndarray,
    pos: np.ndarray,
    vel: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    body_bgr: Tuple[int, int, int],
    led_bgr: Tuple[int, int, int],
    rotor_phase: float,
    scale: float = 0.55,
) -> Optional[Tuple[int, int, float]]:
    """Render articulated 3D X-quadrotor with dynamic attitude bank & spinning rotor discs."""
    speed = float(np.linalg.norm(vel))
    yaw = math.atan2(vel[1], vel[0]) if speed > 0.25 else 0.0
    bank = min(0.38, 0.045 * speed)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(-bank), math.sin(-bank)
    rot_m = np.array(
        [
            [cy * cp, -sy, cy * sp],
            [sy * cp, cy, sy * sp],
            [-sp, 0.0, cp],
        ],
        dtype=np.float64,
    )
    arm = scale
    arms_local = np.array(
        [
            [-arm, -arm, 0.0],
            [arm, arm, 0.0],
            [-arm, arm, 0.0],
            [arm, -arm, 0.0],
            [0.0, 0.0, 0.0],
            [1.15 * arm, 0.0, 0.08 * arm],
        ],
        dtype=np.float64,
    )
    world_pts = pos.reshape(1, 3) + (rot_m @ arms_local.T).T
    uv, zc = project_3d_to_2d(world_pts, R, t, K)
    if np.min(zc) <= 0.5:
        return None
    depth = float(zc[4])
    pts_i = np.round(uv).astype(np.int32)

    # Boom cross arms + fuselage
    cv2.line(img, tuple(pts_i[0]), tuple(pts_i[1]), body_bgr, 3, cv2.LINE_AA)
    cv2.line(img, tuple(pts_i[2]), tuple(pts_i[3]), body_bgr, 3, cv2.LINE_AA)
    cv2.line(img, tuple(pts_i[4]), tuple(pts_i[5]), led_bgr, 2, cv2.LINE_AA)
    fus_r = max(4, int(round(190.0 / depth)))
    cv2.circle(img, tuple(pts_i[4]), fus_r, body_bgr, -1, cv2.LINE_AA)
    cv2.circle(img, tuple(pts_i[4]), fus_r, led_bgr, 1, cv2.LINE_AA)

    # Spinning rotors & corner LED strobes
    rot_r = max(5, int(round(360.0 * scale / depth)))
    for m_idx in range(4):
        mc = tuple(pts_i[m_idx])
        cv2.ellipse(
            img,
            mc,
            (rot_r, max(2, int(0.45 * rot_r))),
            math.degrees(rotor_phase + m_idx * 0.7),
            0,
            360,
            (225, 238, 250),
            1,
            cv2.LINE_AA,
        )
        cv2.circle(img, mc, max(2, fus_r // 2), led_bgr, -1, cv2.LINE_AA)
    return int(pts_i[4, 0]), int(pts_i[4, 1]), depth


def _render_eoir_thermal_pip(
    state: Dict[str, Any],
    arena_cfg: Dict[str, Any],
    focus_target: Optional[Dict[str, Any]],
    focus_drone: Optional[Tuple[str, np.ndarray]],
    frame_idx: int,
    time_scale: float,
    fonts: Dict[str, ImageFont.FreeTypeFont],
) -> np.ndarray:
    """Render genuine synthetic EO/IR FLIR LWIR thermal PiP viewport tracking primary locked target."""
    pip_w, pip_h = 416, 260
    tgt_pos = (
        focus_target['pos']
        if focus_target is not None
        else np.array([30.0, 1.0, 6.0], dtype=np.float64)
    )
    chaser_pos = (
        focus_drone[1]
        if focus_drone is not None
        else (tgt_pos + np.array([-8.5, -3.5, 1.5], dtype=np.float64))
    )
    eye = chaser_pos + np.array([0.0, 0.0, 0.65], dtype=np.float64)
    R, t, K = _build_camera_pose(eye, tgt_pos, pip_w, pip_h, fov_deg=34.0)

    heat = np.full((pip_h, pip_w), 26, dtype=np.uint8)
    # Cool horizon gradient in IR band
    grad = np.linspace(14, 48, pip_h, dtype=np.uint8)[:, None]
    heat = np.broadcast_to(grad, (pip_h, pip_w)).copy()

    # Project nearby urban obstacle silhouettes as cool thermal masses
    for obs in arena_cfg.get('obstacles') or []:
        cx, cy = float(obs.get('cx', 0.0)), float(obs.get('cy', 0.0))
        z0, z1 = float(obs.get('z_min', 0.0)), float(obs.get('z_max', 12.0))
        sx = float(obs.get('sx', 2.0 * float(obs.get('radius', 2.0))))
        sy = float(obs.get('sy', sx))
        corners = np.array(
            [
                [cx - 0.5 * sx, cy - 0.5 * sy, z0],
                [cx + 0.5 * sx, cy - 0.5 * sy, z0],
                [cx + 0.5 * sx, cy + 0.5 * sy, z0],
                [cx - 0.5 * sx, cy + 0.5 * sy, z0],
                [cx - 0.5 * sx, cy - 0.5 * sy, z1],
                [cx + 0.5 * sx, cy - 0.5 * sy, z1],
                [cx + 0.5 * sx, cy + 0.5 * sy, z1],
                [cx - 0.5 * sx, cy + 0.5 * sy, z1],
            ],
            dtype=np.float64,
        )
        uv, zc = project_3d_to_2d(corners, R, t, K)
        if np.all(zc > 0.6):
            hull = cv2.convexHull(np.round(uv).astype(np.int32))
            cv2.fillConvexPoly(heat, hull, 64)

    # Render hot signature for target rotor cores & exhaust plume
    uv_t, zc_t = project_3d_to_2d(tgt_pos.reshape(1, 3), R, t, K)
    tx, ty = int(round(uv_t[0, 0])), int(round(uv_t[0, 1]))
    depth = max(float(zc_t[0]), 1.5)
    core_r = max(7, min(32, int(round(125.0 / depth))))
    cv2.circle(heat, (tx, ty), core_r + 10, 145, -1, cv2.LINE_AA)
    cv2.circle(heat, (tx, ty), core_r, 232, -1, cv2.LINE_AA)
    for ox, oy in [(-core_r, -core_r // 2), (core_r, -core_r // 2), (-core_r, core_r // 2), (core_r, core_r // 2)]:
        cv2.circle(heat, (tx + ox, ty + oy), max(3, core_r // 2), 252, -1, cv2.LINE_AA)
    heat = cv2.GaussianBlur(heat, (7, 7), 0)

    # Subtle sensor scanline shimmer
    noise_band = (frame_idx * 5) % pip_h
    heat[noise_band : min(noise_band + 2, pip_h), :] = np.clip(
        heat[noise_band : min(noise_band + 2, pip_h), :].astype(np.int16) + 18, 0, 255
    ).astype(np.uint8)

    thermal_bgr = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)

    # Crisp FLIR targeting reticle & brackets
    cx_p, cy_p = pip_w // 2, pip_h // 2
    ret_col = (240, 255, 240)
    cv2.line(thermal_bgr, (cx_p - 28, cy_p), (cx_p - 9, cy_p), ret_col, 1, cv2.LINE_AA)
    cv2.line(thermal_bgr, (cx_p + 9, cy_p), (cx_p + 28, cy_p), ret_col, 1, cv2.LINE_AA)
    cv2.line(thermal_bgr, (cx_p, cy_p - 24), (cx_p, cy_p - 7), ret_col, 1, cv2.LINE_AA)
    cv2.line(thermal_bgr, (cx_p, cy_p + 7), (cx_p, cy_p + 24), ret_col, 1, cv2.LINE_AA)

    bx = max(18, min(pip_w - 18, tx))
    by = max(18, min(pip_h - 18, ty))
    br = max(18, core_r + 10)
    lock_col = (0, 220, 255) if time_scale < 0.5 else (80, 255, 160)
    for sx_d, sy_d in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        cv2.line(thermal_bgr, (bx + sx_d * br, by + sy_d * br), (bx + sx_d * (br - 8), by + sy_d * br), lock_col, 2)
        cv2.line(thermal_bgr, (bx + sx_d * br, by + sy_d * br), (bx + sx_d * br, by + sy_d * (br - 8)), lock_col, 2)

    # PIL telemetry text header & footer inside PiP
    pil_im = Image.fromarray(cv2.cvtColor(thermal_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im, 'RGBA')
    draw.rectangle([(0, 0), (pip_w, 26)], fill=(10, 12, 18, 195))
    draw.rectangle([(0, pip_h - 24), (pip_w, pip_h)], fill=(10, 12, 18, 195))
    tid_lbl = str(focus_target['target_id']) if focus_target is not None else 'NO ACTIVE TGT'
    rng = float(np.linalg.norm(tgt_pos - chaser_pos))
    draw.text((8, 5), f'EO/IR THERMAL // LWIR LOCK: {tid_lbl}', font=fonts['micro'], fill=(255, 215, 120, 255))
    slo_tag = '  [0.25x SLOW-MO]' if time_scale < 0.5 else ''
    draw.text(
        (8, pip_h - 19),
        f'RNG: {rng:5.1f}m   CHASER: {focus_drone[0] if focus_drone else "--"}{slo_tag}',
        font=fonts['micro'],
        fill=(180, 245, 255, 255),
    )
    out = cv2.cvtColor(np.asarray(pil_im), cv2.COLOR_RGB2BGR)
    cv2.rectangle(out, (0, 0), (pip_w - 1, pip_h - 1), (95, 210, 255), 2)
    return out


def _draw_minimap(
    arena_cfg: Dict[str, Any],
    state: Dict[str, Any],
    eye: np.ndarray,
    cam_tgt: np.ndarray,
    fonts: Dict[str, ImageFont.FreeTypeFont],
) -> np.ndarray:
    """Render top-down orthographic arena tactical radar inset."""
    mw, mh = 260, 224
    panel = np.full((mh, mw, 3), 18, dtype=np.uint8)
    b = arena_cfg.get('arena_bounds') or {}
    x_min = float(b.get('x_min', -12.0))
    x_max = float(b.get('x_max', 78.0))
    y_min = float(b.get('y_min', -26.0))
    y_max = float(b.get('y_max', 26.0))

    def to_px(wx: float, wy: float) -> Tuple[int, int]:
        px = int(round(12 + (wx - x_min) / max(x_max - x_min, 1.0) * (mw - 24)))
        py = int(round(mh - 26 - (wy - y_min) / max(y_max - y_min, 1.0) * (mh - 50)))
        return px, py

    for obs in arena_cfg.get('obstacles') or []:
        p = to_px(float(obs.get('cx', 0.0)), float(obs.get('cy', 0.0)))
        r = max(3, int(round(0.5 * float(obs.get('sx', 2.0 * float(obs.get('radius', 2.0)))) * 2.3)))
        cv2.rectangle(panel, (p[0] - r, p[1] - r), (p[0] + r, p[1] + r), (58, 66, 78), -1)
    for nfz in arena_cfg.get('no_fly_zones') or []:
        p = to_px(float(nfz.get('cx', 0.0)), float(nfz.get('cy', 0.0)))
        r = max(4, int(round(float(nfz.get('radius', 4.0)) * 2.4)))
        cv2.circle(panel, p, r, (50, 50, 175), 1, cv2.LINE_AA)

    # Camera FOV wedge
    e_p = to_px(float(eye[0]), float(eye[1]))
    t_p = to_px(float(cam_tgt[0]), float(cam_tgt[1]))
    cv2.line(panel, e_p, t_p, (90, 140, 170), 1, cv2.LINE_AA)
    cv2.circle(panel, e_p, 3, (120, 210, 255), -1)

    for did, d in state['drones'].items():
        dp = to_px(float(d['pos'][0]), float(d['pos'][1]))
        cv2.circle(panel, dp, 4, (255, 210, 60), -1, cv2.LINE_AA)
    for tgt in state['targets']:
        tp = to_px(float(tgt['pos'][0]), float(tgt['pos'][1]))
        if tgt['status'] == 'ACTIVE':
            cv2.drawMarker(panel, tp, (60, 80, 255), cv2.MARKER_DIAMOND, 9, 2, cv2.LINE_AA)
        else:
            cv2.drawMarker(panel, tp, (120, 120, 130), cv2.MARKER_TILTED_CROSS, 7, 1, cv2.LINE_AA)

    pil_im = Image.fromarray(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im, 'RGBA')
    draw.rectangle([(0, 0), (mw, 22)], fill=(8, 12, 20, 210))
    draw.text((8, 4), 'TACTICAL ARENA RADAR (TOP)', font=fonts['micro'], fill=(145, 225, 255, 255))
    draw.text((8, mh - 18), 'UAV: CYAN   TGT: CRIMSON   NFZ: RED', font=fonts['micro'], fill=(175, 185, 200, 255))
    out = cv2.cvtColor(np.asarray(pil_im), cv2.COLOR_RGB2BGR)
    cv2.rectangle(out, (0, 0), (mw - 1, mh - 1), (85, 130, 170), 1)
    return out


def render_single_frame(
    width: int,
    height: int,
    sky_bg: np.ndarray,
    mesh_lib: ArenaMeshLibrary,
    arena_cfg: Dict[str, Any],
    state: Dict[str, Any],
    trails: Dict[str, List[np.ndarray]],
    tgt_trails: Dict[str, List[np.ndarray]],
    eye: np.ndarray,
    cam_tgt: np.ndarray,
    fov_deg: float,
    act_idx: int,
    act_title: str,
    time_scale: float,
    u_total: float,
    frame_idx: int,
    kills: List[Tuple[float, str, np.ndarray]],
    active_kill: Optional[Tuple[float, str, np.ndarray]],
    fonts: Dict[str, ImageFont.FreeTypeFont],
    hud_enabled: bool = True,
    t_start: float = 0.0,
) -> np.ndarray:
    """Compose one 1920x1080 3D perspective + AR HUD + EO/IR thermal PiP frame."""
    frame = sky_bg.copy()
    R, t, K = _build_camera_pose(eye, cam_tgt, width, height, fov_deg=fov_deg)

    # 1. Ground grid plane (z = 0)
    grid_lines: List[Tuple[np.ndarray, np.ndarray]] = []
    for gx in range(-15, 86, 5):
        grid_lines.append((np.array([gx, -28.0, 0.0]), np.array([gx, 28.0, 0.0])))
    for gy in range(-25, 30, 5):
        grid_lines.append((np.array([-15.0, gy, 0.0]), np.array([85.0, gy, 0.0])))
    for pa, pb in grid_lines:
        ca = R @ pa + t
        cb = R @ pb + t
        seg = _clip_seg_near(ca, cb, 0.5)
        if seg is None:
            continue
        p0, p1 = seg
        u0 = int(round(K[0, 0] * (p0[0] / p0[2]) + K[0, 2]))
        v0 = int(round(K[1, 1] * (p0[1] / p0[2]) + K[1, 2]))
        u1 = int(round(K[0, 0] * (p1[0] / p1[2]) + K[0, 2]))
        v1 = int(round(K[1, 1] * (p1[1] / p1[2]) + K[1, 2]))
        cv2.line(frame, (u0, v0), (u1, v1), (56, 62, 74), 1, cv2.LINE_AA)

    # Helipads on ground plane
    raw_pads = arena_cfg.get('helipads') or []
    pad_items = raw_pads.values() if isinstance(raw_pads, dict) else raw_pads
    for hp in pad_items:
        if isinstance(hp, dict):
            hx, hy = float(hp.get('x', 0.0)), float(hp.get('y', 0.0))
        elif isinstance(hp, (list, tuple)) and len(hp) >= 2:
            hx, hy = float(hp[0]), float(hp[1])
        else:
            continue
        ang = np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)
        ring = np.column_stack([hx + 1.15 * np.cos(ang), hy + 1.15 * np.sin(ang), np.full(16, 0.03)])
        uv, zc = project_3d_to_2d(ring, R, t, K)
        if np.all(zc > 0.6):
            pts = np.round(uv).astype(np.int32)
            cv2.polylines(frame, [pts], True, (75, 225, 130), 2, cv2.LINE_AA)

    # 2. Painter's sorted PBR-shaded 3D Urban Obstacles
    draw_polys: List[Tuple[float, np.ndarray, Tuple[int, int, int], bool]] = []
    for world_poly, normal, base_bgr, is_roof in mesh_lib.faces:
        centroid = np.mean(world_poly, axis=0)
        view_vec = eye - centroid
        if not is_roof and float(np.dot(normal, view_vec)) < -0.1:
            continue
        cam_poly = (R @ world_poly.T).T + t.reshape(1, 3)
        if float(np.max(cam_poly[:, 2])) < 0.4:
            continue
        clipped = _clip_poly_near(cam_poly, 0.4)
        if clipped is None:
            continue
        u = K[0, 0] * (clipped[:, 0] / clipped[:, 2]) + K[0, 2]
        v = K[1, 1] * (clipped[:, 1] / clipped[:, 2]) + K[1, 2]
        pts2d = np.round(np.column_stack([u, v])).astype(np.int32)
        if (
            int(np.max(pts2d[:, 0])) < 0
            or int(np.min(pts2d[:, 0])) >= width
            or int(np.max(pts2d[:, 1])) < 0
            or int(np.min(pts2d[:, 1])) >= height
        ):
            continue
        depth = float(np.mean(clipped[:, 2]))
        lambert = max(0.32, float(np.dot(normal, mesh_lib.sun_dir)))
        # Golden sunlight tint + atmospheric distance fog
        fog = min(0.38, depth / 170.0)
        c_b = int(np.clip(base_bgr[0] * lambert * (1.0 - fog) + 90.0 * fog, 0, 255))
        c_g = int(np.clip(base_bgr[1] * (0.95 * lambert + 0.15) * (1.0 - fog) + 115.0 * fog, 0, 255))
        c_r = int(np.clip(base_bgr[2] * (1.12 * lambert + 0.20) * (1.0 - fog) + 165.0 * fog, 0, 255))
        draw_polys.append((depth, pts2d, (c_b, c_g, c_r), is_roof))

    draw_polys.sort(key=lambda item: item[0], reverse=True)
    for _, pts2d, shade_bgr, is_roof in draw_polys:
        cv2.fillConvexPoly(frame, pts2d, shade_bgr, cv2.LINE_AA)
        edge_col = (125, 165, 220) if is_roof else (36, 40, 48)
        cv2.polylines(frame, [pts2d], True, edge_col, 1, cv2.LINE_AA)

    # NFZ glowing perimeter laser rings
    for nfz in arena_cfg.get('no_fly_zones') or []:
        ncx, ncy = float(nfz.get('cx', 0.0)), float(nfz.get('cy', 0.0))
        nr = float(nfz.get('radius', 4.0))
        nz_lo, nz_hi = float(nfz.get('z_min', 0.0)), float(nfz.get('z_max', 14.0))
        ang = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
        for zr in [max(0.2, nz_lo + 0.2), 0.5 * (nz_lo + nz_hi), nz_hi]:
            ring = np.column_stack([ncx + nr * np.cos(ang), ncy + nr * np.sin(ang), np.full(24, zr)])
            uv, zc = project_3d_to_2d(ring, R, t, K)
            if np.all(zc > 0.6):
                cv2.polylines(frame, [np.round(uv).astype(np.int32)], True, (45, 55, 225), 1, cv2.LINE_AA)

    # 3. 3D Glowing Ribbon Trails
    drone_palette = [
        (255, 215, 45),
        (165, 255, 85),
        (255, 165, 70),
        (95, 235, 255),
        (240, 145, 255),
        (135, 255, 195),
    ]
    for idx, (did, pts_hist) in enumerate(sorted(trails.items())):
        if len(pts_hist) < 2:
            continue
        arr = np.asarray(pts_hist, dtype=np.float64)
        uv, zc = project_3d_to_2d(arr, R, t, K)
        valid = zc > 0.6
        col = drone_palette[idx % len(drone_palette)]
        for k in range(len(arr) - 1):
            if valid[k] and valid[k + 1]:
                frac = (k + 1) / float(len(arr))
                c_fade = (int(col[0] * frac), int(col[1] * frac), int(col[2] * frac))
                cv2.line(
                    frame,
                    (int(round(uv[k, 0])), int(round(uv[k, 1]))),
                    (int(round(uv[k + 1, 0])), int(round(uv[k + 1, 1]))),
                    c_fade,
                    2 if frac > 0.5 else 1,
                    cv2.LINE_AA,
                )

    tgt_colors = {
        'red_moving_target': (45, 55, 255),
        'amber_escort_target': (40, 175, 255),
        'slate_recon_target': (210, 85, 245),
    }
    for tid, t_hist in sorted(tgt_trails.items()):
        if len(t_hist) < 2:
            continue
        arr = np.asarray(t_hist, dtype=np.float64)
        uv, zc = project_3d_to_2d(arr, R, t, K)
        col = tgt_colors.get(tid, (60, 80, 255))
        for k in range(len(arr) - 1):
            if zc[k] > 0.6 and zc[k + 1] > 0.6:
                cv2.line(
                    frame,
                    (int(round(uv[k, 0])), int(round(uv[k, 1]))),
                    (int(round(uv[k + 1, 0])), int(round(uv[k + 1, 1]))),
                    col,
                    2,
                    cv2.LINE_AA,
                )

    # 4. Render 3D Hostile Targets & Interceptors
    tgt_screen_coords: Dict[str, Tuple[int, int, float]] = {}
    for tgt in state['targets']:
        tid = tgt['target_id']
        col = tgt_colors.get(tid, (45, 65, 255))
        if tgt['status'] == 'ACTIVE':
            res = _draw_quadrotor_3d(
                frame,
                tgt['pos'],
                tgt['vel'],
                R,
                t,
                K,
                body_bgr=(35, 38, 48),
                led_bgr=col,
                rotor_phase=frame_idx * 0.65,
                scale=0.68,
            )
            if res is not None:
                tgt_screen_coords[tid] = res

    drone_screen_coords: Dict[str, Tuple[int, int, float]] = {}
    for idx, (did, dval) in enumerate(sorted(state['drones'].items())):
        col = drone_palette[idx % len(drone_palette)]
        res = _draw_quadrotor_3d(
            frame,
            dval['pos'],
            dval['vel'],
            R,
            t,
            K,
            body_bgr=(215, 222, 232),
            led_bgr=col,
            rotor_phase=frame_idx * 0.82 + idx * 0.4,
            scale=0.55,
        )
        if res is not None:
            drone_screen_coords[did] = res

    # Strike Explosion / Shockwave FX around active kill events
    sim_t = float(state['sim_t'])
    for k_t, k_tid, k_pos in kills:
        dt_k = sim_t - k_t
        if -0.25 <= dt_k <= 1.50:
            uv_k, zc_k = project_3d_to_2d(k_pos.reshape(1, 3), R, t, K)
            if float(zc_k[0]) > 0.6:
                kx, ky = int(round(uv_k[0, 0])), int(round(uv_k[0, 1]))
                prog_k = max(0.0, min(1.0, (dt_k + 0.25) / 1.75))
                ring_r = int(round(26 + 175 * prog_k))
                fade = max(0.20, 1.0 - 0.8 * prog_k)
                core_r = max(10, int(round(36 * (1.0 - 0.65 * prog_k))))
                cv2.circle(
                    frame,
                    (kx, ky),
                    core_r + 10,
                    (40, int(145 * fade), int(255 * fade)),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    frame,
                    (kx, ky),
                    core_r,
                    (195, int(240 * fade), 255),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    frame,
                    (kx, ky),
                    ring_r,
                    (int(75 * fade), int(205 * fade), 255),
                    max(2, int(5 * fade)),
                    cv2.LINE_AA,
                )
                cv2.circle(
                    frame,
                    (kx, ky),
                    max(6, int(round(ring_r * 0.55))),
                    (170, 245, 255),
                    2,
                    cv2.LINE_AA,
                )
                for sp_i in range(20):
                    ang_sp = 2.0 * math.pi * sp_i / 20.0 + k_t
                    dist_sp = ring_r * (0.45 + 0.70 * ((sp_i % 4) / 3.0))
                    sx_p = int(round(kx + dist_sp * math.cos(ang_sp)))
                    sy_p = int(round(ky + dist_sp * math.sin(ang_sp)))
                    cv2.line(
                        frame,
                        (kx + int(0.35 * dist_sp * math.cos(ang_sp)), ky + int(0.35 * dist_sp * math.sin(ang_sp))),
                        (sx_p, sy_p),
                        (int(85 * fade), int(220 * fade), 255),
                        max(1, int(3 * fade)),
                        cv2.LINE_AA,
                    )

    if not hud_enabled:
        return frame

    # 5. AR Tactical HUD Projection Overlays (3D->2D lead vectors, lock reticles, CBF halo)
    wta = state.get('wta_assignments') or {}
    active_targets = [t_it for t_it in state['targets'] if t_it['status'] == 'ACTIVE']
    for did, tid in wta.items():
        if did in drone_screen_coords and tid in tgt_screen_coords:
            dx_s, dy_s, _ = drone_screen_coords[did]
            tx_s, ty_s, _ = tgt_screen_coords[tid]
            cv2.line(frame, (dx_s, dy_s), (tx_s, ty_s), (110, 235, 255), 1, cv2.LINE_AA)

    for tid, (tx_s, ty_s, depth_t) in tgt_screen_coords.items():
        box_r = max(20, min(46, int(round(290.0 / depth_t))))
        pts_d = np.array(
            [
                [tx_s, ty_s - box_r],
                [tx_s + box_r, ty_s],
                [tx_s, ty_s + box_r],
                [tx_s - box_r, ty_s],
            ],
            dtype=np.int32,
        )
        cv2.polylines(frame, [pts_d], True, (65, 110, 255), 2, cv2.LINE_AA)

    # Closest inter-drone pair CBF telemetry link
    d_items = sorted(state['drones'].items())
    min_sep = 999.0
    min_pair: Optional[Tuple[str, str]] = None
    for i in range(len(d_items)):
        for j in range(i + 1, len(d_items)):
            d1, v1 = d_items[i]
            d2, v2 = d_items[j]
            dist = float(np.linalg.norm(v1['pos'] - v2['pos']))
            if v1['pos'][2] > 0.8 and v2['pos'][2] > 0.8 and dist < min_sep:
                min_sep = dist
                min_pair = (d1, d2)
    if min_pair is not None and min_sep < 6.0 and min_pair[0] in drone_screen_coords and min_pair[1] in drone_screen_coords:
        p1_s = drone_screen_coords[min_pair[0]][:2]
        p2_s = drone_screen_coords[min_pair[1]][:2]
        cv2.line(frame, p1_s, p2_s, (110, 255, 175), 1, cv2.LINE_AA)

    # 6. Composite EO/IR Thermal PiP (top-right) & Mini Tactical Radar (bottom-left)
    primary_tgt = active_targets[0] if active_targets else (state['targets'][0] if state['targets'] else None)
    focus_drone = None
    if primary_tgt is not None and d_items:
        focus_drone = min(
            ((did, dv['pos']) for did, dv in d_items),
            key=lambda item: float(np.linalg.norm(item[1] - primary_tgt['pos'])),
        )
    pip = _render_eoir_thermal_pip(state, arena_cfg, primary_tgt, focus_drone, frame_idx, time_scale, fonts)
    pip_x, pip_y = width - pip.shape[1] - 28, 84
    frame[pip_y : pip_y + pip.shape[0], pip_x : pip_x + pip.shape[1]] = pip

    minimap = _draw_minimap(arena_cfg, state, eye, cam_tgt, fonts)
    mm_x, mm_y = 28, height - minimap.shape[0] - 80
    frame[mm_y : mm_y + minimap.shape[0], mm_x : mm_x + minimap.shape[1]] = minimap

    # 7. PIL Top Banner & Bottom Scrubber + AR Labels (single fast RGBA composite)
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, 'RGBA')

    # Top Command Banner (0..64)
    draw.rectangle([(0, 0), (width, 64)], fill=(8, 12, 22, 205))
    draw.line([(0, 64), (width, 64)], fill=(75, 195, 255, 210), width=2)
    draw.text((28, 10), 'AEGIS SWARM // COUNTER-UAS INTERCEPT SHOWCASE', font=fonts['title'], fill=(235, 245, 255, 255))
    draw.text((28, 38), act_title, font=fonts['hud'], fill=(105, 225, 255, 255))

    rel_t = max(0.0, sim_t - (t_start if t_start > 100.0 else 0.0))
    slo_badge_col = (255, 95, 75, 235) if time_scale < 0.5 else (42, 130, 175, 215)
    slo_text = 'TIME SCALE: 0.25x SLOW-MO' if time_scale < 0.5 else f'MISSION T+{rel_t:05.2f}s  [1.00x]'
    badge_w = int(round(draw.textlength(slo_text, font=fonts['badge']))) + 32
    draw.rounded_rectangle([(825, 14), (825 + badge_w, 50)], radius=6, fill=slo_badge_col)
    draw.text((841, 23), slo_text, font=fonts['badge'], fill=(255, 255, 255, 255))

    kills_so_far = sum(1 for k_t, _, _ in kills if k_t <= sim_t + 1e-3)
    total_tgts = max(len(state['targets']), 3)
    sep_txt = f'{min_sep:4.2f}m' if min_sep < 900.0 else ' >10m'
    stats_txt = (
        f'INTERCEPTORS: {len(state["drones"])}/6   '
        f'KILLS: {kills_so_far}/{total_tgts}   '
        f'MIN SEP: {sep_txt}   '
        f'CBF: ACTIVE'
    )
    draw.text((1205, 23), stats_txt, font=fonts['hud'], fill=(170, 245, 210, 255))

    # 3D->2D AR Callout Labels for drones & active targets
    for did, (ux, uy, _) in drone_screen_coords.items():
        if 80 < uy < height - 90 and 20 < ux < width - 80:
            draw.text((ux + 12, uy - 14), did.upper(), font=fonts['micro'], fill=(175, 245, 255, 235))
    for tid, (tx_s, ty_s, _) in tgt_screen_coords.items():
        if 80 < ty_s < height - 90 and 20 < tx_s < width - 140:
            draw.text((tx_s + 26, ty_s - 18), f'HOSTILE:{tid}', font=fonts['micro'], fill=(255, 130, 120, 245))

    # Bottom 5-Act Timeline Scrubber (y: 1018..1056)
    bar_x0, bar_x1, bar_y0, bar_y1 = 28, width - 28, height - 54, height - 18
    draw.rounded_rectangle([(bar_x0, bar_y0), (bar_x1, bar_y1)], radius=6, fill=(10, 14, 24, 210))
    prog_w = int(round((bar_x1 - bar_x0 - 8) * u_total))
    draw.rectangle([(bar_x0 + 4, bar_y1 - 8), (bar_x0 + 4 + prog_w, bar_y1 - 3)], fill=(80, 220, 255, 245))
    act_labels = ['I. SCRAMBLE', 'II. STEREO TRACK', 'III. WOLFPACK', 'IV. 0.25x STRIKE', 'V. BDA & RTB']
    act_splits = [0.0, 0.18, 0.38, 0.60, 0.82, 1.0]
    for a_i, lbl in enumerate(act_labels):
        ax = bar_x0 + 16 + int(round((bar_x1 - bar_x0 - 32) * act_splits[a_i]))
        is_cur = (a_i + 1) == act_idx
        fill_c = (255, 220, 95, 255) if is_cur else (140, 155, 175, 210)
        draw.text((ax, bar_y0 + 7), lbl, font=fonts['small'], fill=fill_c)

    if active_kill is not None:
        hit_msg = f'TERMINAL HIT CONFIRMED // {active_kill[1].upper()} // 0.25x REPLAY'
        hit_w = int(round(draw.textlength(hit_msg, font=fonts['badge']))) + 36
        draw.rounded_rectangle([(316, height - 122), (316 + hit_w, height - 74)], radius=6, fill=(140, 28, 22, 225))
        draw.text(
            (334, height - 106),
            hit_msg,
            font=fonts['badge'],
            fill=(255, 235, 180, 255),
        )

    ov_np = np.asarray(overlay)
    alpha = ov_np[:, :, 3:4].astype(np.float32) / 255.0
    ov_bgr = ov_np[:, :, :3][:, :, ::-1].astype(np.float32)
    blended = frame.astype(np.float32) * (1.0 - alpha) + ov_bgr * alpha
    return np.clip(blended, 0, 255).astype(np.uint8)


def render_showcase_cinema(
    trace_path: Path,
    summary_path: Optional[Path],
    obstacles_path: Path,
    output_mp4: Path,
    keyframes_dir: Path,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    duration_s: float = 20.0,
) -> Dict[str, Any]:
    """Validate mission trace and render 1920x1080 H.264 showcase MP4 + 5 act keyframe PNGs."""
    records, summary = load_and_validate_trace(trace_path, summary_path)
    arena_cfg = _load_arena_cfg(obstacles_path)
    mesh_lib = ArenaMeshLibrary(arena_cfg)
    sky_bg = _build_sky_background(width, height)
    fonts = _load_fonts()

    kills = _extract_kill_events(records)
    schedule = _build_timeline_schedule(records, kills, duration_s=duration_s, fps=fps)
    total_frames = len(schedule)

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    keyframes_dir.mkdir(parents=True, exist_ok=True)
    tmp_mp4 = output_mp4.with_name(f'.{output_mp4.stem}.tmp{output_mp4.suffix}')
    tmp_mp4.unlink(missing_ok=True)

    ffmpeg_bin = shutil.which('ffmpeg') or '/usr/bin/ffmpeg'
    cmd = [
        ffmpeg_bin,
        '-y',
        '-f',
        'rawvideo',
        '-pix_fmt',
        'bgr24',
        '-s',
        f'{width}x{height}',
        '-r',
        str(fps),
        '-i',
        '-',
        '-c:v',
        'libx264',
        '-pix_fmt',
        'yuv420p',
        '-preset',
        'fast',
        '-crf',
        '18',
        '-g',
        str(max(15, fps)),
        '-keyint_min',
        '15',
        '-bsf:v',
        'filter_units=remove_types=6',
        '-movflags',
        '+faststart',
        str(tmp_mp4),
    ]

    keyframe_targets = {
        int(round(0.10 * (total_frames - 1))): keyframes_dir / 'swarm_showcase_act1_scramble.png',
        int(round(0.28 * (total_frames - 1))): keyframes_dir / 'swarm_showcase_act2_stereo_track.png',
        int(round(0.49 * (total_frames - 1))): keyframes_dir / 'swarm_showcase_act3_wolfpack_intercept.png',
        int(round(0.71 * (total_frames - 1))): keyframes_dir / 'swarm_showcase_act4_slowmo_strike.png',
        int(round(0.92 * (total_frames - 1))): keyframes_dir / 'swarm_showcase_act5_rtb.png',
    }
    saved_keyframes: List[str] = []

    trails: Dict[str, List[np.ndarray]] = {}
    tgt_trails: Dict[str, List[np.ndarray]] = {}

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        for f_idx, (sim_t, act_idx, act_title, time_scale, active_kill) in enumerate(schedule):
            u_total = f_idx / float(max(total_frames - 1, 1))
            state = _interpolate_trace_state(records, sim_t)

            for did, dv in state['drones'].items():
                hist = trails.setdefault(did, [])
                hist.append(dv['pos'].copy())
                if len(hist) > 36:
                    del hist[0]
            for tgt in state['targets']:
                if tgt['status'] == 'ACTIVE':
                    th = tgt_trails.setdefault(tgt['target_id'], [])
                    th.append(tgt['pos'].copy())
                    if len(th) > 36:
                        del th[0]

            eye, cam_tgt, fov_deg = _choreograph_camera(act_idx, u_total, state, active_kill)
            img = render_single_frame(
                width=width,
                height=height,
                sky_bg=sky_bg,
                mesh_lib=mesh_lib,
                arena_cfg=arena_cfg,
                state=state,
                trails=trails,
                tgt_trails=tgt_trails,
                eye=eye,
                cam_tgt=cam_tgt,
                fov_deg=fov_deg,
                act_idx=act_idx,
                act_title=act_title,
                time_scale=time_scale,
                u_total=u_total,
                frame_idx=f_idx,
                kills=kills,
                active_kill=active_kill,
                fonts=fonts,
                hud_enabled=True,
                t_start=float(records[0]['timestamp']),
            )
            if f_idx in keyframe_targets:
                kf_path = keyframe_targets[f_idx]
                cv2.imwrite(str(kf_path), img)
                saved_keyframes.append(str(kf_path))
            proc.stdin.write(img.tobytes())

        proc.stdin.close()
        stderr_data = proc.stderr.read().decode('utf-8', errors='replace') if proc.stderr else ''
        rc = proc.wait(timeout=120)
        if rc != 0:
            tmp_mp4.unlink(missing_ok=True)
            raise RuntimeError(f'ffmpeg encoding failed (rc={rc}): {stderr_data[-500:]}')
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        tmp_mp4.unlink(missing_ok=True)
        raise

    tmp_mp4.replace(output_mp4)
    return {
        'output_mp4': str(output_mp4),
        'keyframes': saved_keyframes,
        'frames': total_frames,
        'fps': fps,
        'resolution': f'{width}x{height}',
        'summary_targets_neutralized': summary.get('targets_neutralized'),
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Render Project 2 multi-UAV blockbuster cinema MP4 & keyframes.')
    p.add_argument('--trace', type=Path, default=DEFAULT_TRACE_PATH, help='Input mission trace JSONL path.')
    p.add_argument('--summary', type=Path, default=DEFAULT_SUMMARY_PATH, help='Input mission summary JSON path.')
    p.add_argument('--obstacles', type=Path, default=_find_default_obstacles_yaml(), help='obstacles.yaml path.')
    p.add_argument(
        '--output',
        type=Path,
        default=DEFAULT_OUTPUT_DIR / 'swarm_showcase_cinema.mp4',
        help='Output MP4 path.',
    )
    p.add_argument('--keyframes-dir', type=Path, default=None, help='Directory for 5 act keyframe PNGs.')
    p.add_argument('--width', type=int, default=1920)
    p.add_argument('--height', type=int, default=1080)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--duration', type=float, default=18.0)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    kf_dir = args.keyframes_dir or args.output.parent
    try:
        info = render_showcase_cinema(
            trace_path=args.trace,
            summary_path=args.summary,
            obstacles_path=args.obstacles,
            output_mp4=args.output,
            keyframes_dir=kf_dir,
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration_s=args.duration,
        )
    except Exception as exc:
        print(f'ERROR [cinema_renderer]: {exc}', file=sys.stderr)
        return 1

    print(json.dumps(info, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
