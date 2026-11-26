"""MP4 media probing and frame-level visual defect validator (ffprobe + OpenCV/ffmpeg)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class VideoProbeResult:
    """Metadata extracted from ffprobe."""

    exists: bool
    width: int
    height: int
    fps: float
    codec: str
    pix_fmt: str
    duration_s: float
    nb_frames: int
    keyframe_count: int
    errors: List[str]


@dataclass
class VideoQualityResult:
    """Combined ffprobe and sampled-frame defect verification result."""

    ok: bool
    probe: VideoProbeResult
    sampled_frames: int
    black_frames: int
    uniform_frames: int
    frozen_pairs: int
    min_luma_mean: float
    min_luma_std: float
    errors: List[str]


def _parse_fps(rate_str: Optional[str]) -> float:
    if not rate_str or rate_str == "0/0":
        return 0.0
    try:
        frac = Fraction(rate_str)
        return float(frac)
    except (ValueError, ZeroDivisionError):
        return 0.0


def probe_video(mp4_path: Union[str, Path]) -> VideoProbeResult:
    """Probe MP4 stream metadata and keyframe count via ffprobe."""
    path = Path(mp4_path)
    if not path.is_file() or path.stat().st_size == 0:
        return VideoProbeResult(
            exists=False,
            width=0,
            height=0,
            fps=0.0,
            codec="",
            pix_fmt="",
            duration_s=0.0,
            nb_frames=0,
            keyframe_count=0,
            errors=[f"Video file missing or empty: {path}"],
        )

    errors: List[str] = []
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,codec_name,pix_fmt,nb_frames,nb_read_frames",
        "-show_entries",
        "format=duration",
        "-count_frames",
        "-of",
        "json",
        str(path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30.0, check=False)
    if proc.returncode != 0:
        return VideoProbeResult(
            exists=True,
            width=0,
            height=0,
            fps=0.0,
            codec="",
            pix_fmt="",
            duration_s=0.0,
            nb_frames=0,
            keyframe_count=0,
            errors=[f"ffprobe failed (rc={proc.returncode}): {proc.stderr.strip()}"],
        )

    try:
        info = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return VideoProbeResult(
            exists=True,
            width=0,
            height=0,
            fps=0.0,
            codec="",
            pix_fmt="",
            duration_s=0.0,
            nb_frames=0,
            keyframe_count=0,
            errors=[f"Failed to parse ffprobe JSON: {exc}"],
        )

    streams = info.get("streams") or []
    if not streams:
        return VideoProbeResult(
            exists=True,
            width=0,
            height=0,
            fps=0.0,
            codec="",
            pix_fmt="",
            duration_s=0.0,
            nb_frames=0,
            keyframe_count=0,
            errors=["No video stream found in container"],
        )

    st = streams[0]
    width = int(st.get("width") or 0)
    height = int(st.get("height") or 0)
    fps = _parse_fps(st.get("avg_frame_rate")) or _parse_fps(st.get("r_frame_rate"))
    codec = str(st.get("codec_name") or "")
    pix_fmt = str(st.get("pix_fmt") or "")

    nb_frames_raw = st.get("nb_frames") or st.get("nb_read_frames") or "0"
    try:
        nb_frames = int(nb_frames_raw)
    except (TypeError, ValueError):
        nb_frames = 0

    dur_raw = (info.get("format") or {}).get("duration") or "0"
    try:
        duration_s = float(dur_raw)
    except (TypeError, ValueError):
        duration_s = 0.0

    kf_cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_packets",
        "-show_entries",
        "packet=flags",
        "-of",
        "csv=p=0",
        str(path),
    ]
    kf_proc = subprocess.run(kf_cmd, capture_output=True, text=True, timeout=30.0, check=False)
    keyframe_count = 0
    if kf_proc.returncode == 0:
        keyframe_count = sum(
            1
            for ln in kf_proc.stdout.splitlines()
            if ln.strip() and "K" in ln.strip()
        )
    else:
        errors.append(f"Keyframe ffprobe failed: {kf_proc.stderr.strip()}")

    return VideoProbeResult(
        exists=True,
        width=width,
        height=height,
        fps=fps,
        codec=codec,
        pix_fmt=pix_fmt,
        duration_s=duration_s,
        nb_frames=nb_frames,
        keyframe_count=keyframe_count,
        errors=errors,
    )


def _sample_frames_ffmpeg(
    mp4_path: Path,
    width: int,
    height: int,
    nb_frames: int,
    sample_count: int,
) -> List[np.ndarray]:
    """Fallback frame sampler using ffmpeg rawvideo gray pipe."""
    if width <= 0 or height <= 0 or nb_frames <= 0:
        return []
    indices = sorted(
        {
            min(nb_frames - 1, max(0, int(round(i * (nb_frames - 1) / max(1, sample_count - 1)))))
            for i in range(min(sample_count, nb_frames))
        }
    )
    select_expr = "+".join(f"eq(n\\,{idx})" for idx in indices)
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(mp4_path),
        "-vf",
        f"select='{select_expr}',format=gray",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=60.0, check=False)
    if proc.returncode != 0:
        return []
    frame_bytes = width * height
    total = len(proc.stdout) // frame_bytes
    frames: List[np.ndarray] = []
    for k in range(total):
        start = k * frame_bytes
        chunk = proc.stdout[start:start + frame_bytes]
        arr = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width))
        frames.append(arr)
    return frames


def _sample_gray_frames(
    mp4_path: Path,
    probe: VideoProbeResult,
    sample_count: int,
) -> List[np.ndarray]:
    nb = max(1, probe.nb_frames)
    n_samples = min(max(1, sample_count), nb)
    if n_samples == 1:
        indices = [0]
    else:
        indices = [
            int(round(i * (nb - 1) / (n_samples - 1))) for i in range(n_samples)
        ]

    grays: List[np.ndarray] = []
    cap = cv2.VideoCapture(str(mp4_path))
    if cap.isOpened():
        try:
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
                ok, frame = cap.read()
                if not ok or frame is None:
                    grays = []
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                grays.append(gray)
        finally:
            cap.release()

    if len(grays) < len(indices):
        fallback = _sample_frames_ffmpeg(
            mp4_path, probe.width, probe.height, probe.nb_frames, n_samples
        )
        if fallback:
            return fallback

    return grays


def check_video_quality(
    mp4_path: Union[str, Path],
    min_w: int = 1920,
    min_h: int = 1080,
    min_fps: float = 24.0,
    sample_frames: int = 12,
    min_duration_s: float = 1.0,
    expected_codecs: Tuple[str, ...] = ("h264",),
    expected_pix_fmts: Tuple[str, ...] = ("yuv420p",),
    black_mean_thresh: float = 8.0,
    uniform_std_thresh: float = 5.0,
    frozen_mad_thresh: float = 0.5,
    max_frozen_pairs: int = 0,
) -> VideoQualityResult:
    """Validate MP4 technical parameters and detect black, uniform, or frozen frames."""
    path = Path(mp4_path)
    probe = probe_video(path)
    errors: List[str] = list(probe.errors)

    if not probe.exists:
        return VideoQualityResult(
            ok=False,
            probe=probe,
            sampled_frames=0,
            black_frames=0,
            uniform_frames=0,
            frozen_pairs=0,
            min_luma_mean=0.0,
            min_luma_std=0.0,
            errors=errors,
        )

    if probe.width < min_w or probe.height < min_h:
        errors.append(
            f"Resolution {probe.width}x{probe.height} below minimum {min_w}x{min_h}"
        )
    if probe.fps < min_fps:
        errors.append(f"FPS {probe.fps:.3f} below minimum {min_fps:.3f}")
    if expected_codecs and probe.codec not in expected_codecs:
        errors.append(f"Codec '{probe.codec}' not in {expected_codecs}")
    if expected_pix_fmts and probe.pix_fmt not in expected_pix_fmts:
        errors.append(f"Pixel format '{probe.pix_fmt}' not in {expected_pix_fmts}")
    if probe.duration_s < min_duration_s:
        errors.append(
            f"Duration {probe.duration_s:.3f}s below minimum {min_duration_s:.3f}s"
        )
    if probe.nb_frames <= 0:
        errors.append(f"Invalid frame count nb_frames={probe.nb_frames}")
    if probe.keyframe_count < 1:
        errors.append("Video contains 0 keyframes")

    grays = _sample_gray_frames(path, probe, sample_frames)
    if not grays:
        errors.append("Failed to decode any sampled frames from video")
        return VideoQualityResult(
            ok=False,
            probe=probe,
            sampled_frames=0,
            black_frames=0,
            uniform_frames=0,
            frozen_pairs=0,
            min_luma_mean=0.0,
            min_luma_std=0.0,
            errors=errors,
        )

    black_frames = 0
    uniform_frames = 0
    frozen_pairs = 0
    min_luma_mean = float("inf")
    min_luma_std = float("inf")
    prev_gray: Optional[np.ndarray] = None

    for gray in grays:
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        dark_ratio = float(np.mean(gray < 10))
        min_luma_mean = min(min_luma_mean, mean_val)
        min_luma_std = min(min_luma_std, std_val)

        if mean_val < black_mean_thresh or dark_ratio > 0.98:
            black_frames += 1
        if std_val < uniform_std_thresh:
            uniform_frames += 1

        if prev_gray is not None:
            mad = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
            if mad < frozen_mad_thresh:
                frozen_pairs += 1
        prev_gray = gray

    if black_frames > 0:
        errors.append(f"Detected {black_frames} black frame(s) out of {len(grays)} sampled")
    if uniform_frames > 0:
        errors.append(
            f"Detected {uniform_frames} uniform/solid frame(s) (min_std={min_luma_std:.2f})"
        )
    if frozen_pairs > max_frozen_pairs:
        errors.append(
            f"Detected {frozen_pairs} frozen consecutive frame pair(s) (max allowed={max_frozen_pairs})"
        )

    return VideoQualityResult(
        ok=(len(errors) == 0),
        probe=probe,
        sampled_frames=len(grays),
        black_frames=black_frames,
        uniform_frames=uniform_frames,
        frozen_pairs=frozen_pairs,
        min_luma_mean=min_luma_mean if min_luma_mean != float("inf") else 0.0,
        min_luma_std=min_luma_std if min_luma_std != float("inf") else 0.0,
        errors=errors,
    )
