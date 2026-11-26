"""Tier 1 functional opaque-box tests for F10: Blockbuster Cinema Rendering & Pipeline."""

from __future__ import annotations

import ast
import io
import math
import os
import re
import tokenize
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image
import pytest

from helpers.hardware_git_guard import check_cuda_env_excludes_gpu3
from helpers.media_validator import check_video_quality, probe_video
from helpers.ros_probe import run_command_with_gpu_guard

pytestmark = [
    pytest.mark.tier1,
    pytest.mark.feature(id="F10"),
]


def _find_showcase_runner(ros2_ws_root: Path) -> Path:
    pkg = ros2_ws_root / "src" / "flight_ctrl_pkg"
    primary = pkg / "scripts" / "run_showcase_pipeline.sh"
    fallback = pkg / "showcase" / "run_showcase_pipeline.sh"
    if primary.is_file():
        return primary
    if fallback.is_file():
        return fallback
    raise AssertionError(
        f"Missing reproducible showcase script at {primary} (or fallback {fallback})"
    )


def _discover_m5_keyframe_pngs(showcase_output_dir: Path) -> List[Path]:
    assert showcase_output_dir.is_dir(), f"Missing showcase output dir: {showcase_output_dir}"
    candidates: List[Path] = []
    for p in sorted(showcase_output_dir.glob("*.png")):
        name = p.name.lower()
        if name.startswith("sim_") or name.startswith("project2_"):
            continue
        if "swarm_showcase" in name or "act" in name or "keyframe" in name:
            candidates.append(p)
    return candidates


_GPU3_UUID = "gpu-617ae266-eed3-ef41-ae0f-5f25c730f5d5"
_CVD_ASSIGN_RE = re.compile(
    r"""(?:^|[\s;|&(])(?:export\s+)?CUDA_VISIBLE_DEVICES\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s;|&)#]+))"""
)
_ACTIVE_GUARD_RE = re.compile(
    r"(?:^|[;\n|&]\s*|\s)(?:source|\.)\s+[^\n]*(?:gpu3_guard\.sh|env\.sh)\b"
    r"|(?:^|[;\n|&]\s*)(?:GPU3_GUARD_HEADLESS=\S+\s+)?(?:exec\s+|bash\s+)?\S*gpu3_guard\.sh\b"
)


def _strip_bash_comments(text: str) -> str:
    """Strip unquoted bash # comments while preserving ${#...}, ${1#...}, and quoted #."""
    out: List[str] = []
    in_single = False
    in_double = False
    escaped = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if escaped:
            out.append(ch)
            escaped = False
            i += 1
            continue
        if ch == "\\" and not in_single:
            out.append(ch)
            escaped = True
            i += 1
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            out.append(ch)
            i += 1
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
            i += 1
            continue
        if ch == "#" and not in_single and not in_double:
            prev = text[i - 1] if i > 0 else "\n"
            is_param_len = i >= 2 and text[i - 2:i] == "${"
            if not is_param_len and prev in " \t\r\n;|&(){":
                while i < n and text[i] != "\n":
                    i += 1
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _assert_script_guards_gpu3(runner_sh: Path, stripped_text: str) -> None:
    """Assert non-comment script text never assigns GPU 3 and activates GPU-3 guard."""
    for match in _CVD_ASSIGN_RE.finditer(stripped_text):
        raw_val = (
            match.group(1)
            if match.group(1) is not None
            else (match.group(2) if match.group(2) is not None else match.group(3))
        )
        check_val = raw_val
        if "$" in raw_val:
            default_match = re.search(r"\$\{[^}:]+(?::[-=]|[-=])([^}]+)\}", raw_val)
            if not default_match:
                continue
            check_val = default_match.group(1)
        tokens = [tok.strip().lower() for tok in check_val.split(",") if tok.strip()]
        assert tokens, f"Empty CUDA_VISIBLE_DEVICES assignment in {runner_sh}: {match.group(0)!r}"
        for tok in tokens:
            assert tok not in {"3", "all", _GPU3_UUID} and "617ae266-eed3-ef41-ae0f-5f25c730f5d5" not in tok, (
                f"Forbidden GPU-3 / all CUDA_VISIBLE_DEVICES assignment in {runner_sh}: {match.group(0)!r}"
            )
        assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": check_val}), (
            f"CUDA_VISIBLE_DEVICES does not exclude GPU 3 in {runner_sh}: {match.group(0)!r}"
        )

    has_active_guard = bool(_ACTIVE_GUARD_RE.search(stripped_text)) or (
        bool(re.search(r"(?:^|[\s;|&])(?:export\s+)?LD_PRELOAD\s*=[^\n]*libegl_gpu_filter\.so", stripped_text))
        and bool(_CVD_ASSIGN_RE.search(stripped_text))
    )
    assert has_active_guard, (
        f"{runner_sh} must actively source gpu3_guard.sh or set LD_PRELOAD=libegl_gpu_filter.so + CUDA_VISIBLE_DEVICES"
    )


def _strip_python_comments_and_docstrings(src: str, path: Path) -> str:
    """Return executable Python code tokens with comments and docstrings removed."""
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        raise AssertionError(f"Syntax error in showcase module {path}: {exc}") from exc

    doc_starts = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                val = node.body[0].value
                if isinstance(val, ast.Constant) and isinstance(val.value, str):
                    doc_starts.add((val.lineno, val.col_offset))

    kept: List[str] = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and tok.start in doc_starts:
            continue
        if tok.string:
            kept.append(tok.string)
    return " ".join(kept)


@pytest.mark.requires_m5
def test_f10_reproducible_pipeline_script_and_gpu3_guard(ros2_ws_root: Path) -> None:
    """Verify run_showcase_pipeline.sh exists, is executable, passes bash -n, and guards GPU-3."""
    runner_sh = _find_showcase_runner(ros2_ws_root)
    assert os.access(runner_sh, os.X_OK), f"{runner_sh} is not executable"

    bash_check = run_command_with_gpu_guard(["bash", "-n", str(runner_sh)], timeout_s=15.0)
    assert bash_check.returncode == 0, f"bash -n failed on {runner_sh}: {bash_check.stderr}"

    text = runner_sh.read_text(encoding="utf-8", errors="replace")
    stripped = _strip_bash_comments(text)
    _assert_script_guards_gpu3(runner_sh, stripped)

    guard_sh = ros2_ws_root / "src" / "flight_ctrl_pkg" / "scripts" / "gpu3_guard.sh"
    assert guard_sh.is_file(), f"Missing {guard_sh}"
    guard_probe = run_command_with_gpu_guard(
        [
            "bash",
            "-lc",
            (
                f"unset LD_PRELOAD; export CUDA_VISIBLE_DEVICES=3; "
                f"source '{guard_sh}' && "
                f"printf 'CVD=%s\\nLDP=%s\\n' \"$CUDA_VISIBLE_DEVICES\" \"${{LD_PRELOAD:-}}\""
            ),
        ],
        timeout_s=10.0,
        extra_env={"CUDA_VISIBLE_DEVICES": "3"},
    )
    assert not guard_probe.timed_out and guard_probe.returncode == 0, (
        f"gpu3_guard.sh failed under poisoned CUDA_VISIBLE_DEVICES=3: {guard_probe.stderr}"
    )
    env_kv = dict(ln.split("=", 1) for ln in guard_probe.stdout.splitlines() if "=" in ln)
    assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": env_kv.get("CVD", "")}), (
        f"Effective CUDA_VISIBLE_DEVICES failed GPU-3 exclusion: {env_kv}"
    )
    assert "libegl_gpu_filter.so" in env_kv.get("LDP", ""), (
        f"Effective LD_PRELOAD missing libegl_gpu_filter.so: {env_kv}"
    )

    if re.search(r"--help\b", stripped):
        help_res = run_command_with_gpu_guard(
            ["bash", str(runner_sh), "--help"],
            cwd=ros2_ws_root,
            timeout_s=10.0,
            extra_env={"CUDA_VISIBLE_DEVICES": "2"},
        )
        assert not help_res.timed_out and help_res.returncode == 0, (
            f"{runner_sh} --help failed: rc={help_res.returncode} stderr={help_res.stderr}"
        )

    showcase_dir = ros2_ws_root / "src" / "flight_ctrl_pkg" / "showcase"
    assert showcase_dir.is_dir(), f"Missing showcase directory: {showcase_dir}"
    py_files = list(showcase_dir.glob("*.py"))
    assert len(py_files) >= 1, f"Expected >= 1 Python module in {showcase_dir}"


@pytest.mark.requires_m5
def test_f10_final_mp4_exists_and_stream_spec(showcase_output_dir: Path) -> None:
    """Verify swarm_showcase_cinema.mp4 meets >=1920x1080, >=30fps, h264, yuv420p Contract #7."""
    mp4 = showcase_output_dir / "swarm_showcase_cinema.mp4"
    assert mp4.is_file(), f"Missing final showcase MP4: {mp4}"
    assert mp4.stat().st_size > 1_000_000, f"Showcase MP4 suspiciously small: {mp4.stat().st_size}"

    probe = probe_video(mp4)
    assert probe.exists is True
    assert probe.errors == [], f"ffprobe errors: {probe.errors}"
    assert probe.width >= 1920 and probe.height >= 1080, (
        f"Resolution {probe.width}x{probe.height} below 1920x1080"
    )
    assert probe.fps >= 30.0, f"FPS {probe.fps:.2f} below Contract #7 >= 30.0 fps"
    assert probe.codec == "h264", f"Unexpected codec: {probe.codec}"
    assert probe.pix_fmt == "yuv420p", f"Unexpected pix_fmt: {probe.pix_fmt}"
    assert probe.duration_s >= 10.0, f"Duration {probe.duration_s:.2f}s too short"
    assert probe.nb_frames >= 300, f"Frame count {probe.nb_frames} < 300"
    assert probe.keyframe_count >= 2, f"Expected >= 2 keyframes, got {probe.keyframe_count}"


@pytest.mark.requires_m5
def test_f10_final_mp4_visual_quality_and_bitstream_integrity(
    showcase_output_dir: Path,
) -> None:
    """Verify zero black/uniform/frozen frames, clean ffmpeg decode, and monotonic PTS."""
    mp4 = showcase_output_dir / "swarm_showcase_cinema.mp4"
    quality = check_video_quality(
        mp4,
        min_w=1920,
        min_h=1080,
        min_fps=30.0,
        sample_frames=24,
        min_duration_s=10.0,
        max_frozen_pairs=0,
    )
    assert quality.ok is True, f"Video quality failed: {quality.errors}"
    assert quality.sampled_frames == 24
    assert quality.black_frames == 0
    assert quality.uniform_frames == 0
    assert quality.frozen_pairs == 0
    assert quality.min_luma_mean >= 8.0
    assert quality.min_luma_std >= 5.0

    dec = run_command_with_gpu_guard(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-f", "null", "-"],
        timeout_s=120.0,
    )
    assert dec.timed_out is False
    assert dec.returncode == 0, f"ffmpeg bitstream decode failed: {dec.stderr}"
    assert dec.stderr.strip() == "", f"ffmpeg emitted decode errors: {dec.stderr}"

    dts_cmd = run_command_with_gpu_guard(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=dts_time",
            "-of",
            "csv=p=0",
            str(mp4),
        ],
        timeout_s=60.0,
    )
    assert not dts_cmd.timed_out and dts_cmd.returncode == 0, (
        f"ffprobe packet=dts_time failed: {dts_cmd.stderr}"
    )
    dts_lines = [ln.strip() for ln in dts_cmd.stdout.splitlines() if ln.strip()]
    assert len(dts_lines) >= 300, f"Expected >= 300 video packets, got {len(dts_lines)}"
    dts_vals = [float(ln) for ln in dts_lines]
    assert all(math.isfinite(v) for v in dts_vals), "Non-finite packet dts_time detected"
    for i in range(1, len(dts_vals)):
        assert dts_vals[i] >= dts_vals[i - 1], (
            f"Non-monotonic decode DTS at packet {i}: {dts_vals[i - 1]} -> {dts_vals[i]}"
        )

    pts_cmd = run_command_with_gpu_guard(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_frames",
            "-show_entries",
            "frame=best_effort_timestamp_time",
            "-of",
            "csv=p=0",
            str(mp4),
        ],
        timeout_s=180.0,
    )
    assert not pts_cmd.timed_out and pts_cmd.returncode == 0, (
        f"ffprobe frame=best_effort_timestamp_time failed: {pts_cmd.stderr}"
    )
    pts_lines = [ln.strip() for ln in pts_cmd.stdout.splitlines() if ln.strip()]
    assert len(pts_lines) >= 300, f"Expected >= 300 decoded frames, got {len(pts_lines)}"
    pts_vals = [float(ln) for ln in pts_lines]
    assert all(math.isfinite(v) for v in pts_vals), "Non-finite frame PTS detected"
    for i in range(1, len(pts_vals)):
        assert pts_vals[i] >= pts_vals[i - 1], (
            f"Non-monotonic presentation PTS at frame {i}: {pts_vals[i - 1]} -> {pts_vals[i]}"
        )


@pytest.mark.requires_m5
def test_f10_act_keyframe_pngs_present_hd_and_distinct(
    showcase_output_dir: Path,
) -> None:
    """Verify >= 5 act keyframe PNGs exist, are valid >=1920x1080 images, and are distinct."""
    pngs = _discover_m5_keyframe_pngs(showcase_output_dir)
    assert len(pngs) >= 5, (
        f"Expected >= 5 act keyframe PNGs in {showcase_output_dir}, found {[p.name for p in pngs]}"
    )

    loaded_grays: List[np.ndarray] = []
    for png in pngs:
        with Image.open(png) as img:
            w, h = img.size
            assert w >= 1920 and h >= 1080, f"{png.name} size {w}x{h} below 1920x1080"
        arr = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        assert arr is not None and arr.size > 0, f"cv2 failed to decode {png}"
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr))
        assert mean_v >= 8.0, f"{png.name} appears black (mean={mean_v:.2f})"
        assert std_v >= 5.0, f"{png.name} appears uniform (std={std_v:.2f})"
        loaded_grays.append(arr)

    for i in range(len(loaded_grays) - 1):
        a, b = loaded_grays[i], loaded_grays[i + 1]
        if a.shape != b.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        diff = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
        assert diff >= 1.0, (
            f"Consecutive act keyframes {pngs[i].name} and {pngs[i + 1].name} are duplicates (diff={diff:.3f})"
        )


@pytest.mark.requires_m5
def test_f10_cinema_slowmo_eoir_pip_and_ar_hud_artifacts(
    ros2_ws_root: Path, showcase_output_dir: Path
) -> None:
    """Verify M5 showcase pipeline implements 0.25x slow-mo, EO/IR PiP, and 3D-to-2D AR HUD."""
    showcase_dir = ros2_ws_root / "src" / "flight_ctrl_pkg" / "showcase"
    assert showcase_dir.is_dir(), f"Missing showcase package dir: {showcase_dir}"
    py_paths = [
        p for p in sorted(showcase_dir.rglob("*.py")) if "__pycache__" not in p.parts
    ]
    assert py_paths, f"No Python modules in {showcase_dir}"
    stripped_sources = [
        _strip_python_comments_and_docstrings(
            p.read_text(encoding="utf-8", errors="replace"), p
        )
        for p in py_paths
    ]
    combined = "\n".join(stripped_sources)

    assert re.search(r"1(?:\.0+)?\s*/\s*4(?:\.0+)?\b|0\.25|slow", combined, re.IGNORECASE), (
        "Expected 0.25x / 1/4 slow-motion replay capability in showcase pipeline code"
    )
    assert re.search(r"thermal|eo[_/]?ir|flir|ir_|pip|inset", combined, re.IGNORECASE), (
        "Expected EO/IR thermal PiP/inset compositing in showcase pipeline code"
    )
    assert re.search(r"hud|project|overlay|ar_", combined, re.IGNORECASE), (
        "Expected 3D-to-2D AR HUD projection/overlay in showcase pipeline code"
    )

    pngs = _discover_m5_keyframe_pngs(showcase_output_dir)
    assert len(pngs) >= 5, "Missing 5-act keyframe PNGs for HUD/PiP inspection"
    edge_scores = []
    for png in pngs[:5]:
        gray = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        assert gray is not None
        edges = cv2.Canny(gray, 50, 150)
        edge_scores.append(float(np.mean(edges > 0)))
    assert max(edge_scores) > 0.005, (
        f"Expected crisp HUD/scene edges across keyframes, got edge ratios {edge_scores}"
    )


@pytest.mark.requires_m1
def test_f10_arena_director_camera_thermal_and_lens_flare_sdf(
    ros2_ws_root: Path,
) -> None:
    """Verify urban_counter_swarm_arena.sdf configures director_rgb, director_thermal & lens flare."""
    sdf_path = (
        ros2_ws_root
        / "src"
        / "flight_ctrl_pkg"
        / "sim"
        / "worlds"
        / "urban_counter_swarm_arena.sdf"
    )
    assert sdf_path.is_file(), f"Missing SDF world: {sdf_path}"
    root = ET.parse(sdf_path).getroot()
    world = root.find("world") or root

    world_plugins = [p.attrib.get("filename", "") for p in world.findall("plugin")]
    assert any("particle-emitter" in fn for fn in world_plugins), (
        f"Missing particle-emitter world plugin: {world_plugins}"
    )

    director = None
    for m in world.findall("model"):
        if m.attrib.get("name") == "director_cam":
            director = m
            break
    assert director is not None, "Missing director_cam model in urban_counter_swarm_arena.sdf"

    sensors = {s.attrib.get("name"): s for s in director.findall(".//sensor")}
    assert "director_rgb" in sensors, f"Missing director_rgb sensor: {list(sensors)}"
    rgb = sensors["director_rgb"]
    w = int(rgb.findtext("./camera/image/width", default="0"))
    h = int(rgb.findtext("./camera/image/height", default="0"))
    rate = float(rgb.findtext("./update_rate", default="0"))
    assert w >= 1920 and h >= 1080, f"director_rgb resolution {w}x{h} < 1920x1080"
    assert rate >= 30.0, f"director_rgb update_rate {rate} < 30"

    flare_plugins = [
        p for p in rgb.findall("plugin") if "lens-flare" in p.attrib.get("filename", "")
    ]
    assert flare_plugins, "Missing gz-sim-lens-flare-system plugin on director_rgb"
    assert flare_plugins[0].findtext("light_name", default="").strip() == "sunUTC"

    assert "director_thermal" in sensors, f"Missing director_thermal sensor: {list(sensors)}"
    therm = sensors["director_thermal"]
    assert therm.attrib.get("type") == "thermal"
    therm_plugins = [p.attrib.get("filename", "") for p in therm.findall("plugin")]
    assert any("thermal-sensor" in fn for fn in therm_plugins), (
        f"Missing thermal sensor plugin on director_thermal: {therm_plugins}"
    )
