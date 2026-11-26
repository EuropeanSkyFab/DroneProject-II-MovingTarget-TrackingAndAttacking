"""Tier 2 BVA & corner-case opaque-box tests for F10: Cinema Rendering & MP4 Pipeline."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from helpers.hardware_git_guard import check_cuda_env_excludes_gpu3
from helpers.media_validator import check_video_quality, probe_video
from helpers.ros_probe import run_command_with_gpu_guard

pytestmark = [
    pytest.mark.tier2,
    pytest.mark.feature(id="F10"),
]


def _make_testsrc_mp4(
    out_path: Path,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    duration_s: float = 1.2,
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
    src_expr: str = "testsrc2",
    vf_filter: str | None = None,
) -> None:
    sep = ":" if "=" in src_expr else "="
    src_filter = f"{src_expr}{sep}size={width}x{height}:rate={fps}:duration={duration_s}"
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        src_filter,
    ]
    if vf_filter:
        cmd.extend(["-vf", vf_filter])
    cmd.extend(
        [
            "-c:v",
            codec,
            "-pix_fmt",
            pix_fmt,
            str(out_path),
        ]
    )
    res = run_command_with_gpu_guard(cmd, timeout_s=60.0)
    assert res.returncode == 0, f"ffmpeg failed creating {out_path}: {res.stderr}"


def _encode_bgr_frames_to_mp4(
    frames: list[np.ndarray],
    out_path: Path,
    fps: int = 30,
) -> None:
    h, w = frames[0].shape[:2]
    tmp_raw = out_path.with_suffix(".raw.bgr")
    with tmp_raw.open("wb") as fh:
        for fr in frames:
            fh.write(fr.tobytes())
    cmd = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        str(tmp_raw),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]
    res = run_command_with_gpu_guard(cmd, timeout_s=60.0)
    tmp_raw.unlink(missing_ok=True)
    assert res.returncode == 0, f"ffmpeg rawvideo encode failed: {res.stderr}"


@pytest.mark.pre_integration_safe
def test_f10_bva_resolution_1078p_vs_1080p(tmp_path: Path) -> None:
    """Verify resolution boundary: 1918x1080 and 1920x1078 fail while 1920x1080 passes."""
    bad_w = tmp_path / "res_1918x1080.mp4"
    bad_h = tmp_path / "res_1920x1078.mp4"
    good = tmp_path / "res_1920x1080.mp4"

    _make_testsrc_mp4(bad_w, width=1918, height=1080)
    _make_testsrc_mp4(bad_h, width=1920, height=1078)
    _make_testsrc_mp4(good, width=1920, height=1080)

    r_w = check_video_quality(bad_w, min_w=1920, min_h=1080, min_fps=30.0, sample_frames=6)
    r_h = check_video_quality(bad_h, min_w=1920, min_h=1080, min_fps=30.0, sample_frames=6)
    r_ok = check_video_quality(good, min_w=1920, min_h=1080, min_fps=30.0, sample_frames=6)

    assert r_w.ok is False and any("1918x1080" in e for e in r_w.errors)
    assert r_h.ok is False and any("1920x1078" in e for e in r_h.errors)
    assert r_ok.ok is True, f"Valid 1920x1080 failed: {r_ok.errors}"


@pytest.mark.pre_integration_safe
def test_f10_bva_fps_23_24_29_30_thresholds(tmp_path: Path) -> None:
    """Verify exact FPS thresholds: 23<24, 24>=24 & <30, 29<30, 30>=30."""
    clips = {}
    for fps in (23, 24, 29, 30):
        p = tmp_path / f"fps_{fps}.mp4"
        _make_testsrc_mp4(p, fps=fps, duration_s=1.2)
        clips[fps] = p

    assert check_video_quality(clips[23], min_fps=24.0, sample_frames=6).ok is False
    assert check_video_quality(clips[24], min_fps=24.0, sample_frames=6).ok is True
    assert check_video_quality(clips[24], min_fps=30.0, sample_frames=6).ok is False
    assert check_video_quality(clips[29], min_fps=30.0, sample_frames=6).ok is False
    assert check_video_quality(clips[30], min_fps=30.0, sample_frames=6).ok is True


@pytest.mark.pre_integration_safe
def test_f10_bva_blackout_uniform_frozen_slowmo_codec_and_corrupt_mp4(
    tmp_path: Path,
) -> None:
    """Verify partial blackout, uniform solid, frozen vs 0.25x slow-mo, codec/pix_fmt, corrupt MP4."""
    # 1. Partial blackout (frames 12..23 of 36 black) and 0.25x slow-mo moving bar
    blackout_frames = []
    slowmo_frames = []
    h, w = 1080, 1920
    for idx in range(36):
        fr = np.full((h, w, 3), 40, dtype=np.uint8)
        x0 = 100 + idx * 12  # crisp moving target (~12 px/frame across 1920px; sampled step > 0.5 MAD)
        cv2.rectangle(fr, (x0, 200), (x0 + 220, 880), (235, 210, 80), -1)
        cv2.putText(fr, f"F{idx}", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        slowmo_frames.append(fr)
        if 12 <= idx <= 23:
            blackout_frames.append(np.zeros((h, w, 3), dtype=np.uint8))
        else:
            blackout_frames.append(fr)

    blackout_mp4 = tmp_path / "partial_blackout.mp4"
    slowmo_mp4 = tmp_path / "slowmo_valid.mp4"
    _encode_bgr_frames_to_mp4(blackout_frames, blackout_mp4, fps=30)
    _encode_bgr_frames_to_mp4(slowmo_frames, slowmo_mp4, fps=30)

    r_bo = check_video_quality(blackout_mp4, min_fps=30.0, sample_frames=12)
    assert r_bo.ok is False and r_bo.black_frames >= 1, f"Partial blackout missed: {r_bo}"

    r_sm = check_video_quality(slowmo_mp4, min_fps=30.0, sample_frames=12)
    assert r_sm.ok is True and r_sm.frozen_pairs == 0, f"Slow-mo rejected falsely: {r_sm}"

    # 2. Uniform solid gray & frozen clip
    solid_mp4 = tmp_path / "solid_gray.mp4"
    _make_testsrc_mp4(solid_mp4, src_expr="color=c=0x808080")
    r_solid = check_video_quality(solid_mp4, min_fps=30.0, sample_frames=6)
    assert r_solid.ok is False and r_solid.uniform_frames >= 1

    frozen_mp4 = tmp_path / "frozen.mp4"
    _make_testsrc_mp4(
        frozen_mp4,
        src_expr="smptebars",
        vf_filter="trim=end_frame=1,loop=loop=35:size=1:start=0",
    )
    r_frozen = check_video_quality(frozen_mp4, min_fps=30.0, sample_frames=6)
    assert r_frozen.ok is False and r_frozen.frozen_pairs >= 1

    # 3. Codec & pix_fmt mismatch + duration < 1.0s
    bad_pix = tmp_path / "yuv444p.mp4"
    bad_codec = tmp_path / "mpeg4.mp4"
    short_mp4 = tmp_path / "short.mp4"
    _make_testsrc_mp4(bad_pix, pix_fmt="yuv444p")
    _make_testsrc_mp4(bad_codec, codec="mpeg4")
    _make_testsrc_mp4(short_mp4, duration_s=0.8)

    assert check_video_quality(bad_pix, min_fps=30.0, sample_frames=4).ok is False
    assert check_video_quality(bad_codec, min_fps=30.0, sample_frames=4).ok is False
    assert check_video_quality(short_mp4, min_fps=30.0, min_duration_s=1.0, sample_frames=4).ok is False

    # 4. Empty & corrupt container
    empty_mp4 = tmp_path / "empty.mp4"
    empty_mp4.write_bytes(b"")
    corrupt_mp4 = tmp_path / "corrupt.mp4"
    corrupt_mp4.write_bytes(b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00corrupt_payload")

    assert probe_video(empty_mp4).exists is False
    assert check_video_quality(empty_mp4).ok is False
    assert check_video_quality(corrupt_mp4).ok is False


@pytest.mark.pre_integration_safe
def test_f10_bva_keyframe_png_missing_corrupt_and_sub_hd_rejection(
    tmp_path: Path,
) -> None:
    """Verify missing act count (<5), corrupt bytes, sub-HD, black, and uniform PNG rejection."""
    def _validate_png_set(paths: list[Path]) -> list[str]:
        errs: list[str] = []
        if len(paths) < 5:
            errs.append(f"Expected >=5 act PNGs, got {len(paths)}")
        for p in paths:
            try:
                with Image.open(p) as im:
                    w, h = im.size
                if w < 1920 or h < 1080:
                    errs.append(f"{p.name}: {w}x{h} < 1920x1080")
                arr = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                if arr is None:
                    errs.append(f"{p.name}: unreadable")
                elif float(np.mean(arr)) < 8.0 or float(np.std(arr)) < 5.0:
                    errs.append(f"{p.name}: black/uniform")
            except Exception as exc:
                errs.append(f"{p.name}: {exc}")
        return errs

    valid_pngs = []
    for i in range(5):
        p = tmp_path / f"swarm_showcase_act{i + 1}.png"
        arr = np.full((1080, 1920, 3), 30 + i * 10, dtype=np.uint8)
        cv2.rectangle(arr, (200 + i * 150, 200), (600 + i * 150, 700), (220, 180, 60), -1)
        Image.fromarray(arr).save(p)
        valid_pngs.append(p)
    assert _validate_png_set(valid_pngs) == []
    assert _validate_png_set(valid_pngs[:4]) != []

    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not_a_png")
    sub_hd = tmp_path / "sub_hd.png"
    Image.new("RGB", (1918, 1080), (120, 80, 40)).save(sub_hd)
    black = tmp_path / "black.png"
    Image.new("RGB", (1920, 1080), (0, 0, 0)).save(black)
    solid = tmp_path / "solid.png"
    Image.new("RGB", (1920, 1080), (128, 128, 128)).save(solid)

    for bad in (corrupt, sub_hd, black, solid):
        assert _validate_png_set(valid_pngs[:4] + [bad]) != [], f"Failed to reject {bad.name}"


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
    out: list[str] = []
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
    """Reject quoted/unquoted/comma-separated GPU-3 assignments and require active guard."""
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
                f"Forbidden GPU-3 / all assignment in {runner_sh}: {match.group(0)!r}"
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


@pytest.mark.requires_m5
def test_f10_bva_run_showcase_pipeline_enforces_gpu3_guard(
    ros2_ws_root: Path,
) -> None:
    """Verify M5 run_showcase_pipeline.sh & gpu3_guard.sh reject poisoned CUDA_VISIBLE_DEVICES=3."""
    pkg = ros2_ws_root / "src" / "flight_ctrl_pkg"
    runner_sh = pkg / "scripts" / "run_showcase_pipeline.sh"
    if not runner_sh.is_file():
        runner_sh = pkg / "showcase" / "run_showcase_pipeline.sh"
    assert runner_sh.is_file(), f"Missing M5 run_showcase_pipeline.sh in {pkg}"

    text = runner_sh.read_text(encoding="utf-8", errors="replace")
    stripped = _strip_bash_comments(text)
    _assert_script_guards_gpu3(runner_sh, stripped)

    guard_sh = pkg / "scripts" / "gpu3_guard.sh"
    assert guard_sh.is_file(), f"Missing {guard_sh}"
    for poisoned, expected in (
        ("3", "2"),
        ("0,1,2,3", "0,1,2"),
        ("GPU-617ae266-eed3-ef41-ae0f-5f25c730f5d5", "2"),
    ):
        probe = run_command_with_gpu_guard(
            [
                "bash",
                "-lc",
                (
                    f"unset LD_PRELOAD; export CUDA_VISIBLE_DEVICES='{poisoned}'; "
                    f"source '{guard_sh}' && "
                    f"printf 'CVD=%s\\nLDP=%s\\n' \"$CUDA_VISIBLE_DEVICES\" \"${{LD_PRELOAD:-}}\""
                ),
            ],
            timeout_s=10.0,
            extra_env={"CUDA_VISIBLE_DEVICES": poisoned},
        )
        assert not probe.timed_out and probe.returncode == 0, (
            f"gpu3_guard.sh failed for {poisoned}: {probe.stderr}"
        )
        kv = dict(ln.split("=", 1) for ln in probe.stdout.splitlines() if "=" in ln)
        assert kv.get("CVD") == expected, (
            f"gpu3_guard.sh did not sanitize {poisoned} -> {expected}: {kv}"
        )
        assert check_cuda_env_excludes_gpu3({"CUDA_VISIBLE_DEVICES": kv.get("CVD", "")})
        assert "libegl_gpu_filter.so" in kv.get("LDP", ""), (
            f"gpu3_guard.sh missing libegl_gpu_filter.so in LD_PRELOAD: {kv}"
        )

    if re.search(r"--dry-run\b", stripped):
        dry = run_command_with_gpu_guard(
            ["bash", str(runner_sh), "--dry-run"],
            cwd=ros2_ws_root,
            timeout_s=15.0,
            extra_env={"CUDA_VISIBLE_DEVICES": "3"},
        )
        assert not dry.timed_out and dry.returncode == 0, (
            f"{runner_sh} --dry-run failed: rc={dry.returncode} stderr={dry.stderr}"
        )
    elif re.search(r"--help\b", stripped):
        hlp = run_command_with_gpu_guard(
            ["bash", str(runner_sh), "--help"],
            cwd=ros2_ws_root,
            timeout_s=10.0,
            extra_env={"CUDA_VISIBLE_DEVICES": "2"},
        )
        assert not hlp.timed_out and hlp.returncode == 0, (
            f"{runner_sh} --help failed: rc={hlp.returncode} stderr={hlp.stderr}"
        )


@pytest.mark.requires_m5
def test_f10_bva_corrupt_trace_jsonl_fails_gracefully_without_fake_mp4(
    tmp_path: Path, ros2_ws_root: Path
) -> None:
    """Verify M5 showcase pipeline/renderer fails non-zero on corrupt trace without fake MP4."""
    pkg = ros2_ws_root / "src" / "flight_ctrl_pkg"
    runner_sh = pkg / "scripts" / "run_showcase_pipeline.sh"
    if not runner_sh.is_file():
        runner_sh = pkg / "showcase" / "run_showcase_pipeline.sh"
    assert runner_sh.is_file(), f"Missing M5 run_showcase_pipeline.sh in {pkg}"

    corrupt_trace = tmp_path / "corrupt_trace.jsonl"
    corrupt_trace.write_text('{"timestamp": 1.0, "drones": "bad"}\n{truncated\n', encoding="utf-8")
    missing_summary = tmp_path / "missing_summary.json"
    out_dir = tmp_path / "showcase_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / "swarm_showcase_cinema.mp4"

    stripped = _strip_bash_comments(runner_sh.read_text(encoding="utf-8", errors="replace"))
    cmd = ["bash", str(runner_sh)]
    if re.search(r"--help\b", stripped):
        help_res = run_command_with_gpu_guard(
            ["bash", str(runner_sh), "--help"],
            cwd=ros2_ws_root,
            timeout_s=10.0,
            extra_env={"CUDA_VISIBLE_DEVICES": "2"},
        )
        if (
            not help_res.timed_out
            and help_res.returncode == 0
            and "--trace" in help_res.stdout
            and re.search(r"(?:^|\s)--output(?:\s|=|,|$)", help_res.stdout)
        ):
            cmd.extend(["--trace", str(corrupt_trace), "--output", str(out_mp4)])

    # Invoke the M5 pipeline pointing at corrupt trace & isolated tmp_path output via contracted env vars
    res = run_command_with_gpu_guard(
        cmd,
        cwd=ros2_ws_root,
        timeout_s=30.0,
        extra_env={
            "CUDA_VISIBLE_DEVICES": "2",
            "PROJECT2_TRACE_PATH": str(corrupt_trace),
            "PROJECT2_SUMMARY_PATH": str(missing_summary),
            "SHOWCASE_OUTPUT_DIR": str(out_dir),
        },
        source_env=True,
    )
    combined_out = f"{res.stderr}\n{res.stdout}"
    assert not res.timed_out, (
        f"Pipeline timed out instead of failing cleanly on corrupt trace: {res.stderr}"
    )
    assert not re.search(
        r"unrecognized\s+(?:option|argument)|unknown\s+option|invalid\s+option|illegal\s+option",
        combined_out,
        re.IGNORECASE,
    ), f"Pipeline failed due to CLI flag parsing rather than corrupt trace: {combined_out}"
    assert not (
        res.returncode == 2
        and "usage:" in combined_out.lower()
        and not re.search(r"trace|json|corrupt|parse|decode", combined_out, re.IGNORECASE)
    ), f"Pipeline exited with CLI usage error instead of trace parse error: {combined_out}"
    assert res.returncode != 0, (
        f"Pipeline unexpectedly exited 0 on corrupt trace: stdout={res.stdout}"
    )
    assert not out_mp4.exists() and not list(out_dir.glob("*.mp4")), (
        f"Pipeline produced a fake MP4 from corrupt trace.jsonl in {out_dir}"
    )
