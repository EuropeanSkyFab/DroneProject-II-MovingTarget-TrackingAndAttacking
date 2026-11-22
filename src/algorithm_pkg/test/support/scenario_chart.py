"""情景测试的证据图：逐帧记录 + 用例结束自动出 PNG。

这是测试产物，不是手工讲解图。用例只断言数据；图由 fixture 在 teardown
生成，路径打到终端，默认写到 /tmp/algorithm_scenario_charts。
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scenario-chart-mpl")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

SC_FONT = Path(tempfile.gettempdir()) / "scenario-chart-font" / "NotoSansCJKsc-Regular.otf"
if not SC_FONT.exists():
    try:
        from fontTools.ttLib import TTCollection

        SC_FONT.parent.mkdir(parents=True, exist_ok=True)
        collection = TTCollection(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
        )
        collection.fonts[2].save(str(SC_FONT))
    except Exception:
        SC_FONT = None
if SC_FONT is not None and SC_FONT.exists():
    font_manager.fontManager.addfont(str(SC_FONT))
    family = ["Noto Sans CJK SC", "DejaVu Sans"]
else:
    family = ["Droid Sans Fallback", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = family
plt.rcParams["axes.unicode_minus"] = False

LEVEL_ORDER = {
    "NORMAL": 0,
    "EXTRAPOLATING": 1,
    "HOLDING": 2,
    "NO_PLAN": 3,
}
LEVEL_LABELS = ["L0 NORMAL", "L1 EXTRAPOLATING", "L2 HOLDING", "L3 NO_PLAN"]
LEVEL_COLORS = {
    "NORMAL": "#2F5D8A",
    "EXTRAPOLATING": "#A65A2A",
    "HOLDING": "#7A7A7A",
    "NO_PLAN": "#B03A3A",
}

# 三条线：颜色 + 线型 + 标记三重区分。
STYLE_OWN = dict(color="#333333", linestyle="-", linewidth=0.7)
STYLE_TARGET = dict(color="#0072B2", linestyle="--", linewidth=0.7)
STYLE_POINT = dict(
    color="#D55E00", linestyle=":", linewidth=0.7, marker="o",
    markersize=2.4, markerfacecolor="white", markeredgewidth=0.6,
)


@dataclass
class Frame:
    t: float
    own: Optional[tuple[float, float, float]]
    target: Optional[tuple[float, float, float]]
    truth: Optional[tuple[float, float, float]]
    point: Optional[tuple[float, float, float]]
    level: str
    reason: str
    series: str


@dataclass
class ScenarioChart:
    name: str
    out_dir: Optional[str] = None
    _frames: dict[str, list[Frame]] = field(default_factory=dict)
    _bands: list[tuple[float, float, str, str]] = field(default_factory=list)
    _events: list[tuple[float, str]] = field(default_factory=list)
    _title: str = ""
    _note: str = ""

    def add(
        self,
        t: float,
        *,
        own=None,
        target=None,
        truth=None,
        point=None,
        level: str = "NORMAL",
        reason: str = "",
        series: str = "default",
    ) -> None:
        """记录一帧；own/target/truth/point 是 ENU 三维点。"""
        self._frames.setdefault(series, []).append(
            Frame(float(t), own, target, truth, point, level, reason, series)
        )

    def band(self, t0: float, t1: float, label: str, color: str) -> None:
        """给复杂场景加阶段底色（目标丢失、己方断流等）。"""
        self._bands.append((float(t0), float(t1), label, color))

    def event(self, t: float, label: str) -> None:
        """给复杂场景加关键事件竖线。"""
        self._events.append((float(t), label))

    def title(self, text: str) -> None:
        """图标题；不写时用 series + 最终 reason。"""
        self._title = text

    def note(self, text: str) -> None:
        """图底注：参数、拐点原因、数据来源。"""
        self._note = text

    def save(self) -> list[Path]:
        """为每个 series 写一张图；没有记录则报错。"""
        if not self._frames:
            raise RuntimeError(
                f"{self.name}: 情景测试没有记录任何帧，图表缺失"
            )
        base = Path(self.out_dir) if self.out_dir else Path(
            os.environ.get(
                "SCENARIO_CHART_DIR",
                str(Path(tempfile.gettempdir()) / "algorithm_scenario_charts"),
            )
        )
        base.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for series, frames in self._frames.items():
            safe_series = "".join(
                char if char.isalnum() or char in "-_" else "_"
                for char in series
            )
            path = base / f"{self.name}__{safe_series}.png"
            self._plot(frames, series, path, self._bands, self._events,
                       self._title, self._note)
            paths.append(path)
        return paths

    @staticmethod
    def _plot(
        frames: list[Frame],
        series: str,
        path: Path,
        bands: list[tuple[float, float, str, str]],
        events: list[tuple[float, str]],
        title_text: str,
        note_text: str,
    ) -> None:
        has_truth = any(frame.truth is not None for frame in frames)
        show_pub = has_truth
        rows = 5 if show_pub else 2
        fig, axes = plt.subplots(
            rows, 1, figsize=(7.4, 7.0 if show_pub else 3.8),
            sharex=True,
            gridspec_kw={
                "height_ratios": [2.0, 0.8, 0.7, 1.0, 0.4][:rows],
                "hspace": 0.12,
            },
        )
        ax_pos = axes[0]
        ax_dist = axes[1] if show_pub else None
        ax_err = axes[2] if show_pub else None
        ax_level = axes[3] if show_pub else axes[-1]
        ax_pub = axes[4] if show_pub else None
        times = [frame.t for frame in frames]

        target_key = "truth" if has_truth else "target"
        target_label = "目标真值" if has_truth else "目标"
        for key, label, style in (
            ("own", "己方", STYLE_OWN),
            (target_key, target_label, STYLE_TARGET),
            ("point", "发点", STYLE_POINT),
        ):
            xs = [getattr(frame, key)[0] if getattr(frame, key) else None
                  for frame in frames]
            valid = [(t, x) for t, x in zip(times, xs) if x is not None]
            if valid:
                ax_pos.plot(
                    [t for t, _ in valid], [x for _, x in valid],
                    label=label, **style,
                )

        if ax_dist is not None:
            dist = []
            for frame in frames:
                if frame.own is not None and frame.truth is not None:
                    dx = frame.own[0] - frame.truth[0]
                    dy = frame.own[1] - frame.truth[1]
                    dz = frame.own[2] - frame.truth[2]
                    dist.append((frame.t, (dx * dx + dy * dy + dz * dz) ** 0.5))
            if dist:
                ax_dist.plot([t for t, _ in dist], [d for _, d in dist],
                             color="#555555", linewidth=0.7)
            ax_dist.set_ylabel("己方-目标\n真值距离（m）")
            ax_dist.grid(axis="y", linestyle="--", linewidth=0.35, alpha=0.35)

        if ax_err is not None:
            errors = []
            for frame in frames:
                if frame.own is not None and frame.point is not None:
                    errors.append((frame.t, frame.point[0] - frame.own[0]))
            if errors:
                ax_err.plot([t for t, _ in errors], [e for _, e in errors],
                            color="#7A5195", linewidth=0.7)
            ax_err.axhline(0.0, color="#888888", linewidth=0.5,
                           linestyle=(0, (2.0, 1.5)))
            ax_err.set_ylabel("发点-己方\nx 差（m）")
            ax_err.grid(axis="y", linestyle="--", linewidth=0.35, alpha=0.35)

        # 阶段底色：目标丢失、己方断流等。
        for t0, t1, label, color in bands:
            for axis in axes:
                axis.axvspan(t0, t1, color=color, alpha=0.32, zorder=0,
                             label=label if axis is ax_pos else None)
            letter = label.split()[0]
            if letter in ("A", "B"):
                ax_pos.text(
                    (t0 + t1) / 2.0, 1.02, f"[{letter}]",
                    transform=ax_pos.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8,
                    fontweight="bold", color="#333333",
                )
        # 关键事件竖线；文字统一放图底。
        for t, _label in events:
            for axis in axes:
                axis.axvline(t, color="#666666", linewidth=0.6,
                             linestyle=(0, (3.0, 1.8)), zorder=2)

        levels = [LEVEL_ORDER.get(frame.level, -1) for frame in frames]
        ax_level.step(times, levels, where="post", color="#333333", linewidth=0.8)
        for frame, level in zip(frames, levels):
            ax_level.scatter(
                [frame.t], [level], s=16,
                color=LEVEL_COLORS.get(frame.level, "#333333"), zorder=3,
            )
        ax_level.set_yticks(range(len(LEVEL_LABELS)))
        ax_level.set_yticklabels(LEVEL_LABELS, fontsize=8)
        ax_level.set_ylim(-0.4, 3.4)
        ax_level.grid(axis="y", linestyle="--", linewidth=0.35, alpha=0.35)

        if ax_pub is not None:
            published = [1 if frame.point is not None else 0 for frame in frames]
            ax_pub.step(times, published, where="post",
                        color="#333333", linewidth=0.4)
            for frame, value in zip(frames, published):
                if value == 0:
                    ax_pub.scatter([frame.t], [value], s=10,
                                   color="#B03A3A", zorder=3)
            ax_pub.set_yticks([0, 1])
            ax_pub.set_yticklabels(["不发点", "发点"], fontsize=8)
            ax_pub.set_ylim(-0.2, 1.2)
            ax_pub.set_xlabel("时间（s）")
            ax_pub.grid(axis="y", linestyle="--", linewidth=0.35, alpha=0.35)
        else:
            ax_level.set_xlabel("时间（s）")

        ax_pos.set_ylabel("ENU x（m）")
        ax_pos.grid(axis="y", linestyle="--", linewidth=0.35, alpha=0.35)
        ax_pos.legend(loc="best", frameon=False, fontsize=7)
        for axis in axes:
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)

        final_reason = frames[-1].reason
        title = title_text or f"{series}｜{final_reason}"
        fig.suptitle(title, fontsize=9.5, x=0.01, ha="left")
        bottom = 0.17 if (note_text or events) else 0.12
        fig.subplots_adjust(left=0.14, right=0.98, top=0.85,
                            bottom=bottom, hspace=0.14)
        note_lines = []
        if note_text:
            note_lines.append(note_text)
        if events:
            note_lines.append(
                "事件：" + "；".join(f"{t:.2f} s {label}" for t, label in events)
            )
        if note_lines:
            fig.text(0.14, 0.012, "\n".join(note_lines), fontsize=6.8,
                     ha="left", va="bottom", color="#333333",
                     linespacing=1.35)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
