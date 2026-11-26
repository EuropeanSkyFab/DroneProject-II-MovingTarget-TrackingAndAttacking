"""情景测试证据图：过程图、判定图、汇总柱状图，用例结束自动出 PNG。

这是测试产物，不是手工讲解图。用例只断言数据；图由 fixture 在 teardown
生成，路径打到终端，默认写到 /tmp/algorithm_scenario_charts。

图型：

- 过程图：时间跨度足够长时使用。A 距离、B 俯视图、C 速度、D 级别。
- 判定图：单帧或极短序列使用。x-y 几何、关键点、标注，不画时间轴。
- 柱状图：多用例汇总使用，由 ``configure(mode="bar", bar=...)`` 指定。
- 3D：z 变化超过 ``Z_3D_THRESHOLD`` 时另出同名 ``__3d`` 图。
- 双机图：两机角色固定时，一张图同时画主攻/警戒轨迹和距离。
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from math import hypot
from pathlib import Path
from typing import Optional

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "scenario-chart-mpl")
)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
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

Z_3D_THRESHOLD = 0.5
SHORT_SPAN_SEC = 0.25

LEVEL_ORDER = {
    "NORMAL": 0,
    "EXTRAPOLATING": 1,
    "HOLDING": 2,
    "NO_PLAN": 3,
}
LEVEL_LABELS = ["L0 NORMAL", "L1 EXTRAPOLATING", "L2 HOLDING", "L3 NO_PLAN"]
LEVEL_COLORS = {
    "NORMAL": "#2F5D8A",
    "EXTRAPOLATING": "#E69F00",
    "HOLDING": "#7A7A7A",
    "NO_PLAN": "#B03A3A",
}
LEVEL_NAMES = {
    "NORMAL": "NORMAL",
    "EXTRAPOLATING": "L1",
    "HOLDING": "L2",
    "NO_PLAN": "L3",
}

OWN_COLOR = "#2F5D8A"
TARGET_COLOR = "#D55E00"
PLAN_COLOR = "#C0392B"
GREEN = "#2E7D32"
GRID = dict(color="#BBBBBB", linewidth=0.5, linestyle=(0, (2, 2)), alpha=0.6)


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
    own_truth: Optional[tuple[float, float, float]] = None


def _dist(a, b) -> float:
    return hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _z_span(frames: list[Frame]) -> float:
    values = []
    for frame in frames:
        for point in (_vec(frame), _truth(frame), frame.point):
            if point is not None:
                values.append(point[2])
    return max(values) - min(values) if values else 0.0


def _vec(frame: Frame):
    return frame.own_truth if frame.own_truth is not None else frame.own


def _truth(frame: Frame):
    return frame.truth if frame.truth is not None else frame.target


def _style(ax) -> None:
    ax.grid(axis="y", **GRID)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.5)


def _panel_title(ax, text: str) -> None:
    ax.set_title(text, fontsize=9.5, loc="left")


class ScenarioChart:
    """一条用例的证据图。

    用例按顺序调用 ``add`` 记录帧，按需调用 ``band`` / ``event`` / ``configure``。
    ``save`` 时按时间跨度自动选过程图或判定图；``mode="bar"`` 时画汇总柱状图。
    """

    def __init__(self, name: str, out_dir: Optional[str] = None) -> None:
        self.name = name
        self.out_dir = out_dir
        self._frames: dict[str, list[Frame]] = {}
        self._bands: list[tuple[float, float, str, str]] = []
        self._events: list[tuple[float, str]] = []
        self._headline = ""
        self._subtitle = ""
        self._source = ""
        self._mode = "auto"
        self._annotations: list[dict] = []
        self._decision: dict = {}
        self._bar: dict = {}

    # ------------------------------------------------------------------
    # 记录
    # ------------------------------------------------------------------
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
        own_truth=None,
    ) -> None:
        """记录一帧；own/target/truth/point 是 ENU 三维点。"""
        self._frames.setdefault(series, []).append(
            Frame(
                float(t), own, target, truth, point, level, reason, series,
                own_truth,
            )
        )

    def band(self, t0: float, t1: float, label: str, color: str) -> None:
        """给过程图加阶段底色。"""
        self._bands.append((float(t0), float(t1), label, color))

    def event(self, t: float, label: str) -> None:
        """给过程图加关键事件竖线。"""
        self._events.append((float(t), label))

    def title(self, text: str) -> None:
        """兼容直接调用：没有 headline 时作为主标题。"""
        if not self._headline:
            self._headline = text

    def note(self, text: str) -> None:
        """兼容直接调用：作为副标题/图底说明。"""
        if not self._subtitle:
            self._subtitle = text

    def configure(
        self,
        *,
        headline: Optional[str] = None,
        subtitle: Optional[str] = None,
        source: Optional[str] = None,
        mode: Optional[str] = None,
        annotations: Optional[list[dict]] = None,
        decision: Optional[dict] = None,
        bar: Optional[dict] = None,
    ) -> None:
        """设置报告图的标题、图型、标注或柱状数据。"""
        if headline is not None:
            self._headline = headline
        if subtitle is not None:
            self._subtitle = subtitle
        if source is not None:
            self._source = source
        if mode is not None:
            self._mode = mode
        if annotations is not None:
            self._annotations = annotations
        if decision is not None:
            self._decision = decision
        if bar is not None:
            self._bar = bar

    # ------------------------------------------------------------------
    # 输出
    # ------------------------------------------------------------------
    def save(self) -> list[Path]:
        """按 series 或 bar 数据写图；没有记录则报错。"""
        if not self._frames and not self._bar:
            raise RuntimeError(
                f"{self.name}: 情景测试没有记录任何帧或柱状数据，图表缺失"
            )
        base = Path(self.out_dir) if self.out_dir else Path(
            os.environ.get(
                "SCENARIO_CHART_DIR",
                str(Path(tempfile.gettempdir()) / "algorithm_scenario_charts"),
            )
        )
        base.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []

        if self._bar:
            path = base / f"{self.name}__summary.png"
            self._plot_bar(path)
            paths.append(path)
            return paths

        for series, frames in self._frames.items():
            safe_series = "".join(
                char if char.isalnum() or char in "-_" else "_"
                for char in series
            )
            path = base / f"{self.name}__{safe_series}.png"
            paths.extend(self._plot_series(frames, series, path))
        return paths

    # ------------------------------------------------------------------
    # 选图型
    # ------------------------------------------------------------------
    def _plot_series(
        self, frames: list[Frame], series: str, path: Path
    ) -> list[Path]:
        start = frames[0].t
        span = frames[-1].t - start
        if self._mode == "decision" or (
            self._mode == "auto"
            and (span < SHORT_SPAN_SEC or len(frames) <= 3)
        ):
            self._plot_decision(frames, series, path)
            return [path]
        self._plot_process(frames, series, path)
        paths = [path]
        if _z_span(frames) > Z_3D_THRESHOLD:
            extra = path.with_name(f"{path.stem}__3d.png")
            self._plot_3d(frames, series, extra)
            paths.append(extra)
        return paths

    def _titles(self, series: str) -> tuple[str, str]:
        headline = self._headline or f"{series}｜{self._final_reason(series)}"
        if len(self._frames) > 1 and series not in headline:
            headline = f"{headline}｜{series}"
        return headline, self._subtitle

    def _final_reason(self, series: str) -> str:
        frames = self._frames.get(series, [])
        return frames[-1].reason if frames else ""

    # ------------------------------------------------------------------
    # 过程图
    # ------------------------------------------------------------------
    def _plot_process(self, frames: list[Frame], series: str, path: Path) -> None:
        start = frames[0].t
        times = [frame.t - start for frame in frames]
        own = [_vec(frame) for frame in frames]
        truth = [_truth(frame) for frame in frames]
        points = [frame.point for frame in frames]
        levels = [LEVEL_ORDER.get(frame.level, -1) for frame in frames]

        dist = [
            _dist(o, g) for o, g in zip(own, truth) if o is not None and g is not None
        ]
        speed = _speed_series(own, times)
        has_3d = _z_span(frames) > Z_3D_THRESHOLD

        headline, subtitle = self._titles(series)
        fig = plt.figure(figsize=(9.6, 7.4))
        grid = fig.add_gridspec(
            2, 2, height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.26,
            left=0.08, right=0.97, top=0.82, bottom=0.16,
        )
        ax_dist = fig.add_subplot(grid[0, 0])
        ax_xy = fig.add_subplot(grid[0, 1])
        ax_speed = fig.add_subplot(grid[1, 0])
        ax_level = fig.add_subplot(grid[1, 1])

        for ax in (ax_dist, ax_speed):
            for t0, t1, _label, color in self._bands:
                ax.axvspan(t0 - start, t1 - start, color=color, alpha=0.35,
                           zorder=0)

        if dist:
            dist_top = max(1.0, max(dist) * 1.14)
            ax_dist.plot(times[:len(dist)], dist, color="#333333", linewidth=1.1)
            threshold = 0.5 if dist_top >= 0.5 else dist_top * 0.5
            ax_dist.axhline(threshold, color=LEVEL_COLORS["NO_PLAN"],
                            linewidth=0.7, linestyle=(0, (4, 3)))
            ax_dist.text(times[-1], threshold + 0.15, "命中门限 0.5 m",
                         ha="right", va="bottom", fontsize=8,
                         color=LEVEL_COLORS["NO_PLAN"])
            min_index = min(range(len(dist)), key=dist.__getitem__)
            ax_dist.plot(times[min_index], dist[min_index], "o", color=GREEN,
                         markersize=4.2, zorder=4)
            span = times[-1] if times else 1.0
            if min_index > len(times) * 0.65:
                text_x = span * 0.45
            elif min_index < len(times) * 0.35:
                text_x = span * 0.45
            else:
                text_x = min(times[min_index] + span * 0.08, span * 0.75)
            text_y = min(dist[min_index] + dist_top * 0.25, dist_top * 0.72)
            ax_dist.annotate(
                f"最近 {dist[min_index]:.2f} m @ {times[min_index]:.2f} s",
                xy=(times[min_index], dist[min_index]),
                xytext=(text_x, text_y),
                fontsize=8, color=GREEN,
                arrowprops=dict(arrowstyle="-", color=GREEN, linewidth=0.6),
            )
            ax_dist.set_ylim(0, dist_top)
        _panel_title(ax_dist, "A. 距离：最近点与 0.5 m 门限")
        ax_dist.set_ylabel("己方-目标\n距离（m）", fontsize=9)
        _style(ax_dist)

        own_xy = [(p[0], p[1], p[2]) for p in own if p is not None]
        truth_xy = [(p[0], p[1], p[2]) for p in truth if p is not None]
        plan_xy = [(p[0], p[1], p[2]) for p in points if p is not None]
        ax_xy.plot([p[0] for p in own_xy], [p[1] for p in own_xy],
                   color=OWN_COLOR, linewidth=2.0, label="己方轨迹")
        ax_xy.plot([p[0] for p in truth_xy], [p[1] for p in truth_xy],
                   color=TARGET_COLOR, linewidth=1.1,
                   linestyle=(0, (5, 2)), label="目标真值")
        ax_xy.plot([p[0] for p in plan_xy], [p[1] for p in plan_xy],
                   color=PLAN_COLOR, linestyle="none", marker="o",
                   markersize=1.8, markevery=4, markerfacecolor="white",
                   markeredgewidth=0.5, label="规划发点")
        ax_xy.set_xlabel("东向 x（m）", fontsize=9)
        ax_xy.set_ylabel("北向 y（m）", fontsize=9)
        ax_xy.set_aspect("auto")
        if has_3d:
            _panel_title(ax_xy, "B. 俯视图：z 有变化，3D 见同名 __3d 图")
        else:
            _panel_title(ax_xy, "B. 俯视图：z 不变，x/y 非等比例")
        ys = [p[1] for p in own_xy + truth_xy + plan_xy]
        if ys and max(ys) - min(ys) < 0.5:
            center = (max(ys) + min(ys)) / 2.0
            ax_xy.set_ylim(center - 2.0, center + 2.0)
            ax_xy.text(0.02, 0.04, "y 方向不变", transform=ax_xy.transAxes,
                       fontsize=7.5, color="#666666")
        ax_xy.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=False, fontsize=7.2)
        _style(ax_xy)

        if max(speed) > 0.05:
            ax_speed.plot(times[:len(speed)], speed, color="#6A3D9A",
                          linewidth=1.0)
            peak_index = max(range(len(speed)), key=speed.__getitem__)
            ax_speed.plot(times[peak_index], speed[peak_index], "o",
                          color="#6A3D9A", markersize=4)
            ax_speed.annotate(
                f"峰值 {speed[peak_index]:.2f} m/s",
                xy=(times[peak_index], speed[peak_index]),
                xytext=(max(0.2, times[peak_index] - 2.6),
                        max(0.15, speed[peak_index] - 1.0)),
                fontsize=8, color="#6A3D9A",
                arrowprops=dict(arrowstyle="-", color="#6A3D9A",
                                linewidth=0.6),
            )
            ax_speed.set_ylim(0, max(0.6, max(speed) * 1.2))
            ax_speed.set_ylabel("己方水平\n速度（m/s）", fontsize=9)
            _panel_title(ax_speed, "C. 速度：峰值与变化")
        else:
            deviation = [
                frame.point[0] - vec[0]
                for frame, vec in zip(frames, own)
                if frame.point is not None and vec is not None
            ]
            if deviation:
                ax_speed.plot(times[:len(deviation)], deviation,
                              color="#7A5195", linewidth=1.1)
                ax_speed.axhline(0.0, color="#888888", linewidth=0.5,
                                 linestyle=(0, (2, 1.5)))
                ax_speed.set_ylabel("发点-己方\nx 差（m）", fontsize=9)
            _panel_title(ax_speed, "C. 发点偏差：看领先还是滞后")
        _style(ax_speed)

        ax_level.step(times, levels, where="post", color="#333333",
                      linewidth=0.5)
        for frame, relative_t, level in zip(frames, times, levels):
            ax_level.scatter(relative_t, level, s=7,
                             color=LEVEL_COLORS.get(frame.level, "#333333"),
                             zorder=3, linewidths=0)
        ax_level.set_yticks(range(len(LEVEL_LABELS)))
        ax_level.set_yticklabels(LEVEL_LABELS, fontsize=8)
        ax_level.set_ylim(-0.5, 3.5)
        ax_level.set_xlabel("相对时间（s）", fontsize=9)
        _panel_title(ax_level, "D. 级别：L1/L2/L3 时间线")
        _style(ax_level)
        ax_level.set_xlabel("相对时间（s）", fontsize=9)

        for event_t, _label in self._events:
            relative_event = event_t - start
            for ax in (ax_dist, ax_speed, ax_level):
                ax.axvline(relative_event, color="#555555", linewidth=0.6,
                           linestyle=(0, (3, 2)), zorder=2)
        for ax in (ax_dist, ax_speed, ax_level):
            ax.set_xlim(0, times[-1] if times else 1.0)

        for item in self._annotations:
            target_ax = {
                "A": ax_dist, "B": ax_xy, "C": ax_speed, "D": ax_level,
            }.get(item.get("panel", "A"), ax_dist)
            target_ax.annotate(
                item["text"], xy=(item["x"], item["y"]),
                xytext=item.get("text_xy", (item["x"] + 0.6, item["y"] + 0.6)),
                fontsize=item.get("size", 8), color=item.get("color", "#333333"),
                arrowprops=dict(arrowstyle="-", color=item.get("color", "#333333"),
                                linewidth=0.6)
                if item.get("arrow", True) else None,
            )

        _add_band_legend(fig, self._bands)
        fig.suptitle(headline, fontsize=12.5, x=0.08, ha="left", y=0.965)
        if subtitle:
            fig.text(0.08, 0.895, subtitle, fontsize=8.2, ha="left",
                     color="#333333")
        footer = self._footer(series, frames)
        if self._events:
            footer += "\n事件：" + "；".join(
                f"{event_t - start:.2f} s {label}"
                for event_t, label in self._events
            )
        fig.text(0.08, 0.015, footer, fontsize=7.2, ha="left",
                 color="#555555", linespacing=1.35)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 3D 图
    # ------------------------------------------------------------------
    def _plot_3d(self, frames: list[Frame], series: str, path: Path) -> None:
        start = frames[0].t
        times = [frame.t - start for frame in frames]
        own = [_vec(frame) for frame in frames if _vec(frame) is not None]
        truth = [_truth(frame) for frame in frames if _truth(frame) is not None]
        plan = [frame.point for frame in frames if frame.point is not None]
        headline, subtitle = self._titles(series)

        fig = plt.figure(figsize=(10.6, 5.8))
        grid = fig.add_gridspec(
            1, 2, width_ratios=[1.15, 1.0], wspace=0.12,
            left=0.05, right=0.98, top=0.80, bottom=0.14,
        )
        ax = fig.add_subplot(grid[0, 0], projection="3d")
        ax2 = fig.add_subplot(grid[0, 1])

        ax.plot([p[0] for p in own], [p[1] for p in own],
                [p[2] for p in own], color=OWN_COLOR, linewidth=2.0,
                label="己方")
        ax.plot([p[0] for p in truth], [p[1] for p in truth],
                [p[2] for p in truth], color=TARGET_COLOR, linewidth=1.2,
                linestyle=(0, (5, 2)), label="目标真值")
        ax.plot([p[0] for p in own], [p[1] for p in own],
                [0.0 for _ in own], color=OWN_COLOR, linewidth=0.8, alpha=0.25)
        ax.plot([p[0] for p in truth], [p[1] for p in truth],
                [0.0 for _ in truth], color=TARGET_COLOR, linewidth=0.8,
                alpha=0.25, linestyle=(0, (5, 2)))
        ax.scatter([p[0] for p in plan], [p[1] for p in plan],
                   [p[2] for p in plan], color=PLAN_COLOR, s=5, marker="o",
                   label="规划发点")
        if own:
            ax.scatter([own[0][0]], [own[0][1]], [own[0][2]], s=40,
                       color=OWN_COLOR, edgecolor="white", zorder=5,
                       label="起点")
        if truth:
            ax.scatter([truth[0][0]], [truth[0][1]], [truth[0][2]], s=40,
                       color=TARGET_COLOR, edgecolor="white", zorder=5,
                       label="目标起点")

        values = [p for p in own + truth + plan]
        xs = [p[0] for p in values]
        ys = [p[1] for p in values]
        zs = [p[2] for p in values]
        x_pad = max((max(xs) - min(xs)) * 0.08, 1.0)
        y_pad = max((max(ys) - min(ys)) * 0.25, 2.0)
        z_pad = max((max(zs) - min(zs)) * 0.08, 1.0)
        ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
        ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)
        if max(ys) - min(ys) < 0.5:
            ax.set_yticks([0.0])
        ax.set_zlim(min(zs) - z_pad, max(zs) + z_pad)
        ax.set_box_aspect((
            max(max(xs) - min(xs), 2.0),
            max(max(ys) - min(ys), 4.0),
            max(max(zs) - min(zs), 2.0),
        ))
        ax.set_xlabel("x 东（m）", labelpad=5)
        ax.set_ylabel("y 北（m）", labelpad=7)
        ax.set_zlabel("z 高（m）", labelpad=2)
        ax.view_init(elev=22, azim=-62)
        ax.tick_params(labelsize=7)
        ax.set_title("A. 3D 路径 + 地面投影", fontsize=9.5, loc="left")
        ax.legend(loc="upper left", frameon=False, fontsize=7)

        ax2.plot([p[1] for p in truth], [p[0] for p in truth],
                 color=TARGET_COLOR, linewidth=1.1,
                 linestyle=(0, (5, 2)), label="目标真值")
        ax2.plot([p[1] for p in own], [p[0] for p in own],
                 color=OWN_COLOR, linewidth=2.0, label="己方")
        if own:
            ax2.scatter([own[0][1]], [own[0][0]], s=35, color=OWN_COLOR,
                        edgecolor="white", zorder=5)
        y_span = max(p[1] for p in truth + own) - min(p[1] for p in truth + own)
        if y_span < 0.5:
            ax2.set_xlim(-2.0, 2.0)
            ax2.text(0.02, 0.96, "y 方向不变", transform=ax2.transAxes,
                     fontsize=7.5, color="#666666", va="top")
        ax2.set_xlabel("北向 y（m）", fontsize=9)
        ax2.set_ylabel("东向 x（m）", fontsize=9)
        ax2.set_title("B. x-y 顶视图", fontsize=9.5, loc="left")
        ax2.grid(**GRID)
        ax2.legend(loc="upper left", frameon=False, fontsize=7)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.text(
            0.98, 0.04,
            f"z 跨度 {min(zs):.1f}～{max(zs):.1f} m；"
            f"x/y 非等比例。",
            transform=ax2.transAxes, ha="right", va="bottom",
            fontsize=7.8, color="#555555",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F4F6F8",
                      edgecolor="#D5DCE3", linewidth=0.5),
        )

        fig.suptitle(headline, fontsize=12.5, x=0.05, ha="left", y=0.965)
        if subtitle:
            fig.text(0.05, 0.865, subtitle, fontsize=8.4, ha="left",
                     color="#333333")
        fig.text(0.05, 0.035, self._footer(series, frames), fontsize=7.2,
                 ha="left", color="#555555")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 判定图
    # ------------------------------------------------------------------
    def _plot_decision(self, frames: list[Frame], series: str, path: Path) -> None:
        headline, subtitle = self._titles(series)
        decision = self._decision
        pursuit = decision.get("pursuit")
        if pursuit:
            fig = plt.figure(figsize=(9.8, 6.0))
            grid = fig.add_gridspec(
                1, 2, width_ratios=[1.25, 1.0], wspace=0.25,
                left=0.07, right=0.97, top=0.78, bottom=0.16,
            )
            ax = fig.add_subplot(grid[0, 0])
            ax_pursuit = fig.add_subplot(grid[0, 1])
        else:
            fig, ax = plt.subplots(figsize=(9.4, 5.6))
            fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.16)
            ax_pursuit = None

        own = [_vec(frame) for frame in frames if _vec(frame) is not None]
        truth = [
            _truth(frame) for frame in frames if _truth(frame) is not None
        ]
        plan = [frame.point for frame in frames if frame.point is not None]
        if own:
            ax.plot([p[0] for p in own], [p[1] for p in own],
                    color=OWN_COLOR, linewidth=2.0, label="己方轨迹")
            ax.plot(own[0][0], own[0][1], "s", color=OWN_COLOR, markersize=6)
        if truth:
            ax.plot([p[0] for p in truth], [p[1] for p in truth],
                    color=TARGET_COLOR, linewidth=1.1,
                    linestyle=(0, (5, 2)), label="目标真值")
            ax.plot(truth[0][0], truth[0][1], "o", color=TARGET_COLOR,
                    markersize=6)
        if plan:
            ax.plot([p[0] for p in plan], [p[1] for p in plan],
                    color=PLAN_COLOR, linestyle="none", marker="o",
                    markersize=5, markerfacecolor="white", markeredgewidth=0.8,
                    label="规划发点")

        extras = decision.get("extra_points", [])
        if own and not extras:
            ax.annotate(f"己方 ({own[0][0]:.1f},{own[0][1]:.1f})",
                        xy=own[0][:2], xytext=(own[0][0] + 0.4, own[0][1] + 1.2),
                        fontsize=8.5, color=OWN_COLOR)
        if truth and not extras:
            ax.annotate(f"目标 ({truth[0][0]:.1f},{truth[0][1]:.1f})",
                        xy=truth[0][:2],
                        xytext=(truth[0][0] + 0.4, truth[0][1] + 2.2),
                        fontsize=8.5, color=TARGET_COLOR)
        if plan:
            target_point = plan[-1]
            ax.annotate(
                f"发点 ({target_point[0]:.2f},{target_point[1]:.2f})",
                xy=target_point[:2],
                xytext=(target_point[0] + 0.3, target_point[1] - 1.2),
                fontsize=8.5, color=PLAN_COLOR,
            )

        final = frames[-1]
        ax.text(
            0.02, 0.96,
            f"最终：{final.level} / {final.reason}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F4F6F8",
                      edgecolor="#D5DCE3", linewidth=0.5),
        )

        for point in extras:
            x, y, label, color = point[:4]
            marker = point[4] if len(point) > 4 else "o"
            text_xy = point[5] if len(point) > 5 else (x + 0.8, y + 0.8)
            ax.plot(x, y, marker=marker, color=color, markersize=10)
            ax.annotate(label, xy=(x, y), xytext=text_xy,
                        fontsize=8.5, color=color)
        for arrow in decision.get("arrows", []):
            x0, y0, x1, y1, label, color = arrow
            ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        linewidth=1.2))
            if label:
                ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.7, label,
                        fontsize=8, color=color, ha="center")
        for text in decision.get("texts", []):
            x, y, value = text[:3]
            ax.text(x, y, value, fontsize=8.5, color="#333333")

        ax.set_xlabel("东向 x（m）", fontsize=9.5)
        ax.set_ylabel("北向 y（m）", fontsize=9.5)
        ax.set_aspect("auto")
        ax.grid(**GRID)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="best", frameon=False, fontsize=8)
        ax.set_title("A. 几何：关键点与轨迹（z 不变时省略高度）",
                     fontsize=9.5, loc="left")

        if decision.get("note"):
            ax.text(
                0.98, 0.04, decision["note"], transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, color="#333333",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8E8",
                          edgecolor="#D0B070", linewidth=0.6),
            )
        if "xlim" in decision:
            ax.set_xlim(*decision["xlim"])
        else:
            xs = [p[0] for p in own + truth + plan]
            if xs:
                pad = max((max(xs) - min(xs)) * 0.18, 2.0)
                ax.set_xlim(min(xs) - pad, max(xs) + pad)
        if "ylim" in decision:
            ax.set_ylim(*decision["ylim"])
        else:
            ys = [p[1] for p in own + truth + plan]
            if ys:
                pad = max((max(ys) - min(ys)) * 0.18, 2.0)
                ax.set_ylim(min(ys) - pad, max(ys) + pad)

        if ax_pursuit is not None:
            duration = float(pursuit.get("duration", 8.0))
            target_x0 = float(pursuit["target_x0"])
            target_speed = float(pursuit["target_speed"])
            own_speed = float(pursuit["own_speed"])
            time = np.linspace(0.0, duration, 64)
            target_reach = target_x0 + target_speed * time
            own_reach = own_speed * time
            ax_pursuit.plot(time, target_reach, color=TARGET_COLOR,
                            linewidth=1.6, label="目标")
            ax_pursuit.plot(time, own_reach, color=OWN_COLOR,
                            linewidth=1.8, label="己方最远（直追）")
            ax_pursuit.fill_between(time, own_reach, target_reach,
                                    color="#F2D7C8", alpha=0.55)
            ax_pursuit.annotate(
                pursuit.get(
                    "note",
                    f"{duration:.0f} s 时目标 {target_reach[-1]:.0f} m，"
                    f"己方最多 {own_reach[-1]:.0f} m",
                ),
                xy=(time[-1], target_reach[-1]),
                xytext=(duration * 0.35, target_reach[-1] * 0.75),
                fontsize=8.5, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#555555",
                                linewidth=0.6),
            )
            ax_pursuit.text(
                duration * 0.25, own_reach[-1] * 0.35,
                "两线不相交\n→ 拦截方程无正根",
                fontsize=9, color=LEVEL_COLORS["NO_PLAN"],
            )
            ax_pursuit.set_xlim(0.0, duration)
            ax_pursuit.set_ylim(0.0, target_reach[-1] * 1.1)
            ax_pursuit.set_xlabel("时间（s）", fontsize=9.5)
            ax_pursuit.set_ylabel("东向距离 x（m）", fontsize=9.5)
            ax_pursuit.set_title("B. 追击能力：目标一直更快",
                                 fontsize=9.5, loc="left")
            ax_pursuit.legend(loc="upper left", frameon=False, fontsize=8)
            ax_pursuit.grid(**GRID)
            ax_pursuit.spines["top"].set_visible(False)
            ax_pursuit.spines["right"].set_visible(False)

        fig.suptitle(headline, fontsize=12.5, x=0.07, ha="left", y=0.965)
        if subtitle:
            fig.text(0.07, 0.865 if ax_pursuit is not None else 0.875,
                     subtitle, fontsize=8.4, ha="left", color="#333333")
        fig.text(0.07, 0.035, self._footer(series, frames), fontsize=7.2,
                 ha="left", color="#555555")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 汇总柱状图
    # ------------------------------------------------------------------
    def _plot_bar(self, path: Path) -> None:
        bar = self._bar
        categories = bar["categories"]
        series = bar["series"]
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        fig.subplots_adjust(left=0.10, right=0.97, top=0.76, bottom=0.20)
        x = np.arange(len(categories))
        bottom = np.zeros(len(categories))
        for item in series:
            values = np.array(item["values"])
            bars = ax.bar(x, values, bottom=bottom, width=0.52,
                          color=item["color"], edgecolor="white",
                          linewidth=0.8, label=item["label"])
            for patch, value in zip(bars, values):
                if value:
                    ax.text(patch.get_x() + patch.get_width() / 2,
                            patch.get_y() + value / 2, f"{value}",
                            ha="center", va="center", fontsize=8,
                            color="white" if item["color"] != "#E69F00" else "#333333")
            bottom += values
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylabel(bar.get("ylabel", "帧数"), fontsize=9.5)
        ax.set_ylim(0, max(bottom) * 1.25 if len(bottom) else 1.0)
        _style(ax)
        ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)
        headline = self._headline or self.name
        fig.suptitle(headline, fontsize=12, x=0.10, ha="left", y=0.965)
        if self._subtitle:
            fig.text(0.10, 0.865, self._subtitle, fontsize=8.2, ha="left",
                     color="#333333")
        fig.text(0.10, 0.04, self._footer("summary", []), fontsize=7.2,
                 ha="left", color="#555555")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    def _footer(self, series: str, frames: list[Frame]) -> str:
        parts = []
        if self._source:
            parts.append(self._source)
        parts.append(f"图：{self.name}｜{series}")
        parts.append("图只作证据，不参与断言")
        return "；".join(parts)


def _speed_series(own: list, times: list[float]) -> list[float]:
    values: list[float] = []
    for index, point in enumerate(own):
        if point is None:
            values.append(0.0)
            continue
        previous = own[index - 1] if index > 0 else None
        dt = times[index] - times[index - 1] if index > 0 else 0.0
        if previous is None or dt <= 0.0:
            values.append(0.0)
        else:
            values.append(_dist(point, previous) / dt)
    return values


def _add_band_legend(fig, bands) -> None:
    seen = set()
    handles = []
    labels = []
    for _t0, _t1, label, color in bands:
        if label in seen:
            continue
        seen.add(label)
        handles.append(plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.65))
        labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.08, 0.055),
                   frameon=False, fontsize=7.5, ncol=3)


class DualChart:
    """双机证据图：主攻/警戒两条轨迹、目标距离和角色时间线。

    只记录两架飞机，不画三机；三机联调由中枢负责。
    """

    def __init__(self, name: str, out_dir: Optional[str] = None) -> None:
        self.name = name
        self.out_dir = out_dir
        self._frames: list[dict] = []
        self._headline = ""
        self._subtitle = ""

    def configure(
        self,
        *,
        headline: Optional[str] = None,
        subtitle: Optional[str] = None,
    ) -> None:
        if headline is not None:
            self._headline = headline
        if subtitle is not None:
            self._subtitle = subtitle

    def add(
        self,
        t: float,
        *,
        primary,
        secondary,
        target,
        primary_point=None,
        secondary_point=None,
        primary_level: str = "NORMAL",
        secondary_level: str = "NORMAL",
    ) -> None:
        self._frames.append({
            "t": float(t),
            "primary": primary,
            "secondary": secondary,
            "target": target,
            "primary_point": primary_point,
            "secondary_point": secondary_point,
            "primary_level": primary_level,
            "secondary_level": secondary_level,
        })

    def save(self) -> list[Path]:
        if not self._frames:
            raise RuntimeError(f"{self.name}: 双机测试没有记录任何帧，图表缺失")
        base = Path(self.out_dir) if self.out_dir else Path(
            os.environ.get(
                "SCENARIO_CHART_DIR",
                str(Path(tempfile.gettempdir()) / "algorithm_scenario_charts"),
            )
        )
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{self.name}__dual.png"
        self._plot(path)
        return [path]

    def _plot(self, path: Path) -> None:
        frames = self._frames
        times = [f["t"] - frames[0]["t"] for f in frames]
        primary = [f["primary"] for f in frames]
        secondary = [f["secondary"] for f in frames]
        target = frames[-1]["target"]
        z_values = [
            point[2]
            for values in (primary, secondary)
            for point in values
            if point is not None
        ]
        if target is not None:
            z_values.append(target[2])
        has_3d = max(z_values) - min(z_values) > Z_3D_THRESHOLD

        if has_3d:
            fig = plt.figure(figsize=(13.4, 5.4))
            grid = fig.add_gridspec(
                1, 3, width_ratios=[1.2, 1.0, 1.0], wspace=0.28,
                left=0.05, right=0.98, top=0.80, bottom=0.15,
            )
            ax3d = fig.add_subplot(grid[0, 0], projection="3d")
            ax_top = fig.add_subplot(grid[0, 1])
            ax_dist = fig.add_subplot(grid[0, 2])
        else:
            fig = plt.figure(figsize=(10.6, 5.2))
            grid = fig.add_gridspec(
                1, 2, width_ratios=[1.1, 1.0], wspace=0.28,
                left=0.06, right=0.98, top=0.80, bottom=0.15,
            )
            ax3d = None
            ax_top = fig.add_subplot(grid[0, 0])
            ax_dist = fig.add_subplot(grid[0, 1])

        colors = {"primary": "#2F5D8A", "secondary": "#D2691E"}
        labels = {"primary": "主攻", "secondary": "警戒"}

        if ax3d is not None:
            for key, points, color in (
                ("primary", primary, colors["primary"]),
                ("secondary", secondary, colors["secondary"]),
            ):
                valid = [p for p in points if p is not None]
                if not valid:
                    continue
                ax3d.plot(
                    [p[0] for p in valid], [p[1] for p in valid],
                    [p[2] for p in valid], color=color, linewidth=2.0,
                    label=labels[key],
                )
                ax3d.plot(
                    [p[0] for p in valid], [p[1] for p in valid],
                    [0.0 for _ in valid], color=color, linewidth=0.7,
                    alpha=0.2,
                )
                ax3d.scatter(
                    [valid[0][0]], [valid[0][1]], [valid[0][2]],
                    color=color, s=25, edgecolor="white",
                )
                ax3d.scatter(
                    [valid[-1][0]], [valid[-1][1]], [valid[-1][2]],
                    color=color, s=55, marker="^", edgecolor="white",
                )
            if target is not None:
                ax3d.scatter(
                    [target[0]], [target[1]], [target[2]],
                    color="#D62728", marker="*", s=200, label="目标",
                )
            y_values = [
                point[1]
                for points in (primary, secondary)
                for point in points
                if point is not None
            ]
            if y_values and max(y_values) - min(y_values) < 0.5:
                ax3d.set_ylim(-2.0, 2.0)
                ax3d.set_yticks([0.0])
            ax3d.set_xlabel("东向 x（m）", labelpad=4)
            ax3d.set_ylabel("北向 y（m）", labelpad=6)
            ax3d.set_zlabel("高度 z（m）", labelpad=2)
            ax3d.view_init(elev=22, azim=-62)
            ax3d.tick_params(labelsize=7)
            _panel_title(ax3d, "A. 双机 3D 路径")
            ax3d.legend(loc="upper left", frameon=False, fontsize=7)

        for key, points, color in (
            ("primary", primary, colors["primary"]),
            ("secondary", secondary, colors["secondary"]),
        ):
            valid = [p for p in points if p is not None]
            ax_top.plot([p[0] for p in valid], [p[1] for p in valid],
                        color=color, linewidth=2.0, label=labels[key])
            ax_top.scatter([valid[0][0]], [valid[0][1]], color=color, s=20)
            ax_top.scatter([valid[-1][0]], [valid[-1][1]], color=color,
                           s=45, marker="^", zorder=5)
            if target is not None:
                ax_top.plot(
                    [valid[-1][0], target[0]], [valid[-1][1], target[1]],
                    color=color, linestyle=":", linewidth=1.0,
                )
                end_dist = _dist(valid[-1], target)
                ax_top.annotate(
                    f"{labels[key]} {end_dist:.1f} m",
                    xy=(valid[-1][0], valid[-1][1]),
                    xytext=(valid[-1][0] + 0.5, valid[-1][1] + 0.5),
                    fontsize=7.2, color=color,
                )
        if target is not None:
            ax_top.scatter([target[0]], [target[1]], color="#D62728",
                          marker="*", s=170, label="目标")
            ax_top.add_patch(plt.Circle(
                (target[0], target[1]), 5.0, fill=False,
                linestyle="--", linewidth=0.8, color="#888888",
            ))
        ax_top.set_xlabel("东向 x（m）", fontsize=9)
        ax_top.set_ylabel("北向 y（m）", fontsize=9)
        ax_top.set_aspect("equal", adjustable="box")
        ax_top.grid(True, alpha=0.3)
        ax_top.legend(loc="upper left", fontsize=7, frameon=False)
        _panel_title(ax_top, "B. x-y 顶视图（虚线为 5 m 环绕圈）")

        for key, points, color in (
            ("primary", primary, colors["primary"]),
            ("secondary", secondary, colors["secondary"]),
        ):
            if target is None:
                continue
            dist = [_dist(p, target) for p in points if p is not None]
            ax_dist.plot(times[:len(dist)], dist, color=color, linewidth=1.6,
                         label=labels[key])
            if dist:
                ax_dist.annotate(
                    f"{dist[-1]:.2f} m",
                    xy=(times[len(dist) - 1], dist[-1]),
                    xytext=(max(0.1, times[len(dist) - 1] - 1.2),
                            max(0.2, dist[-1] + 1.0)),
                    fontsize=7.5, color=color,
                )
        ax_dist.axhline(0.5, color="#B03A3A", linewidth=0.7,
                        linestyle=(0, (4, 3)), label="命中门限 0.5 m")
        ax_dist.set_xlabel("相对时间（s）", fontsize=9)
        ax_dist.set_ylabel("到目标距离（m）", fontsize=9)
        ax_dist.set_ylim(bottom=0)
        ax_dist.grid(axis="y", **GRID)
        ax_dist.spines["top"].set_visible(False)
        ax_dist.spines["right"].set_visible(False)
        ax_dist.legend(loc="upper right", fontsize=7, frameon=False)
        _panel_title(ax_dist, "C. 到目标距离")

        fig.suptitle(self._headline or f"{self.name}｜双机", fontsize=12.5,
                     x=0.05, ha="left", y=0.965)
        if self._subtitle:
            fig.text(0.05, 0.865, self._subtitle, fontsize=8.2, ha="left",
                     color="#333333")
        fig.text(0.05, 0.035, f"图：{self.name}｜双机主攻/警戒；图只作证据。",
                 fontsize=7.2, ha="left", color="#555555")
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
