#!/usr/bin/env python3
"""把 Gazebo 探针 CSV 画成图：固定点、速度/跳变、延迟对比。

图是证据，不参与断言。输出 PNG/SVG/PDF 到 --out。
用法（宿主机）：
    PYTHONPATH=src/algorithm_pkg:src/algorithm_pkg/test \
      python3 src/algorithm_pkg/scripts/p7/plot_gazebo.py
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "algorithm_pkg" / "test"))

from support.scenario_chart import plt  # noqa: E402  (设置中文字体)


def _rows(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _steady_jump(run: Path, prefix: str) -> float:
    plans = _rows(run / f"{prefix}_plan.csv")
    jumps = []
    for a, b in zip(plans, plans[1:]):
        if float(a["t_mode"]) > 1.0 and a["action"] == b["action"]:
            jumps.append(
                math.hypot(
                    float(b["x"]) - float(a["x"]),
                    float(b["y"]) - float(a["y"]),
                )
            )
    return max(jumps) if jumps else 0.0


def _speed(run: Path, prefix: str) -> tuple[float, float]:
    states = [
        r for r in _rows(run / f"{prefix}_state.csv")
        if float(r["t_mode"]) > 4.0
    ]
    if not states:
        return 0.0, 0.0
    speeds = [float(r["speed"]) for r in states]
    return max(speeds), sum(speeds) / len(speeds)


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(out / f"{name}.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _anchor(run: Path, prefix: str, title: str, out: Path, name: str) -> None:
    plans = _rows(run / f"{prefix}_plan.csv")
    states = _rows(run / f"{prefix}_state.csv")
    if not plans or not states:
        return
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
    for ax, key, ylabel in (
        (axes[0], "x", "x (m)"),
        (axes[1], "y", "y (m)"),
    ):
        ax.plot(
            [float(r["t_mode"]) for r in states],
            [float(r[key]) for r in states],
            color="#333333", linewidth=0.8, label="己方",
        )
        ax.plot(
            [float(r["t_mode"]) for r in plans],
            [float(r[key]) for r in plans],
            color="#D55E00", linestyle=":", linewidth=0.9,
            marker="o", markersize=2, markerfacecolor="white",
            label="规划发点",
        )
        ax.set_ylabel(ylabel)
        ax.grid(True, linewidth=0.3, alpha=0.5)
    axes[0].set_title(title)
    axes[1].set_xlabel("模式后时间 (s)")
    axes[0].legend(loc="best", fontsize=8)
    fig.text(
        0.01, 0.01,
        "数据：.p5/evidence/runs/p7；图由 scripts/p7/plot_gazebo.py 生成。",
        fontsize=7,
    )
    _save(fig, out, name)


def _bar_compare(items, title, ylabel, out, name) -> None:
    labels = [item[0] for item in items]
    values = [item[1] for item in items]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(labels, values, color="#4C72B0", width=0.55)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value,
            f"{value:.2f}", ha="center", va="bottom", fontsize=8,
        )
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", linewidth=0.3, alpha=0.5)
    fig.text(
        0.01, 0.01,
        "数据：.p5/evidence/runs/p7；单次 Gazebo 运行；图由 scripts/p7/plot_gazebo.py 生成。",
        fontsize=7,
    )
    _save(fig, out, name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Gazebo 探针出图")
    parser.add_argument("--dir", default=str(ROOT / ".p5" / "p7_run"))
    parser.add_argument(
        "--out",
        default=str(Path(tempfile.gettempdir()) / "p7_gazebo_figures"),
    )
    args = parser.parse_args()
    run_dir = Path(args.dir)
    out = Path(args.out)

    _anchor(
        run_dir, "hover_track", "HOVER 固定点：发点固定，己方继续移动",
        out, "hover_anchor",
    )
    _anchor(
        run_dir, "l2_track", "L2 固定点：发点固定，中枢视觉优先使己方继续前冲",
        out, "l2_anchor",
    )

    speed_items = [
        ("TRACK 0.74", _speed(run_dir / "track_0_74", "speed_track")[0]),
        ("TRACK 0.14", _speed(run_dir / "new_track_014", "speed_track")[0]),
        ("ATTACK 0.62\n12.4 m/s", _speed(run_dir / "new_attack_124_062", "speed_attack")[0]),
        ("ATTACK 0.74\n14.8 m/s", _speed(run_dir / "attack_14_8", "speed_attack")[0]),
    ]
    _bar_compare(speed_items, "实际最大速度", "m/s", out, "speed_compare")

    jump_items = [
        ("TRACK 0.74", _steady_jump(run_dir / "track_0_74", "speed_track")),
        ("TRACK 0.14", _steady_jump(run_dir / "new_track_014", "speed_track")),
        ("ATTACK 0.62", _steady_jump(run_dir / "new_attack_124_062", "speed_attack")),
        ("ATTACK 0.74", _steady_jump(run_dir / "attack_14_8", "speed_attack")),
    ]
    _bar_compare(jump_items, "规划点稳态帧间最大跳变", "m", out, "jump_compare")

    delay_items = [
        ("0.4 s", _speed(run_dir / "delay_0_4", "delay_intercept")[0]),
        ("0.5 s", _speed(run_dir / "new_delay_05", "delay_intercept")[0]),
        ("1.1 s", _speed(run_dir / "delay_1_1", "delay_intercept")[0]),
    ]
    _bar_compare(delay_items, "目标延迟下的实际最大速度", "m/s", out, "delay_speed")

    print(f"图输出: {out}")


if __name__ == "__main__":
    main()
