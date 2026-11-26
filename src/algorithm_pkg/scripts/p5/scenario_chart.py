#!/usr/bin/env python3
"""场景结果图：把 CSV 画成两张图（轨迹 + 距离），不依赖 ROS。

每场景一张图，左边“飞机和目标怎么走”，右边“离目标多远”；
另有一张四场景汇总图。用法：

```bash
MPLCONFIGDIR=/tmp/mplconfig PYTHONPATH=src/algorithm_pkg python3 \
    src/algorithm_pkg/scripts/p5/scenario_chart.py
```

图是产物，和 build 同类，不入库：默认输出 `.p5/evidence/scenarios/figures/`。

配色沿用第 9.4 节约定：绿=命中/PASS，红=失败/超限，橙=ATTACK/风险，
蓝=正常，灰=辅助。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager

from algorithm_pkg.scenario_defs import (
    SCENARIOS,
    Scenario,
    evaluate,
    gate_mask,
    get_scenario,
    verdict,
)

FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
)

BLUE = "#1f6feb"
ORANGE = "#f0883e"
GREEN = "#2ea043"
RED = "#d73a4a"
GRAY = "#8b949e"
SHADE = "#ffd8a8"

THRESHOLD = 0.5


def _setup_fonts() -> None:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Noto Sans CJK SC",
                "Noto Sans CJK JP",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.facecolor": "white",
        }
    )


def _read(path: str):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _load(directory: str, scenario_id: str):
    state_path = os.path.join(directory, f"{scenario_id}_state.csv")
    plan_path = os.path.join(directory, f"{scenario_id}_plan.csv")
    if not os.path.exists(state_path):
        return None, None
    state = [
        (
            float(row["t"]),
            float(row["drone_x"]),
            float(row["drone_y"]),
            float(row["drone_z"]),
            float(row["speed"]),
        )
        for row in _read(state_path)
    ]
    plan = []
    if os.path.exists(plan_path):
        plan = [
            (
                float(row["t"]),
                float(row["x"]),
                float(row["y"]),
                float(row["z"]),
                row["action"],
            )
            for row in _read(plan_path)
        ]
    return state, plan


def _first_time(plan, action: str) -> Optional[float]:
    for t, _x, _y, _z, name in plan:
        if name == action:
            return t
    return None


def _hover_time(plan) -> Optional[float]:
    """场景结束时刻：第一条 HOVER 计划点。"""
    for t, _x, _y, _z, action in plan:
        if action == "HOVER":
            return t
    return None


def _target_path(scenario: Scenario, times: List[float]):
    return [scenario.target_at(t) for t in times]


def _plot_scenario(
    scenario: Scenario,
    state,
    plan,
    metrics,
    passed: bool,
    out_path: str,
) -> None:
    mode_time = _first_time(plan, "INTERCEPT") or _first_time(plan, "ATTACK") or 0.0
    attack_time = _first_time(plan, "ATTACK")
    v_max = 14.8 if scenario.mode == "ATTACK" else 7.6

    mask = gate_mask(scenario, state, v_max=v_max)
    # ! 判定只对 INTERCEPT 生效；切了 ATTACK 之后的帧不再算触发。
    if attack_time is not None:
        mask = [hit and row[0] < attack_time for hit, row in zip(mask, state)]

    times = [row[0] for row in state]
    drone_x = [row[1] for row in state]
    drone_y = [row[2] for row in state]
    dists = []
    for row in state:
        tx, ty, tz = scenario.target_at(row[0])
        dists.append(
            ((row[1] - tx) ** 2 + (row[2] - ty) ** 2 + (row[3] - tz) ** 2) ** 0.5
        )
    target_xy = _target_path(scenario, times)

    figure, (left, right) = plt.subplots(
        1, 2, figsize=(13.2, 5.0), gridspec_kw={"width_ratios": [1.0, 1.25]}
    )

    # ---------------- 左：俯视轨迹 ----------------
    left.plot(
        [p[0] for p in target_xy], [p[1] for p in target_xy],
        color=ORANGE, linewidth=2.0, linestyle="--", label="目标航迹",
    )
    left.plot(drone_x, drone_y, color=BLUE, linewidth=1.6, label="飞机航迹")
    left.plot(drone_x[0], drone_y[0], "o", color=BLUE, markersize=6)
    left.annotate(
        "起飞", (drone_x[0], drone_y[0]), textcoords="offset points",
        xytext=(6, 6), fontsize=9, color=GRAY,
    )
    best = min(range(len(state)), key=lambda i: dists[i])
    left.plot(drone_x[best], drone_y[best], "*", color=GREEN, markersize=15,
              markeredgecolor="white", zorder=5)
    left.annotate(
        f"最近 {dists[best]:.3f} m\n@ {times[best]:.1f} s",
        (drone_x[best], drone_y[best]), textcoords="offset points",
        xytext=(8, -18), fontsize=9, color=GREEN, fontweight="bold",
    )
    left.set_xlabel("东向 x / m")
    left.set_ylabel("北向 y / m")
    left.set_title("飞机和目标怎么走（俯视，横纵非等比例）", fontsize=11, loc="left")
    left.legend(loc="upper left", fontsize=9, framealpha=0.9)

    # ---------------- 右：距离随时间 ----------------
    right.plot(times, [max(d, 1e-3) for d in dists], color=BLUE, linewidth=1.6,
               label="飞机到目标的距离")
    right.axhline(THRESHOLD, color=RED, linestyle="--", linewidth=1.1)
    right.text(
        times[-1], THRESHOLD * 1.08, f"命中门限 {THRESHOLD} m",
        color=RED, fontsize=9, ha="right", va="bottom",
    )

    fired = [row[0] for row, hit in zip(state, mask) if hit]
    if fired:
        start = fired[0]
        right.axvspan(start, fired[-1], color=SHADE, alpha=0.55, linewidth=0,
                      label=f"判定“拐不过来” {len(fired)} 帧")

    for label, when, color in (
        ("切 INTERCEPT", mode_time, BLUE),
        ("切 ATTACK", attack_time, ORANGE),
    ):
        if when is not None and when > 0:
            right.axvline(when, color=color, linewidth=1.0, linestyle=":")
            right.annotate(
                label, (when, 0.30), xycoords=("data", "axes fraction"),
                textcoords="offset points", xytext=(3, 0),
                fontsize=8.5, color=color, rotation=90, va="bottom",
            )

    hover = _hover_time(plan)
    if hover is not None:
        right.axvline(hover, color=GRAY, linewidth=1.1, linestyle="--")
        right.annotate(
            "场景结束", (hover, 0.72), xycoords=("data", "axes fraction"),
            textcoords="offset points", xytext=(-4, 0),
            fontsize=8.5, color=GRAY, rotation=90, ha="right", va="bottom",
        )

    right.axvline(best and times[best] or 0.0, color=GREEN, linewidth=1.2)
    right.plot([times[best]], [max(dists[best], 1e-3)], "*", color=GREEN,
               markersize=13, markeredgecolor="white", zorder=5)

    right.set_yscale("log")
    right.set_xlabel("时间 / s")
    right.set_ylabel("距离 / m（对数轴）")
    right.set_title("离目标多远（越低越接近；阴影=算法判定拐不过来）",
                    fontsize=11, loc="left")
    right.legend(loc="upper right", fontsize=9, framealpha=0.9)

    mark = "PASS" if passed else "FAIL"
    figure.suptitle(
        f"{scenario.id}｜{scenario.title}｜期望 {scenario.expect} → {mark}",
        fontsize=13, fontweight="bold", x=0.01, ha="left",
        color=GREEN if passed else RED,
    )
    figure.text(
        0.01, 0.015,
        f"数据：.p5/evidence/scenarios/{scenario.id}_state.csv（50 Hz）与 _plan.csv（20 Hz）　"
        f"最近 {metrics.min_dist:.3f} m，过冲 {metrics.overshoot:.2f} m，"
        f"Vmax {metrics.max_speed:.2f} m/s。场景结束切 HOVER，之后目标继续飞，"
        f"距离增大不代表失败。",
        fontsize=8.5, color=GRAY,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def _plot_dashboard(rows, out_path: str) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 7.6))
    for axis, item in zip(axes.ravel(), rows):
        scenario = item["scenario"]
        state = item["state"]
        times = [row[0] for row in state]
        dists = []
        for row in state:
            tx, ty, tz = scenario.target_at(row[0])
            dists.append(
                ((row[1] - tx) ** 2 + (row[2] - ty) ** 2 + (row[3] - tz) ** 2) ** 0.5
            )
        axis.plot(times, [max(d, 1e-3) for d in dists], color=BLUE, linewidth=1.5)
        axis.axhline(THRESHOLD, color=RED, linestyle="--", linewidth=1.0)
        hover = _hover_time(item["plan"])
        if hover is not None:
            axis.axvline(hover, color=GRAY, linestyle="--", linewidth=1.0)
        axis.set_yscale("log")
        axis.set_ylim(0.03, 300)
        axis.set_xlabel("时间 / s", fontsize=9)
        axis.set_ylabel("距离 / m", fontsize=9)
        mark = "PASS" if item["passed"] else "FAIL"
        axis.set_title(
            f"{scenario.id}｜{scenario.title}｜{mark}",
            fontsize=10.5, loc="left",
            color=GREEN if item["passed"] else RED,
        )
        axis.annotate(
            f"最近 {item['metrics'].min_dist:.3f} m",
            xy=(0.98, 0.06), xycoords="axes fraction",
            ha="right", fontsize=9.5,
            color=GREEN if item["metrics"].min_dist < THRESHOLD else ORANGE,
        )
    figure.suptitle(
        "四场景：离目标距离随时间（红虚线=0.5 m 命中门限，灰虚线=场景结束）",
        fontsize=13.5, fontweight="bold", x=0.01, ha="left",
    )
    figure.text(
        0.01, 0.005,
        "每条曲线谷底越接近 0 说明越贴近目标；曲线末端上扬是场景切 HOVER 后目标继续飞，不是失败。",
        fontsize=9, color=GRAY,
    )
    figure.tight_layout(rect=(0, 0.02, 1, 0.94))
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="把场景 CSV 画成图")
    parser.add_argument("--dir", default=".p5/evidence/scenarios", help="CSV 目录")
    parser.add_argument("--out", default=".p5/evidence/scenarios/figures", help="图片输出目录")
    parser.add_argument("--id", default="", help="场景 id，逗号分隔；默认全部")
    args = parser.parse_args(argv)

    _setup_fonts()
    os.makedirs(args.out, exist_ok=True)
    chosen = (
        [get_scenario(item.strip()) for item in args.id.split(",") if item.strip()]
        if args.id
        else list(SCENARIOS)
    )

    rows: List[Dict[str, object]] = []
    for scenario in chosen:
        state, plan = _load(args.dir, scenario.id)
        if state is None:
            print(f"跳过 {scenario.id}：没有 CSV", file=sys.stderr)
            continue
        mode_time = _first_time(plan, "INTERCEPT") or _first_time(plan, "ATTACK") or 0.0
        metrics = evaluate(
            scenario, state, plan, start_t=mode_time,
            v_max=14.8 if scenario.mode == "ATTACK" else 7.6,
        )
        passed, _ = verdict(scenario, metrics)
        out_path = os.path.join(args.out, f"p6_scenario_{scenario.id}.png")
        _plot_scenario(scenario, state, plan, metrics, passed, out_path)
        print(f"已生成 {out_path}")
        rows.append(
            {
                "scenario": scenario,
                "state": state,
                "plan": plan,
                "metrics": metrics,
                "passed": passed,
            }
        )

    if rows:
        dashboard = os.path.join(args.out, "p6_dashboard.png")
        _plot_dashboard(rows, dashboard)
        print(f"已生成 {dashboard}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
