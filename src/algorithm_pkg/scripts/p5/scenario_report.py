#!/usr/bin/env python3
"""场景结果查看器：读已落盘的 CSV，打印判定表。

不需要 ROS、不需要容器。跑完场景后随时可以重看：

```bash
PYTHONPATH=src/algorithm_pkg /usr/bin/python3 \
    src/algorithm_pkg/scripts/p5/scenario_report.py --dir .p5/evidence/scenarios
```

指标口径与执行器完全一致：都用 ``scenario_defs.evaluate`` 与
``scenario_defs.verdict``，不另写一套统计。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

from algorithm_pkg.scenario_defs import (
    SCENARIOS,
    get_scenario,
    evaluate,
    verdict,
)


def _read_csv(path: str):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _load(directory: str, scenario_id: str):
    state_path = os.path.join(directory, f"{scenario_id}_state.csv")
    plan_path = os.path.join(directory, f"{scenario_id}_plan.csv")
    if not os.path.exists(state_path):
        return None, None, state_path
    state = [
        (
            float(row["t"]),
            float(row["drone_x"]),
            float(row["drone_y"]),
            float(row["drone_z"]),
            float(row["speed"]),
        )
        for row in _read_csv(state_path)
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
            for row in _read_csv(plan_path)
        ]
    return state, plan, state_path


def _first_mode_time(plan) -> float:
    for t, _x, _y, _z, action in plan:
        if action in ("INTERCEPT", "ATTACK"):
            return t
    return 0.0


def report(directory: str, ids: Optional[List[str]] = None) -> int:
    chosen = [get_scenario(item) for item in ids] if ids else list(SCENARIOS)
    print(f"数据目录: {os.path.abspath(directory)}\n")
    header = (
        f"{'场景':<20}{'期望':<12}{'最近距离':>10}{'过冲':>9}"
        f"{'Vmax':>7}{'触发':>6}{'尾随':>6}  {'判定'}"
    )
    print(header)
    print("-" * len(header))

    rows: List[Dict[str, object]] = []
    for scenario in chosen:
        state, plan, state_path = _load(directory, scenario.id)
        if state is None:
            print(f"{scenario.id:<20}{scenario.expect:<12}{'':>10}{'':>9}{'':>7}{'':>6}{'':>6}  缺数据: {state_path}")
            continue
        metrics = evaluate(
            scenario,
            state,
            plan,
            start_t=_first_mode_time(plan),
            v_max=14.8 if scenario.mode == "ATTACK" else 7.6,
        )
        passed, _detail = verdict(scenario, metrics)
        mark = "PASS" if passed else "FAIL"
        print(
            f"{scenario.id:<20}{scenario.expect:<12}"
            f"{metrics.min_dist:>9.3f}m{metrics.overshoot:>8.2f}m"
            f"{metrics.max_speed:>7.2f}{metrics.gate_frames:>6d}"
            f"{metrics.tail_frames:>6d}  {mark}"
        )
        rows.append(
            {
                "scenario": scenario,
                "metrics": metrics,
                "state": state,
                "passed": passed,
            }
        )

    if not rows:
        print("\n没有可用的 CSV。先跑场景：")
        print("  python3 .../scenario_suite.py --run all --orchestrate")
        return 1

    print("\n明细")
    for item in rows:
        scenario = item["scenario"]
        metrics = item["metrics"]
        state = item["state"]
        takeoff = next((row[0] for row in state if row[3] > 0.5), None)
        print(f"{scenario.id}  {scenario.title}")
        print(f"  目的：{scenario.purpose}")
        print(
            f"  起飞 {takeoff:.1f} s | 最近 {metrics.min_dist:.3f} m @ "
            f"{metrics.min_dist_t:.1f} s | 过冲 {metrics.overshoot:.2f} m | "
            f"Vmax {metrics.max_speed:.2f} m/s"
        )
        print(
            f"  判定口径：期望 {scenario.expect}；"
            f"复算触发 {metrics.gate_frames} 帧，疑似尾随点 {metrics.tail_frames} 帧"
        )
        if scenario.note:
            print(f"  备注：{scenario.note}")
        print()

    passed = sum(1 for item in rows if item["passed"])
    print(f"合计 {passed}/{len(rows)} PASS")
    return 0 if passed == len(rows) else 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="查看已落盘的场景结果")
    parser.add_argument(
        "--dir",
        default=os.environ.get("SCENARIO_OUT_DIR", ".p5/evidence/scenarios"),
        help="CSV 目录",
    )
    parser.add_argument("--id", default="", help="场景 id，逗号分隔；默认全部")
    args = parser.parse_args(argv)
    ids = [item.strip() for item in args.id.split(",") if item.strip()]
    return report(args.dir, ids or None)


if __name__ == "__main__":
    sys.exit(main())
