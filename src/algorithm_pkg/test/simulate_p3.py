#!/usr/bin/env python3
"""P3 三个复杂场景的时间推进仿真（纯 Python，不启动 ROS）。

仿真链路与 ``algorithm_node.py`` 的真实调用顺序一致：

    目标真实位置
        -> 每 50ms 采样一次
        -> P1 predict_target()
        -> P3 plan_with_degrade()（内部走 P2 plan() + 降级 + 护栏）
        -> 一架简化的飞机按 v_max 飞向发布点
        -> 记录 own / level / reason / point

为了贴近现状，中枢“收到不到新点就保留上一帧最多 2 秒”的行为也模拟了：
``PlanResult.point is None`` 时，飞机继续朝上一帧指令点飞，最多 2 秒。

运行：
    PYTHONPATH=src/algorithm_pkg /usr/bin/python3 \
        src/algorithm_pkg/test/simulate_p3.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import deque
from dataclasses import dataclass
from math import isfinite, sqrt
from pathlib import Path
from typing import Callable, Optional, Sequence

# Matplotlib 默认配置目录在部分环境不可写；必须在 import pyplot 前设置。
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "p3-mpl-config")
)


def _pyplot():
    """延迟导入 matplotlib 的 Agg 后端。

    pytest 的场景用例只读数字、不画图；把 matplotlib 放在函数里，
    测试环境没有 GUI/绘图库时也能跑，导入提速。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


from algorithm_pkg.algorithm_core import (  # noqa: E402
    TargetSample,
    TargetState,
    predict_target,
)
from algorithm_pkg.degrade_fsm import DegradeEvent, DegradeFSM  # noqa: E402
from algorithm_pkg.degrade_policy import plan_with_degrade  # noqa: E402
from algorithm_pkg.plan_params import PlanParams  # noqa: E402
from algorithm_pkg.plan_types import PlanResult  # noqa: E402
from algorithm_pkg.validators import EgoStatus, TargetStatus  # noqa: E402

DT = 0.05  # 20Hz，和飞控/发布周期一致
HISTORY_SEC = 3.0  # 与 algorithm_node.py 的 history_sec 一致
MAX_PRED_AGE = 1.0  # P1 max_age
MAX_TARGET_SPEED = 15.0  # P1 max_speed
CENTER_RETENTION = 2.0  # 中枢保留上一帧规划点的时间

Vec3 = tuple[float, float, float]

LEVEL_COLORS = {
    "NORMAL": "#1f77b4",
    "EXTRAPOLATING": "#ff7f0e",
    "HOLDING": "#d62728",
    "NO_PLAN": "#111111",
}


@dataclass(frozen=True)
class TickRecord:
    """一帧的时间推进记录。

    既服务于整段场景，也服务于 ``--tick`` 的单帧调试：把这一帧
    的输入检测、FSM 迁移、P3 结果、适配层兜底点和中枢实际指令
    全部留下，避免调试时再手工复现。
    """

    t: float
    own_before: Vec3
    own: Vec3
    target: Optional[TargetState]
    target_status: Optional[TargetStatus]
    ego_status: Optional[EgoStatus]
    event: Optional[DegradeEvent]
    target_true: Optional[Vec3]
    target_visible: bool
    result: PlanResult
    plan_point: Optional[Vec3]
    published: Optional[Vec3]
    command_source: str
    using_retained: bool
    ego_dropped: bool


def _norm(value: Vec3) -> float:
    """三维向量长度。"""
    return sqrt(value[0] ** 2 + value[1] ** 2 + value[2] ** 2)


def _distance(a: Vec3, b: Vec3) -> float:
    """两点距离。"""
    return _norm((a[0] - b[0], a[1] - b[1], a[2] - b[2]))


def _move_toward(own: Vec3, goal: Vec3, max_step: float) -> Vec3:
    """朝目标点飞，最多走 max_step 米，不允许越过目标点。"""
    delta = (goal[0] - own[0], goal[1] - own[1], goal[2] - own[2])
    length = _norm(delta)
    if length <= 1e-9:
        return own
    step = min(length, max_step)
    scale = step / length
    return (
        own[0] + delta[0] * scale,
        own[1] + delta[1] * scale,
        own[2] + delta[2] * scale,
    )


def _clamp_hold(own: Vec3, params: PlanParams) -> Vec3:
    """模拟 algorithm_node.py 的 ATTACK“当前位置保持点”兜底。"""
    return (
        max(-params.max_xy, min(params.max_xy, own[0])),
        max(-params.max_xy, min(params.max_xy, own[1])),
        max(params.min_alt, min(params.max_alt, own[2])),
    )


def _center_track_offset(own: Vec3, goal: Vec3, reference: Vec3) -> Vec3:
    """模拟中枢 TRACK 分支的 5m 环绕，避免仿真把“跟踪”飞成撞机。

    中枢在有视觉时用 ``latest_vision_target`` 算距离，没有视觉时才用
    规划点；这里用 ``reference`` 区分。环绕偏移加在目标点上，不属于
    P3，但真实飞机运动必须带上，否则轨迹会误导。
    """
    dx = reference[0] - own[0]
    dy = reference[1] - own[1]
    horizontal = sqrt(dx * dx + dy * dy)
    if not 0.01 < horizontal < 5.0:
        return goal
    ux, uy = dx / horizontal, dy / horizontal
    tangent_x, tangent_y = -uy, ux
    force = (5.0 - horizontal) * 1.5
    return (
        goal[0] + tangent_x * force,
        goal[1] + tangent_y * force,
        goal[2],
    )


# ------------------------------ 场景定义 ------------------------------


def _target_a(t: float) -> Vec3:
    """场景 A：目标先向东 3 m/s，2.0s 后转为向北 3 m/s。"""
    if t < 2.0:
        return (20.0 + 3.0 * t, 0.0, 10.0)
    return (26.0, 3.0 * (t - 2.0), 10.0)


def _visible_a(t: float) -> bool:
    """场景 A：目标被遮挡 0.6s，其余时间可见。"""
    return not (2.0 <= t < 2.6)


def _target_b(t: float) -> Vec3:
    """场景 B：目标同向远离，9 m/s，比己方 V_max=6.0 m/s 快。"""
    return (20.0 + 9.0 * t, 0.0, 10.0)


def _visible_b(t: float) -> bool:
    """场景 B：目标丢失 1.5s。"""
    return not (3.0 <= t < 4.5)


def _target_c(t: float) -> Vec3:
    """场景 C：目标先向北 4 m/s，2.0s 后改向东 4 m/s（慢于 V_max，可被 ATTACK 追上）。"""
    if t < 2.0:
        return (30.0, 4.0 * t, 10.0)
    return (30.0 + 4.0 * (t - 2.0), 8.0, 10.0)


def _visible_c(t: float) -> bool:
    """场景 C：目标在转弯期间被遮挡 0.8s。"""
    return not (2.0 <= t < 2.8)


def _ego_dropout_c(t: float) -> bool:
    """场景 C：己方状态在 2.3～2.8s 断流 0.5s。"""
    return 2.3 <= t < 2.8


# ------------------------------ 仿真主循环 ------------------------------


def simulate(
    *,
    name: str,
    action: str,
    params: PlanParams,
    own_start: Vec3,
    duration: float,
    target_fn: Callable[[float], Vec3],
    visible_fn: Callable[[float], bool],
    ego_dropout_fn: Callable[[float], bool],
) -> list[TickRecord]:
    """按 20Hz 推进一个场景，返回每一帧的记录。"""
    fsm = DegradeFSM(params)
    samples: deque[TargetSample] = deque()
    own = own_start
    prev_result: Optional[PlanResult] = None

    # 中枢缓存：视觉原始目标、规划点；两者都保留 2 秒。
    vision_target: Optional[Vec3] = None
    vision_time = -1e9
    plan_target: Optional[Vec3] = None
    plan_time = -1e9
    records: list[TickRecord] = []

    steps = int(round(duration / DT)) + 1
    for index in range(steps):
        now = index * DT
        visible = visible_fn(now)
        target_true = target_fn(now) if visible else None

        if visible:
            samples.append(
                TargetSample(
                    stamp=now,
                    x=target_true[0],
                    y=target_true[1],
                    z=target_true[2],
                )
            )
        while samples and now - samples[0].stamp > HISTORY_SEC:
            samples.popleft()

        target = None
        if samples:
            target = predict_target(
                samples,
                now,
                max_age=MAX_PRED_AGE,
                max_speed=MAX_TARGET_SPEED,
            )

        ego_dropped = ego_dropout_fn(now)
        own_seen = None if ego_dropped else own

        result = plan_with_degrade(
            action,
            own_seen,
            target,
            fsm,
            params,
            now,
            prev_result,
            ego_stamp=None if ego_dropped else now,
        )
        prev_result = result

        plan_point: Optional[Vec3] = None
        if result.point is not None:
            plan_point = result.point
        elif (
            action == "ATTACK"
            and result.level == "NORMAL"
            and own_seen is not None
        ):
            # 现状适配层：ATTACK 无解/目标失效时补一个当前位置保持点。
            plan_point = _clamp_hold(own_seen, params)

        if plan_point is not None:
            plan_target = plan_point
            plan_time = now
        if visible:
            vision_target = target_true
            vision_time = now

        vision_fresh = (
            vision_target is not None
            and now - vision_time <= CENTER_RETENTION
        )
        plan_fresh = (
            plan_target is not None and now - plan_time <= CENTER_RETENTION
        )

        published: Optional[Vec3] = None
        command_source = "none"
        source_time = now
        if action == "TRACK":
            # ! 中枢 TRACK 分支优先 latest_vision_target，其次 my_plan。
            # ! 这意味着 P3 的 TRACK 外推/保持点只有在视觉丢失 >2s 后才生效。
            if vision_fresh:
                published = _center_track_offset(
                    own, vision_target, vision_target
                )
                command_source = "vision"
                source_time = vision_time
            elif plan_fresh:
                published = _center_track_offset(
                    own, plan_target, plan_target
                )
                command_source = "plan"
                source_time = plan_time
        else:
            if plan_fresh:
                published = plan_target
                command_source = "plan"
                source_time = plan_time

        using_retained = (
            published is not None
            and now - source_time > 1.5 * DT
        )
        if using_retained:
            command_source += "_retained"

        own_before = own
        if published is not None:
            speed = params.v_max_for(action)
            own = _move_toward(own, published, speed * DT)

        records.append(
            TickRecord(
                t=now,
                own_before=own_before,
                own=own,
                target=target,
                target_status=fsm.last_target_status,
                ego_status=fsm.last_ego_status,
                event=fsm.last_event,
                target_true=target_true,
                target_visible=visible,
                result=result,
                plan_point=plan_point,
                published=published,
                command_source=command_source,
                using_retained=using_retained,
                ego_dropped=ego_dropped,
            )
        )

    return records


# ------------------------------ 结果检查 ------------------------------


def summarize(name: str, records: list[TickRecord]) -> dict:
    """统计级别、发布点、跳变、最小距离和异常原因。

    跳变分三个口径，避免把适配层补的保持点混进“P3 算法跳变”：

    - ``p3_jump``：``PlanResult.point``，纯 P3 输出；
    - ``plan_jump``：``plan_point``，含适配层 ATTACK 保持点，即准备发布点；
    - ``published_jump``：``published``，中枢实际指令（TRACK 会叠加视觉/环绕）。

    为兼容旧脚本保留 ``max_jump=plan_jump`` 和 ``command_jump=published_jump``。
    """
    levels: dict[str, int] = {}
    reasons: dict[str, int] = {}
    p3_count = 0
    plan_count = 0
    published_count = 0
    p3_jump = 0.0
    plan_jump = 0.0
    published_jump = 0.0
    plan_vs_published = 0.0
    last_p3: Optional[Vec3] = None
    last_plan: Optional[Vec3] = None
    last_published: Optional[Vec3] = None
    min_distance = float("inf")
    retained_ticks = 0
    retained_distance = 0.0
    vision_override_ticks = 0
    tail_points = 0
    previous_own: Optional[Vec3] = None

    for record in records:
        levels[record.result.level] = levels.get(record.result.level, 0) + 1
        reasons[record.result.reason] = (
            reasons.get(record.result.reason, 0) + 1
        )
        if record.result.point is not None:
            p3_count += 1
            if last_p3 is not None:
                p3_jump = max(
                    p3_jump, _distance(last_p3, record.result.point)
                )
            last_p3 = record.result.point
        if record.plan_point is not None:
            plan_count += 1
            if last_plan is not None:
                plan_jump = max(
                    plan_jump, _distance(last_plan, record.plan_point)
                )
            last_plan = record.plan_point
        if record.published is not None:
            published_count += 1
            if last_published is not None:
                published_jump = max(
                    published_jump, _distance(last_published, record.published)
                )
            last_published = record.published
            if record.result.reason == "fallback_tail":
                tail_points += 1
        if (
            record.plan_point is not None
            and record.published is not None
            and record.command_source.startswith("vision")
        ):
            vision_override_ticks += 1
            plan_vs_published = max(
                plan_vs_published,
                _distance(record.plan_point, record.published),
            )
        if record.target_true is not None:
            min_distance = min(
                min_distance, _distance(record.own, record.target_true)
            )
        if record.using_retained:
            retained_ticks += 1
            if previous_own is not None:
                retained_distance += _distance(
                    previous_own, record.own
                )
        previous_own = record.own

    return {
        "name": name,
        "levels": levels,
        "reasons": reasons,
        "p3_count": p3_count,
        "plan_count": plan_count,
        "published": published_count,
        "p3_jump": p3_jump,
        "plan_jump": plan_jump,
        "published_jump": published_jump,
        # 兼容旧脚本/文档的字段名。
        "max_jump": plan_jump,
        "command_jump": published_jump,
        "plan_vs_published": plan_vs_published,
        "plan_vs_command": plan_vs_published,
        "vision_override_ticks": vision_override_ticks,
        "min_distance": min_distance,
        "retained_ticks": retained_ticks,
        "retained_distance": retained_distance,
        "tail_points": tail_points,
    }


def check_common(records: list[TickRecord]) -> list[str]:
    """所有场景共用的硬检查。"""
    issues: list[str] = []
    for record in records:
        point = record.published
        if point is not None:
            if not all(isfinite(component) for component in point):
                issues.append(f"t={record.t:.2f}: 输出点含 NaN/Inf")
                break
            if abs(point[0]) > 1000.0 or abs(point[1]) > 1000.0:
                issues.append(f"t={record.t:.2f}: 输出点越 XY 边界")
                break
            if point[2] < 3.0 or point[2] > 50.0:
                issues.append(f"t={record.t:.2f}: 输出点越高度边界")
                break
        if (
            record.result.level == "NO_PLAN"
            and record.plan_point is not None
        ):
            issues.append(f"t={record.t:.2f}: NO_PLAN 却仍生成新 P3 点")
            break
    return issues


def check_scenario(
    scenario: dict,
    records: list[TickRecord],
    summary: dict,
    *,
    tuned: bool = True,
) -> list[str]:
    """按场景返回未满足的预期项；空列表表示通过。

    ``tuned=False``（``max_step=None``）时跳过跳变断言，只保留场景语义
    与中枢保留时长检查，用于 ``--params default`` 的观察输出。
    """
    failures = list(check_common(records))
    if not records:
        failures.append("没有记录")
        return failures

    if tuned:
        max_step = scenario["max_step"]
        if summary["p3_jump"] > max_step + 1e-6:
            failures.append(
                f"P3 输出点跳变 {summary['p3_jump']:.3f}m > "
                f"max_step {max_step:.2f}m"
            )
        if summary["plan_jump"] > max_step + 1e-6:
            failures.append(
                f"准备发布点跳变 {summary['plan_jump']:.3f}m > "
                f"max_step {max_step:.2f}m"
            )
        # 标定后 ATTACK 的保留盲飞应接近 0；default 模式只观察，不判。
        if (
            scenario["name"] == "C"
            and summary["retained_distance"] > 1.0
        ):
            failures.append(
                "C: 标定后保留期间盲飞 "
                f"{summary['retained_distance']:.2f}m > 1.0m"
            )
    if summary["retained_ticks"] * DT > CENTER_RETENTION + DT + 1e-6:
        failures.append(
            f"中枢保留 {summary['retained_ticks'] * DT:.2f}s > "
            f"{CENTER_RETENTION:.1f}s"
        )

    final_level = records[-1].result.level
    name = scenario["name"]
    if name == "A":
        if "EXTRAPOLATING" not in summary["levels"]:
            failures.append("A: 目标遮挡后没有进入 L1 外推")
        if "NO_PLAN" in summary["levels"]:
            failures.append("A: 不应出现 L3")
        if final_level != "NORMAL":
            failures.append(f"A: 最终未回 NORMAL（{final_level}）")
        if summary["vision_override_ticks"] == 0:
            failures.append("A: 中枢没有采用视觉目标（TRACK 视觉优先）")
        if summary["retained_ticks"] == 0:
            failures.append("A: 没有出现中枢保留旧指令")
    elif name == "B":
        if summary["tail_points"] == 0:
            failures.append("B: INTERCEPT 追不上时应发尾随点")
        if "HOLDING" not in summary["levels"]:
            failures.append("B: 目标丢失 1.5s 应进入 L2 保持")
        if "NO_PLAN" in summary["levels"]:
            failures.append("B: 不应出现 L3")
        if final_level != "NORMAL":
            failures.append(f"B: 最终未回 NORMAL（{final_level}）")
    elif name == "C":
        if summary["tail_points"] != 0:
            failures.append("C: ATTACK 不允许尾随")
        if "NO_PLAN" not in summary["levels"]:
            failures.append("C: 己方断流应进入 L3")
        if summary["retained_ticks"] == 0:
            failures.append("C: L3 期间中枢应保留旧指令")
        if final_level != "NORMAL":
            failures.append(f"C: 最终未回 NORMAL（{final_level}）")
    return failures


def transitions(records: list[TickRecord]) -> list[str]:
    """输出级别发生变化的时刻，便于对照策略表。"""
    events: list[str] = []
    previous = None
    for record in records:
        if record.result.level != previous:
            events.append(
                f"t={record.t:4.2f}s  {record.result.level:<14} "
                f"anomaly={record.result.anomaly or '-':<13} "
                f"reason={record.result.reason}"
            )
            previous = record.result.level
    return events


# ------------------------------ 绘图 ------------------------------


def _shade_intervals(
    ax,
    records: Sequence[TickRecord],
    predicate: Callable[[TickRecord], bool],
    color: str,
    label: str,
    alpha: float,
) -> None:
    """把连续满足条件的帧合并成一条浅色带，并在图例里只加一次。"""
    start: Optional[float] = None
    first = True
    for record in records:
        active = predicate(record)
        if active and start is None:
            start = record.t
        elif not active and start is not None:
            ax.axvspan(
                start,
                record.t,
                color=color,
                alpha=alpha,
                linewidth=0,
                label=label if first else "_nolegend_",
                zorder=0,
            )
            first = False
            start = None
    if start is not None:
        ax.axvspan(
            start,
            records[-1].t + DT,
            color=color,
            alpha=alpha,
            linewidth=0,
            label=label if first else "_nolegend_",
            zorder=0,
        )


def plot_scene(
    name: str,
    title: str,
    records: list[TickRecord],
    target_path: list[Vec3],
    out_path: Path,
) -> None:
    """画 XY 轨迹 + 距离/级别时间线。"""
    plt = _pyplot()
    figure, (ax_path, ax_time) = plt.subplots(
        1, 2, figsize=(12.0, 4.8), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    # 左图：真实目标路径、可见采样、无人机路径、发布点。
    ax_path.plot(
        [point[0] for point in target_path],
        [point[1] for point in target_path],
        "--",
        color="0.55",
        linewidth=1.4,
        label="target true path",
    )
    visible_points = [
        record.target_true
        for record in records
        if record.target_true is not None
    ]
    ax_path.scatter(
        [point[0] for point in visible_points],
        [point[1] for point in visible_points],
        s=8,
        color="0.45",
        alpha=0.55,
        label="target visible sample",
    )

    # 无人机路径按级别分段着色；每个级别只加一次图例。
    labeled_levels: set[str] = set()
    for level, color in LEVEL_COLORS.items():
        xs: list[float] = []
        ys: list[float] = []
        for record in records:
            if record.result.level == level:
                xs.append(record.own[0])
                ys.append(record.own[1])
            elif xs:
                label = level if level not in labeled_levels else None
                ax_path.plot(
                    xs, ys, color=color, linewidth=2.2, label=label
                )
                labeled_levels.add(level)
                xs, ys = [], []
        if xs:
            label = level if level not in labeled_levels else None
            ax_path.plot(xs, ys, color=color, linewidth=2.2, label=label)
            labeled_levels.add(level)

    commands = [record.published for record in records if record.published]
    ax_path.scatter(
        [point[0] for point in commands],
        [point[1] for point in commands],
        marker=".",
        s=6,
        color="#9467bd",
        alpha=0.45,
        label="center command",
    )
    ax_path.scatter(
        [records[0].own[0]], [records[0].own[1]],
        marker="o", s=55, color="black", label="start",
    )
    ax_path.scatter(
        [records[-1].own[0]], [records[-1].own[1]],
        marker="*", s=120, color="black", label="end",
    )
    ax_path.set_xlabel("x / m")
    ax_path.set_ylabel("y / m")
    ax_path.set_title(title + "\nleft: XY trajectory, right: level & distance")
    ax_path.axis("equal")
    ax_path.grid(True, alpha=0.3)
    ax_path.legend(fontsize=7, loc="best")

    # 右图：级别时间线 + 与目标的距离。
    level_order = {
        "NORMAL": 0,
        "EXTRAPOLATING": 1,
        "HOLDING": 2,
        "NO_PLAN": 3,
    }
    times = [record.t for record in records]
    level_values = [level_order[record.result.level] for record in records]
    ax_time.step(
        times,
        level_values,
        where="post",
        color="#1f77b4",
        linewidth=1.6,
        label="degrade level",
    )
    ax_time.set_yticks(list(level_order.values()))
    ax_time.set_yticklabels(list(level_order.keys()), fontsize=8)
    ax_time.set_xlabel("t / s")
    ax_time.set_ylabel("P3 level")
    ax_time.set_title("P3 level / target distance")
    ax_time.grid(True, alpha=0.3)

    ax_distance = ax_time.twinx()
    distance_times = [
        record.t for record in records if record.target_true is not None
    ]
    distance_values = [
        _distance(record.own, record.target_true)
        for record in records
        if record.target_true is not None
    ]
    ax_distance.plot(
        distance_times,
        distance_values,
        color="#d62728",
        linewidth=1.3,
        label="own-target distance",
    )
    ax_distance.set_ylabel("distance to target / m", color="#d62728")
    ax_distance.tick_params(axis="y", colors="#d62728")

    # 断流/遮挡区间用浅色带标出；连续帧合并，图例只出现一次。
    _shade_intervals(
        ax_time,
        records,
        lambda record: record.ego_dropped,
        "#111111",
        "ego dropout",
        0.10,
    )
    _shade_intervals(
        ax_time,
        records,
        lambda record: record.target_visible is False
        and record.ego_dropped is False,
        "#ff7f0e",
        "target occlusion",
        0.10,
    )

    handles, labels = ax_time.get_legend_handles_labels()
    extra_handles, extra_labels = ax_distance.get_legend_handles_labels()
    ax_time.legend(
        handles + extra_handles,
        labels + extra_labels,
        fontsize=7,
        loc="best",
    )

    figure.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


# ------------------------------ 单帧调试/CSV ------------------------------


def _fmt_point(point: Optional[Vec3], digits: int = 2) -> str:
    """把三维点格式化成一行短文本；None 显示为 ``-``。"""
    if point is None:
        return "-"
    return (
        f"({point[0]:.{digits}f},{point[1]:.{digits}f},"
        f"{point[2]:.{digits}f})"
    )


def _fmt_status(status: Optional[object]) -> str:
    """格式化 TargetStatus / EgoStatus；两者都有 code/age/reason。"""
    if status is None:
        return "-"
    code = getattr(getattr(status, "code", None), "value", None)
    age = getattr(status, "age", None)
    reason = getattr(status, "reason", None)
    age_text = "inf" if age is None else f"{age:.2f}s"
    return f"{code} age={age_text} reason={reason}"


def print_tick_detail(
    name: str,
    records: Sequence[TickRecord],
    tick: int,
) -> None:
    """打印单帧调试视图：输入检测 -> FSM -> P3 -> 适配层 -> 中枢。"""
    if not records:
        print(f"场景 {name}: 没有记录")
        return
    tick = max(0, min(len(records) - 1, tick))
    print("-" * 78)
    print(
        f"单帧调试：场景 {name}，tick={tick}/{len(records) - 1}，"
        f"t={records[tick].t:.2f}s"
    )
    print(
        f"{'idx':>3} {'t':>5} {'own_before':>21} {'target_true':>21} "
        f"{'level':<14} {'anomaly':<13} {'plan_point':>21} "
        f"{'published':>21} source"
    )
    start = max(0, tick - 3)
    end = min(len(records), tick + 4)
    for index in range(start, end):
        record = records[index]
        marker = ">>" if index == tick else "  "
        print(
            f"{marker}{index:3d} {record.t:5.2f} "
            f"{_fmt_point(record.own_before):>21} "
            f"{_fmt_point(record.target_true):>21} "
            f"{record.result.level:<14} "
            f"{record.result.anomaly or '-':<13} "
            f"{_fmt_point(record.plan_point):>21} "
            f"{_fmt_point(record.published):>21} "
            f"{record.command_source}"
        )

    record = records[tick]
    print("选中帧详情：")
    print(
        f"  输入己方={_fmt_point(record.own_before)}，"
        f"目标真实={_fmt_point(record.target_true)}，"
        f"target_visible={record.target_visible}，"
        f"ego_dropped={record.ego_dropped}"
    )
    if record.target is None:
        print("  P1 目标状态：None（没有可用采样）")
    else:
        target = record.target
        print(
            f"  P1 目标状态：pos={_fmt_point(target.position)} "
            f"vel={_fmt_point(target.velocity)} age={target.age:.2f}s "
            f"valid={target.valid} velocity_valid={target.velocity_valid}"
        )
    print(f"  目标检测：{_fmt_status(record.target_status)}")
    print(f"  己方检测：{_fmt_status(record.ego_status)}")
    event = record.event
    if event is not None:
        recovery = ""
        if event.recover_frames is not None:
            recovery = (
                f" recover_frames={event.recover_frames}"
                f" recover_elapsed={event.recover_elapsed}"
            )
        print(
            f"  FSM 迁移：{event.previous} -> {event.current} "
            f"reason={event.reason} changed={event.changed} "
            f"anomaly={event.anomaly or '-'}{recovery}"
        )
    print(
        f"  P3 结果：action={record.result.action} "
        f"level={record.result.level} "
        f"anomaly={record.result.anomaly or '-'} "
        f"reason={record.result.reason}"
    )
    print(
        f"    point={_fmt_point(record.result.point)} "
        f"reachable={record.result.reachable} "
        f"t_go={record.result.t_go} "
        f"clipped={record.result.clipped} "
        f"output_stamp={record.result.output_stamp}"
    )
    if record.result.guard_notes:
        print(f"    guard_notes={record.result.guard_notes}")
    if record.result.recovered_from is not None:
        print(
            f"    recovered_from={record.result.recovered_from} "
            f"recovery_frames={record.result.recovery_frames}"
        )
    print(
        f"  适配层兜底点 plan_point={_fmt_point(record.plan_point)}"
        "（ATTACK 无解时为当前位置保持点）"
    )
    print(
        f"  中枢实际指令 published={_fmt_point(record.published)} "
        f"source={record.command_source} "
        f"using_retained={record.using_retained}"
    )


CSV_FIELDS = (
    "scenario",
    "t",
    "own_before_x",
    "own_before_y",
    "own_before_z",
    "own_x",
    "own_y",
    "own_z",
    "target_visible",
    "target_true_x",
    "target_true_y",
    "target_true_z",
    "target_age",
    "target_valid",
    "target_velocity_valid",
    "target_code",
    "target_reason",
    "ego_code",
    "ego_reason",
    "level",
    "anomaly",
    "reason",
    "reachable",
    "t_go",
    "clipped",
    "p3_x",
    "p3_y",
    "p3_z",
    "plan_x",
    "plan_y",
    "plan_z",
    "published_x",
    "published_y",
    "published_z",
    "command_source",
    "using_retained",
    "ego_dropped",
    "event_previous",
    "event_current",
    "event_reason",
    "guard_notes",
)


def _point_fields(prefix: str, point: Optional[Vec3]) -> dict[str, object]:
    """把点拆成三列，None 写空字符串。"""
    if point is None:
        return {f"{prefix}_x": "", f"{prefix}_y": "", f"{prefix}_z": ""}
    return {
        f"{prefix}_x": point[0],
        f"{prefix}_y": point[1],
        f"{prefix}_z": point[2],
    }


def _code_text(status: Optional[object]) -> str:
    """取状态的 code 字符串；None 为空。"""
    if status is None:
        return ""
    code = getattr(getattr(status, "code", None), "value", None)
    return "" if code is None else code


def write_csv(
    path: Path,
    scenario_records: Sequence[tuple[str, Sequence[TickRecord]]],
) -> None:
    """把每帧记录写成 CSV，便于 pandas/Excel 做单帧对照。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for name, records in scenario_records:
            for record in records:
                target = record.target
                target_true = record.target_true
                event = record.event
                row: dict[str, object] = {
                    "scenario": name,
                    "t": record.t,
                    "target_visible": record.target_visible,
                    "target_age": "" if target is None else target.age,
                    "target_valid": "" if target is None else target.valid,
                    "target_velocity_valid": (
                        "" if target is None else target.velocity_valid
                    ),
                    "target_code": _code_text(record.target_status),
                    "target_reason": (
                        "" if record.target_status is None
                        else record.target_status.reason
                    ),
                    "ego_code": _code_text(record.ego_status),
                    "ego_reason": (
                        "" if record.ego_status is None
                        else record.ego_status.reason
                    ),
                    "level": record.result.level,
                    "anomaly": record.result.anomaly or "",
                    "reason": record.result.reason,
                    "reachable": record.result.reachable,
                    "t_go": (
                        "" if record.result.t_go is None
                        else record.result.t_go
                    ),
                    "clipped": record.result.clipped,
                    "command_source": record.command_source,
                    "using_retained": record.using_retained,
                    "ego_dropped": record.ego_dropped,
                    "event_previous": "" if event is None else event.previous,
                    "event_current": "" if event is None else event.current,
                    "event_reason": "" if event is None else event.reason,
                    "guard_notes": "|".join(record.result.guard_notes),
                }
                row.update(_point_fields("own_before", record.own_before))
                row.update(_point_fields("own", record.own))
                row.update(_point_fields("target_true", target_true))
                row.update(_point_fields("p3", record.result.point))
                row.update(_point_fields("plan", record.plan_point))
                row.update(_point_fields("published", record.published))
                writer.writerow(row)


# ------------------------------ 场景/入口 ------------------------------


def build_scenarios() -> list[dict]:
    """返回三个可复现场景的定义。"""
    return [
        {
            "name": "A",
            "title": "A: TRACK + 0.6s occlusion + target turn",
            "action": "TRACK",
            "own_start": (0.0, 0.0, 10.0),
            "duration": 5.0,
            "max_step": 0.30,  # 6.0 m/s * 0.05s
            "target_fn": _target_a,
            "visible_fn": _visible_a,
            "ego_dropout_fn": lambda t: False,
        },
        {
            "name": "B",
            "title": "B: INTERCEPT unreachable + 1.5s target loss",
            "action": "INTERCEPT",
            "own_start": (0.0, 0.0, 10.0),
            "duration": 6.0,
            "max_step": 0.30,  # 6.0 m/s * 0.05s
            "target_fn": _target_b,
            "visible_fn": _visible_b,
            "ego_dropout_fn": lambda t: False,
        },
        {
            "name": "C",
            "title": "C: ATTACK + target turn + 0.5s ego dropout",
            "action": "ATTACK",
            "own_start": (0.0, 0.0, 10.0),
            "duration": 7.0,
            "max_step": 0.30,  # 6.0 m/s * 0.05s
            "target_fn": _target_c,
            "visible_fn": _visible_c,
            "ego_dropout_fn": _ego_dropout_c,
        },
    ]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """解析命令行；不带参数时行为与原来一致（三个场景全跑）。"""
    parser = argparse.ArgumentParser(
        description="P3 场景仿真：整段回放、单帧调试、CSV 导出。",
    )
    parser.add_argument(
        "--scenario",
        default="all",
        help="场景名：A、B、C 或 all（默认 all）",
    )
    parser.add_argument(
        "--tick",
        type=int,
        default=None,
        help="打印该帧前后的输入/检测/FSM/输出详情；配合 --scenario 使用",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="把选中场景的每帧记录写到 CSV",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="把场景摘要、判定与迁移写成 JSON（供 CI/下游消费）",
    )
    parser.add_argument(
        "--params",
        choices=("tuned", "default"),
        default="tuned",
        help="用标定 max_step 还是节点默认 max_step=None 的记录做输出/绘图",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不生成 PNG（单帧调试时更快）",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出场景后退出",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """跑选中场景，打印摘要/迁移/单帧详情，按需导出 CSV、JSON 和轨迹图。

    Returns:
        0：所有选中场景判定通过；1：存在 FAIL。
    """
    args = parse_args(argv)
    scenarios = build_scenarios()
    if args.list:
        for scenario in scenarios:
            print(
                f"{scenario['name']}: {scenario['title']} "
                f"({scenario['action']}, {scenario['duration']:.1f}s)"
            )
        return 0

    selected = scenarios
    if args.scenario.lower() != "all":
        selected = [
            scenario
            for scenario in scenarios
            if scenario["name"].lower() == args.scenario.lower()
        ]
        if not selected:
            raise SystemExit(
                f"未知场景 {args.scenario!r}；可选 A/B/C/all"
            )

    root = Path(__file__).resolve().parents[3]
    out_dir = root / "文档" / "figures"

    all_ok = True
    output_records: list[tuple[str, list[TickRecord]]] = []
    payload: dict[str, object] = {
        "params": args.params,
        "ok": True,
        "scenarios": [],
    }
    for scenario in selected:
        # 先按节点默认 max_step=None 跑一遍，看看“护栏关闭”时的指令跳变。
        default_params = PlanParams(
            v_max=PlanParams().v_max,
            v_max_intercept=PlanParams().v_max_intercept,
            v_max_attack=PlanParams().v_max_attack,
        )
        default_records = simulate(
            name=scenario["name"],
            action=scenario["action"],
            params=default_params,
            own_start=scenario["own_start"],
            duration=scenario["duration"],
            target_fn=scenario["target_fn"],
            visible_fn=scenario["visible_fn"],
            ego_dropout_fn=scenario["ego_dropout_fn"],
        )
        default_summary = summarize(scenario["name"], default_records)

        # 再按“v_max * dt”标定 max_step，验证 P3 护栏能把跳变压住。
        tuned_params = PlanParams(max_step=scenario["max_step"])
        tuned_records = simulate(
            name=scenario["name"],
            action=scenario["action"],
            params=tuned_params,
            own_start=scenario["own_start"],
            duration=scenario["duration"],
            target_fn=scenario["target_fn"],
            visible_fn=scenario["visible_fn"],
            ego_dropout_fn=scenario["ego_dropout_fn"],
        )
        tuned_summary = summarize(scenario["name"], tuned_records)

        records = (
            default_records if args.params == "default" else tuned_records
        )
        summary = summarize(scenario["name"], records)
        active_label = (
            "默认 max_step=None"
            if args.params == "default"
            else f"标定 max_step={scenario['max_step']:.2f}m"
        )

        print("=" * 78)
        print(f"场景 {scenario['name']}：{scenario['title']}")
        print(
            f"动作={scenario['action']}, 时长={scenario['duration']:.1f}s, "
            f"帧数={len(records)}, 当前输出参数={active_label}"
        )
        print(
            f"级别分布={summary['levels']}, "
            f"准备发布点={summary['plan_count']} 帧, "
            f"中枢实际指令={summary['published']} 帧, "
            f"最小目标距离={summary['min_distance']:.2f}m"
        )
        print(
            f"P3 输出点最大跳变（纯算法）：默认 max_step=None -> "
            f"{default_summary['p3_jump']:.3f}m；"
            f"标定 max_step={scenario['max_step']:.2f}m -> "
            f"{tuned_summary['p3_jump']:.3f}m"
        )
        print(
            f"准备发布点最大跳变（含适配层保持点）="
            f"{summary['plan_jump']:.3f}m"
        )
        print(
            f"中枢实际指令点最大跳变={summary['published_jump']:.3f}m"
            "（TRACK 场景会叠加 5m 环绕，且发生在 P3 护栏之后）"
        )
        print(
            f"尾随点帧数={summary['tail_points']}, "
            f"中枢保留指令帧数={summary['retained_ticks']} "
            f"（标定后保留期间飞行 "
            f"{tuned_summary['retained_distance']:.2f}m；"
            f"默认 max_step=None 时 "
            f"{default_summary['retained_distance']:.2f}m）, "
            f"TRACK 视觉覆盖帧数={summary['vision_override_ticks']}"
        )
        if summary["vision_override_ticks"]:
            print(
                "P3 点与中枢实际指令最大偏差="
                f"{summary['plan_vs_command']:.3f}m"
                "（中枢 TRACK 优先视觉，P3 的 TRACK 点被覆盖）"
            )

        print("级别迁移：")
        for event in transitions(records):
            print("  " + event)

        failures = check_scenario(
            scenario, records, summary, tuned=args.params != "default"
        )
        scenario_ok = not failures
        all_ok = all_ok and scenario_ok
        if scenario_ok:
            print("场景判定：PASS（硬检查 + 跳变 + 场景语义）")
        else:
            print("场景判定：FAIL")
            for failure in failures:
                print("  - " + failure)

        payload["scenarios"].append(
            {
                "name": scenario["name"],
                "action": scenario["action"],
                "ok": scenario_ok,
                "failures": failures,
                "summary": summary,
                "transitions": transitions(records),
            }
        )

        if scenario["action"] == "TRACK":
            print(
                "观察：中枢 TRACK 优先 latest_vision_target；P3 的 TRACK "
                "外推/保持点只有在视觉丢失 >2s 后才会被采用。"
            )
        if summary["retained_ticks"] > 0:
            print(
                "观察：L2/L3 期间中枢会按 2s 保留继续发旧指令；"
                "max_step 标定后旧指令通常就在附近，未标定则可能继续飞远。"
            )

        if args.tick is not None:
            print_tick_detail(scenario["name"], records, args.tick)

        output_records.append((scenario["name"], records))

        if not args.no_plot:
            target_path = [
                scenario["target_fn"](index * DT)
                for index in range(
                    int(round(scenario["duration"] / DT)) + 1
                )
            ]
            suffix = "_default" if args.params == "default" else ""
            out_path = (
                out_dir
                / f"p3_scene_{scenario['name'].lower()}{suffix}.png"
            )
            plot_scene(
                scenario["name"],
                scenario["title"],
                records,
                target_path,
                out_path,
            )
            print(f"轨迹图：{out_path}")

    if args.csv is not None:
        write_csv(args.csv, output_records)
        print(f"CSV：{args.csv}（{len(output_records)} 个场景）")

    print("=" * 78)
    print("总判定：" + ("全部 PASS" if all_ok else "存在 FAIL"))

    payload["ok"] = all_ok
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"JSON：{args.json}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
