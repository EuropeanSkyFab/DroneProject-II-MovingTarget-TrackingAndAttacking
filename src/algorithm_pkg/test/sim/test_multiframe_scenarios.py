"""3 条多帧场景：短遮挡、追不上尾随、ATTACK 保持。

三条测试按 20 Hz 推进，统一走 ``adapter_core.plan_drone``，参数用固定基线。
己方每帧朝发点最多走 ``v_max*dt``，只作为闭环推进；不模拟中枢 2 s 保留和
飞控动力学。图表由 ``scenario_chart`` fixture 在用例结束后生成。
"""

from collections import Counter
from dataclasses import dataclass
from math import hypot, radians
from typing import Callable, Optional

import pytest

from algorithm_pkg.adapter_core import (
    DroneRuntime,
    TargetBuffer,
    plan_drone,
    should_publish,
)
from algorithm_pkg.algorithm_core import TargetState
from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.plan_types import PlanResult

BASELINE_PARAMS = PlanParams(
    v_max=4.50,
    v_max_intercept=7.60,
    v_max_attack=14.80,
    a_max=9.79,
    yaw_rate_max=radians(172.4),
    max_step=0.74,
)

DT = 0.05
T_START = 100.0
OWN_START = (0.0, 0.0, 10.0)

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Frame:
    """一帧的输入、输出与己方真实位置。"""

    t: float
    now: float
    own: Optional[Vec3]
    own_truth: Vec3
    target: Optional[TargetState]
    truth: Vec3
    result: PlanResult
    ego_lost: bool


def _distance(a: Vec3, b: Vec3) -> float:
    return hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _step_toward(own: Vec3, point: Vec3, limit: float) -> Vec3:
    """朝发点走，最多 ``limit`` 米；到达就不动。"""
    delta = (point[0] - own[0], point[1] - own[1], point[2] - own[2])
    length = hypot(delta[0], delta[1], delta[2])
    if length <= 1e-9:
        return own
    step = min(limit, length)
    return (
        own[0] + delta[0] / length * step,
        own[1] + delta[1] / length * step,
        own[2] + delta[2] / length * step,
    )


def _target_a(t: float) -> Vec3:
    """A：目标先向东 3 m/s，2.0 s 后转为向北 3 m/s。"""
    if t < 2.0:
        return (20.0 + 3.0 * t, 0.0, 10.0)
    return (26.0, 3.0 * (t - 2.0), 10.0)


def _visible_a(t: float) -> bool:
    """A：目标被遮挡 0.6 s。"""
    return not (2.0 <= t < 2.6)


def _target_b(t: float) -> Vec3:
    """B：目标同向远离，9 m/s，快于 INTERCEPT 档 7.60 m/s。"""
    return (20.0 + 9.0 * t, 0.0, 10.0)


def _visible_b(t: float) -> bool:
    """B：目标丢失 1.5 s。"""
    return not (3.0 <= t < 4.5)


def _target_c(t: float) -> Vec3:
    """C：目标先向北 4 m/s，2.0 s 后改向东 4 m/s。"""
    if t < 2.0:
        return (30.0, 4.0 * t, 10.0)
    return (30.0 + 4.0 * (t - 2.0), 8.0, 10.0)


def _visible_c(t: float) -> bool:
    """C：目标在转弯期间被遮挡 0.8 s。"""
    return not (2.0 <= t < 2.8)


def _ego_lost_c(t: float) -> bool:
    """C：己方状态在 2.3～2.8 s 断流 0.5 s。"""
    return 2.3 <= t < 2.8


def _simulate(
    action: str,
    duration: float,
    target_fn: Callable[[float], Vec3],
    visible_fn: Callable[[float], bool],
    ego_lost_fn: Optional[Callable[[float], bool]] = None,
) -> list[Frame]:
    """按 20 Hz 推进一帧规划，己方朝发点移动。

    目标不可见时给 ``plan_drone`` 传 ``target=None``，降级由 P3 自己判断；
    己方断流时传 ``own=None``、``ego_stamp=None``。
    """
    buffer = TargetBuffer()
    runtime = DroneRuntime.create(BASELINE_PARAMS)
    own = OWN_START
    frames: list[Frame] = []

    total = int(round(duration / DT)) + 1
    for index in range(total):
        now = T_START + DT * index
        t = now - T_START
        truth = target_fn(t)
        visible = visible_fn(t)
        if visible:
            buffer.offer(now, truth, now)
        target = (
            buffer.predict(now, max_age=1.0, max_speed=15.0)
            if visible
            else None
        )

        ego_lost = bool(ego_lost_fn(t)) if ego_lost_fn is not None else False
        own_arg = None if ego_lost else own
        result = plan_drone(
            runtime,
            action,
            own_arg,
            target,
            None if ego_lost else now,
            now,
            BASELINE_PARAMS,
        )
        frames.append(
            Frame(
                t=round(t, 2),
                now=now,
                own=own_arg,
                own_truth=own,
                target=target,
                truth=truth,
                result=result,
                ego_lost=ego_lost,
            )
        )

        if result.point is not None and not ego_lost:
            own = _step_toward(own, result.point, BASELINE_PARAMS.v_max * DT)

    return frames


def _records(frames: list[Frame]) -> dict[float, Frame]:
    return {frame.t: frame for frame in frames}


def _add_to_chart(scenario_chart, frames: list[Frame], series: str) -> None:
    """把逐帧记录交给 fixture；图在用例结束后生成。"""
    for frame in frames:
        scenario_chart.add(
            frame.now,
            own=frame.own,
            target=frame.target.position if frame.target else None,
            truth=frame.truth,
            point=frame.result.point,
            level=frame.result.level,
            reason=frame.result.reason,
            series=series,
            own_truth=frame.own_truth,
        )


def test_multiframe_track_short_occlusion(measured, scenario_chart):
    """场景：A 场景改造。TRACK，目标东 3 m/s，2.0 s 转北 3 m/s，2.0～2.6 s 遮挡。

    预期：2.00 s 第 1 帧 NORMAL 不发点；2.05 s 进 L1 并继续发外推点；
    2.65 s 因目标回来进入恢复确认 HOLDING；2.75 s 回 NORMAL；
    遮挡期间不进入 L2（0.6 s < t_extrap_max=1.0 s）。
    """
    frames = _simulate("TRACK", 5.0, _target_a, _visible_a)
    records = _records(frames)

    scenario_chart.configure(
        headline="短遮挡 0.6 s：TRACK 进 L1 继续外推，2.65 s 恢复确认后回 NORMAL",
        subtitle="目标 (20,0,10) 东 3 m/s，2.0 s 转北 3 m/s，2.0～2.6 s 遮挡；"
                 "参数：固定基线 v_max=4.50 m/s、max_step=0.74 m/帧。",
    )
    scenario_chart.note(
        "参数：固定基线 v_max=4.50、max_step=0.74、max_pred_age=0.5、"
        "t_extrap_max=1.0；action=TRACK。\n"
        "拐点：100.00 目标丢失；100.05 进 L1；102.65 进 HOLDING；"
        "102.75 回 NORMAL。"
    )
    scenario_chart.band(
        T_START + 2.0, T_START + 2.6, "A 目标遮挡", "#F2D7C8"
    )
    for event_t, label in (
        (T_START + 2.0, "目标丢失"),
        (T_START + 2.05, "进 L1"),
        (T_START + 2.65, "进 HOLDING"),
        (T_START + 2.75, "回 NORMAL"),
    ):
        scenario_chart.event(event_t, label)
    _add_to_chart(scenario_chart, frames, "A_track")

    first_missing = records[2.0].result
    assert first_missing.level == "NORMAL"
    assert first_missing.reason == "target_missing"
    assert not should_publish(first_missing)

    extrapolating = records[2.05].result
    assert extrapolating.level == "EXTRAPOLATING"
    assert should_publish(extrapolating)
    # L1 点应领先己方，说明仍在按目标速度外推，而不是停在原地。
    assert extrapolating.point[0] > records[2.05].own[0]

    holding = records[2.65].result
    assert holding.level == "HOLDING"
    assert holding.reason == "hold_fixed_anchor"
    assert holding.point == pytest.approx(records[2.65].own)

    recovered = records[2.75].result
    assert recovered.level == "NORMAL"
    assert recovered.reason == "track_target_point"
    assert should_publish(recovered)
    # 目标已转向北，恢复后的发点 y 应跟随。
    assert recovered.point[1] > 0.0

    measured(
        f"2.05 s 进 L1，点 {tuple(round(v, 2) for v in extrapolating.point)}；"
        f"2.65 s HOLDING 锚点 {tuple(round(v, 2) for v in holding.point)}；"
        f"2.75 s 回 NORMAL，点 y={recovered.point[1]:.2f}"
    )


def test_multiframe_intercept_unreachable_tail(measured, scenario_chart):
    """场景：B 场景改造。INTERCEPT，目标 9 m/s 远离，3.0～4.5 s 丢失。

    预期：可见期因目标快于 v_max_intercept 持续返回尾随点
    （reason=fallback_tail，reachable=False）；3.00 s 目标丢失第 1 帧
    NORMAL 不发点；3.05 s 进 L1；4.00 s 进 L2 并固定锚点；
    4.50 s 目标回来，4.60 s 回 NORMAL 后仍发尾随点。
    """
    frames = _simulate("INTERCEPT", 6.0, _target_b, _visible_b)
    records = _records(frames)

    scenario_chart.configure(
        headline="追不上：INTERCEPT 持续发尾随点，目标丢失 1.5 s 进 L2 固定锚点",
        subtitle="目标 (20,0,10) 以 9 m/s 远离，v_max_intercept=7.60 m/s，"
                 "d_tail=4.0 m；3.0～4.5 s 丢失；max_step=0.74 m/帧。",
    )
    scenario_chart.note(
        "参数：固定基线 v_max_intercept=7.60、max_step=0.74、d_tail=4.0；"
        "目标 9.00 m/s，快于拦截档。\n"
        "拐点：100.00 velocity_unavailable；100.05 fallback_tail；"
        "103.00 目标丢失；103.05 进 L1；104.00 进 L2；"
        "104.60 回 NORMAL。"
    )
    scenario_chart.band(
        T_START + 3.0, T_START + 4.5, "B 目标丢失", "#F2D7C8"
    )
    for event_t, label in (
        (T_START + 3.0, "目标丢失"),
        (T_START + 3.05, "进 L1"),
        (T_START + 4.0, "进 L2"),
        (T_START + 4.5, "目标回来"),
        (T_START + 4.6, "回 NORMAL"),
    ):
        scenario_chart.event(event_t, label)
    _add_to_chart(scenario_chart, frames, "B_intercept")

    no_velocity = records[0.0].result
    assert no_velocity.reason == "velocity_unavailable"
    assert not should_publish(no_velocity)

    tail = records[0.05].result
    assert tail.reason == "fallback_tail"
    assert should_publish(tail)
    assert tail.reachable is False
    assert tail.t_go is None

    first_missing = records[3.0].result
    assert first_missing.level == "NORMAL"
    assert first_missing.reason == "target_missing"
    assert not should_publish(first_missing)

    extrapolating = records[3.05].result
    assert extrapolating.level == "EXTRAPOLATING"
    assert should_publish(extrapolating)

    holding = records[4.0].result
    assert holding.level == "HOLDING"
    assert holding.reason == "hold_fixed_anchor"
    assert holding.point == pytest.approx(records[4.0].own)
    # 锚点不跟着目标继续向东跑。
    assert holding.point[0] < records[4.0].truth[0]

    recovered = records[4.6].result
    assert recovered.level == "NORMAL"
    assert recovered.reason == "fallback_tail"
    assert should_publish(recovered)
    assert recovered.reachable is False

    measured(
        f"0.05 s 尾随点 {tuple(round(v, 2) for v in tail.point)}；"
        f"4.00 s L2 锚点 {tuple(round(v, 2) for v in holding.point)}，"
        f"目标已到 x={records[4.0].truth[0]:.2f}；"
        f"4.60 s 恢复尾随"
    )


def test_multiframe_attack_loss_dropout(measured, scenario_chart):
    """场景：C 场景改造。ATTACK，目标北 4 m/s 后转东，2.0～2.8 s 遮挡，
    2.3～2.8 s 己方断流。

    预期：0.00 s、2.00 s 出现 hold_attack_no_solution；2.05 s 进 L1；
    2.35 s 己方断流触发 L3（no_plan_ego_lost），L3 期间不发点；
    2.85 s 恢复 HOLDING 并固定锚点；2.95 s 回 NORMAL。
    """
    frames = _simulate("ATTACK", 7.0, _target_c, _visible_c, _ego_lost_c)
    records = _records(frames)

    scenario_chart.configure(
        headline="ATTACK 目标转向：遮挡 0.8 s 进 L1，己方断流 0.5 s 进 L3，"
                 "2.85 s 恢复悬停",
        subtitle="目标 (30,0,10) 北 4 m/s，2.0 s 转东 4 m/s；2.0～2.8 s 遮挡，"
                 "2.3～2.8 s 己方断流；v_max_attack=14.80 m/s，max_step=0.74 m/帧。",
        annotations=[
            {
                "panel": "B", "x": 31.0, "y": 4.0,
                "text": "目标：先向北 4 m/s，2 s 后转东 4 m/s",
                "text_xy": (31.5, 4.8), "color": "#D55E00",
                "arrow": False,
            },
            {
                "panel": "B", "x": 12.0, "y": 4.5,
                "text": "蓝线与橙点几乎重合：己方沿发点飞",
                "text_xy": (20.0, 1.0), "color": "#555555",
                "arrow": False,
            },
        ],
    )
    scenario_chart.note(
        "参数：固定基线 v_max_attack=14.80、max_step=0.74、max_pred_age=0.5；"
        "action=ATTACK。\n"
        "拐点：100.00 hold_attack_no_solution；100.05 拦截点；"
        "102.00 再保持；102.05 进 L1；102.35 进 L3；"
        "102.85 进 HOLDING；102.95 回 NORMAL。"
    )
    scenario_chart.band(
        T_START + 2.0, T_START + 2.8, "A 目标遮挡", "#F2D7C8"
    )
    scenario_chart.band(
        T_START + 2.3, T_START + 2.8, "B 己方断流", "#D9D9D9"
    )
    for event_t, label in (
        (T_START + 2.0, "目标遮挡"),
        (T_START + 2.05, "进 L1"),
        (T_START + 2.35, "进 L3"),
        (T_START + 2.8, "己方恢复"),
        (T_START + 2.85, "进 HOLDING"),
        (T_START + 2.95, "回 NORMAL"),
    ):
        scenario_chart.event(event_t, label)
    _add_to_chart(scenario_chart, frames, "C_attack")

    attack_hold = records[0.0].result
    assert attack_hold.reason == "hold_attack_no_solution"
    assert should_publish(attack_hold)
    assert attack_hold.action == "ATTACK"
    assert attack_hold.point == pytest.approx(OWN_START)

    assert records[0.05].result.reason == "intercept_point"

    second_hold = records[2.0].result
    assert second_hold.reason == "hold_attack_no_solution"
    assert should_publish(second_hold)

    extrapolating = records[2.05].result
    assert extrapolating.level == "EXTRAPOLATING"
    assert extrapolating.reason == "intercept_point"

    no_plan = records[2.35].result
    assert no_plan.level == "NO_PLAN"
    assert no_plan.reason == "no_plan_ego_lost"
    assert not should_publish(no_plan)
    for t in (2.35, 2.5, 2.8):
        assert not should_publish(records[t].result)

    holding = records[2.85].result
    assert holding.level == "HOLDING"
    assert holding.reason == "hold_fixed_anchor"
    assert holding.point == pytest.approx(records[2.85].own)

    recovered = records[2.95].result
    assert recovered.level == "NORMAL"
    assert recovered.reason == "intercept_point"

    measured(
        f"0.00 s ATTACK 保持点 {tuple(round(v, 2) for v in attack_hold.point)}；"
        f"2.35 s 进 L3 不发点；2.85 s HOLDING 锚点 "
        f"{tuple(round(v, 2) for v in holding.point)}；"
        f"2.95 s 回 NORMAL"
    )


def test_multiframe_level_coverage_summary(measured, scenario_chart):
    """场景：汇总 A/B/C 三条多帧用例的异常级别帧数。

    预期：三条用例都覆盖 L1；A/B 覆盖 L2；L3 只由 C 覆盖。
    """
    scenarios = [
        ("A 短遮挡", _simulate("TRACK", 5.0, _target_a, _visible_a)),
        ("B 追不上", _simulate("INTERCEPT", 6.0, _target_b, _visible_b)),
        (
            "C ATTACK",
            _simulate(
                "ATTACK", 7.0, _target_c, _visible_c, _ego_lost_c
            ),
        ),
    ]
    levels = ["EXTRAPOLATING", "HOLDING", "NO_PLAN"]
    counts = []
    for _name, frames in scenarios:
        counter = Counter(frame.result.level for frame in frames)
        counts.append([counter.get(level, 0) for level in levels])

    assert all(row[0] > 0 for row in counts)
    assert counts[0][1] > 0 and counts[1][1] > 0 and counts[2][1] > 0
    assert counts[0][2] == 0
    assert counts[1][2] == 0
    assert counts[2][2] > 0

    scenario_chart.configure(
        mode="bar",
        headline="多帧覆盖：L1/L2 由三条用例覆盖，L3 只由 C 覆盖",
        subtitle="只画异常级别帧数，NORMAL 未画；每条用例 20 Hz。",
        source="数据来源：test_multiframe_scenarios.py 的 A/B/C 三条用例",
        bar={
            "categories": [name for name, _ in scenarios],
            "ylabel": "异常级别帧数（20 Hz）",
            "series": [
                {
                    "label": "L1 外推",
                    "color": "#E69F00",
                    "values": [row[0] for row in counts],
                },
                {
                    "label": "L2 保持",
                    "color": "#7A7A7A",
                    "values": [row[1] for row in counts],
                },
                {
                    "label": "L3 不发点",
                    "color": "#B03A3A",
                    "values": [row[2] for row in counts],
                },
            ],
        },
    )

    measured(
        f"A L1/L2/L3={counts[0]}；B={counts[1]}；C={counts[2]}"
    )
