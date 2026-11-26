"""视觉延迟门禁：0.5 s 边界、0.55 s 进 L1、1.5 s 进 L2。

固定延迟保留源时间戳，等价于目标流晚 0.5 s 到达。用例按 20 Hz 跑完整
ROS 链路，出证据图；不接 Gazebo，也不改正式节点。
"""

from collections import Counter
from math import hypot, radians

import pytest

from algorithm_pkg.adapter_core import (
    DroneRuntime,
    TargetBuffer,
    plan_drone,
)
from algorithm_pkg.plan_params import PlanParams
from support.vision_mock import MockVision

BASELINE_PARAMS = PlanParams(
    v_max=4.50,
    v_max_intercept=7.60,
    v_max_attack=14.80,
    a_max=9.79,
    yaw_rate_max=radians(172.4),
    max_step=0.74,
    max_pred_age=0.45,
)

DT = 0.05
T_START = 100.0
RUN_SECONDS = 5.0
WARMUP = 0.4
TARGET_START_X = 20.0
TARGET_SPEED = 2.0
OWN = (0.0, 0.0, 5.0)


def _truth(t: float) -> tuple[float, float, float]:
    return (
        TARGET_START_X + TARGET_SPEED * (t - T_START),
        0.0,
        5.0,
    )


def _error(position, truth) -> float:
    return hypot(
        position[0] - truth[0],
        position[1] - truth[1],
        position[2] - truth[2],
    )


def _run_case(delay: float, scenario_chart) -> list[dict]:
    """跑一段固定延迟的目标流，返回预热后的逐帧记录。"""
    mock = MockVision(delay=delay)
    buffer = TargetBuffer(history_sec=3.0)
    runtime = DroneRuntime.create(BASELINE_PARAMS)
    records: list[dict] = []

    scenario_chart.title(f"视觉延迟门禁：{delay:.2f} s")
    if delay > 1.0:
        headline = f"视觉延迟 {delay:.2f} s：超过 t_extrap_max 1.0 s，进 L2 固定锚点"
    elif delay >= 0.5:
        headline = f"视觉延迟 {delay:.2f} s：超过预测门限，进 L1 继续外推，不降 L2"
    else:
        headline = f"视觉延迟 {delay:.2f} s：仍在门限内，保持 NORMAL"
    scenario_chart.configure(
        headline=headline,
        subtitle="目标流固定延迟，保留源头 stamp；预热 0.4 s 后统计；"
                 "action=TRACK，20 Hz。",
    )
    scenario_chart.band(
        T_START,
        T_START + RUN_SECONDS,
        f"目标流延迟 {delay:.2f} s",
        "#E8E8E8",
    )
    scenario_chart.note(
        "参数：固定基线 v_max=4.50、max_step=0.74、max_pred_age=0.5、"
        "t_extrap_max=1.0；action=TRACK。\n"
        "延迟保留源头 stamp；预热 0.4 s 后才计入统计。"
    )

    prefill = int(round(delay / DT)) + 3
    total = int(round(RUN_SECONDS / DT)) + 1
    transitions: dict[str, float] = {}
    for index in range(-prefill, total):
        now = T_START + DT * index
        sample = mock.sample(now, _truth)
        if sample is not None:
            stamp, point = sample
            buffer.offer(stamp, point, now)

        if now < T_START:
            continue

        truth = _truth(now)
        target = buffer.predict(now, max_age=1.0, max_speed=15.0)
        result = plan_drone(
            runtime,
            "TRACK",
            OWN,
            target,
            ego_stamp=now,
            now=now,
            params=BASELINE_PARAMS,
        )
        if result.level not in transitions:
            transitions[result.level] = now - T_START

        scenario_chart.add(
            now,
            own=OWN,
            target=target.position if target is not None else None,
            truth=truth,
            point=result.point,
            level=result.level,
            reason=result.reason,
            series=f"delay_{delay:.2f}s",
        )
        records.append(
            {
                "t": now - T_START,
                "level": result.level,
                "reason": result.reason,
                "point": result.point,
                "target": target,
                "truth": truth,
            }
        )

    for level, t in transitions.items():
        scenario_chart.event(T_START + t, f"进 {level}")
    return records


def _post_warmup(records: list[dict]) -> list[dict]:
    return [record for record in records if record["t"] >= WARMUP]


def _max_error(records: list[dict]) -> float:
    errors = [
        _error(record["target"].position, record["truth"])
        for record in records
        if record["target"] is not None
    ]
    assert errors, "目标流没有产生任何预测结果"
    return max(errors)


def test_vision_delay_050s_over_threshold(measured, scenario_chart):
    """场景：目标流固定晚 0.5 s 到达，生产阈值 max_pred_age=0.45，TRACK。

    预期：0.5 s 已超过 0.45 s 门限，进 L1 外推；不出现 HOLDING/NO_PLAN；
    每帧都有点；预测误差 <=1 m。
    """
    records = _post_warmup(_run_case(0.5, scenario_chart))
    levels = {record["level"] for record in records}

    assert "EXTRAPOLATING" in levels
    assert "HOLDING" not in levels
    assert "NO_PLAN" not in levels
    assert records[-1]["level"] == "EXTRAPOLATING"
    assert all(record["point"] is not None for record in records)
    assert _max_error(records) <= 1.0

    measured(
        f"0.50 s 延迟（阈值 0.45）：级别 {dict(Counter(r['level'] for r in records))}，"
        f"最大预测误差 {_max_error(records):.3f} m"
    )


def test_vision_delay_055s_enters_l1(measured, scenario_chart):
    """场景：目标流固定晚 0.55 s 到达，20 Hz 跑 5 s，TRACK。

    预期：超过 max_pred_age 0.5 s，连续 2 帧后进 L1 外推；不出现 L2；
    外推点仍发布，预测误差 <=1 m。
    """
    records = _post_warmup(_run_case(0.55, scenario_chart))
    levels = {record["level"] for record in records}

    assert "EXTRAPOLATING" in levels
    assert "HOLDING" not in levels
    assert records[-1]["level"] == "EXTRAPOLATING"
    assert all(record["point"] is not None for record in records)
    assert _max_error(records) <= 1.0

    measured(
        f"0.55 s 延迟：级别 {dict(Counter(r['level'] for r in records))}，"
        f"最大预测误差 {_max_error(records):.3f} m"
    )


def test_vision_delay_150s_enters_l2(measured, scenario_chart):
    """场景：目标流固定晚 1.5 s 到达，20 Hz 跑 5 s，TRACK。

    预期：超过 t_extrap_max 1.0 s，进 L2 保持；保持点不再跟目标，
    取己方当前位置；不出现 NO_PLAN。
    """
    records = _post_warmup(_run_case(1.5, scenario_chart))
    levels = {record["level"] for record in records}

    assert "HOLDING" in levels
    assert "NO_PLAN" not in levels
    assert records[-1]["level"] == "HOLDING"
    assert records[-1]["point"] == pytest.approx(OWN)
    assert all(record["point"] is not None for record in records)

    measured(
        f"1.50 s 延迟：级别 {dict(Counter(r['level'] for r in records))}，"
        f"末帧保持点 {records[-1]['point']}，距目标真值 "
        f"{_error(records[-1]['point'], records[-1]['truth']):.2f} m"
    )
