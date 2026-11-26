"""多帧 L3 兜底：目标坏、己方好时发当前位置；目标恢复后回到 NORMAL。

这条用例覆盖适配层与 FSM 的连续行为，并出证据图。单帧分支见
``test/unit/test_no_plan_hold.py``。
"""

from math import hypot, isnan, radians

import pytest

from algorithm_pkg.adapter_core import (
    DroneRuntime,
    plan_drone,
    should_publish,
)
from algorithm_pkg.algorithm_core import TargetState
from algorithm_pkg.plan_params import PlanParams

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
RUN_SECONDS = 5.5
INVALID_START = 2.0
INVALID_END = 3.0
TARGET_START_X = 20.0
TARGET_SPEED = 2.0
OWN = (0.0, 0.0, 5.0)


def _target(now: float, truth: tuple[float, float, float], invalid: bool) -> TargetState:
    """构造一帧目标；invalid=True 时速度塞 NaN，让 P3 判 INVALID。"""
    velocity = (float("nan"), 0.0, 0.0) if invalid else (TARGET_SPEED, 0.0, 0.0)
    return TargetState(
        position=truth,
        velocity=velocity,
        stamp=now,
        age=0.0,
        valid=True,
        velocity_valid=not invalid,
    )


def test_l3_target_invalid_holds_current_ego(measured, scenario_chart):
    """场景：TRACK；目标以 2 m/s 沿 +x 飞，2.0～3.0 s 速度变 NaN；己方在 (0,0,5) 不动。

    预期：2.0 s 第 1 帧异常仍 NORMAL、不发点；2.05 s 进 NO_PLAN，适配层
    发己方当前位置，reason=hold_no_plan_invalid；3.0 s 目标恢复后回
    HOLDING，3.2 s 前回 NORMAL；兜底期间不发上一帧目标点。
    """
    runtime = DroneRuntime.create(BASELINE_PARAMS)
    records = {}

    scenario_chart.configure(
        headline="L3 上一帧点兜底：目标速度变 NaN，己方有效，发固定保持点",
        subtitle="目标 2 m/s 沿 +x，2.0～3.0 s 速度 NaN；己方固定 (0,0,5)；"
                 "第 1 帧 NORMAL，第 2 帧 L3，3.0 s 恢复。",
    )
    scenario_chart.band(
        T_START + INVALID_START,
        T_START + INVALID_END,
        "目标速度 NaN",
        "#F2D7C8",
    )
    scenario_chart.note(
        "参数：固定基线 v_max=4.50、max_step=0.74；action=TRACK；"
        "己方固定 (0,0,5)。\n"
        "2.00 s 目标速度 NaN，第 1 帧仍 NORMAL；2.05 s 进 L3，"
        "适配层发己方当前位置；3.00 s 目标恢复，先 HOLDING 再 NORMAL。"
    )
    for event_t, label in (
        (T_START + 2.0, "目标变 NaN"),
        (T_START + 2.05, "进 L3 兜底"),
        (T_START + 3.0, "目标恢复"),
    ):
        scenario_chart.event(event_t, label)

    total_frames = int(round(RUN_SECONDS / DT)) + 1
    for index in range(total_frames):
        now = T_START + DT * index
        t = now - T_START
        truth = (TARGET_START_X + TARGET_SPEED * t, 0.0, 5.0)
        invalid = INVALID_START <= t < INVALID_END
        target = _target(now, truth, invalid)

        result = plan_drone(
            runtime,
            "TRACK",
            OWN,
            target,
            ego_stamp=now,
            now=now,
            params=BASELINE_PARAMS,
        )
        records[round(t, 2)] = (result, truth)

        scenario_chart.add(
            now,
            own=OWN,
            target=target.position,
            truth=truth,
            point=result.point,
            level=result.level,
            reason=result.reason,
            series="no_plan_hold",
        )

    before = records[1.0][0]
    assert before.level == "NORMAL"
    assert should_publish(before)

    first_bad = records[2.0][0]
    assert first_bad.level == "NORMAL"
    assert not should_publish(first_bad)

    fallback = records[2.05][0]
    assert fallback.level == "NO_PLAN"
    assert fallback.reason == "hold_no_plan_invalid"
    assert fallback.point == pytest.approx(OWN)
    assert fallback.source_stamp == pytest.approx(102.05)
    assert should_publish(fallback)

    during = records[2.5][0]
    assert during.level == "NO_PLAN"
    assert during.point == pytest.approx(OWN)
    assert not any(
        isnan(component) for component in during.point
    )
    assert hypot(
        during.point[0] - records[2.5][1][0],
        during.point[1] - records[2.5][1][1],
        during.point[2] - records[2.5][1][2],
    ) > 1.0

    assert records[3.0][0].level == "HOLDING"
    assert records[3.2][0].level == "NORMAL"
    assert should_publish(records[3.2][0])

    fallback_frames = sum(
        1
        for t, (result, _) in records.items()
        if 2.05 <= t < 3.0 and result.reason == "hold_no_plan_invalid"
    )
    assert fallback_frames >= 15

    measured(
        f"2.00 s 第 1 帧 {first_bad.level}/{first_bad.reason} 不发点；"
        f"2.05 s 进 L3，兜底 {fallback_frames} 帧发己方位置；"
        f"3.00 s 恢复 {records[3.0][0].level}，3.20 s {records[3.2][0].level}"
    )
