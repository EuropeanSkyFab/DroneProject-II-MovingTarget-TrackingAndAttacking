"""L3 兜底：目标坏、己方新鲜时发当前位置；己方失效时不发。

多帧图见 ``test/sim/test_l3_hold_fallback.py``；本文件只覆盖适配层分支。
"""

import pytest

from algorithm_pkg.adapter_core import (
    DroneRuntime,
    payload,
    plan_drone,
    should_publish,
)
from algorithm_pkg.algorithm_core import TargetState
from algorithm_pkg.plan_params import PlanParams

PARAMS = PlanParams()
OWN = (10.0, 20.0, 5.0)


def _invalid_target(now: float) -> TargetState:
    """位置正常、速度 NaN：P3 判 INVALID，进 L3。"""
    return TargetState(
        position=(30.0, 30.0, 5.0),
        velocity=(float("nan"), 0.0, 0.0),
        stamp=now,
        age=0.0,
        valid=True,
        velocity_valid=False,
    )


def test_l3_target_invalid_holds_current_ego(measured):
    """场景：TRACK，目标速度 NaN 连续 2 帧，己方位置 (10,20,5) 持续有效。

    预期：第 1 帧仍 NORMAL、不发点；第 2 帧进 NO_PLAN，适配层补己方当前
    位置，reason=hold_no_plan_invalid，stamp 用当前己方时刻，仍发布。
    """
    runtime = DroneRuntime.create(PARAMS)
    results = []
    for index in range(2):
        now = 100.0 + 0.05 * index
        results.append(
            plan_drone(
                runtime,
                "TRACK",
                OWN,
                _invalid_target(now),
                ego_stamp=now,
                now=now,
                params=PARAMS,
            )
        )

    first, second = results
    assert first.level == "NORMAL"
    assert first.point is None
    assert not should_publish(first)

    assert second.level == "NO_PLAN"
    assert second.reason == "hold_no_plan_invalid"
    assert second.point == pytest.approx(OWN)
    assert second.anomaly == "INVALID"
    assert second.source_stamp == pytest.approx(100.05)
    assert should_publish(second)

    data = payload("drone_1", 100.05, second)
    assert (data["x"], data["y"], data["z"]) == pytest.approx(OWN)
    assert data["stamp"] == pytest.approx(100.05)

    measured(
        f"第 1 帧 {first.level}/{first.reason} 不发点；"
        f"第 2 帧 {second.level}/{second.reason} 发己方位置，"
        f"stamp={data['stamp']:.2f}"
    )


def test_l3_ego_lost_does_not_publish_hold(measured):
    """场景：TRACK，目标速度 NaN，己方状态同时缺失。

    预期：第 2 帧进 NO_PLAN，但己方不新鲜，适配层不发保持点；
    reason=no_plan_ego_lost，point=None。
    """
    runtime = DroneRuntime.create(PARAMS)
    result = None
    for index in range(2):
        now = 100.0 + 0.05 * index
        result = plan_drone(
            runtime,
            "TRACK",
            None,
            _invalid_target(now),
            ego_stamp=None,
            now=now,
            params=PARAMS,
        )

    assert result.level == "NO_PLAN"
    assert result.reason == "no_plan_ego_lost"
    assert result.point is None
    assert not should_publish(result)

    measured(f"己方缺失时 {result.level}/{result.reason}，point=None")
