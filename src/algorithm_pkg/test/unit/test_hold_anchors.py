"""L2/L3/HOVER 固定锚点：进入时捕获一次，之后不跟当前位置。"""

import pytest

from algorithm_pkg.adapter_core import DroneRuntime, plan_drone, should_publish
from algorithm_pkg.algorithm_core import TargetState
from algorithm_pkg.plan_params import PlanParams

PARAMS = PlanParams()


def _fresh_target(now: float, position=(30.0, 0.0, 5.0)) -> TargetState:
    return TargetState(
        position=position,
        velocity=(0.0, 0.0, 0.0),
        stamp=now,
        age=0.0,
        valid=True,
        velocity_valid=True,
    )


def _invalid_target(now: float) -> TargetState:
    return TargetState(
        position=(30.0, 0.0, 5.0),
        velocity=(float("nan"), 0.0, 0.0),
        stamp=now,
        age=0.0,
        valid=True,
        velocity_valid=False,
    )


def test_l2_anchor_holds_entry_position(measured):
    """场景：TRACK 没有目标，进 L2；己方位置从 (10,20,5) 漂到 (14,20,5)。

    预期：L2 第一帧捕获 (10,20,5)，之后发点仍是它；reason=hold_fixed_anchor，
    stamp 用当前己方时刻。
    """
    runtime = DroneRuntime.create(PARAMS)
    own = (10.0, 20.0, 5.0)

    first = plan_drone(runtime, "TRACK", own, None, 100.0, 100.0, PARAMS)
    assert first.level == "NORMAL"
    assert not should_publish(first)

    second = plan_drone(runtime, "TRACK", own, None, 100.05, 100.05, PARAMS)
    assert second.level == "HOLDING"
    assert second.point == pytest.approx(own)
    assert second.reason == "hold_fixed_anchor"

    moved = (14.0, 20.0, 5.0)
    third = plan_drone(runtime, "TRACK", moved, None, 100.10, 100.10, PARAMS)
    assert third.point == pytest.approx(own)
    assert third.source_stamp == pytest.approx(100.10)

    measured(
        f"L2 锚点 {second.point}；己方漂到 {moved} 后发点仍是 {third.point}"
    )


def test_l3_anchor_holds_first_fallback_position(measured):
    """场景：TRACK 目标速度 NaN；己方从 (10,20,5) 漂到 (14,20,5)。

    预期：第 2 帧进 L3，锚点取 (10,20,5)；第 3 帧己方已移动，发点仍是
    (10,20,5)，reason=hold_no_plan_invalid。
    """
    runtime = DroneRuntime.create(PARAMS)
    own = (10.0, 20.0, 5.0)

    plan_drone(runtime, "TRACK", own, _invalid_target(100.0), 100.0, 100.0, PARAMS)
    fallback = plan_drone(
        runtime, "TRACK", own, _invalid_target(100.05), 100.05, 100.05, PARAMS
    )
    assert fallback.level == "NO_PLAN"
    assert fallback.reason == "hold_no_plan_invalid"
    assert fallback.point == pytest.approx(own)

    moved = (14.0, 20.0, 5.0)
    later = plan_drone(
        runtime, "TRACK", moved, _invalid_target(100.10), 100.10, 100.10, PARAMS
    )
    assert later.point == pytest.approx(own)
    assert later.source_stamp == pytest.approx(100.10)

    measured(f"L3 锚点 {fallback.point}；己方漂到 {moved} 后发点 {later.point}")


def test_hover_anchor_is_cleared_and_recaptured(measured):
    """场景：HOVER 捕获 (10,20,5)，己方漂到 (14,20,5)；切 TRACK 再切回 HOVER。

    预期：HOVER 期间发点固定 (10,20,5)；切 TRACK 清空；再进 HOVER 捕获
    当前 (20,20,5)。
    """
    runtime = DroneRuntime.create(PARAMS)

    first = plan_drone(runtime, "HOVER", (10.0, 20.0, 5.0), None, 100.0, 100.0, PARAMS)
    second = plan_drone(runtime, "HOVER", (14.0, 20.0, 5.0), None, 100.05, 100.05, PARAMS)
    assert first.point == pytest.approx((10.0, 20.0, 5.0))
    assert second.point == pytest.approx((10.0, 20.0, 5.0))

    track = plan_drone(
        runtime,
        "TRACK",
        (14.0, 20.0, 5.0),
        _fresh_target(100.10, (30.0, 0.0, 5.0)),
        100.10,
        100.10,
        PARAMS,
    )
    assert track.level == "NORMAL"
    assert runtime.hover_anchor is None

    rehover = plan_drone(
        runtime, "HOVER", (20.0, 20.0, 5.0), None, 100.15, 100.15, PARAMS
    )
    assert rehover.point == pytest.approx((20.0, 20.0, 5.0))

    measured("HOVER 锚点 10 -> 切 TRACK 清空 -> 再 HOVER 锚点 20")


def test_hover_anchor_clamps_altitude_to_envelope():
    """场景：己方短暂低于 3 m，HOVER 也要捕获固定水平点，高度抬到 3 m。"""
    runtime = DroneRuntime.create(PARAMS)
    low = (10.0, 20.0, 1.0)

    first = plan_drone(runtime, "HOVER", low, None, 100.0, 100.0, PARAMS)
    assert first.point == pytest.approx((10.0, 20.0, 3.0))

    moved = (14.0, 20.0, 4.0)
    second = plan_drone(runtime, "HOVER", moved, None, 100.05, 100.05, PARAMS)
    assert second.point == pytest.approx((10.0, 20.0, 3.0))


def test_hover_anchor_clamps_above_max_altitude():
    """场景：己方高于 50 m，HOVER 锚点收到 max_alt，不跟着继续爬。"""
    runtime = DroneRuntime.create(PARAMS)
    high = (10.0, 20.0, 60.0)

    first = plan_drone(runtime, "HOVER", high, None, 100.0, 100.0, PARAMS)
    assert first.point == pytest.approx((10.0, 20.0, 50.0))

    second = plan_drone(runtime, "HOVER", (14.0, 20.0, 65.0), None, 100.05, 100.05, PARAMS)
    assert second.point == pytest.approx((10.0, 20.0, 50.0))
