"""适配层纯逻辑测试（不启动 ROS）。

覆盖 ``algorithm_pkg.adapter_core``：时间戳门控、目标缓冲、目标源选择、
接口 C 载荷、ATTACK 无解保持点、己方缺失/时间戳倒退的降级行为。

运行：

    PYTHONPATH=src/algorithm_pkg python3 -m pytest -q \
        src/algorithm_pkg/test/test_node_adapter.py
"""

from __future__ import annotations

import pytest

from algorithm_pkg.adapter_core import (
    ATTACK_HOLD_REASON,
    STAMP_FUTURE,
    STAMP_NON_FINITE,
    STAMP_REVERSED,
    STAMP_ZERO,
    POINT_NON_FINITE,
    DroneRuntime,
    TargetBuffer,
    choose_target,
    level_transition,
    payload,
    plan_drone,
    should_publish,
    stamp_problem,
)
from algorithm_pkg.algorithm_core import TargetState
from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.plan_types import PlanResult


def _params(**overrides) -> PlanParams:
    """测试用参数：默认单帧跳变 0.30m，与 launch 默认一致。"""
    values = {"max_step": 0.30}
    values.update(overrides)
    return PlanParams(**values)


def _target(
    now: float,
    position: tuple[float, float, float] = (20.0, 0.0, 10.0),
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> TargetState:
    """构造一个“刚采到、速度已估出”的目标状态。"""
    return TargetState(
        position=position,
        velocity=velocity,
        stamp=now,
        age=0.0,
        valid=True,
        velocity_valid=True,
    )


def test_stamp_problem_classifies_zero_future_reversed(measured):
    """场景：目标帧时间戳为 0、超前、倒退、正常。

    预期：分别返回 stamp_zero / stamp_future / stamp_reversed / None。
    """
    now = 10.0
    actual = [
        stamp_problem(0.0, now, None),
        stamp_problem(11.0, now, None),
        stamp_problem(9.0, now, 9.5),
        stamp_problem(9.5, now, 9.0),
    ]
    measured(
        f"zero={actual[0]}, future={actual[1]}, "
        f"reversed={actual[2]}, ok={actual[3]}"
    )
    assert actual == [STAMP_ZERO, STAMP_FUTURE, STAMP_REVERSED, None]


def test_stamp_problem_rejects_non_finite(measured):
    """场景：时间戳为 NaN。

    预期：返回 stamp_non_finite，不进入预测。
    """
    actual = stamp_problem(float("nan"), 10.0)
    measured(f"problem={actual}")
    assert actual == STAMP_NON_FINITE


def test_target_buffer_drops_reversed_without_moving_last_stamp(measured):
    """场景：先接受 stamp=10.0，再来一帧 stamp=9.5。

    预期：倒退帧被丢弃，last_stamp 仍是 10.0，采样数不增加。
    """
    buffer = TargetBuffer(history_sec=3.0)
    accepted = buffer.offer(10.0, (20.0, 0.0, 10.0), now=10.0)
    dropped = buffer.offer(9.5, (19.0, 0.0, 10.0), now=10.0)
    measured(
        f"accepted={accepted}, dropped={dropped}, "
        f"last_stamp={buffer.last_stamp}, samples={len(buffer.samples)}"
    )
    assert accepted is None
    assert dropped == STAMP_REVERSED
    assert buffer.last_stamp == 10.0
    assert len(buffer.samples) == 1


def test_target_buffer_rejects_future_and_zero(measured):
    """场景：来帧时间戳为 11.0（now=10）和 0。

    预期：future/zero 都丢弃，缓冲保持为空。
    """
    buffer = TargetBuffer()
    future = buffer.offer(11.0, (20.0, 0.0, 10.0), now=10.0)
    zero = buffer.offer(0.0, (20.0, 0.0, 10.0), now=10.0)
    measured(
        f"future={future}, zero={zero}, samples={len(buffer.samples)}"
    )
    assert future == STAMP_FUTURE
    assert zero == STAMP_ZERO
    assert len(buffer.samples) == 0


def test_target_buffer_rejects_non_finite_point(measured):
    """场景：时间戳合法但坐标含 NaN。

    预期：返回 point_non_finite，不写入采样。
    """
    buffer = TargetBuffer()
    problem = buffer.offer(10.0, (float("nan"), 0.0, 10.0), now=10.0)
    measured(f"problem={problem}, samples={len(buffer.samples)}")
    assert problem == POINT_NON_FINITE
    assert len(buffer.samples) == 0


def test_choose_target_prefers_fresh_vision(measured):
    """场景：视觉与外部源都新鲜。

    预期：优先用视觉的预测点。
    """
    vision = TargetBuffer()
    external = TargetBuffer()
    vision.offer(10.0, (20.0, 0.0, 10.0), now=10.0)
    external.offer(10.0, (30.0, 0.0, 10.0), now=10.0)
    state = choose_target(
        vision,
        external,
        now=10.0,
        target_timeout=0.5,
        max_age=1.0,
        max_speed=15.0,
    )
    measured(f"target_position={state.position if state else None}")
    assert state is not None
    assert state.position[0] == pytest.approx(20.0)


def test_choose_target_falls_back_to_external(measured):
    """场景：视觉最后一帧是 1.0s 前，外部源刚更新。

    预期：视觉超过 target_timeout=0.5s，改用外部目标。
    """
    vision = TargetBuffer()
    external = TargetBuffer()
    vision.offer(9.0, (20.0, 0.0, 10.0), now=9.0)
    external.offer(10.0, (30.0, 0.0, 10.0), now=10.0)
    state = choose_target(
        vision,
        external,
        now=10.0,
        target_timeout=0.5,
        max_age=1.0,
        max_speed=15.0,
    )
    measured(f"target_position={state.position if state else None}")
    assert state is not None
    assert state.position[0] == pytest.approx(30.0)


def test_payload_uses_publish_time_stamp(measured):
    """场景：目标 stamp=100.0，发布时刻 now=123.456。

    预期：接口 C 的 stamp 是 123.456，不是目标时间戳。
    """
    result = PlanResult(
        action="TRACK",
        raw_action="TRACK",
        point=(1.0, 2.0, 3.0),
        reachable=True,
        t_go=None,
        reason="track_target_point",
    )
    data = payload("drone_1", 123.456, result)
    measured(f"payload={data}")
    assert data["stamp"] == pytest.approx(123.456)
    assert data["x"] == pytest.approx(1.0)
    assert data["drone_id"] == "drone_1"


def test_payload_rejects_missing_point(measured):
    """场景：PlanResult 没有点。

    预期：payload 抛 ValueError，避免发布空坐标。
    """
    result = PlanResult(
        action="HOVER",
        raw_action="HOVER",
        point=None,
        reachable=False,
        t_go=None,
        reason="hover_center_generated",
    )
    with pytest.raises(ValueError):
        payload("drone_1", 1.0, result)
    measured("payload raised ValueError")


def test_attack_no_solution_publishes_hold_point(measured):
    """场景：ATTACK 目标以 30 m/s 远离，己方 12.4 m/s 追不上。

    预期：P2 返回 attack_no_solution；适配层补己方当前位置保持点，
    reason=hold_attack_no_solution，action 仍是 ATTACK。
    """
    runtime = DroneRuntime.create(_params())
    target = _target(
        now=10.0,
        position=(100.0, 0.0, 10.0),
        velocity=(30.0, 0.0, 0.0),
    )
    result = plan_drone(
        runtime,
        "ATTACK",
        own=(0.0, 0.0, 10.0),
        target=target,
        ego_stamp=10.0,
        now=10.0,
        params=_params(),
    )
    measured(
        f"point={result.point}, reason={result.reason}, level={result.level}"
    )
    assert result.point == pytest.approx((0.0, 0.0, 10.0))
    assert result.reason == ATTACK_HOLD_REASON
    assert result.action == "ATTACK"


def test_missing_ego_goes_no_plan_and_does_not_publish(measured):
    """场景：连续两帧没有己方状态，目标是新鲜的。

    预期：第 2 帧进入 NO_PLAN（no_plan_ego_lost），point=None。
    """
    runtime = DroneRuntime.create(_params())
    target = _target(now=10.0)
    first = plan_drone(
        runtime, "TRACK", None, target, None, 10.0, _params()
    )
    second = plan_drone(
        runtime, "TRACK", None, target, None, 10.05, _params()
    )
    measured(
        f"first_level={first.level}, second_level={second.level}, "
        f"second_reason={second.reason}, publish={should_publish(second)}"
    )
    assert second.level == "NO_PLAN"
    assert second.reason == "no_plan_ego_lost"
    assert not should_publish(second)


def test_ego_time_reversed_frame_is_discarded(measured):
    """场景：己方状态 stamp 先 10.0，下一帧倒退到 9.9。

    预期：P3 返回 frame_discarded_time_reversed，不发点，级别不变。
    """
    runtime = DroneRuntime.create(_params())
    target = _target(now=10.0)
    first = plan_drone(
        runtime,
        "TRACK",
        (0.0, 0.0, 10.0),
        target,
        10.0,
        10.0,
        _params(),
    )
    second = plan_drone(
        runtime,
        "TRACK",
        (0.0, 0.0, 10.0),
        target,
        9.9,
        10.1,
        _params(),
    )
    measured(
        f"first_level={first.level}, second_reason={second.reason}, "
        f"second_level={second.level}, publish={should_publish(second)}"
    )
    assert second.reason == "frame_discarded_time_reversed"
    assert second.level == first.level
    assert not should_publish(second)


def test_level_transition_reports_only_on_change(measured):
    """场景：级别从 NORMAL 变到 HOLDING，再保持 HOLDING。

    预期：第一次返回带 reason 的迁移串，第二次返回 None。
    """
    holding = PlanResult(
        action="TRACK",
        raw_action="TRACK",
        point=(0.0, 0.0, 10.0),
        reachable=False,
        t_go=None,
        reason="hold_last_valid",
        level="HOLDING",
    )
    changed = level_transition("NORMAL", holding)
    unchanged = level_transition("HOLDING", holding)
    measured(f"changed={changed}, unchanged={unchanged}")
    assert changed is not None and "NORMAL->HOLDING" in changed
    assert "hold_last_valid" in changed
    assert unchanged is None
