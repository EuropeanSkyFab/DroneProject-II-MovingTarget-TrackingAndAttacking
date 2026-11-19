"""P2 回放测试：连续切换动作时，不能把上一模式的点带过来。

P2 的 plan() 是无状态纯函数；这里仍按“时间顺序”连续调用，
模拟中枢下发指令的方式，避免模块内引入缓存后产生状态泄漏。

运行：
    PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
        src/algorithm_pkg/test
"""

import pytest

from algorithm_pkg.algorithm_core import TargetState
from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.plan_policy import plan


def make_target(position=(10.0, 0.0, 10.0), velocity=(1.0, 0.0, 0.0)):
    """构造回放用的有效目标。"""
    return TargetState(
        position=position,
        velocity=velocity,
        stamp=1.0,
        age=0.0,
        valid=True,
        velocity_valid=True,
    )


def test_full_mode_sequence_keeps_each_result_independent(measured):
    """场景：HOVER → TRACK → INTERCEPT → ATTACK → TRACK → HOVER。

    操作：按顺序连续调用 plan()。
    输入：同一个有效目标和己方位置。
    预期：HOVER 发当前保持点；TRACK/INTERCEPT 发点；
          ATTACK 这里可达，也发点。
    """
    target = make_target()
    own = (0.0, 0.0, 10.0)
    actions = ["HOVER", "TRACK", "INTERCEPT", "ATTACK", "TRACK", "HOVER"]
    results = [plan(action, own, target) for action in actions]

    assert results[0].should_publish
    assert results[0].reason == "hover_hold"
    assert results[1].should_publish
    assert results[1].reason == "track_target_point"
    assert results[2].should_publish
    assert results[2].reason == "intercept_point"
    assert results[3].should_publish
    assert results[3].reason == "intercept_point"
    assert results[4].should_publish
    assert results[4].reason == "track_target_point"
    assert results[5].should_publish
    assert results[5].reason == "hover_hold"

    measured(
        " -> ".join(
            f"{result.action}:{result.reason}:"
            f"{'publish' if result.should_publish else 'no'}"
            for result in results
        )
    )


def test_unknown_action_between_valid_actions_does_not_break_next(measured):
    """场景：TRACK 后插一个拼错的动作，再回到 TRACK。

    操作：连续调用 plan("TRACK")、plan("TRACKK")、plan("TRACK")。
    输入：同一个有效目标。
    预期：中间被拒绝；前后两个 TRACK 都正常发点。
    """
    target = make_target()
    own = (0.0, 0.0, 10.0)

    first = plan("TRACK", own, target)
    bad = plan("TRACKK", own, target)
    second = plan("TRACK", own, target)

    assert first.should_publish
    assert bad.point is None
    assert bad.reason == "unknown_action"
    assert second.should_publish
    assert second.point == pytest.approx(first.point)

    measured(
        f"first={first.should_publish}, bad={bad.reason}, "
        f"second={second.should_publish}"
    )


def test_attack_failure_then_track_does_not_leak_tail(measured):
    """场景：ATTACK 追不上，随后切 TRACK。

    操作：先 ATTACK（攻击速度 6），再 TRACK。
    输入：目标 (10,0,10)，速度 (8,0,0)。
    预期：ATTACK 不发点；TRACK 发目标点 (10,0,10)，
          不能把 INTERCEPT 风格的尾随点 (6,0,10) 带过来。
    """
    target = make_target(velocity=(8.0, 0.0, 0.0))
    own = (0.0, 0.0, 10.0)
    params = PlanParams(v_max_attack=6.0)

    attack = plan("ATTACK", own, target, params)
    track = plan("TRACK", own, target, params)

    assert attack.point is None
    assert attack.reason == "attack_no_solution"
    assert track.point == pytest.approx((10.0, 0.0, 10.0))
    assert track.reason == "track_target_point"

    measured(
        f"ATTACK={attack.reason}, TRACK={track.reason}, "
        f"point={track.point}"
    )


def test_intercept_tail_then_attack_then_track(measured):
    """场景：INTERCEPT 追不上 → ATTACK 追不上 → TRACK。

    操作：连续调用三个动作。
    输入：目标 (10,0,10)，速度 (8,0,0)，拦截/攻击速度都压到 6。
    预期：INTERCEPT 发尾随点 (6,0,10)；
          ATTACK 不发点；TRACK 发目标点 (10,0,10)。
    """
    target = make_target(velocity=(8.0, 0.0, 0.0))
    own = (0.0, 0.0, 10.0)
    params = PlanParams(v_max_intercept=6.0, v_max_attack=6.0)

    intercept = plan("INTERCEPT", own, target, params)
    attack = plan("ATTACK", own, target, params)
    track = plan("TRACK", own, target, params)

    assert intercept.point == pytest.approx((6.0, 0.0, 10.0))
    assert intercept.reason == "fallback_tail"
    assert attack.point is None
    assert attack.reason == "attack_no_solution"
    assert track.point == pytest.approx((10.0, 0.0, 10.0))

    measured(
        f"INTERCEPT={intercept.reason}@{intercept.point}, "
        f"ATTACK={attack.reason}, TRACK={track.reason}@{track.point}"
    )


def test_rapid_flapping_does_not_reuse_previous_point(measured):
    """场景：战术指令快速抖动 TRACK/INTERCEPT/ATTACK/TRACK/ATTACK/TRACK。

    操作：每次都用同一目标连续调用。
    输入：目标 (10,0,10)，速度 (1,0,0)。
    预期：每次结果只由本次 action 决定；TRACK 始终是目标点，
          INTERCEPT/ATTACK 是拦截点，不出现上一帧缓存。
    """
    target = make_target()
    own = (0.0, 0.0, 10.0)
    actions = ["TRACK", "INTERCEPT", "ATTACK", "TRACK", "ATTACK", "TRACK"]
    results = [plan(action, own, target) for action in actions]

    for result in results:
        assert result.should_publish
        if result.action == "TRACK":
            assert result.point == pytest.approx(target.position)
        else:
            assert result.reason in ("intercept_point", "attack_no_solution")

    measured(
        " | ".join(
            f"{result.action}:{result.reason}:{result.point}"
            for result in results
        )
    )
