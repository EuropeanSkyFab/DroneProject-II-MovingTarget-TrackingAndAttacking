"""P2 策略测试：7 个动作分别该发点、还是不发电。

运行：
    PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
        src/algorithm_pkg/test
"""

import math

import pytest

from algorithm_pkg.algorithm_core import TargetState
from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.plan_policy import plan


def make_target(
    position=(10.0, 0.0, 10.0),
    velocity=(0.0, 0.0, 0.0),
    *,
    age=0.0,
    valid=True,
    velocity_valid=True,
):
    """构造测试用目标状态。"""
    return TargetState(
        position=position,
        velocity=velocity,
        stamp=1.0,
        age=age,
        valid=valid,
        velocity_valid=velocity_valid,
    )


def fmt_vec(value):
    """把三维向量格式化，便于终端摘要阅读。"""
    if value is None:
        return "None"
    return f"({value[0]:.3f}, {value[1]:.3f}, {value[2]:.3f})"


def test_center_generated_actions_have_no_point(measured):
    """场景：TAKEOFF/HOVER/RETURN 的坐标由中枢生成。

    操作：分别调用 plan()。
    输入：action 为 TAKEOFF/HOVER/RETURN，目标有效，己方在原点。
    预期：point=None，reason 分别说明“中枢负责”。
    """
    target = make_target()
    own = (0.0, 0.0, 10.0)
    expected = {
        "TAKEOFF": "takeoff_center_generated",
        "HOVER": "hover_center_generated",
        "RETURN": "return_center_generated",
    }

    for action, reason in expected.items():
        result = plan(action, own, target)
        assert result.point is None
        assert result.reason == reason
        assert not result.should_publish

    measured(
        "TAKEOFF/HOVER/RETURN -> point=None；"
        "原因分别是 takeoff/hover/return_center_generated"
    )


def test_resume_patrol_is_explicitly_unspecified(measured):
    """场景：接口里提到了 RESUME_PATROL，但没有定义它的坐标规则。

    操作：调用 plan("RESUME_PATROL", ...)。
    输入：有效目标和己方位置。
    预期：point=None，reason=resume_patrol_unspecified，不猜行为。
    """
    result = plan(
        "RESUME_PATROL",
        (0.0, 0.0, 10.0),
        make_target(),
    )

    assert result.point is None
    assert result.reason == "resume_patrol_unspecified"

    measured(f"reason={result.reason}, point={result.point}")


def test_unknown_action_is_not_normalized_to_hover(measured):
    """场景：上游把 action 拼错，不能让飞机继续执行旧模式。

    操作：调用 plan("TRACKK", ...)。
    输入：未知动作 TRACKK。
    预期：point=None，reason=unknown_action。
    """
    result = plan(
        "TRACKK",
        (0.0, 0.0, 10.0),
        make_target(),
    )

    assert result.action == "TRACKK"
    assert result.point is None
    assert result.reason == "unknown_action"

    measured(f"action={result.action}, reason={result.reason}")


def test_none_or_blank_action_is_unknown(measured):
    """场景：上游传了 None、空串或纯空格。

    操作：分别调用 plan()。
    输入：action 为 None、""、"   "。
    预期：三次都拒绝，不抛异常，reason=unknown_action。
    """
    target = make_target()
    own = (0.0, 0.0, 10.0)

    for action in (None, "", "   "):
        result = plan(action, own, target)
        assert result.point is None
        assert result.reason == "unknown_action"

    measured("None / 空串 / 空格 -> point=None, reason=unknown_action")


def test_track_allows_single_frame_target(measured):
    """场景：视觉只有一帧，速度无效，但 TRACK 只需要位置。

    操作：调用 plan("TRACK", ...)，目标 velocity_valid=False。
    输入：own=(0,0,10)，target=(10,0,10)，速度按 0 处理。
    预期：允许发目标点 (10,0,10)，reason=track_target_point。
    """
    target = make_target(
        position=(10.0, 0.0, 10.0),
        velocity=(0.0, 0.0, 0.0),
        velocity_valid=False,
    )
    result = plan("TRACK", (0.0, 0.0, 10.0), target)

    assert result.should_publish
    assert result.reason == "track_target_point"
    assert result.point == pytest.approx((10.0, 0.0, 10.0))

    measured(f"point={fmt_vec(result.point)}, reason={result.reason}")


def test_intercept_rejects_invalid_velocity(measured):
    """场景：目标只有一帧，速度还没估出来。

    操作：调用 plan("INTERCEPT", ...)，目标 velocity_valid=False。
    输入：有效位置，但 velocity_valid=False。
    预期：point=None，reason=velocity_unavailable。
    """
    target = make_target(
        velocity_valid=False,
    )
    result = plan("INTERCEPT", (0.0, 0.0, 10.0), target)

    assert result.point is None
    assert result.reason == "velocity_unavailable"

    measured(f"point={result.point}, reason={result.reason}")


def test_attack_rejects_invalid_velocity(measured):
    """场景：ATTACK 比 INTERCEPT 更危险，更不能拿假速度算点。

    操作：调用 plan("ATTACK", ...)，目标 velocity_valid=False。
    输入：有效位置，但 velocity_valid=False。
    预期：point=None，reason=velocity_unavailable。
    """
    target = make_target(
        velocity_valid=False,
    )
    result = plan("ATTACK", (0.0, 0.0, 10.0), target)

    assert result.point is None
    assert result.reason == "velocity_unavailable"

    measured(f"point={result.point}, reason={result.reason}")


def test_missing_target_has_no_point(measured):
    """场景：目标还没进入视野。

    操作：调用 plan("TRACK", ...)，target=None。
    输入：target=None。
    预期：point=None，reason=target_missing。
    """
    result = plan("TRACK", (0.0, 0.0, 10.0), None)

    assert result.point is None
    assert result.reason == "target_missing"

    measured(f"point={result.point}, reason={result.reason}")


def test_invalid_target_has_no_point(measured):
    """场景：目标预测已经过期，valid=False。

    操作：调用 plan("TRACK", ...)。
    输入：target.valid=False。
    预期：point=None，reason=target_invalid。
    """
    target = make_target(valid=False)
    result = plan("TRACK", (0.0, 0.0, 10.0), target)

    assert result.point is None
    assert result.reason == "target_invalid"

    measured(f"point={result.point}, reason={result.reason}")


def test_target_older_than_p2_limit_has_no_point(measured):
    """场景：P1 认为目标还在 1 秒内，但 P2 只允许 0.5 秒。

    操作：调用 plan("TRACK", ...)，target.age=0.6。
    输入：target.valid=True，age=0.6，max_pred_age=0.5。
    预期：point=None，reason=target_too_old。
    """
    target = make_target(age=0.6)
    result = plan("TRACK", (0.0, 0.0, 10.0), target)

    assert result.point is None
    assert result.reason == "target_too_old"

    measured(f"age={target.age:.2f}s, reason={result.reason}")


def test_target_at_p2_age_boundary_is_allowed(measured):
    """场景：目标年龄刚好等于 max_pred_age。

    操作：调用 plan("TRACK", ...)，target.age=0.5。
    输入：target.valid=True，age=0.5，max_pred_age=0.5。
    预期：允许发点（边界取等号）。
    """
    target = make_target(age=0.5)
    result = plan("TRACK", (0.0, 0.0, 10.0), target)

    assert result.should_publish
    assert result.point == pytest.approx((10.0, 0.0, 10.0))

    measured(f"age={target.age:.2f}s, point={fmt_vec(result.point)}")


def test_missing_own_position_has_no_point(measured):
    """场景：己方状态还没到，不能算拦截点。

    操作：调用 plan("INTERCEPT", ...)，own=None。
    输入：own=None，目标有效。
    预期：point=None，reason=own_invalid。
    """
    result = plan("INTERCEPT", None, make_target())

    assert result.point is None
    assert result.reason == "own_invalid"

    measured(f"point={result.point}, reason={result.reason}")


def test_nan_own_position_has_no_point(measured):
    """场景：己方位置里混入 NaN。

    操作：调用 plan("TRACK", ...)，own=(NaN,0,10)。
    输入：含 NaN 的 own。
    预期：point=None，reason=own_invalid。
    """
    result = plan(
        "TRACK",
        (math.nan, 0.0, 10.0),
        make_target(),
    )

    assert result.point is None
    assert result.reason == "own_invalid"

    measured(f"point={result.point}, reason={result.reason}")


def test_attack_unreachable_does_not_return_tail_point(measured):
    """场景：ATTACK 目标更快，直线追不上。

    操作：调用 plan("ATTACK", ...)，把攻击速度压到 6 m/s。
    输入：own=(0,0,10)，target=(10,0,10)，v_t=(8,0,0)。
    预期：point=None，reason=attack_no_solution；不能退化成尾随点。
    """
    target = make_target(velocity=(8.0, 0.0, 0.0))
    params = PlanParams(v_max_attack=6.0)
    result = plan("ATTACK", (0.0, 0.0, 10.0), target, params)

    assert result.point is None
    assert result.reason == "attack_no_solution"

    measured(f"point={result.point}, reason={result.reason}")


def test_intercept_unreachable_returns_tail_point(measured):
    """场景：INTERCEPT 追不上，允许退化为尾随点。

    操作：调用 plan("INTERCEPT", ...)，把拦截速度压到 6 m/s。
    输入：own=(0,0,10)，target=(10,0,10)，v_t=(8,0,0)，d_tail=4。
    预期：point=(6,0,10)，reachable=False，reason=fallback_tail。
    """
    target = make_target(velocity=(8.0, 0.0, 0.0))
    params = PlanParams(v_max_intercept=6.0)
    result = plan("INTERCEPT", (0.0, 0.0, 10.0), target, params)

    assert result.should_publish
    assert not result.reachable
    assert result.reason == "fallback_tail"
    assert result.point == pytest.approx((6.0, 0.0, 10.0))

    measured(
        f"point={fmt_vec(result.point)}, reachable={result.reachable}, "
        f"reason={result.reason}"
    )


def test_track_point_is_clipped_to_safe_altitude(measured):
    """场景：TRACK 目标点只有 1 米高，低于 5 米安全高度。

    操作：调用 plan("TRACK", ...)。
    输入：target=(10,0,1)。
    预期：point=(10,0,5)，clipped=True。
    """
    target = make_target(position=(10.0, 0.0, 1.0))
    result = plan("TRACK", (0.0, 0.0, 10.0), target)

    assert result.point == pytest.approx((10.0, 0.0, 5.0))
    assert result.clipped

    measured(f"point={fmt_vec(result.point)}, clipped={result.clipped}")


def test_attack_point_uses_attack_altitude(measured):
    """场景：ATTACK 可以打低空目标，最低高度是 3 米而不是 5 米。

    操作：调用 plan("ATTACK", ...)。
    输入：target=(10,0,1)，目标静止。
    预期：point 的 z 被抬到 3，clipped=True。
    """
    target = make_target(position=(10.0, 0.0, 1.0))
    result = plan("ATTACK", (0.0, 0.0, 1.0), target)

    assert result.point == pytest.approx((10.0, 0.0, 3.0))
    assert result.clipped

    measured(f"point={fmt_vec(result.point)}, clipped={result.clipped}")


def test_max_step_clips_and_marks_result(measured):
    """场景：目标点很远，但配置要求单帧最多走 1 米。

    操作：调用 plan("TRACK", ...)，max_step=1。
    输入：own=(0,0,10)，target=(10,0,10)。
    预期：point=(1,0,10)，clipped=True。
    """
    target = make_target(position=(10.0, 0.0, 10.0))
    params = PlanParams(max_step=1.0)
    result = plan("TRACK", (0.0, 0.0, 10.0), target, params)

    assert result.point == pytest.approx((1.0, 0.0, 10.0))
    assert result.clipped

    measured(f"point={fmt_vec(result.point)}, clipped={result.clipped}")
