"""P3 输出护栏测试：不得越界、不得跳变、不得发 NaN、stamp 必须递增。

每个用例只测 ``safety_guard.py`` 的纯函数。

运行：
    PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
        src/algorithm_pkg/test
"""

import math

import pytest

from algorithm_pkg.algorithm_core import TargetState
from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.safety_guard import apply_safety_guard


def fmt_vec(value):
    """把三维向量格式化成便于摘要阅读的字符串。"""
    if value is None:
        return "None"
    return f"({value[0]:.3f}, {value[1]:.3f}, {value[2]:.3f})"


def make_target(position=(10.0, 0.0, 10.0)):
    """构造护栏最小距离检查用的目标。"""
    return TargetState(
        position=position,
        velocity=(0.0, 0.0, 0.0),
        stamp=1.0,
        age=0.0,
        valid=True,
        velocity_valid=True,
    )


def test_nan_and_inf_points_are_rejected(measured):
    """场景：规划点里混入 NaN 或 Inf。

    操作：分别调用 apply_safety_guard()。
    输入：(NaN,0,10) 和 (0,0,Inf)。
    预期：两次都拒绝，point=None，reason=guard_non_finite。
    """
    for point in ((math.nan, 0.0, 10.0), (0.0, 0.0, math.inf)):
        result = apply_safety_guard(
            point,
            None,
            "TRACK",
            PlanParams(),
            1.0,
        )
        assert not result.accepted
        assert result.point is None
        assert result.rejected_reason == "guard_non_finite"

    measured("NaN / Inf -> guard_non_finite，point=None")


def test_none_point_and_unsupported_mode_are_rejected(measured):
    """场景：本帧没有点，或动作不是三个规划动作。

    操作：分别调用 apply_safety_guard(None, ...) 和 mode=HOVER。
    输入：None 点；HOVER。
    预期：分别返回 guard_no_point / guard_unsupported_mode。
    """
    no_point = apply_safety_guard(None, None, "TRACK", PlanParams(), 1.0)
    hover = apply_safety_guard(
        (1.0, 2.0, 3.0),
        None,
        "HOVER",
        PlanParams(),
        1.0,
    )

    assert no_point.rejected_reason == "guard_no_point"
    assert hover.rejected_reason == "guard_unsupported_mode"

    measured(
        f"None -> {no_point.rejected_reason}, "
        f"HOVER -> {hover.rejected_reason}"
    )


def test_xy_is_clamped_before_output(measured):
    """场景：规划点超出水平 ±1000 米。

    操作：调用 apply_safety_guard()，mode=TRACK。
    输入：point=(2000,-2000,10)。
    预期：point=(1000,-1000,10)，clipped=True，notes 记录原值。
    """
    result = apply_safety_guard(
        (2000.0, -2000.0, 10.0),
        None,
        "TRACK",
        PlanParams(),
        1.0,
    )

    assert result.accepted
    assert result.point == pytest.approx((1000.0, -1000.0, 10.0))
    assert result.clipped
    assert result.notes[0].startswith("xy_clamped")

    measured(f"point={fmt_vec(result.point)}, notes={result.notes}")


def test_altitude_depends_on_mode(measured):
    """场景：TRACK 要求 5 米安全高度，ATTACK 允许 3 米，上限都是 50 米。

    操作：分别调用 apply_safety_guard()。
    输入：TRACK/INTERCEPT 的 z=1；ATTACK 的 z=1；TRACK 的 z=60。
    预期：z 分别裁到 5、5、3、50。
    """
    track = apply_safety_guard(
        (0.0, 0.0, 1.0), None, "TRACK", PlanParams(), 1.0
    )
    intercept = apply_safety_guard(
        (0.0, 0.0, 1.0), None, "INTERCEPT", PlanParams(), 1.0
    )
    attack = apply_safety_guard(
        (0.0, 0.0, 1.0), None, "ATTACK", PlanParams(), 1.0
    )
    too_high = apply_safety_guard(
        (0.0, 0.0, 60.0), None, "TRACK", PlanParams(), 1.0
    )

    assert track.point == pytest.approx((0.0, 0.0, 5.0))
    assert intercept.point == pytest.approx((0.0, 0.0, 5.0))
    assert attack.point == pytest.approx((0.0, 0.0, 3.0))
    assert too_high.point == pytest.approx((0.0, 0.0, 50.0))

    measured(
        f"TRACK z=1->{track.point[2]}, INTERCEPT z=1->{intercept.point[2]}, "
        f"ATTACK z=1->{attack.point[2]}, 上限->{too_high.point[2]}"
    )


def test_step_limit_is_between_goals_not_from_own(measured):
    """场景：上一帧目标点在 (0,0,10)，这一帧要跳到 (10,0,10)。

    操作：调用 apply_safety_guard()，max_step=1。
    输入：p_goal=(10,0,10)，p_prev=(0,0,10)。
    预期：点被裁到距上一目标点 1 米处，即 (1,0,10)。
    """
    params = PlanParams(max_step=1.0)
    result = apply_safety_guard(
        (10.0, 0.0, 10.0),
        (0.0, 0.0, 10.0),
        "TRACK",
        params,
        1.0,
    )

    assert result.point == pytest.approx((1.0, 0.0, 10.0))
    assert result.clipped
    assert any(note.startswith("step_clamped") for note in result.notes)

    measured(f"point={fmt_vec(result.point)}, notes={result.notes}")


def test_step_limit_none_means_no_clip(measured):
    """场景：没有配置 max_step，也没有历史点。

    操作：调用 apply_safety_guard()，max_step=None。
    输入：p_goal=(100,0,10)。
    预期：点原样输出，不标 clipped。
    """
    result = apply_safety_guard(
        (100.0, 0.0, 10.0),
        (0.0, 0.0, 10.0),
        "TRACK",
        PlanParams(),
        1.0,
    )

    assert result.point == pytest.approx((100.0, 0.0, 10.0))
    assert not result.clipped

    measured(f"point={fmt_vec(result.point)}, clipped={result.clipped}")


def test_min_distance_guard_rejects_intercept_and_attack(measured):
    """场景：拦截/攻击点落在目标当前位置上。

    操作：调用 apply_safety_guard()，target=(10,0,10)，p_goal=(10,0,10)。
    输入：min_distance=0.5。
    预期：INTERCEPT/ATTACK 都拒绝，reason=guard_target_too_close。
    """
    for action in ("INTERCEPT", "ATTACK"):
        result = apply_safety_guard(
            (10.0, 0.0, 10.0),
            None,
            action,
            PlanParams(),
            1.0,
            target=make_target(),
        )
        assert not result.accepted
        assert result.rejected_reason == "guard_target_too_close"

    measured("INTERCEPT/ATTACK 与目标重合 -> guard_target_too_close")


def test_min_distance_boundary_is_rejected(measured):
    """场景：点与目标的距离刚好等于 min_distance=0.5。

    操作：调用 apply_safety_guard()，mode=ATTACK。
    输入：p_goal=(10.5,0,10)，target=(10,0,10)。
    预期：按 P2 的“<= min_distance 算太近”一致处理，拒绝该点。
    """
    result = apply_safety_guard(
        (10.5, 0.0, 10.0),
        None,
        "ATTACK",
        PlanParams(),
        1.0,
        target=make_target(),
    )

    assert not result.accepted
    assert result.rejected_reason == "guard_target_too_close"

    measured(f"distance=0.5 -> {result.rejected_reason}")


def test_min_distance_does_not_drop_track_point(measured):
    """场景：TRACK 的目标预测点就是目标当前位置。

    操作：调用 apply_safety_guard()，mode=TRACK。
    输入：p_goal 与 target 重合。
    预期：接受；TRACK 近距环绕由中枢处理，护栏不能改变 P2 契约。
    """
    result = apply_safety_guard(
        (10.0, 0.0, 10.0),
        None,
        "TRACK",
        PlanParams(),
        1.0,
        target=make_target(),
    )

    assert result.accepted
    assert result.point == pytest.approx((10.0, 0.0, 10.0))

    measured(f"TRACK point={fmt_vec(result.point)}（不因近距被拒）")


def test_stamp_must_strictly_increase(measured):
    """场景：ROS 时钟抖动，本帧 now 与上一帧 stamp 相同。

    操作：调用 apply_safety_guard()，now=1.0，last_stamp=1.0。
    输入：相同时间戳。
    预期：输出 stamp 严格大于 1.0，并记录 stamp_adjusted。
    """
    result = apply_safety_guard(
        (0.0, 0.0, 10.0),
        None,
        "TRACK",
        PlanParams(),
        1.0,
        last_stamp=1.0,
    )

    assert result.accepted
    assert result.stamp > 1.0
    assert any(note.startswith("stamp_adjusted") for note in result.notes)

    measured(f"stamp={result.stamp:.6f}, notes={result.notes}")


def test_guard_order_xy_then_z_then_step(measured):
    """场景：点同时越 XY 边界、超高、且相对上一目标点跳变过大。

    操作：调用 apply_safety_guard()，max_step=5。
    输入：p_goal=(2000,0,60)，p_prev=(0,0,40)。
    预期：notes 顺序为 XY -> 高度 -> 跳变；最终点仍在包线内。
    """
    params = PlanParams(max_step=5.0)
    result = apply_safety_guard(
        (2000.0, 0.0, 60.0),
        (0.0, 0.0, 40.0),
        "TRACK",
        params,
        1.0,
    )

    assert result.accepted
    note_kinds = [note.split(":")[0] for note in result.notes]
    assert note_kinds[:2] == ["xy_clamped", "z_clamped"]
    assert any(kind == "step_clamped" for kind in note_kinds)
    assert abs(result.point[0]) <= params.max_xy
    assert abs(result.point[1]) <= params.max_xy
    assert params.safe_alt <= result.point[2] <= params.max_alt

    measured(
        f"point={fmt_vec(result.point)}, notes={result.notes}"
    )


def test_recovery_uses_tighter_step_limit(measured):
    """场景：恢复期第一帧，max_step 要临时收紧到 max_step_recover。

    操作：调用 apply_safety_guard()，recovering=True。
    输入：max_step=1.0，max_step_recover=0.2，p_prev=(0,0,10)，
          p_goal=(10,0,10)。
    预期：点被裁到 0.2 米处，而不是 1.0 米处。
    """
    params = PlanParams(max_step=1.0, max_step_recover=0.2)
    result = apply_safety_guard(
        (10.0, 0.0, 10.0),
        (0.0, 0.0, 10.0),
        "TRACK",
        params,
        1.0,
        recovering=True,
    )

    assert result.point == pytest.approx((0.2, 0.0, 10.0))
    assert any("0.200" in note for note in result.notes)

    measured(f"恢复期 point={fmt_vec(result.point)}")


def test_explicit_step_limit_overrides_params(measured):
    """场景：调用方已经算好本帧的跳变上限。

    操作：调用 apply_safety_guard()，step_limit=0.5。
    输入：params.max_step=1.0。
    预期：按 0.5 裁剪，而不是 1.0。
    """
    result = apply_safety_guard(
        (10.0, 0.0, 10.0),
        (0.0, 0.0, 10.0),
        "TRACK",
        PlanParams(max_step=1.0),
        1.0,
        step_limit=0.5,
    )

    assert result.point == pytest.approx((0.5, 0.0, 10.0))

    measured(f"step_limit=0.5 -> point={fmt_vec(result.point)}")


def test_non_finite_now_is_rejected(measured):
    """场景：发布时刻本身是 NaN，不能生成合法 stamp。

    操作：调用 apply_safety_guard()，now=NaN。
    输入：有效点 + NaN 时刻。
    预期：拒绝，reason=guard_stamp_non_finite。
    """
    result = apply_safety_guard(
        (0.0, 0.0, 10.0),
        None,
        "TRACK",
        PlanParams(),
        math.nan,
    )

    assert not result.accepted
    assert result.rejected_reason == "guard_stamp_non_finite"

    measured(f"reason={result.rejected_reason}")


def test_step_limit_uses_action_specific_value(measured):
    """场景：TRACK/ATTACK 配置不同的单帧位移上限。

    预期：同一条从 (0,0,10) 到 (10,0,10) 的目标点，TRACK 最多走 0.14 m，
    ATTACK 最多走 0.62 m，不受兜底 1.0 m 影响。
    """
    params = PlanParams(
        max_step=1.0,
        max_step_track=0.14,
        max_step_attack=0.62,
    )
    prev = (0.0, 0.0, 10.0)
    goal = (10.0, 0.0, 10.0)

    track = apply_safety_guard(goal, prev, "TRACK", params, now=1.0)
    attack = apply_safety_guard(goal, prev, "ATTACK", params, now=1.0)

    assert track.point is not None
    assert attack.point is not None
    assert math.dist(track.point, prev) == pytest.approx(0.14, abs=1e-9)
    assert math.dist(attack.point, prev) == pytest.approx(0.62, abs=1e-9)

    measured(
        f"TRACK {math.dist(track.point, prev):.3f} m，"
        f"ATTACK {math.dist(attack.point, prev):.3f} m"
    )
