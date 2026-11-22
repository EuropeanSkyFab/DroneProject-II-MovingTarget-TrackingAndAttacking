"""P1 纯算法测试：用可读的场景说明每个函数的行为。

本文件只测 algorithm_core.py 的纯函数，不启动 ROS、不连飞控。

运行：
    PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
        src/algorithm_pkg/test

成功时终端末尾会输出“算法实测结果”，按“场景 / 预期 / 实测”列出每个测试。

测试分三组：
1. 目标预测：速度估计、过期、异常数据
2. 拦截求解：迎头、等速、追不上、ATTACK、TRACK
3. 安全约束：高度、水平边界、max_step、非法输入
"""

import math

import pytest

from algorithm_pkg.algorithm_core import (
    TargetSample,
    TargetState,
    apply_constraints,
    predict_target,
    solve_intercept,
)


def make_state(position, velocity, valid=True):
    """构造测试用目标状态。

    stamp/age 不参与 solve_intercept 的几何计算，固定填 1.0/0.0；
    valid=False 用来模拟“目标预测已过期”。
    """
    return TargetState(
        position=position,
        velocity=velocity,
        stamp=1.0,
        age=0.0,
        valid=valid,
        velocity_valid=True,
    )


def fmt_vec(value):
    """把三维向量格式化成便于阅读的字符串。"""
    return f"({value[0]:.3f}, {value[1]:.3f}, {value[2]:.3f})"


# ============================================================
# 第一组：目标预测
# 用到 predict_target()
# ============================================================


def test_predict_constant_velocity(measured):
    """场景：目标沿 +x 方向匀速直线飞行，每秒 1 米，高度 10 米。

    操作：调用 predict_target()，用三帧位置估计速度，并外推到 now=3 秒。
    输入：TargetSample(stamp,x,y,z) 分别为
          (0,0,0,10)、(1,1,0,10)、(2,2,0,10)。
    预期：速度 (1,0,0)，now=3 时位置 (3,0,10)，age=1，valid=True。
    """
    samples = [
        TargetSample(0.0, 0.0, 0.0, 10.0),
        TargetSample(1.0, 1.0, 0.0, 10.0),
        TargetSample(2.0, 2.0, 0.0, 10.0),
    ]
    state = predict_target(samples, now=3.0, max_age=1.0)

    assert state is not None
    assert state.valid
    assert state.velocity_valid
    assert state.velocity == pytest.approx((1.0, 0.0, 0.0))
    assert state.position == pytest.approx((3.0, 0.0, 10.0))
    assert state.age == pytest.approx(1.0)

    measured(
        f"v={fmt_vec(state.velocity)}, p@now={fmt_vec(state.position)}, "
        f"age={state.age:.3f}s, valid={state.valid}"
    )


def test_predict_single_sample_has_zero_velocity(measured):
    """场景：视觉只给出第一帧，还没有第二帧可以估速度。

    操作：调用 predict_target()，只传入一帧。
    输入：TargetSample(stamp=5.0, x=1, y=2, z=3)，now=5.1。
    预期：位置 (1,2,3)，速度 (0,0,0)，velocity_valid=False，valid=True。
    """
    state = predict_target([TargetSample(5.0, 1.0, 2.0, 3.0)], now=5.1)

    assert state is not None
    assert state.velocity == (0.0, 0.0, 0.0)
    assert not state.velocity_valid
    assert state.valid

    measured(
        f"v={fmt_vec(state.velocity)}, "
        f"velocity_valid={state.velocity_valid}"
    )


def test_predict_near_duplicate_timestamps_marks_velocity_invalid(
    measured,
):
    """场景：两帧时间戳几乎相同，位置差 1 米，但时间差只有 2e-9 秒。

    操作：调用 predict_target()，用两帧估计速度。
    输入：stamp=0、x=0；stamp=2e-9、x=1；now=2e-9。
    预期：斜率不可信，速度置 0，位置取最新帧 (1,0,0)，
          velocity_valid=False。
    """
    samples = [
        TargetSample(0.0, 0.0, 0.0, 0.0),
        TargetSample(2e-9, 1.0, 0.0, 0.0),
    ]
    state = predict_target(samples, now=2e-9)

    assert state is not None
    assert state.velocity == (0.0, 0.0, 0.0)
    assert not state.velocity_valid
    assert state.position == pytest.approx((1.0, 0.0, 0.0))

    measured(
        f"v={fmt_vec(state.velocity)}, p={fmt_vec(state.position)}, "
        f"velocity_valid={state.velocity_valid}"
    )


def test_predict_stale_target_is_invalid(measured):
    """场景：目标数据已经过期，不能再用于规划。

    操作：调用 predict_target()，把 max_age 设为 0.5 秒。
    输入：唯一一帧 stamp=0；now=2.0，数据年龄 2 秒。
    预期：state.valid=False。
    """
    state = predict_target(
        [TargetSample(0.0, 0.0, 0.0, 10.0)],
        now=2.0,
        max_age=0.5,
    )

    assert state is not None
    assert not state.valid

    measured(
        f"age={state.age:.3f}s, valid={state.valid}, velocity_valid="
        f"{state.velocity_valid}"
    )


def test_predict_rejects_overspeed(measured):
    """场景：目标检测结果发生跳变，1 秒内 x 从 0 跳到 20 米。

    操作：调用 predict_target()，把 max_speed 设为 15 m/s。
    输入：stamp=0、x=0；stamp=1、x=20。
    预期：估计速度 20 m/s 超过上限，返回 None。
    """
    samples = [
        TargetSample(0.0, 0.0, 0.0, 0.0),
        TargetSample(1.0, 20.0, 0.0, 0.0),
    ]

    state = predict_target(samples, now=1.0, max_speed=15.0)

    assert state is None

    measured("state=None（速度 20 m/s 超过 15 m/s）")


def test_predict_target_none_without_samples(measured):
    """场景：系统刚启动，还没有收到任何目标采样。

    操作：调用 predict_target()，传入空列表。
    预期：返回 None。
    """
    state = predict_target([], now=1.0)

    assert state is None

    measured("state=None（没有可用采样）")


def test_predict_ignores_future_timestamps(measured):
    """场景：视觉把某一帧的时间戳误设到未来（stamp=100 秒），当前只有 2 秒。

    操作：调用 predict_target()，把一帧正常采样和一帧未来采样一起传入。
    输入：stamp=0、x=0；stamp=100、x=100；now=2，max_age=1。
    预期：未来帧被丢弃，只能用 stamp=0 的帧；位置 (0,0,10)，
          velocity_valid=False；age=2 超过 max_age，valid=False。
    """
    samples = [
        TargetSample(0.0, 0.0, 0.0, 10.0),
        TargetSample(100.0, 100.0, 0.0, 10.0),
    ]
    state = predict_target(samples, now=2.0, max_age=1.0)

    assert state is not None
    assert state.position == pytest.approx((0.0, 0.0, 10.0))
    assert state.velocity == (0.0, 0.0, 0.0)
    assert not state.velocity_valid
    assert not state.valid

    measured(
        f"p@now={fmt_vec(state.position)}, age={state.age:.1f}s, "
        f"valid={state.valid}（未来帧未参与拟合）"
    )


def test_predict_future_only_samples_return_none(measured):
    """场景：所有采样都落在未来，可能是视觉时钟跳变或消息 stamp 写错。

    操作：调用 predict_target()，只传一帧 stamp=100 的采样。
    输入：stamp=100，now=2。
    预期：没有可用采样，返回 None。
    """
    state = predict_target([TargetSample(100.0, 1.0, 2.0, 3.0)], now=2.0)

    assert state is None

    measured("state=None（全部为未来帧）")


def test_predict_duplicate_timestamp_keeps_last_value(measured):
    """场景：同一时刻收到两帧（后一帧是视觉修正值），时间差为 0。

    操作：调用 predict_target()，两帧 stamp 都是 1.0。
    输入：stamp=1、x=0；stamp=1、x=5；now=1.5。
    预期：同一时间戳只保留最后一条，位置 (5,0,10)，速度无效。
    """
    samples = [
        TargetSample(1.0, 0.0, 0.0, 10.0),
        TargetSample(1.0, 5.0, 0.0, 10.0),
    ]
    state = predict_target(samples, now=1.5)

    assert state is not None
    assert state.position == pytest.approx((5.0, 0.0, 10.0))
    assert state.velocity == (0.0, 0.0, 0.0)
    assert not state.velocity_valid
    assert state.stamp == pytest.approx(1.0)

    measured(f"p@now={fmt_vec(state.position)}（保留同 stamp 的最后一帧）")


def test_predict_invalid_arguments_raise(measured):
    """场景：调用方把 max_age 或 max_speed 配成非法值。

    操作：分别用 max_age=-0.1 和 max_speed=0 调用 predict_target()。
    预期：两处都抛 ValueError，而不是静默返回 None。
    """
    with pytest.raises(ValueError):
        predict_target([], now=0.0, max_age=-0.1)
    with pytest.raises(ValueError):
        predict_target([], now=0.0, max_speed=0.0)

    measured("max_age<0 / max_speed=0 -> ValueError")


# ============================================================
# 第二组：拦截求解
# 用到 solve_intercept()
# ============================================================


def test_intercept_head_on(measured):
    """场景：目标从正东 10 米处迎面飞来，己方最大速度 6 m/s。

    操作：调用 solve_intercept()，mode=INTERCEPT，v_max=6。
    输入：own=(0,0,10)，target=(10,0,10)，v_t=(-5,0,0)。
    预期：reachable=True，t_go=10/11 秒，拦截点 (60/11,0,10)。
    """
    target = make_state((10.0, 0.0, 10.0), (-5.0, 0.0, 0.0))
    result = solve_intercept((0.0, 0.0, 10.0), target, 6.0)

    assert result.action == "INTERCEPT"
    assert result.reachable
    assert result.reason == "intercept_point"
    assert result.t_go == pytest.approx(10.0 / 11.0)
    assert result.point == pytest.approx((60.0 / 11.0, 0.0, 10.0))

    measured(
        f"t_go={result.t_go:.3f}s, point={fmt_vec(result.point)}, "
        f"reason={result.reason}"
    )


def test_intercept_equal_speed_uses_linear_equation(measured):
    """场景：目标迎头速度正好等于己方最大速度，二次方程退化成一次方程。

    操作：调用 solve_intercept()，v_max=6。
    输入：own=(0,0,0)，target=(10,0,0)，v_t=(-6,0,0)。
    预期：t_go=5/6 秒，拦截点 (5,0,0)。
    """
    target = make_state((10.0, 0.0, 0.0), (-6.0, 0.0, 0.0))
    result = solve_intercept((0.0, 0.0, 0.0), target, 6.0)

    assert result.reachable
    assert result.t_go == pytest.approx(10.0 / 12.0)
    assert result.point == pytest.approx((5.0, 0.0, 0.0))

    measured(f"t_go={result.t_go:.3f}s, point={fmt_vec(result.point)}")


def test_intercept_unreachable_uses_tail_point(measured):
    """场景：目标同向远离，而且比己方快，追不上。

    操作：调用 solve_intercept()，mode=INTERCEPT，v_max=6，d_tail=4。
    输入：own=(0,0,10)，target=(10,0,10)，v_t=(8,0,0)。
    预期：reachable=False，reason=fallback_tail，
          尾随点 (6,0,10)。
    """
    target = make_state((10.0, 0.0, 10.0), (8.0, 0.0, 0.0))
    result = solve_intercept((0.0, 0.0, 10.0), target, 6.0, d_tail=4.0)

    assert result.action == "INTERCEPT"
    assert not result.reachable
    assert result.reason == "fallback_tail"
    assert result.point == pytest.approx((6.0, 0.0, 10.0))

    measured(
        f"reachable=False, point={fmt_vec(result.point)}, "
        f"reason={result.reason}"
    )


def test_attack_never_falls_back_to_tail(measured):
    """场景：ATTACK 模式下目标更快，拦截方程无解。

    操作：调用 solve_intercept()，mode=ATTACK。
    输入：own=(0,0,10)，target=(10,0,10)，v_t=(8,0,0)。
    预期：point=None，reachable=False，reason=attack_no_solution；
          不能退化成尾随点。
    """
    target = make_state((10.0, 0.0, 10.0), (8.0, 0.0, 0.0))
    result = solve_intercept((0.0, 0.0, 10.0), target, 6.0, mode="ATTACK")

    assert result.action == "ATTACK"
    assert not result.reachable
    assert result.reason == "attack_no_solution"
    assert result.point is None

    measured(f"reachable=False, point={result.point}, reason={result.reason}")


def test_fast_crossing_target_is_unreachable(measured):
    """场景：目标以 10 m/s 垂直横穿，己方最大速度只有 6 m/s。

    操作：调用 solve_intercept()，mode=INTERCEPT，v_max=6，d_tail=4。
    输入：own=(0,0,10)，target=(10,0,10)，v_t=(0,10,0)。
    预期：判别式小于 0，无实数拦截点；退化为尾随点 (10,-4,10)。
    """
    target = make_state((10.0, 0.0, 10.0), (0.0, 10.0, 0.0))
    result = solve_intercept((0.0, 0.0, 10.0), target, 6.0, d_tail=4.0)

    assert not result.reachable
    assert result.reason == "fallback_tail"
    assert result.point == pytest.approx((10.0, -4.0, 10.0))

    measured(
        f"reachable=False, point={fmt_vec(result.point)}, "
        f"reason={result.reason}"
    )


def test_intercept_dynamic_infeasible_uses_tail(measured):
    """场景：目标在己方身后 2 米、同向慢移，几何有解但要求瞬间掉头。

    操作：调用 solve_intercept()，mode=INTERCEPT，v_max=4，
          own_velocity=(4,0,0) 表示己方正以 4 m/s 向东飞。
    输入：own=(0,0,10)，target=(-2,0,10)，v_t=(0.5,0,0)。
          转向角 180°，位移 1.778 m，所需转弯半径 0.889 m；
          4 m/s 时最小转弯半径 1.633 m，物理上转不过来。
    预期：reachable=False，reason=dynamic_infeasible_tail，
          尾随点 (-6,0,10)；不能返回拦截点。
    """
    target = make_state((-2.0, 0.0, 10.0), (0.5, 0.0, 0.0))
    result = solve_intercept(
        (0.0, 0.0, 10.0),
        target,
        4.0,
        own_velocity=(4.0, 0.0, 0.0),
        d_tail=4.0,
    )

    assert result.action == "INTERCEPT"
    assert not result.reachable
    assert result.reason == "dynamic_infeasible_tail"
    assert result.point == pytest.approx((-6.0, 0.0, 10.0))

    measured(
        f"reachable=False, point={fmt_vec(result.point)}, "
        f"reason={result.reason}"
    )


def test_attack_ignores_turn_radius_gate(measured):
    """场景：与掉头用例同一几何，但模式是 ATTACK，且传了己方速度。

    操作：调用 solve_intercept()，mode=ATTACK，v_max=4，
          own_velocity=(4,0,0)。
    输入：own=(0,0,10)，target=(-2,0,10)，v_t=(0.5,0,0)。
    预期：reachable=True，reason=intercept_point；
          ATTACK 不做动力学判定——它的末段语义（撞击还是贴近后保持）
          P5 尚未定案，规划层不能顺手改掉。
    """
    target = make_state((-2.0, 0.0, 10.0), (0.5, 0.0, 0.0))
    result = solve_intercept(
        (0.0, 0.0, 10.0),
        target,
        4.0,
        mode="ATTACK",
        own_velocity=(4.0, 0.0, 0.0),
    )

    assert result.action == "ATTACK"
    assert result.reachable
    assert result.reason == "intercept_point"
    assert result.point == pytest.approx((-1.778, 0.0, 10.0), abs=1e-3)

    measured(f"reachable=True, reason={result.reason}")


def test_intercept_without_ego_velocity_skips_dynamics(measured):
    """场景：没提供己方速度时，动力学判定必须跳过，行为与 P5 完全一致。

    操作：调用 solve_intercept()，mode=INTERCEPT，v_max=4，不传 own_velocity。
    输入：own=(0,0,10)，target=(-2,0,10)，v_t=(0.5,0,0)（掉头几何）。
    预期：reachable=True，reason=intercept_point；
          不知道朝向就不能做动力学判定，不能凭空拒绝。
    """
    target = make_state((-2.0, 0.0, 10.0), (0.5, 0.0, 0.0))
    result = solve_intercept((0.0, 0.0, 10.0), target, 4.0)

    assert result.reachable
    assert result.reason == "intercept_point"
    assert result.point == pytest.approx((-1.778, 0.0, 10.0), abs=1e-3)

    measured(f"reachable=True, reason={result.reason}")


def test_intercept_gentle_turn_stays_reachable(measured):
    """场景：远距 90° 转弯，所需半径远大于最小半径，判定不能误拒。

    操作：调用 solve_intercept()，mode=INTERCEPT，v_max=4，
          own_velocity=(4,0,0)。
    输入：own=(0,0,10)，target=(0,10,10)，v_t=(0,2,0)。
          位移 20 m，转向角 90°，所需半径 14.14 m，大于 1.633 m。
    预期：reachable=True，reason=intercept_point，拦截点 (0,20,10)。
    """
    target = make_state((0.0, 10.0, 10.0), (0.0, 2.0, 0.0))
    result = solve_intercept(
        (0.0, 0.0, 10.0),
        target,
        4.0,
        own_velocity=(4.0, 0.0, 0.0),
    )

    assert result.reachable
    assert result.reason == "intercept_point"
    assert result.t_go == pytest.approx(5.0)
    assert result.point == pytest.approx((0.0, 20.0, 10.0))

    measured(
        f"reachable=True, t_go={result.t_go:.3f}s, "
        f"point={fmt_vec(result.point)}"
    )


def test_slow_ego_velocity_skips_dynamics(measured):
    """场景：己方速度接近 0（起飞/悬停），没有可信朝向，判定必须跳过。

    操作：调用 solve_intercept()，mode=INTERCEPT，v_max=4，
          own_velocity=(0.2,0,0)。
    输入：own=(0,0,10)，target=(-2,0,10)，v_t=(0.5,0,0)（掉头几何）。
    预期：reachable=True，reason=intercept_point；
          低速时最小转弯半径趋 0，强行判定会把所有转弯都放过。
    """
    target = make_state((-2.0, 0.0, 10.0), (0.5, 0.0, 0.0))
    result = solve_intercept(
        (0.0, 0.0, 10.0),
        target,
        4.0,
        own_velocity=(0.2, 0.0, 0.0),
    )

    assert result.reachable
    assert result.reason == "intercept_point"

    measured(f"reachable=True, reason={result.reason}")


def test_track_returns_target_point_without_offset(measured):
    """场景：TRACK 模式，中枢自己会做近距离环绕，规划不重复加偏移。

    操作：调用 solve_intercept()，mode=TRACK。
    输入：own=(0,0,10)，target=(10,0,10)，v_t=(1,0,0)。
    预期：point 直接等于目标预测点 (10,0,10)。
    """
    target = make_state((10.0, 0.0, 10.0), (1.0, 0.0, 0.0))
    result = solve_intercept((0.0, 0.0, 10.0), target, 6.0, mode="TRACK")

    assert result.action == "TRACK"
    assert result.reachable
    assert result.reason == "track_target_point"
    assert result.point == (10.0, 0.0, 10.0)

    measured(f"point={fmt_vec(result.point)}, reason={result.reason}")


def test_invalid_target_gives_no_point(measured):
    """场景：目标预测已经过期，valid=False，不能继续规划。

    操作：调用 solve_intercept()，传入 valid=False 的目标状态。
    输入：target=(10,0,10)，v_t=(0,0,0)，valid=False。
    预期：point=None，reason=target_invalid。
    """
    target = make_state((10.0, 0.0, 10.0), (0.0, 0.0, 0.0), valid=False)
    result = solve_intercept((0.0, 0.0, 10.0), target, 6.0)

    assert not result.reachable
    assert result.point is None
    assert result.reason == "target_invalid"

    measured(f"point=None, reason={result.reason}")


def test_too_close_target_gives_no_point(measured):
    """场景：目标就在己方当前位置，继续输出点会变成撞机。

    操作：调用 solve_intercept()，min_distance=0.5。
    输入：own=(0,0,10)，target=(0,0,10)。
    预期：point=None，reason=target_too_close。
    """
    target = make_state((0.0, 0.0, 10.0), (0.0, 0.0, 0.0))
    result = solve_intercept((0.0, 0.0, 10.0), target, 6.0, min_distance=0.5)

    assert not result.reachable
    assert result.point is None
    assert result.reason == "target_too_close"

    measured(f"point=None, reason={result.reason}")


def test_intercept_accepts_lowercase_mode(measured):
    """场景：上游联调时把动作写成小写并带空格（" intercept "）。

    操作：调用 solve_intercept()，mode 传小写带空格串。
    输入：own=(0,0,10)，target=(10,0,10)，v_t=(-5,0,0)，v_max=6。
    预期：动作归一化为 INTERCEPT，可拦截，t_go=10/11，
          拦截点 (60/11,0,10)。
    """
    target = make_state((10.0, 0.0, 10.0), (-5.0, 0.0, 0.0))
    result = solve_intercept(
        (0.0, 0.0, 10.0), target, 6.0, mode=" intercept "
    )

    assert result.action == "INTERCEPT"
    assert result.reachable
    assert result.reason == "intercept_point"
    assert result.t_go == pytest.approx(10.0 / 11.0)
    assert result.point == pytest.approx((60.0 / 11.0, 0.0, 10.0))

    measured(
        f"action={result.action}, t_go={result.t_go:.3f}s, "
        f"point={fmt_vec(result.point)}"
    )


def test_intercept_stationary_target_beyond_t_go_uses_target_as_tail(measured):
    """场景：目标悬停，但 t_go_max 太短，按己方速度赶不到。

    操作：调用 solve_intercept()，mode=INTERCEPT，t_go_max=0.5。
    输入：own=(0,0,10)，target=(10,0,10)，v_t=(0,0,0)，v_max=6，d_tail=4。
    预期：不可达；目标速度为 0 时没有“后方”方向，
          尾随点退化为目标位置 (10,0,10)，reason=fallback_tail。
    """
    target = make_state((10.0, 0.0, 10.0), (0.0, 0.0, 0.0))
    result = solve_intercept(
        (0.0, 0.0, 10.0),
        target,
        6.0,
        mode="INTERCEPT",
        t_go_max=0.5,
        d_tail=4.0,
    )

    assert not result.reachable
    assert result.reason == "fallback_tail"
    assert result.point == pytest.approx((10.0, 0.0, 10.0))

    measured(
        f"reachable=False, point={fmt_vec(result.point)}, "
        f"reason={result.reason}"
    )


def test_intercept_rejects_non_finite_input(measured):
    """场景：己方位置里混入 NaN（状态消息损坏）。

    操作：调用 solve_intercept()，own=(NaN,0,10)。
    预期：point=None，reachable=False，reason=non_finite_input。
    """
    target = make_state((10.0, 0.0, 10.0), (0.0, 0.0, 0.0))
    result = solve_intercept((math.nan, 0.0, 10.0), target, 6.0)

    assert not result.reachable
    assert result.point is None
    assert result.reason == "non_finite_input"

    measured(f"point=None, reason={result.reason}")


def test_intercept_invalid_arguments_raise(measured):
    """场景：调用方把速度上限、拦截时间或距离参数配成非法值。

    操作：分别用 v_max=0、t_go_max=0、min_distance=-1、d_tail=-1 调用。
    预期：每一处都抛 ValueError，避免带着错误配置继续飞。
    """
    target = make_state((10.0, 0.0, 10.0), (0.0, 0.0, 0.0))
    own = (0.0, 0.0, 10.0)

    with pytest.raises(ValueError):
        solve_intercept(own, target, 0.0)
    with pytest.raises(ValueError):
        solve_intercept(own, target, 6.0, t_go_max=0.0)
    with pytest.raises(ValueError):
        solve_intercept(own, target, 6.0, min_distance=-1.0)
    with pytest.raises(ValueError):
        solve_intercept(own, target, 6.0, d_tail=-1.0)

    measured("v_max=0 / t_go_max=0 / min_distance<0 / d_tail<0 -> ValueError")


# ============================================================
# 第三组：安全约束
# 用到 apply_constraints()
# ============================================================


def test_constraints_attack_uses_min_alt(measured):
    """场景：ATTACK 模式下目标点只有 1 米高，但允许最低 3 米。

    操作：调用 apply_constraints()，mode=ATTACK。
    输入：point=(0,0,1)。
    预期：z 被抬到 3，返回 (0,0,3)。
    """
    point = apply_constraints((0.0, 0.0, 1.0), "ATTACK")

    assert point == pytest.approx((0.0, 0.0, 3.0))

    measured(f"point={fmt_vec(point)}")


def test_constraints_track_uses_safe_alt(measured):
    """场景：TRACK 模式要求保持 5 米安全高度。

    操作：调用 apply_constraints()，mode=TRACK。
    输入：point=(0,0,1)。
    预期：z 被抬到 5，返回 (0,0,5)。
    """
    point = apply_constraints((0.0, 0.0, 1.0), "TRACK")

    assert point == pytest.approx((0.0, 0.0, 5.0))

    measured(f"point={fmt_vec(point)}")


def test_constraints_clamp_xy_and_max_alt(measured):
    """场景：目标点超出水平 ±1000 米和高度 50 米边界。

    操作：调用 apply_constraints()，mode=INTERCEPT。
    输入：point=(2000,-2000,60)。
    预期：裁剪为 (1000,-1000,50)。
    """
    point = apply_constraints((2000.0, -2000.0, 60.0), "INTERCEPT")

    assert point == pytest.approx((1000.0, -1000.0, 50.0))

    measured(f"point={fmt_vec(point)}")


def test_constraints_limit_max_step(measured):
    """场景：目标点距己方 20 米，但每周期最多只允许走 2 米。

    操作：调用 apply_constraints()，own=(0,0,10)，max_step=2。
    输入：point=(20,0,10)。
    预期：点按比例缩短到 (2,0,10)。
    """
    point = apply_constraints(
        (20.0, 0.0, 10.0),
        "TRACK",
        own=(0.0, 0.0, 10.0),
        max_step=2.0,
    )

    assert point == pytest.approx((2.0, 0.0, 10.0))

    measured(f"point={fmt_vec(point)}")


def test_constraints_ignore_unsupported_mode(measured):
    """场景：中枢自己负责 HOVER 的坐标，规划不应该输出点。

    操作：调用 apply_constraints()，mode=HOVER。
    输入：point=(1,2,3)。
    预期：返回 None。
    """
    point = apply_constraints((1.0, 2.0, 3.0), "HOVER")

    assert point is None

    measured("point=None（HOVER 由中枢负责）")


def test_constraints_return_none_for_none_or_nan(measured):
    """场景：本帧没有点，或点里混入 NaN，不能发布给飞控。

    操作：调用 apply_constraints() 两次。
    输入：point=None；point=(NaN,0,10)。
    预期：两次都返回 None。
    """
    assert apply_constraints(None, "TRACK") is None
    assert apply_constraints((math.nan, 0.0, 10.0), "TRACK") is None

    measured("point=None（空点/NaN）")


def test_constraints_reject_invalid_arguments(measured):
    """场景：配置里出现负的高度边界、零水平边界、零步长或矛盾高度。

    操作：用五组非法参数分别调用 apply_constraints()。
    输入：max_xy=0；max_step=0；min_alt=-1；
          min_alt=10/max_alt=5；safe_alt=60/max_alt=50。
    预期：每一组都抛 ValueError，不能带着错误包线继续裁剪。
    """
    with pytest.raises(ValueError):
        apply_constraints((0.0, 0.0, 10.0), "TRACK", max_xy=0.0)
    with pytest.raises(ValueError):
        apply_constraints((0.0, 0.0, 10.0), "TRACK", max_step=0.0)
    with pytest.raises(ValueError):
        apply_constraints((0.0, 0.0, 10.0), "TRACK", min_alt=-1.0)
    with pytest.raises(ValueError):
        apply_constraints(
            (0.0, 0.0, 10.0), "TRACK", min_alt=10.0, max_alt=5.0
        )
    with pytest.raises(ValueError):
        apply_constraints(
            (0.0, 0.0, 10.0), "TRACK", safe_alt=60.0, max_alt=50.0
        )

    measured("max_xy/max_step/高度边界非法 -> ValueError")


def test_constraints_return_none_for_non_finite_own(measured):
    """场景：己方状态消息损坏，own 里混入 NaN，但仍给了 max_step。

    操作：调用 apply_constraints()，own=(NaN,0,10)。
    预期：无法计算步长，返回 None，不发布该点。
    """
    point = apply_constraints(
        (5.0, 0.0, 10.0),
        "TRACK",
        own=(math.nan, 0.0, 10.0),
        max_step=1.0,
    )

    assert point is None

    measured("point=None（own 非有限值）")


def test_predict_rejects_non_finite_now(measured):
    """场景：调用方把当前时刻算成了 NaN（时钟异常）。

    操作：调用 predict_target()，now=NaN。
    预期：直接返回 None，不做任何拟合。
    """
    state = predict_target([TargetSample(0.0, 0.0, 0.0, 10.0)], now=math.nan)

    assert state is None

    measured("state=None（now 非有限值）")


def test_predict_skips_non_finite_samples(measured):
    """场景：一帧里混入 NaN 坐标，另一帧正常。

    操作：调用 predict_target()，两帧 stamp 分别为 0 和 1。
    输入：stamp=0 的 x 是 NaN；stamp=1 的 x=1。
    预期：NaN 帧被跳过，只用 stamp=1 的帧，位置 (1,0,10)。
    """
    samples = [
        TargetSample(0.0, math.nan, 0.0, 10.0),
        TargetSample(1.0, 1.0, 0.0, 10.0),
    ]
    state = predict_target(samples, now=1.0)

    assert state is not None
    assert state.position == pytest.approx((1.0, 0.0, 10.0))
    assert not state.velocity_valid

    measured(f"p@now={fmt_vec(state.position)}（NaN 帧被跳过）")


def test_intercept_unknown_mode_returns_unsupported_mode(measured):
    """场景：调用方把中枢动作（HOVER）误传给了拦截求解。

    操作：调用 solve_intercept()，mode=HOVER。
    预期：不求解，point=None，reason=unsupported_mode。
    """
    target = make_state((10.0, 0.0, 10.0), (0.0, 0.0, 0.0))
    result = solve_intercept((0.0, 0.0, 10.0), target, 6.0, mode="HOVER")

    assert not result.reachable
    assert result.point is None
    assert result.reason == "unsupported_mode"

    measured(f"point=None, reason={result.reason}")
