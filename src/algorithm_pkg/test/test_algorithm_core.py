"""P1 纯算法测试：用可读的场景说明每个函数的行为。

本文件只测 algorithm_core.py 的纯函数，不启动 ROS、不连飞控。

运行：
    PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
        src/algorithm_pkg/test

成功时终端末尾会输出“P1 实测结果”，列出每个测试实际测出的值。

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
