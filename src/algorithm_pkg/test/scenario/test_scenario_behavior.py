"""情景 1～7 的单点行为测试：从公开入口跑完整数据流，每个用例出图。

多帧降级测试在 ``test/sim/test_multiframe_degrade.py``；情景 8～10
不做单点。

每个用例对应一个情景，只断言业务结果（发不发点、点的位置、reason、
级别），时间线通过 ``measured()`` 打到终端。与单元测试的分工：

- 单元测试验证单个函数的边界；
- 本文件验证“连续若干帧、按生产参数”时，整条链路是否还满足约定。

参数取 launch 的运行值，不是 ``PlanParams()`` 的宪章兜底值。
"""

from math import hypot, radians

import pytest

from algorithm_pkg.adapter_core import (
    DroneRuntime,
    TargetBuffer,
    payload,
    plan_drone,
    should_publish,
)
from algorithm_pkg.algorithm_core import TargetSample, predict_target
from algorithm_pkg.plan_params import PlanParams

# launch/algorithm.launch.py 的 运行值：
# 4.50 / 7.60 / 14.80 m/s、9.79 m/s²、172.4°/s、0.74 m/帧。
BASELINE_PARAMS = PlanParams(
    v_max=4.50,
    v_max_intercept=7.60,
    v_max_attack=14.80,
    a_max=9.79,
    yaw_rate_max=radians(172.4),
    max_step=0.74,
)


def _fmt(point):
    """把三维点格式化成终端时间线用的短字符串。"""
    if point is None:
        return "None"
    return f"({point[0]:.2f},{point[1]:.2f},{point[2]:.2f})"


def test_scenario1_hover_holds_entry_anchor(measured, scenario_chart):
    """场景：目标从未出现，drone_1 第 1 帧在 (10, 20, 5) m，之后每帧向东漂 2 m；指令 HOVER。

    预期：第 1 帧锚点为 (10, 20, 5)；之后己方位置变到 (12,20,5)…(18,20,5)，
    发点仍是第 1 帧锚点，reason=hover_hold，stamp 仍用当前己方时刻，级别一直
    NORMAL；目标缺失不进入降级状态机。
    """
    runtime = DroneRuntime.create(BASELINE_PARAMS)
    anchor = (10.0, 20.0, 5.0)
    timeline = []

    scenario_chart.configure(
        headline="HOVER：进入时锚点固定，己方漂 8 m 后发点仍是 (10,20,5)",
        subtitle="参数：固定基线 v_max=4.50 m/s，max_step=0.74 m/帧；"
                 "5 帧，20 Hz；目标缺失不驱动降级状态机。",
    )

    for frame in range(5):
        now = 100.0 + 0.05 * frame
        own = (10.0 + 2.0 * frame, 20.0, 5.0)
        result = plan_drone(
            runtime,
            "HOVER",
            own,
            None,
            ego_stamp=now,
            now=now,
            params=BASELINE_PARAMS,
        )

        assert should_publish(result), f"第 {frame + 1} 帧不该停发"
        assert result.point == pytest.approx(anchor)
        assert result.reason == "hover_hold"
        assert result.level == "NORMAL"
        assert result.action == "HOVER"
        assert result.source_stamp == pytest.approx(now)

        data = payload("drone_1", now, result)
        assert data["action"] == "HOVER"
        assert (data["x"], data["y"], data["z"]) == pytest.approx(anchor)
        assert data["stamp"] == pytest.approx(now)
        scenario_chart.add(
            now, own=own, target=None, point=result.point,
            level=result.level, reason=result.reason, series="HOVER",
        )
        timeline.append(
            f"t={now:.2f}s own_x={own[0]:.1f} 发点_x={data['x']:.1f} "
            f"stamp={data['stamp']:.2f}"
        )

    # HOVER 是中枢动作，P3 直接透传：连续 5 帧没有目标，FSM 也不该被驱动。
    assert runtime.fsm.bad_frames == 0
    assert runtime.fsm.last_event.reason == "init"
    assert runtime.fsm.level.value == "NORMAL"

    measured(" | ".join(timeline))


def test_scenario2_single_frame_target_is_not_chased(measured, scenario_chart):
    """场景：目标只在 t=100.00 s 出现一帧，位置 (30,10,5) m；己方在 (28,10,5) m，随后两帧目标消失。

    预期：单帧估不出速度，velocity_valid=False。TRACK 只发最后位置方向上的
    0.74 m 点；INTERCEPT 首帧以 velocity_unavailable 拒发，目标消失后进 L1
    也只发保持点，不追。
    """
    own = (28.0, 10.0, 5.0)
    scenario_chart.configure(
        headline="单帧目标：估不出速度，TRACK 只发 0.74 m，INTERCEPT 拒发",
        subtitle="目标只在 t=100.00 s 出现一帧；v_max=4.50 m/s，"
                 "v_max_intercept=7.60 m/s，max_step=0.74 m/帧。",
    )
    target = predict_target(
        [TargetSample(stamp=100.0, x=30.0, y=10.0, z=5.0)],
        now=100.0,
        max_age=1.0,
        max_speed=15.0,
    )
    assert target is not None
    assert target.velocity == pytest.approx((0.0, 0.0, 0.0))
    assert target.velocity_valid is False

    track_runtime = DroneRuntime.create(BASELINE_PARAMS)
    intercept_runtime = DroneRuntime.create(BASELINE_PARAMS)
    timeline = []

    track1 = plan_drone(
        track_runtime, "TRACK", own, target, 100.0, 100.0, BASELINE_PARAMS
    )
    intercept1 = plan_drone(
        intercept_runtime, "INTERCEPT", own, target, 100.0, 100.0, BASELINE_PARAMS
    )

    assert should_publish(track1)
    assert track1.reason == "track_target_point"
    assert track1.point == pytest.approx((28.74, 10.0, 5.0))
    assert abs(track1.point[0] - own[0]) == pytest.approx(0.74)
    assert not should_publish(intercept1)
    assert intercept1.reason == "velocity_unavailable"
    assert intercept1.level == "NORMAL"
    scenario_chart.add(100.0, own=own, target=target.position,
                       point=track1.point, level=track1.level,
                       reason=track1.reason, series="TRACK")
    scenario_chart.add(100.0, own=own, target=target.position,
                       point=intercept1.point, level=intercept1.level,
                       reason=intercept1.reason, series="INTERCEPT")
    timeline.append(
        "t=100.00s 目标出现1帧 "
        f"TRACK 发点={_fmt(track1.point)} reason={track1.reason} | "
        f"INTERCEPT 不发 reason={intercept1.reason}"
    )

    track2 = plan_drone(
        track_runtime, "TRACK", own, None, 100.05, 100.05, BASELINE_PARAMS
    )
    intercept2 = plan_drone(
        intercept_runtime, "INTERCEPT", own, None, 100.05, 100.05, BASELINE_PARAMS
    )
    assert not should_publish(track2)
    assert not should_publish(intercept2)
    assert track2.reason == "target_missing"
    assert intercept2.reason == "target_missing"
    assert track2.level == "NORMAL"
    scenario_chart.add(100.05, own=own, target=None, point=track2.point,
                       level=track2.level, reason=track2.reason,
                       series="TRACK")
    scenario_chart.add(100.05, own=own, target=None, point=intercept2.point,
                       level=intercept2.level, reason=intercept2.reason,
                       series="INTERCEPT")
    timeline.append(
        "t=100.05s 目标消失第1帧 "
        "TRACK/INTERCEPT 都不发 reason=target_missing"
    )

    track3 = plan_drone(
        track_runtime, "TRACK", own, None, 100.10, 100.10, BASELINE_PARAMS
    )
    intercept3 = plan_drone(
        intercept_runtime, "INTERCEPT", own, None, 100.10, 100.10, BASELINE_PARAMS
    )
    assert track3.level == "EXTRAPOLATING"
    assert track3.reason == "track_target_point"
    assert track3.point == pytest.approx((28.74, 10.0, 5.0))
    assert intercept3.level == "EXTRAPOLATING"
    assert intercept3.reason == "hold_fallback_velocity_unavailable"
    assert intercept3.point == pytest.approx(own)
    scenario_chart.add(100.10, own=own, target=None, point=track3.point,
                       level=track3.level, reason=track3.reason,
                       series="TRACK")
    scenario_chart.add(100.10, own=own, target=None, point=intercept3.point,
                       level=intercept3.level, reason=intercept3.reason,
                       series="INTERCEPT")
    timeline.append(
        "t=100.10s 目标消失第2帧 进 L1 "
        f"TRACK 发点={_fmt(track3.point)} reason={track3.reason} | "
        f"INTERCEPT 保持={_fmt(intercept3.point)} "
        f"reason={intercept3.reason}"
    )

    # 三帧里没有任何一帧把点推到目标最后位置，更没有按 4 m/s 做提前量。
    assert track1.point != target.position
    assert track3.point != target.position
    assert intercept3.point != target.position

    measured(" | ".join(timeline))


def test_scenario3_constant_velocity_is_estimated_and_extrapolated(measured, scenario_chart):
    """场景：目标从 (20,0,10) m 以 4 m/s 沿 +x 匀速直线飞，20 Hz 连采 5 帧；now=100.30 s 时规划。

    预期：P1 用 5 帧最小二乘拟合出 v=(4,0,0)，把最新采样外推到
    (21.2,0,10)。TRACK 点=外推位置；INTERCEPT 解出 t_go≈5.889 s，
    原始拦截点≈(44.76,0,10)，受 max_step=0.74 裁剪后发 (0.74,0,10)。
    """
    buffer = TargetBuffer()
    for frame in range(5):
        stamp = 100.0 + 0.05 * frame
        buffer.offer(
            stamp,
            (20.0 + 4.0 * (stamp - 100.0), 0.0, 10.0),
            stamp,
        )

    state = buffer.predict(100.30, max_age=1.0, max_speed=15.0)
    assert state is not None
    assert state.velocity == pytest.approx((4.0, 0.0, 0.0))
    scenario_chart.configure(
        headline="匀速目标：5 帧拟合 4 m/s，TRACK 跟外推点，INTERCEPT 取最早根",
        subtitle="目标外推到 (21.2,0,10)；INTERCEPT t_go=5.889 s，"
                 "原始点 (44.76,0,10)，max_step=0.74 裁到 (0.74,0,10)。",
    )
    assert state.position == pytest.approx((21.2, 0.0, 10.0))
    assert state.velocity_valid is True
    assert state.age == pytest.approx(0.10)

    track_runtime = DroneRuntime.create(BASELINE_PARAMS)
    track = plan_drone(
        track_runtime,
        "TRACK",
        (20.9, 0.0, 10.0),
        state,
        100.30,
        100.30,
        BASELINE_PARAMS,
    )
    assert should_publish(track)
    assert track.point == pytest.approx((21.2, 0.0, 10.0))
    assert track.reason == "track_target_point"
    assert track.level == "NORMAL"

    intercept_runtime = DroneRuntime.create(BASELINE_PARAMS)
    intercept = plan_drone(
        intercept_runtime,
        "INTERCEPT",
        (0.0, 0.0, 10.0),
        state,
        100.30,
        100.30,
        BASELINE_PARAMS,
    )
    assert should_publish(intercept)
    assert intercept.reachable is True
    assert intercept.reason == "intercept_point"
    assert intercept.t_go == pytest.approx(5.8889, abs=1e-3)
    raw_point_x = state.position[0] + state.velocity[0] * intercept.t_go
    assert raw_point_x == pytest.approx(44.7556, abs=1e-2)
    assert intercept.point == pytest.approx((0.74, 0.0, 10.0))
    assert intercept.clipped is True
    scenario_chart.add(100.30, own=(20.9, 0.0, 10.0),
                       target=state.position, point=track.point,
                       level=track.level, reason=track.reason,
                       series="TRACK")
    scenario_chart.add(100.30, own=(0.0, 0.0, 10.0),
                       target=state.position, point=intercept.point,
                       level=intercept.level, reason=intercept.reason,
                       series="INTERCEPT")

    measured(
        "5 帧拟合 v=(4.00,0,0)，外推 now=100.30 → (21.20,0,10) | "
        f"TRACK 发点={_fmt(track.point)} reason={track.reason} | "
        f"INTERCEPT t_go={intercept.t_go:.3f}s "
        f"原始点=({raw_point_x:.2f},0,10) "
        f"max_step 裁剪后={_fmt(intercept.point)} reason={intercept.reason}"
    )


def test_scenario4_crossing_target_takes_earliest_intercept_root(measured, scenario_chart):
    """场景：目标在 (10,10,5) m 以 10 m/s 向西横穿，己方在 (0,0,5) m；v_max_intercept=7.60 m/s。

    预期：目标比己方快，但朝己方一侧横穿，拦截方程有两个正根：
    t1≈1.435 s、t2≈3.300 s。代码取最早 t1。原始拦截点约 (-4.35,10,5)，
    距己方 10.90 m = 7.6×1.435；max_step=0.74 裁剪后点距己方 0.74 m。
    """
    state = predict_target(
        [
            TargetSample(stamp=99.95, x=10.5, y=10.0, z=5.0),
            TargetSample(stamp=100.0, x=10.0, y=10.0, z=5.0),
        ],
        now=100.0,
        max_age=1.0,
        max_speed=15.0,
    )
    assert state is not None
    assert state.velocity == pytest.approx((-10.0, 0.0, 0.0))
    assert state.velocity_valid is True

    runtime = DroneRuntime.create(BASELINE_PARAMS)
    result = plan_drone(
        runtime,
        "INTERCEPT",
        (0.0, 0.0, 5.0),
        state,
        100.0,
        100.0,
        BASELINE_PARAMS,
    )

    scenario_chart.configure(
        headline="横穿目标：两个正根，取最早的可达根 t=1.435 s",
        subtitle="目标 (10,10,5) 以 10 m/s 向西横穿，己方 (0,0,5)，"
                 "v_max_intercept=7.60 m/s；原始点 (-4.35,10,5)，裁剪后 0.74 m。",
    )

    assert should_publish(result)
    assert result.reachable is True
    assert result.reason == "intercept_point"
    assert result.t_go == pytest.approx(1.4348, abs=1e-3)
    assert result.t_go < 3.3001

    raw_point = (
        state.position[0] + state.velocity[0] * result.t_go,
        state.position[1] + state.velocity[1] * result.t_go,
        state.position[2] + state.velocity[2] * result.t_go,
    )
    assert raw_point[0] == pytest.approx(-4.3477, abs=1e-2)
    assert raw_point[1] == pytest.approx(10.0)
    assert raw_point[2] == pytest.approx(5.0)
    assert hypot(
        raw_point[0], raw_point[1], raw_point[2] - 5.0
    ) == pytest.approx(BASELINE_PARAMS.v_max_intercept * result.t_go, abs=1e-2)

    assert result.point == pytest.approx(
        (-0.2950, 0.6786, 5.0), abs=1e-3
    )
    assert hypot(
        result.point[0], result.point[1], result.point[2] - 5.0
    ) == pytest.approx(0.74)
    assert result.clipped is True
    scenario_chart.add(100.0, own=(0.0, 0.0, 5.0),
                       target=state.position, point=result.point,
                       level=result.level, reason=result.reason,
                       series="INTERCEPT")

    measured(
        "目标 (10,10,5) 以 10 m/s 向西横穿；两个正根 "
        f"t1=1.435s、t2=3.300s，取 t1={result.t_go:.3f}s | "
        f"原始拦截点=({raw_point[0]:.2f},{raw_point[1]:.2f},{raw_point[2]:.2f}) "
        f"距己方 {BASELINE_PARAMS.v_max_intercept * result.t_go:.2f}m | "
        f"max_step 裁剪后={_fmt(result.point)}，距己方 0.74m"
    )


def test_scenario5_faster_target_without_solution_uses_tail_point(measured, scenario_chart):
    """场景：目标在 (30,0,10) m 以 10 m/s 沿 +x 远离，己方在 (0,0,10) m；v_max_intercept=7.60 m/s，d_tail=4.0 m。连跑两帧。

    预期：目标比己方快且远离，拦截方程没有正根。INTERCEPT 发目标后方
    4 m 的尾随点，reachable=False，但仍发布；max_step=0.74 把本帧点
    裁到 (0.74,0,10)。
    """
    buffer = TargetBuffer()
    buffer.offer(99.95, (29.5, 0.0, 10.0), 99.95)
    buffer.offer(100.0, (30.0, 0.0, 10.0), 100.0)

    state1 = buffer.predict(100.0, max_age=1.0, max_speed=15.0)
    assert state1 is not None
    assert state1.velocity == pytest.approx((10.0, 0.0, 0.0))
    assert state1.velocity_valid is True
    assert state1.position == pytest.approx((30.0, 0.0, 10.0))
    assert BASELINE_PARAMS.v_max_intercept < state1.velocity[0]

    runtime = DroneRuntime.create(BASELINE_PARAMS)
    result1 = plan_drone(
        runtime,
        "INTERCEPT",
        (0.0, 0.0, 10.0),
        state1,
        100.0,
        100.0,
        BASELINE_PARAMS,
    )

    raw_tail1 = (
        state1.position[0] - BASELINE_PARAMS.d_tail,
        state1.position[1],
        state1.position[2],
    )
    assert raw_tail1 == pytest.approx((26.0, 0.0, 10.0))
    assert state1.position[0] - raw_tail1[0] == pytest.approx(
        BASELINE_PARAMS.d_tail
    )
    assert should_publish(result1)
    assert result1.reachable is False
    assert result1.reason == "fallback_tail"
    assert result1.t_go is None
    assert result1.point == pytest.approx((0.74, 0.0, 10.0))
    assert hypot(
        result1.point[0], result1.point[1], result1.point[2] - 10.0
    ) == pytest.approx(0.74)
    assert result1.clipped is True
    scenario_chart.configure(
        headline="目标 10 m/s 远离：INTERCEPT 无解，发目标后方 4 m 尾随点",
        subtitle="目标 (30,0,10) 以 10 m/s 沿 +x 逃离，己方 (0,0,10)，"
                 "v_max_intercept=7.60 m/s，d_tail=4.0 m，max_step=0.74 m/帧。",
        mode="decision",
        decision={
            "extra_points": [
                (0.0, 0.0, "己方 (0,0)", "#2F5D8A", "s"),
                (
                    26.0, 0.0, "尾随点原值 (26,0)", "#2E7D32", "*",
                    (16.0, 2.2),
                ),
            ],
            "arrows": [(30.0, 0.0, 40.0, 0.0, "10 m/s 逃离", "#D55E00")],
            "texts": [
                (11.0, 3.2, "目标速度 10 m/s 大于拦截上限 7.60 m/s"),
                (5.0, -2.6, "拦截方程无正根：尾随点仍发布，下一帧可能重解"),
            ],
            "note": "reachable=False，reason=fallback_tail；"
                    "max_step 把本帧点截到 (0.74,0,10)。",
            "xlim": (-2.0, 42.0),
            "ylim": (-5.0, 5.0),
            "pursuit": {
                "target_x0": 30.0,
                "target_speed": 10.0,
                "own_speed": 7.60,
                "duration": 8.0,
                "note": "8 s 时：目标 110 m，己方最多 60.8 m",
            },
        },
    )
    scenario_chart.add(100.0, own=(0.0, 0.0, 10.0),
                       target=state1.position, point=result1.point,
                       level=result1.level, reason=result1.reason,
                       series="INTERCEPT")

    buffer.offer(100.05, (30.5, 0.0, 10.0), 100.05)
    state2 = buffer.predict(100.05, max_age=1.0, max_speed=15.0)
    assert state2 is not None
    assert state2.position == pytest.approx((30.5, 0.0, 10.0))

    result2 = plan_drone(
        runtime,
        "INTERCEPT",
        (0.0, 0.0, 10.0),
        state2,
        100.05,
        100.05,
        BASELINE_PARAMS,
    )

    raw_tail2 = (
        state2.position[0] - BASELINE_PARAMS.d_tail,
        state2.position[1],
        state2.position[2],
    )
    assert raw_tail2 == pytest.approx((26.5, 0.0, 10.0))
    assert should_publish(result2)
    assert result2.reachable is False
    assert result2.reason == "fallback_tail"
    assert result2.t_go is None
    assert result2.point == pytest.approx((0.74, 0.0, 10.0))
    assert result2.clipped is True
    scenario_chart.add(100.05, own=(0.0, 0.0, 10.0),
                       target=state2.position, point=result2.point,
                       level=result2.level, reason=result2.reason,
                       series="INTERCEPT")

    measured(
        "t=100.00 目标 (30,0,10) 10 m/s 远离；"
        f"尾随原始点={_fmt(raw_tail1)}，reachable=False，"
        f"发点={_fmt(result1.point)} reason={result1.reason} | "
        "t=100.05 目标 (30.5,0,10)；"
        f"尾随原始点={_fmt(raw_tail2)}，"
        f"发点={_fmt(result2.point)} reason={result2.reason}"
    )


def test_scenario6_target_behind_triggers_u_turn_check(measured, scenario_chart):
    """目标跑到己方身后：INTERCEPT 判掉头不可达退尾随点，ATTACK 不做该判定。

    场景：己方在 (0,0,5) m 以 5 m/s 沿 +x 飞；目标在 (-2,0,5) m 以
    4 m/s 沿 -x 远离，20 Hz 两帧拟合出 v=(-4,0,0)。v_max_intercept=7.60
    m/s，a_max=9.79 m/s²，yaw_rate_max=172.4°/s，max_step=0.74 m。

    预期：拦截方程正根 t≈0.556 s，原始拦截点 (-4.22,0,5) 在己方身后，
    要 180° 掉头，所需半径 2.11 m < R_min(5 m/s)=2.55 m，INTERCEPT 退
    目标后方 4 m 的尾随点 (2,0,5)，max_step 裁到 (0.74,0,5)。同一几何
    ATTACK 不做掉头判定，照发拦截点 (-2.74,0,5) 方向的 (0.74 m 内) 点
    (-0.74,0,5)。
    """
    buffer = TargetBuffer()
    buffer.offer(99.95, (-1.8, 0.0, 5.0), 99.95)
    buffer.offer(100.0, (-2.0, 0.0, 5.0), 100.0)
    state = buffer.predict(100.0, max_age=1.0, max_speed=15.0)
    assert state is not None
    assert state.velocity == pytest.approx((-4.0, 0.0, 0.0))
    assert state.velocity_valid is True

    own = (0.0, 0.0, 5.0)
    own_velocity = (5.0, 0.0, 0.0)

    scenario_chart.configure(
        headline="目标跑到身后：INTERCEPT 判掉头不可达，ATTACK 不判",
        subtitle="己方 (0,0,5) 以 5 m/s 向东，目标 (-2,0,5) 以 4 m/s 向西；"
                 "R_req=2.11 m < R_min=2.55 m，INTERCEPT 退尾随点 (2,0,5)。",
    )

    # 正根 t=0.5556 s 对应的拦截点在己方身后，需 180° 掉头。
    raw_intercept_x = state.position[0] + state.velocity[0] * 0.5556
    assert raw_intercept_x == pytest.approx(-4.2222, abs=1e-3)
    chord = abs(raw_intercept_x - own[0])
    required_radius = chord / 2.0
    min_radius = max(
        own_velocity[0] ** 2 / BASELINE_PARAMS.a_max,
        own_velocity[0] / BASELINE_PARAMS.yaw_rate_max,
    )
    assert required_radius == pytest.approx(2.1111, abs=1e-3)
    assert min_radius == pytest.approx(2.5536, abs=1e-3)
    assert required_radius < min_radius

    intercept_runtime = DroneRuntime.create(BASELINE_PARAMS)
    intercept = plan_drone(
        intercept_runtime,
        "INTERCEPT",
        own,
        state,
        100.0,
        100.0,
        BASELINE_PARAMS,
        ego_velocity=own_velocity,
    )

    # 掉头不可达：不发拦截点，发目标运动方向后方的尾随点。
    assert should_publish(intercept)
    assert intercept.reason == "dynamic_infeasible_tail"
    assert intercept.reachable is False
    assert intercept.t_go is None
    assert intercept.level == "NORMAL"
    raw_tail = (
        state.position[0] - (-1.0) * BASELINE_PARAMS.d_tail,
        state.position[1],
        state.position[2],
    )
    assert raw_tail == pytest.approx((2.0, 0.0, 5.0))
    assert intercept.point == pytest.approx((0.74, 0.0, 5.0))
    assert intercept.clipped is True

    # 同一几何换 ATTACK：不做掉头判定，照发拦截点，哪怕它也在身后。
    attack_runtime = DroneRuntime.create(BASELINE_PARAMS)
    attack = plan_drone(
        attack_runtime,
        "ATTACK",
        own,
        state,
        100.0,
        100.0,
        BASELINE_PARAMS,
        ego_velocity=own_velocity,
    )
    assert should_publish(attack)
    assert attack.reason == "intercept_point"
    assert attack.reachable is True
    assert attack.t_go == pytest.approx(0.1852, abs=1e-3)
    raw_attack_x = state.position[0] + state.velocity[0] * attack.t_go
    assert raw_attack_x == pytest.approx(-2.7407, abs=1e-3)
    assert raw_attack_x < own[0]
    assert attack.point == pytest.approx((-0.74, 0.0, 5.0))
    assert attack.clipped is True
    scenario_chart.add(100.0, own=(0.0, 0.0, 5.0),
                       target=state.position, point=intercept.point,
                       level=intercept.level, reason=intercept.reason,
                       series="INTERCEPT")
    scenario_chart.add(100.0, own=(0.0, 0.0, 5.0),
                       target=state.position, point=attack.point,
                       level=attack.level, reason=attack.reason,
                       series="ATTACK")

    measured(
        "目标 (-2,0,5) 以 4 m/s 向西远离，己方 (0,0,5) 以 5 m/s 向东飞 | "
        f"INTERCEPT 原始拦截点=({raw_intercept_x:.2f},0,5)，在身后，"
        f"R_req={required_radius:.2f}m < R_min={min_radius:.2f}m → "
        f"退尾随点={_fmt(raw_tail)}，max_step 裁到={_fmt(intercept.point)} "
        f"reason={intercept.reason} | "
        f"ATTACK 不判掉头，t_go={attack.t_go:.3f}s "
        f"原始点=({raw_attack_x:.2f},0,5)，裁到={_fmt(attack.point)} "
        f"reason={attack.reason}"
    )


def test_scenario7_target_loss_0p3s_stale_sample_vs_l1(measured, scenario_chart):
    """目标消失 0.3 s：缓存样本走 L0 外推，目标流断掉走 L1 外推。

    场景：目标以 2 m/s 沿 +x 飞，最后两帧采样在 t=99.95/100.00 s，
    t=100.00 时在 (20,0,10)。t=100.30 时比较两种输入形态：缓存里仍有
    缓存样本，或目标流已经完全消失（target=None）。

    预期：缓存样本 age=0.30 s < max_pred_age=0.50 s，TRACK 仍在 NORMAL，
    P1 已把位置外推到 (20.60,0,10)。目标流消失时，第 1 帧 NORMAL 不发，
    第 2 帧进 L1；t=100.30 时 L1 用最后可用样本加 CV 外推，得到同一个
    (20.60,0,10)。t_extrap_max=1.0 s 才是 L1 的上限，超过才进 L2。
    """
    buffer = TargetBuffer()
    buffer.offer(99.95, (19.9, 0.0, 10.0), 99.95)
    buffer.offer(100.0, (20.0, 0.0, 10.0), 100.0)

    last = buffer.predict(100.0, max_age=1.0, max_speed=15.0)
    assert last is not None
    assert last.position == pytest.approx((20.0, 0.0, 10.0))
    assert last.velocity == pytest.approx((2.0, 0.0, 0.0))
    assert last.velocity_valid is True

    # 形态一：缓存里还有样本。P1 外推到 now，P2 仍按 age<=0.5 放行。
    stale = buffer.predict(100.30, max_age=1.0, max_speed=15.0)
    assert stale is not None
    assert stale.age == pytest.approx(0.30)
    assert stale.valid is True
    assert stale.position == pytest.approx((20.6, 0.0, 10.0))
    assert stale.age < BASELINE_PARAMS.max_pred_age

    scenario_chart.configure(
        headline="目标消失 0.3 s：缓存样本走 NORMAL，目标流断掉走 L1",
        subtitle="最后样本 (20,0,10)，速度 2 m/s；max_pred_age=0.5 s，"
                 "t_extrap_max=1.0 s；两种输入外推到 (20.6,0,10)。",
    )

    stale_runtime = DroneRuntime.create(BASELINE_PARAMS)
    stale_result = plan_drone(
        stale_runtime,
        "TRACK",
        (20.4, 0.0, 10.0),
        stale,
        100.30,
        100.30,
        BASELINE_PARAMS,
    )
    assert should_publish(stale_result)
    assert stale_result.level == "NORMAL"
    assert stale_result.reason == "track_target_point"
    assert stale_result.point == pytest.approx((20.6, 0.0, 10.0))
    scenario_chart.add(100.30, own=(20.4, 0.0, 10.0),
                       target=stale.position, point=stale_result.point,
                       level=stale_result.level,
                       reason=stale_result.reason, series="A 缓存样本")

    # 形态二：目标流完全消失，target=None，逐帧推进状态机。
    runtime = DroneRuntime.create(BASELINE_PARAMS)

    frame1 = plan_drone(
        runtime, "TRACK", (19.8, 0.0, 10.0), last, 100.0, 100.0, BASELINE_PARAMS
    )
    assert should_publish(frame1)
    assert frame1.level == "NORMAL"
    assert frame1.point == pytest.approx((20.0, 0.0, 10.0))
    scenario_chart.add(100.0, own=(19.8, 0.0, 10.0),
                       target=last.position, point=frame1.point,
                       level=frame1.level, reason=frame1.reason,
                       series="B 目标流断掉")

    frame2 = plan_drone(
        runtime, "TRACK", (19.8, 0.0, 10.0), None, 100.05, 100.05, BASELINE_PARAMS
    )
    assert not should_publish(frame2)
    assert frame2.level == "NORMAL"
    assert frame2.reason == "target_missing"
    scenario_chart.add(100.05, own=(19.8, 0.0, 10.0),
                       target=None, point=frame2.point,
                       level=frame2.level, reason=frame2.reason,
                       series="B 目标流断掉")

    frame3 = plan_drone(
        runtime, "TRACK", (19.9, 0.0, 10.0), None, 100.10, 100.10, BASELINE_PARAMS
    )
    assert should_publish(frame3)
    assert frame3.level == "EXTRAPOLATING"
    assert frame3.reason == "track_target_point"
    assert frame3.point == pytest.approx((20.2, 0.0, 10.0))
    scenario_chart.add(100.10, own=(19.9, 0.0, 10.0),
                       target=None, point=frame3.point,
                       level=frame3.level, reason=frame3.reason,
                       series="B 目标流断掉")

    frame4 = plan_drone(
        runtime, "TRACK", (20.4, 0.0, 10.0), None, 100.30, 100.30, BASELINE_PARAMS
    )
    expected_l1 = (
        last.position[0] + last.velocity[0] * 0.30,
        last.position[1] + last.velocity[1] * 0.30,
        last.position[2] + last.velocity[2] * 0.30,
    )
    assert expected_l1 == pytest.approx((20.6, 0.0, 10.0))
    assert should_publish(frame4)
    assert frame4.level == "EXTRAPOLATING"
    assert frame4.reason == "track_target_point"
    assert frame4.point == pytest.approx(expected_l1)
    assert frame4.point == pytest.approx(stale_result.point)
    assert 100.30 - last.stamp < BASELINE_PARAMS.t_extrap_max
    scenario_chart.add(100.30, own=(20.4, 0.0, 10.0),
                       target=None, point=frame4.point,
                       level=frame4.level, reason=frame4.reason,
                       series="B 目标流断掉")

    measured(
        "t=100.30 缓存样本：age=0.30s < max_pred_age=0.50s "
        f"→ NORMAL，P1 外推到 {_fmt(stale_result.point)} | "
        "目标流断：t=100.05 第1帧 NORMAL 不发；"
        "t=100.10 第2帧进 L1，点 (20.20,0,10)；"
        "t=100.30 L1 用最后样本 (20.00,0,10)+v×0.30 "
        f"= {_fmt(frame4.point)}；t_extrap_max=1.0s 才进 L2"
    )
