"""场景注册表的纯 Python 测试：目标轨迹、指标统计、判定边界。

``scenario_suite.py`` 需要 ROS 与 Gazebo，宿主机跑不了；本文件只测
``algorithm_pkg.scenario_defs`` 里的纯函数，保证场景定义本身不自相矛盾。

运行：
    PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
        src/algorithm_pkg/test
"""

import math

import pytest

from algorithm_pkg.scenario_defs import (
    SCENARIOS,
    Scenario,
    Segment,
    count_tail_points,
    evaluate,
    get_scenario,
    replay_gate,
    verdict,
)

ALLOWED_EXPECT = {"HIT", "MISS", "GATE_FIRED"}
ALLOWED_MODE = {"INTERCEPT", "ATTACK"}


def test_scenario_registry_is_consistent(measured):
    """场景：检查注册表本身是否自洽。

    操作：遍历 ``SCENARIOS``，检查 id 唯一、期望值合法、模式合法、
          track 按时间升序且分段位置连续。
    预期：全部通过；新增场景时忘记排序或写错枚举会在这里失败。
    """
    ids = [scenario.id for scenario in SCENARIOS]
    assert len(ids) == len(set(ids)), f"场景 id 重复: {ids}"

    for scenario in SCENARIOS:
        assert scenario.expect in ALLOWED_EXPECT, scenario.id
        assert scenario.mode in ALLOWED_MODE, scenario.id
        assert scenario.track, scenario.id
        times = [segment.start_t for segment in scenario.track]
        assert times == sorted(times), f"{scenario.id} track 未按时间排序"

        for earlier, later in zip(scenario.track, scenario.track[1:]):
            # 后一段的起点必须等于前一段在该时刻的位置。
            span = later.start_t - earlier.start_t
            expected = tuple(
                earlier.start[axis] + earlier.velocity[axis] * span
                for axis in range(3)
            )
            assert later.start == pytest.approx(expected), (
                f"{scenario.id} 在 t={later.start_t} 处轨迹不连续"
            )

    measured(f"{len(SCENARIOS)} 个场景自洽: {', '.join(ids)}")


def test_crossing_track_positions(measured):
    """场景：4 m/s 横穿场景的目标轨迹。

    操作：读 ``intercept_cross_4`` 的 ``target_at``。
    预期：t=0 与 t=5.9 停在 (20,0,8)；t=6 仍在原点；
          t=11 时沿 +y 走了 20 m；速度在 t=6 前后由 0 变 4。
    """
    scenario = get_scenario("intercept_cross_4")

    assert scenario.target_at(0.0) == pytest.approx((20.0, 0.0, 8.0))
    assert scenario.target_at(5.9) == pytest.approx((20.0, 0.0, 8.0))
    assert scenario.target_at(6.0) == pytest.approx((20.0, 0.0, 8.0))
    assert scenario.target_at(11.0) == pytest.approx((20.0, 20.0, 8.0))
    assert scenario.target_velocity(5.0) == pytest.approx((0.0, 0.0, 0.0))
    assert scenario.target_velocity(7.0) == pytest.approx((0.0, 4.0, 0.0))

    measured(f"t=11 位置 {scenario.target_at(11.0)}")


def test_fly_past_track_passes_origin(measured):
    """场景：迎面掠过场景，目标必须真的飞过己方原点。

    操作：读 ``intercept_uturn`` 的 ``target_at``。
    预期：t=6 在 (20,0,8)，t=11 到 (0,0,8)，t=16 到 (-20,0,8)。
    """
    scenario = get_scenario("intercept_uturn")

    assert scenario.target_at(6.0) == pytest.approx((20.0, 0.0, 8.0))
    assert scenario.target_at(11.0) == pytest.approx((0.0, 0.0, 8.0))
    assert scenario.target_at(16.0) == pytest.approx((-20.0, 0.0, 8.0))

    measured(f"t=16 位置 {scenario.target_at(16.0)}")


def _state_row(t, x, y, z, speed):
    return (t, x, y, z, speed)


def test_evaluate_reports_hit(measured):
    """场景：飞机在最近点贴到目标 0.2 m。

    操作：构造 20 帧状态，最近距离 0.2 m，交给 ``evaluate``。
    预期：``min_dist≈0.2``，``verdict`` 返回 PASS（场景期望 HIT）。
    """
    scenario = get_scenario("intercept_cross_4")
    rows = []
    for index in range(20):
        t = 10.0 + index * 0.1
        tx, ty, tz = scenario.target_at(t)
        offset = 0.2 if index == 10 else 2.0
        rows.append(_state_row(t, tx, ty, tz + offset, 4.0))

    metrics = evaluate(scenario, rows, [], start_t=10.0)
    passed, detail = verdict(scenario, metrics)

    assert metrics.min_dist == pytest.approx(0.2)
    assert passed, detail

    measured(f"最近 {metrics.min_dist:.3f} m, 判定 {passed}")


def test_evaluate_reports_miss(measured):
    """场景：飞机全程离目标 5 m，场景期望 HIT。

    操作：同样构造 20 帧，固定 5 m 偏差。
    预期：``verdict`` 返回 FAIL，说明里带最近距离。
    """
    scenario = get_scenario("intercept_cross_4")
    rows = []
    for index in range(20):
        t = 10.0 + index * 0.1
        tx, ty, tz = scenario.target_at(t)
        rows.append(_state_row(t, tx, ty, tz + 5.0, 4.0))

    metrics = evaluate(scenario, rows, [], start_t=10.0)
    passed, detail = verdict(scenario, metrics)

    assert not passed
    assert "5.000" in detail

    measured(f"最近 {metrics.min_dist:.3f} m, 判定 {passed}")


def test_replay_gate_detects_u_turn(measured):
    """场景：己方向东飞，目标在其正西方 2 m，几何有解但必须掉头。

    操作：构造 3 帧速度 4 m/s 向东的状态，交给 ``replay_gate``。
    预期：3 帧全部判为 ``dynamic_infeasible_tail``；
          这是闭环里看不到 reason 时的替代观测手段。
    """
    scenario = Scenario(
        id="unit_uturn",
        title="掉头",
        purpose="单元测试",
        track=(Segment(0.0, (-2.0, 0.0, 8.0), (0.5, 0.0, 0.0)),),
        mode="INTERCEPT",
        run_sec=1.0,
        expect="GATE_FIRED",
    )
    rows = [
        _state_row(0.0, 0.0, 0.0, 8.0, 4.0),
        _state_row(0.1, 0.4, 0.0, 8.0, 4.0),
        _state_row(0.2, 0.8, 0.0, 8.0, 4.0),
    ]

    count = replay_gate(scenario, rows, v_max=7.6)

    assert count == 3

    measured(f"3 帧中触发 {count} 帧")


def test_count_tail_points_matches_geometry(measured):
    """场景：计划点位于目标后方 4 m。

    操作：构造一条正确尾随点和一条错误计划点，交给 ``count_tail_points``。
    预期：只统计正确的那条。
    """
    scenario = get_scenario("intercept_cross_4")
    t = 12.0
    tx, ty, tz = scenario.target_at(t)
    # 目标沿 +y 飞，尾随点应在 -y 方向 4 m。
    good = (t, tx, ty - 4.0, tz, "INTERCEPT")
    bad = (t, tx, ty + 4.0, tz, "INTERCEPT")

    assert count_tail_points(scenario, [good]) == 1
    assert count_tail_points(scenario, [bad]) == 0

    measured("尾随点判定只认目标后方")


def test_verdict_gate_fired_requires_frames(measured):
    """场景：场景期望 GATE_FIRED，但没有任何一帧触发。

    操作：构造空指标，交给 ``verdict``。
    预期：FAIL；判定不能因为“跑了”就算过。
    """
    scenario = get_scenario("intercept_uturn")
    rows = [_state_row(10.0, 0.0, 0.0, 8.0, 4.0)]
    metrics = evaluate(scenario, rows, [], start_t=10.0)

    passed, detail = verdict(scenario, metrics)

    assert not passed
    assert "触发 0 帧" in detail

    measured(f"判定 {passed}, {detail}")
