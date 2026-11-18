"""P3 异常回放：把第 5 节列的 11 种异常场景按时间顺序跑一遍。

回放只调用纯算法层（``validators`` / ``degrade_fsm`` /
``degrade_policy``），不启动 ROS、不连飞控。

运行：
    PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
        src/algorithm_pkg/test
"""

import math

import pytest

from algorithm_pkg.algorithm_core import TargetState
from algorithm_pkg.degrade_fsm import DegradeFSM, DegradeLevel
from algorithm_pkg.degrade_policy import (
    DISCARDED_TIME_REVERSED,
    plan_with_degrade,
)
from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.validators import (
    ANOMALY_FREQ,
    ANOMALY_INVALID,
    CmdCode,
    validate_cmd,
)


def make_target(
    now,
    *,
    position=(10.0, 0.0, 10.0),
    velocity=(0.0, 0.0, 0.0),
    age=0.0,
    stamp=None,
    valid=None,
):
    """构造目标状态；age>0 时 stamp 自动回退。"""
    stamp_value = now - age if stamp is None else stamp
    if valid is None:
        valid = age <= 1.0
    return TargetState(
        position=position,
        velocity=velocity,
        stamp=stamp_value,
        age=age,
        valid=valid,
        velocity_valid=True,
    )


def fmt_vec(value):
    """把三维点格式化成便于摘要阅读的字符串。"""
    if value is None:
        return "None"
    return f"({value[0]:.3f}, {value[1]:.3f}, {value[2]:.3f})"


def tick(
    fsm,
    params,
    now,
    *,
    action="TRACK",
    own=(0.0, 0.0, 10.0),
    target=None,
    prev_result=None,
):
    """调用一次 plan_with_degrade()。"""
    return plan_with_degrade(
        action,
        own,
        target,
        fsm,
        params,
        now,
        prev_result,
    )


def feed_fresh(fsm, params, times, **kwargs):
    """按给定时刻连续喂新鲜目标，返回最后一个结果。"""
    result = None
    for now in times:
        result = tick(
            fsm,
            params,
            now,
            target=make_target(now),
            prev_result=result,
            **kwargs,
        )
    return result


# ============================================================
# 1. 目标短暂丢失 3 帧 → L1 外推 → 平滑恢复
# ============================================================


def test_replay_short_target_loss_uses_l1_and_recovers(measured):
    """场景：目标正常 5 帧，丢失 3 帧（< t_extrap_max），再恢复。

    操作：10Hz 回放，丢帧期间 target=None。
    输入：max_pred_age=0.5，t_extrap_max=1.0，n_enter=2，n_recover=3。
    预期：丢失期间出现 L1 外推点；恢复经过 HOLDING 确认后回到 NORMAL。
    """
    params = PlanParams()
    fsm = DegradeFSM(params)
    prev = feed_fresh(fsm, params, [index * 0.1 for index in range(5)])

    lost_levels = []
    for index in range(3):
        now = 0.5 + index * 0.1
        prev = tick(fsm, params, now, target=None, prev_result=prev)
        lost_levels.append(prev.level)

    recovery_levels = []
    for index in range(4):
        now = 0.8 + index * 0.1
        prev = tick(
            fsm,
            params,
            now,
            target=make_target(now),
            prev_result=prev,
        )
        recovery_levels.append(prev.level)

    assert DegradeLevel.EXTRAPOLATING.value in lost_levels
    assert prev.level == DegradeLevel.NORMAL.value

    measured(
        f"丢失期={lost_levels}, 恢复期={recovery_levels}, "
        f"最终={prev.level}"
    )


# ============================================================
# 2. 目标丢失 20 帧 → L1 → L2 → 恢复
# ============================================================


def test_replay_long_target_loss_enters_l2_and_recovers(measured):
    """场景：目标正常 5 帧，丢失 20 帧（> t_extrap_max），再恢复。

    操作：10Hz 回放，丢帧 2 秒。
    输入：t_extrap_max=1.0。
    预期：丢失期出现 L1，随后进 L2；恢复经过 HOLDING 后回 NORMAL。
    """
    params = PlanParams()
    fsm = DegradeFSM(params)
    prev = feed_fresh(fsm, params, [index * 0.1 for index in range(5)])

    lost_levels = []
    for index in range(20):
        now = 0.5 + index * 0.1
        prev = tick(fsm, params, now, target=None, prev_result=prev)
        lost_levels.append(prev.level)

    assert DegradeLevel.EXTRAPOLATING.value in lost_levels
    assert lost_levels[-1] == DegradeLevel.HOLDING.value

    recovery_levels = []
    for index in range(6):
        now = 2.5 + index * 0.1
        prev = tick(
            fsm,
            params,
            now,
            target=make_target(now),
            prev_result=prev,
        )
        recovery_levels.append(prev.level)

    assert prev.level == DegradeLevel.NORMAL.value

    measured(
        f"丢失期={lost_levels}, 恢复期={recovery_levels}, "
        f"最终={prev.level}"
    )


# ============================================================
# 3. 目标 NaN 1 帧 → 不崩、不输出 NaN
# ============================================================


def test_replay_single_nan_target_does_not_output_nan(measured):
    """场景：目标正常 5 帧后突然出现 1 帧 NaN。

    操作：1 帧 NaN 后马上恢复正常。
    输入：n_enter=2，单帧非法不立刻进 L3。
    预期：该帧 point=None；不抛异常；下一帧正常发点，且没有 NaN。
    """
    params = PlanParams()
    fsm = DegradeFSM(params)
    prev = feed_fresh(fsm, params, [index * 0.1 for index in range(5)])

    nan_target = TargetState(
        position=(math.nan, 0.0, 10.0),
        velocity=(0.0, 0.0, 0.0),
        stamp=0.5,
        age=0.0,
        valid=True,
        velocity_valid=True,
    )
    bad = tick(fsm, params, 0.5, target=nan_target, prev_result=prev)
    good = tick(
        fsm,
        params,
        0.6,
        target=make_target(0.6),
        prev_result=bad,
    )

    assert bad.point is None
    assert bad.anomaly == ANOMALY_INVALID
    assert good.point is not None
    assert all(math.isfinite(component) for component in good.point)

    measured(
        f"NaN 帧 point={bad.point}, anomaly={bad.anomaly}; "
        f"下一帧 point={fmt_vec(good.point)}"
    )


# ============================================================
# 4. 目标时间戳倒退 3 帧 → 丢弃、不降级
# ============================================================


def test_replay_reversed_stamps_are_dropped_without_degrade(measured):
    """场景：目标时间戳连续倒退 3 帧。

    操作：正常 5 帧后喂 stamp 0.10/0.15/0.20 的坏帧，再恢复。
    输入：上一帧 stamp=0.4。
    预期：3 个坏帧 point=None、reason=frame_discarded_time_reversed、
          级别保持 NORMAL；恢复帧正常发点。
    """
    params = PlanParams()
    fsm = DegradeFSM(params)
    prev = feed_fresh(fsm, params, [index * 0.1 for index in range(5)])

    bad_results = []
    for index, stamp in enumerate((0.10, 0.15, 0.20)):
        now = 0.5 + index * 0.05
        prev = tick(
            fsm,
            params,
            now,
            target=make_target(now, stamp=stamp),
            prev_result=prev,
        )
        bad_results.append(prev)

    recovered = tick(
        fsm,
        params,
        0.65,
        target=make_target(0.65),
        prev_result=prev,
    )

    assert all(
        result.point is None
        and result.reason == DISCARDED_TIME_REVERSED
        and result.level == DegradeLevel.NORMAL.value
        for result in bad_results
    )
    assert recovered.point is not None

    measured(
        f"3 个坏帧={[(r.reason, r.level) for r in bad_results]}, "
        f"恢复点={fmt_vec(recovered.point)}"
    )


# ============================================================
# 5. 己方状态断流 → L3 → 恢复
# ============================================================


def test_replay_ego_lost_enters_l3_and_recovers(measured):
    """场景：目标一直新鲜，但己方状态断流 3 帧。

    操作：先正常 5 帧，再 own=None 3 帧，然后恢复。
    输入：n_enter=2，n_exit=1，n_recover=3。
    预期：断流期进 L3；恢复先回 HOLDING，再回 NORMAL。
    """
    params = PlanParams()
    fsm = DegradeFSM(params)
    prev = feed_fresh(fsm, params, [index * 0.1 for index in range(5)])

    lost_levels = []
    for index in range(3):
        now = 0.5 + index * 0.1
        prev = tick(
            fsm,
            params,
            now,
            own=None,
            target=make_target(now),
            prev_result=prev,
        )
        lost_levels.append(prev.level)

    recovery_levels = []
    for index in range(4):
        now = 0.8 + index * 0.1
        prev = tick(
            fsm,
            params,
            now,
            own=(0.0, 0.0, 10.0),
            target=make_target(now),
            prev_result=prev,
        )
        recovery_levels.append(prev.level)

    assert DegradeLevel.NO_PLAN.value in lost_levels
    assert prev.level == DegradeLevel.NORMAL.value

    measured(
        f"断流期={lost_levels}, 恢复期={recovery_levels}, "
        f"最终={prev.level}"
    )


# ============================================================
# 6. 目标丢失/恢复反复抖动 → 滞回，不横跳
# ============================================================


def test_replay_flapping_target_does_not_oscillate(measured):
    """场景：目标时有时无，每次只丢 1 帧。

    操作：新鲜/丢失交替 10 轮。
    输入：n_enter=2。
    预期：级别始终 NORMAL，不出现“有点/没点”反复横跳。
    """
    params = PlanParams()
    fsm = DegradeFSM(params)
    prev = None
    levels = []

    for index in range(10):
        now = index * 0.2
        prev = tick(
            fsm,
            params,
            now,
            target=make_target(now),
            prev_result=prev,
        )
        levels.append(prev.level)
        prev = tick(
            fsm,
            params,
            now + 0.1,
            target=None,
            prev_result=prev,
        )
        levels.append(prev.level)

    assert all(level == DegradeLevel.NORMAL.value for level in levels)

    measured(f"交替 10 轮后的级别集合={sorted(set(levels))}")


# ============================================================
# 7. ATTACK 目标丢失 → 不违反 ATTACK 语义
# ============================================================


def test_replay_attack_loss_never_returns_tail_point(measured):
    """场景：ATTACK 追不上，随后目标丢失。

    操作：攻击速度 6，目标 8 m/s 远离；正常 5 帧后丢失 20 帧。
    输入：ATTACK 几何无解，P2 绝不尾随。
    预期：任何一帧都不出现 reason=fallback_tail 或尾随点 (6,0,10)；
          L1/L2 用显式保持点。
    """
    params = PlanParams(v_max_attack=6.0)
    fsm = DegradeFSM(params)
    prev = None
    for index in range(5):
        now = index * 0.1
        prev = tick(
            fsm,
            params,
            now,
            action="ATTACK",
            target=make_target(now, velocity=(8.0, 0.0, 0.0)),
            prev_result=prev,
        )

    for index in range(20):
        now = 0.5 + index * 0.1
        prev = tick(
            fsm,
            params,
            now,
            action="ATTACK",
            target=None,
            prev_result=prev,
        )
        assert prev.reason != "fallback_tail"
        assert prev.point != pytest.approx((6.0, 0.0, 10.0))

    assert prev.point == pytest.approx((0.0, 0.0, 10.0))
    assert prev.reason.startswith("hold_")

    measured(
        f"最终 level={prev.level}, reason={prev.reason}, "
        f"point={fmt_vec(prev.point)}"
    )


# ============================================================
# 8. 输出越界/跳变 → 护栏裁剪，不降级
# ============================================================


def test_replay_out_of_bounds_output_is_clipped_not_degraded(measured):
    """场景：TRACK 目标点只有 1 米高，输出需要抬到安全高度。

    操作：TRACK 一帧。
    输入：target=(10,0,1)。
    预期：point z=5，clipped=True，level 仍 NORMAL；护栏只裁剪不降级。
    """
    params = PlanParams()
    fsm = DegradeFSM(params)
    result = tick(
        fsm,
        params,
        1.0,
        target=make_target(1.0, position=(10.0, 0.0, 1.0)),
    )

    assert result.level == DegradeLevel.NORMAL.value
    assert result.point == pytest.approx((10.0, 0.0, 5.0))
    assert result.clipped

    measured(
        f"level={result.level}, point={fmt_vec(result.point)}, "
        f"clipped={result.clipped}"
    )


def test_replay_goal_jump_is_clipped_not_degraded(measured):
    """场景：两帧目标点相差 sqrt(2) 米，超过 max_step=1。

    操作：TRACK 两帧，目标从 (1,0,10) 跳到 (0,1,10)。
    输入：max_step=1.0。
    预期：第二帧被裁到距第一帧 1 米处；level 仍 NORMAL。
    """
    params = PlanParams(max_step=1.0)
    fsm = DegradeFSM(params)

    first = tick(
        fsm,
        params,
        1.0,
        target=make_target(1.0, position=(1.0, 0.0, 10.0)),
    )
    second = tick(
        fsm,
        params,
        1.05,
        target=make_target(1.05, position=(0.0, 1.0, 10.0)),
        prev_result=first,
    )

    expected = (1.0 - 0.5 ** 0.5, 0.5 ** 0.5, 10.0)
    assert second.point == pytest.approx(expected)
    assert second.level == DegradeLevel.NORMAL.value
    assert second.clipped

    measured(
        f"first={fmt_vec(first.point)}, second={fmt_vec(second.point)}, "
        f"level={second.level}"
    )


# ============================================================
# 9. 恢复第一帧跳变 → max_step_recover 生效
# ============================================================


def test_replay_recovery_first_frame_uses_recover_step(measured):
    """场景：目标从 L2 恢复，恢复第一帧相对上一目标点只能走 0.2 米。

    操作：建立上一目标点 (0.5,0,10)，丢到 L2，再让目标出现在 (5,0,10)。
    输入：max_step=1.0，max_step_recover=0.2，n_recover=3。
    预期：回 NORMAL 第一帧 point=(0.7,0,10)。
    """
    params = PlanParams(
        max_step=1.0,
        max_step_recover=0.2,
        n_recover=3,
        n_recover_step=3,
        max_pred_age=0.5,
        t_extrap_max=1.0,
    )
    fsm = DegradeFSM(params)
    prev = None
    own = (0.0, 0.0, 10.0)

    for index in range(5):
        now = index * 0.05
        prev = tick(
            fsm,
            params,
            now,
            own=own,
            target=make_target(
                now, position=(0.5, 0.0, 10.0)
            ),
            prev_result=prev,
        )
    own = (0.5, 0.0, 10.0)

    for index in range(50):
        now = 0.25 + index * 0.05
        prev = tick(
            fsm,
            params,
            now,
            own=own,
            target=None,
            prev_result=prev,
        )
        if prev.level == DegradeLevel.HOLDING.value:
            break

    for index in range(3):
        now = 1.8 + index * 0.05
        prev = tick(
            fsm,
            params,
            now,
            own=own,
            target=make_target(
                now, position=(5.0, 0.0, 10.0)
            ),
            prev_result=prev,
        )

    assert prev.level == DegradeLevel.NORMAL.value
    assert prev.point == pytest.approx((0.7, 0.0, 10.0))

    measured(
        f"level={prev.level}, point={fmt_vec(prev.point)}"
    )


# ============================================================
# 10. 指令 JSON 非法 → no_plan + reason；下一帧合法恢复
# ============================================================


def test_replay_invalid_cmd_then_valid_recovers(measured):
    """场景：/strategy/cmd 的 JSON 解析失败，下一帧又合法。

    操作：先 validate_cmd 坏 JSON；再 validate_cmd 合法 JSON；
          最后用合法动作跑一帧 plan_with_degrade。
    输入："{not json" 和 {"drone_id":"drone_1","action":"TRACK"}。
    预期：坏指令 reason=cmd_json_invalid；合法指令 code=FRESH；
          规划层按 TRACK 正常发点。
    """
    bad_cmd = validate_cmd("{not json")
    good_cmd = validate_cmd(
        '{"drone_id":"drone_1","action":"TRACK"}'
    )

    params = PlanParams()
    fsm = DegradeFSM(params)
    result = plan_with_degrade(
        good_cmd.action,
        (0.0, 0.0, 10.0),
        make_target(1.0),
        fsm,
        params,
        1.0,
    )

    assert bad_cmd.code is CmdCode.INVALID
    assert bad_cmd.reason == "cmd_json_invalid"
    assert good_cmd.code is CmdCode.FRESH
    assert result.point is not None

    measured(
        f"bad={bad_cmd.reason}, good={good_cmd.code.value}, "
        f"plan={result.reason}"
    )


def test_replay_unknown_action_is_l3_no_plan(measured):
    """场景：合法 JSON 但 action 拼错。

    操作：validate_cmd 后把动作交给 plan_with_degrade。
    输入：action="TRACKK" / "ATTACKK"。
    预期：规划层返回 level=NO_PLAN + reason=unknown_action，不出点。
    """
    cmd = validate_cmd(
        '{"drone_id":"drone_1","action":"ATTACKK"}'
    )
    params = PlanParams()
    fsm = DegradeFSM(params)
    result = plan_with_degrade(
        cmd.action,
        (0.0, 0.0, 10.0),
        make_target(1.0),
        fsm,
        params,
        1.0,
    )

    assert cmd.code is CmdCode.INVALID
    assert cmd.reason == "cmd_unknown_action"
    assert result.level == DegradeLevel.NO_PLAN.value
    assert result.point is None
    assert result.reason == "unknown_action"

    measured(
        f"cmd={cmd.reason}, plan={result.level}/{result.reason}"
    )


# ============================================================
# 11. 目标频率 10Hz → 1Hz → 可解释
# ============================================================


def test_replay_frequency_drop_is_explainable(measured):
    """场景：目标从 10Hz 掉到 1Hz，频率检测开启和关闭各跑一遍。

    操作：先 10 帧 10Hz，再连续 8 帧 1Hz。
    输入：freq_anomaly_enabled=True/False，阈值 0.5，窗口 10。
    预期：开启时第一帧 1Hz 命中 FREQ_ANOMALY，窗口滑过后恢复；
          关闭时始终不命中；两种情况 level 都是 NORMAL。
    """
    enabled_params = PlanParams(
        freq_anomaly_enabled=True,
        freq_anomaly_threshold=0.5,
        freq_window_size=10,
    )
    enabled_fsm = DegradeFSM(enabled_params)
    for index in range(10):
        tick(
            enabled_fsm,
            enabled_params,
            index * 0.1,
            target=make_target(index * 0.1),
        )

    enabled_results = []
    for index in range(8):
        now = 1.9 + index * 1.0
        enabled_results.append(
            tick(
                enabled_fsm,
                enabled_params,
                now,
                target=make_target(now),
            )
        )

    disabled_params = PlanParams(
        freq_anomaly_enabled=False,
        freq_window_size=10,
    )
    disabled_fsm = DegradeFSM(disabled_params)
    for index in range(10):
        tick(
            disabled_fsm,
            disabled_params,
            index * 0.1,
            target=make_target(index * 0.1),
        )
    disabled_results = [
        tick(
            disabled_fsm,
            disabled_params,
            1.9 + index * 1.0,
            target=make_target(1.9 + index * 1.0),
        )
        for index in range(8)
    ]

    assert enabled_results[0].anomaly == ANOMALY_FREQ
    assert enabled_results[-1].anomaly is None
    assert all(result.anomaly is None for result in disabled_results)
    assert all(
        result.level == DegradeLevel.NORMAL.value
        for result in enabled_results + disabled_results
    )

    measured(
        "enabled 首帧 anomaly="
        f"{enabled_results[0].anomaly}, 末帧 anomaly="
        f"{enabled_results[-1].anomaly}; disabled 全部=None"
    )
