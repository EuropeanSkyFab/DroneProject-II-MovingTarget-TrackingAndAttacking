"""P3 分级降级输出测试：四个级别 × 七动作的关键路径。

只测 ``degrade_policy.plan_with_degrade()``，不启动 ROS。

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
    NO_PLAN_EGO_LOST,
    NO_PLAN_TARGET_INVALID,
    plan_with_degrade,
)
from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.plan_types import PlanResult
from algorithm_pkg.validators import (
    ANOMALY_CMD_INVALID,
    ANOMALY_FREQ,
    ANOMALY_STALE,
    ANOMALY_TIME_REVERSED,
)


def make_target(
    now,
    *,
    position=(10.0, 0.0, 10.0),
    velocity=(1.0, 0.0, 0.0),
    age=0.0,
    velocity_valid=True,
    stamp=None,
):
    """构造目标状态；age>0 表示最后一次采样已经过去多久。"""
    return TargetState(
        position=position,
        velocity=velocity,
        stamp=now - age if stamp is None else stamp,
        age=age,
        valid=age <= 1.0,
        velocity_valid=velocity_valid,
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


def test_normal_track_calls_p2_and_sets_stamp(measured):
    """场景：输入完全正常。

    操作：TRACK 一帧，调用 plan_with_degrade()。
    输入：own=(0,0,10)，target=(10,0,10)，now=1.0。
    预期：level=NORMAL，point=(10,0,10)，output_stamp=1.0，无异常。
    """
    params = PlanParams()
    fsm = DegradeFSM(params)
    result = tick(
        fsm,
        params,
        1.0,
        target=make_target(1.0),
    )

    assert result.level == DegradeLevel.NORMAL.value
    assert result.anomaly is None
    assert result.point == pytest.approx((10.0, 0.0, 10.0))
    assert result.output_stamp == pytest.approx(1.0)
    assert result.reason == "track_target_point"

    measured(
        f"level={result.level}, point={fmt_vec(result.point)}, "
        f"stamp={result.output_stamp}"
    )


def test_normal_attack_no_solution_stays_normal(measured):
    """场景：ATTACK 追不上，但输入本身没有异常。

    操作：调用 plan_with_degrade()，攻击速度压到 6。
    输入：target 速度 8 m/s。
    预期：point=None，reason=attack_no_solution，level=NORMAL；
          降级层不为几何无解背锅。
    """
    params = PlanParams(v_max_attack=6.0)
    fsm = DegradeFSM(params)
    result = tick(
        fsm,
        params,
        1.0,
        action="ATTACK",
        target=make_target(1.0, velocity=(8.0, 0.0, 0.0)),
    )

    assert result.point is None
    assert result.reason == "attack_no_solution"
    assert result.level == DegradeLevel.NORMAL.value

    measured(f"level={result.level}, reason={result.reason}")


def test_hover_returns_current_position_without_fsm_change(measured):
    """场景：动作是 HOVER，宪章要求原地保持不动。

    操作：调用 plan_with_degrade("HOVER", ...)。
    输入：own=(0,0,10)，目标有效。
    预期：point=己方当前位置，reason=hover_hold，level=NORMAL；
          FSM 级别和计数不变。
    """
    params = PlanParams()
    fsm = DegradeFSM(params)
    result = tick(
        fsm,
        params,
        1.0,
        action="HOVER",
        target=make_target(1.0),
    )

    assert result.point == pytest.approx((0.0, 0.0, 10.0))
    assert result.reason == "hover_hold"
    assert result.level == DegradeLevel.NORMAL.value
    assert fsm.level is DegradeLevel.NORMAL
    assert fsm.bad_frames == 0

    measured(f"point={fmt_vec(result.point)}, reason={result.reason}, fsm.level={fsm.level.value}")


def test_unknown_action_is_l3_cmd_invalid(measured):
    """场景：动作拼错，不能默认 HOVER。

    操作：调用 plan_with_degrade("TRACKK", ...)。
    输入：未知动作。
    预期：level=NO_PLAN，anomaly=CMD_INVALID，point=None；
          FSM 不被未知动作驱动。
    """
    params = PlanParams()
    fsm = DegradeFSM(params)
    result = tick(
        fsm,
        params,
        1.0,
        action="TRACKK",
        target=make_target(1.0),
    )

    assert result.level == DegradeLevel.NO_PLAN.value
    assert result.anomaly == ANOMALY_CMD_INVALID
    assert result.reason == "unknown_action"
    assert result.point is None
    assert fsm.level is DegradeLevel.NORMAL

    measured(
        f"level={result.level}, anomaly={result.anomaly}, "
        f"reason={result.reason}"
    )


def test_l1_track_uses_extrapolated_target(measured):
    """场景：目标短暂丢失，仍在 t_extrap_max=1.0 内。

    操作：前 5 帧正常，随后两帧目标年龄 0.7 秒。
    输入：过期目标的位置已经由 P1 外推到 now。
    预期：第 2 个过期帧进入 L1；TRACK 发外推点，不掉成 no_plan。
    """
    params = PlanParams()
    fsm = DegradeFSM(params)

    for index in range(5):
        tick(fsm, params, index * 0.05, target=make_target(index * 0.05))

    first = tick(
        fsm,
        params,
        0.8,
        target=make_target(0.8, position=(10.8, 0.0, 10.0), age=0.6),
    )
    second = tick(
        fsm,
        params,
        0.9,
        target=make_target(0.9, position=(10.9, 0.0, 10.0), age=0.7),
        prev_result=first,
    )

    assert second.level == DegradeLevel.EXTRAPOLATING.value
    assert second.anomaly == ANOMALY_STALE
    assert second.point == pytest.approx((10.9, 0.0, 10.0))

    measured(
        f"first.level={first.level}, second.level={second.level}, "
        f"point={fmt_vec(second.point)}"
    )


def test_l1_attack_no_solution_never_returns_tail(measured):
    """场景：ATTACK 追不上，目标又短暂丢失。

    操作：攻击速度 6，目标 8 m/s 远离；先正常 5 帧，再两帧 age=0.7。
    输入：ATTACK 几何无解。
    预期：L1 打不出拦截点时用显式保持点；不能输出尾随点 (6,0,10)。
    """
    params = PlanParams(v_max_attack=6.0)
    fsm = DegradeFSM(params)

    for index in range(5):
        tick(
            fsm,
            params,
            index * 0.05,
            action="ATTACK",
            target=make_target(
                index * 0.05, velocity=(8.0, 0.0, 0.0)
            ),
        )

    first = tick(
        fsm,
        params,
        0.8,
        action="ATTACK",
        target=make_target(
            0.8, velocity=(8.0, 0.0, 0.0), age=0.6
        ),
    )
    second = tick(
        fsm,
        params,
        0.9,
        action="ATTACK",
        target=make_target(
            0.9, velocity=(8.0, 0.0, 0.0), age=0.7
        ),
        prev_result=first,
    )

    assert second.level == DegradeLevel.EXTRAPOLATING.value
    assert second.reason == "hold_fallback_attack_no_solution"
    assert second.point == pytest.approx((0.0, 0.0, 10.0))
    assert second.point != pytest.approx((6.0, 0.0, 10.0))

    measured(
        f"level={second.level}, reason={second.reason}, "
        f"point={fmt_vec(second.point)}（不是尾随点）"
    )


def test_l1_intercept_still_allows_tail_point(measured):
    """场景：INTERCEPT 追不上，目标又短暂丢失。

    操作：拦截速度 6，目标 8 m/s 远离；进入 L1 后看输出。
    输入：INTERCEPT 几何无解。
    预期：level=L1，reason=fallback_tail，点 (6,0,10)；
          INTERCEPT 的尾随语义在降级层仍然保留。
    """
    params = PlanParams(v_max_intercept=6.0)
    fsm = DegradeFSM(params)

    for index in range(5):
        tick(
            fsm,
            params,
            index * 0.05,
            action="INTERCEPT",
            target=make_target(
                index * 0.05, velocity=(8.0, 0.0, 0.0)
            ),
        )
    first = tick(
        fsm,
        params,
        0.8,
        action="INTERCEPT",
        target=make_target(
            0.8, velocity=(8.0, 0.0, 0.0), age=0.6
        ),
    )
    second = tick(
        fsm,
        params,
        0.9,
        action="INTERCEPT",
        target=make_target(
            0.9, velocity=(8.0, 0.0, 0.0), age=0.7
        ),
        prev_result=first,
    )

    assert second.level == DegradeLevel.EXTRAPOLATING.value
    assert second.reason == "fallback_tail"
    assert second.point == pytest.approx((6.0, 0.0, 10.0))

    measured(
        f"level={second.level}, reason={second.reason}, "
        f"point={fmt_vec(second.point)}"
    )


def test_l2_holds_last_valid_point(measured):
    """场景：目标彻底丢失，超过 t_extrap_max，进入 L2。

    操作：先正常 5 帧，再连续丢帧直到 HOLDING。
    输入：hold_point_source=last_valid。
    预期：reason=hold_last_valid，输出的点等于进 L2 前的最后一个有效点。
    """
    params = PlanParams()
    fsm = DegradeFSM(params)

    for index in range(5):
        tick(fsm, params, index * 0.05, target=make_target(index * 0.05))

    prev = None
    hold_result = None
    before_hold = None
    for index in range(50):
        now = 0.25 + index * 0.05
        before_hold = prev
        prev = tick(fsm, params, now, target=None, prev_result=prev)
        if prev.level == DegradeLevel.HOLDING.value:
            hold_result = prev
            break

    assert hold_result is not None
    assert hold_result.reason == "hold_last_valid"
    assert hold_result.point == pytest.approx(before_hold.point)

    measured(
        f"level={hold_result.level}, reason={hold_result.reason}, "
        f"point={fmt_vec(hold_result.point)}"
    )


def test_l2_can_hold_current_ego(measured):
    """场景：配置 hold_point_source=current_ego。

    操作：目标丢失到 L2，己方在 (1,2,10)。
    输入：hold_point_source=current_ego。
    预期：reason=hold_current_ego，点就是己方当前位置。
    """
    params = PlanParams(hold_point_source="current_ego")
    fsm = DegradeFSM(params)

    for index in range(5):
        tick(
            fsm,
            params,
            index * 0.05,
            own=(1.0, 2.0, 10.0),
            target=make_target(index * 0.05),
        )

    result = None
    for index in range(50):
        result = tick(
            fsm,
            params,
            0.25 + index * 0.05,
            own=(1.0, 2.0, 10.0),
            target=None,
            prev_result=result,
        )
        if result.level == DegradeLevel.HOLDING.value:
            break

    assert result.reason == "hold_current_ego"
    assert result.point == pytest.approx((1.0, 2.0, 10.0))

    measured(
        f"level={result.level}, reason={result.reason}, "
        f"point={fmt_vec(result.point)}"
    )


def test_l3_ego_lost_returns_explicit_no_plan(measured):
    """场景：己方状态断流。

    操作：先正常 5 帧，再连续两帧 own=None。
    输入：n_enter=2。
    预期：level=NO_PLAN，reason=no_plan_ego_lost，point=None。
    """
    params = PlanParams()
    fsm = DegradeFSM(params)

    for index in range(5):
        tick(fsm, params, index * 0.05, target=make_target(index * 0.05))

    first = tick(
        fsm,
        params,
        0.5,
        own=None,
        target=make_target(0.5),
    )
    second = tick(
        fsm,
        params,
        0.55,
        own=None,
        target=make_target(0.55),
        prev_result=first,
    )

    assert second.level == DegradeLevel.NO_PLAN.value
    assert second.reason == NO_PLAN_EGO_LOST
    assert second.point is None

    measured(
        f"level={second.level}, reason={second.reason}, "
        f"point={second.point}"
    )


def test_l3_invalid_target_returns_no_plan(measured):
    """场景：目标位置是 NaN。

    操作：连续两帧喂 NaN 目标。
    输入：n_enter=2。
    预期：level=NO_PLAN，reason=no_plan_target_invalid。
    """
    params = PlanParams()
    fsm = DegradeFSM(params)
    bad = TargetState(
        position=(math.nan, 0.0, 10.0),
        velocity=(0.0, 0.0, 0.0),
        stamp=1.0,
        age=0.0,
        valid=True,
        velocity_valid=True,
    )

    first = tick(fsm, params, 1.0, target=bad)
    second = tick(fsm, params, 1.05, target=bad, prev_result=first)

    assert second.level == DegradeLevel.NO_PLAN.value
    assert second.reason == NO_PLAN_TARGET_INVALID
    assert second.point is None

    measured(
        f"level={second.level}, reason={second.reason}, "
        f"point={second.point}"
    )


def test_time_reversed_frame_is_discarded_without_level_change(measured):
    """场景：目标时间戳从 0.2 倒退到 0.1。

    操作：先正常 5 帧，再喂一帧时间戳倒退的目标，最后恢复正常。
    输入：last_stamp=0.2，坏帧 stamp=0.1。
    预期：坏帧 point=None、reason=frame_discarded_time_reversed、
          level 仍 NORMAL；下一帧正常发点。
    """
    params = PlanParams()
    fsm = DegradeFSM(params)

    for index in range(5):
        tick(fsm, params, index * 0.05, target=make_target(index * 0.05))

    reversed_frame = tick(
        fsm,
        params,
        0.4,
        target=make_target(0.4, stamp=0.1),
    )
    recovered = tick(
        fsm,
        params,
        0.45,
        target=make_target(0.45),
        prev_result=reversed_frame,
    )

    assert reversed_frame.point is None
    assert reversed_frame.reason == DISCARDED_TIME_REVERSED
    assert reversed_frame.level == DegradeLevel.NORMAL.value
    assert reversed_frame.anomaly == ANOMALY_TIME_REVERSED
    assert recovered.point is not None

    measured(
        f"坏帧={reversed_frame.reason}/{reversed_frame.level}, "
        f"下一帧 point={fmt_vec(recovered.point)}"
    )


def test_freq_anomaly_is_reported_without_forcing_degrade(measured):
    """场景：目标从 10Hz 掉到 1Hz，频率检测开启。

    操作：前 10 帧 10Hz，之后一帧间隔 1 秒。
    输入：freq_anomaly_enabled=True，阈值 0.5。
    预期：anomaly=FREQ_ANOMALY，但数据仍新鲜，level 保持 NORMAL；
          频率异常是诊断标记，不替代目标的时效/合法性判断。
    """
    params = PlanParams(
        freq_anomaly_enabled=True,
        freq_anomaly_threshold=0.5,
        freq_window_size=10,
    )
    fsm = DegradeFSM(params)

    for index in range(10):
        tick(
            fsm,
            params,
            index * 0.1,
            target=make_target(index * 0.1),
        )

    result = tick(
        fsm,
        params,
        1.9,
        target=make_target(1.9),
    )

    assert result.anomaly == ANOMALY_FREQ
    assert result.level == DegradeLevel.NORMAL.value
    assert result.point is not None

    measured(
        f"anomaly={result.anomaly}, level={result.level}, "
        f"point={fmt_vec(result.point)}"
    )


def test_recovery_first_frame_uses_tighter_max_step(measured):
    """场景：从 L2 恢复到 NORMAL，第一帧相对上一目标点不能跳太远。

    操作：建立上一目标点 (0.5,0,10)；丢失到 L2；目标重新出现在
          (5,0,10)，连续 3 帧有效后回 NORMAL。
    输入：max_step=1.0，max_step_recover=0.2，n_recover=3。
    预期：回 NORMAL 第一帧的点距上一目标点 0.2 米，而不是 1.0 米。
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

    for index in range(5):
        tick(
            fsm,
            params,
            index * 0.05,
            own=(0.0, 0.0, 10.0),
            target=make_target(
                index * 0.05,
                position=(0.5, 0.0, 10.0),
                velocity=(0.0, 0.0, 0.0),
            ),
        )

    prev = None
    for index in range(50):
        prev = tick(
            fsm,
            params,
            0.25 + index * 0.05,
            own=(0.5, 0.0, 10.0),
            target=None,
            prev_result=prev,
        )
        if prev.level == DegradeLevel.HOLDING.value:
            break

    assert fsm.last_valid_point == pytest.approx((0.5, 0.0, 10.0))

    levels = []
    for index in range(3):
        now = 1.8 + index * 0.05
        prev = tick(
            fsm,
            params,
            now,
            own=(0.5, 0.0, 10.0),
            target=make_target(
                now, position=(5.0, 0.0, 10.0), velocity=(0.0, 0.0, 0.0)
            ),
            prev_result=prev,
        )
        levels.append(prev.level)

    assert levels[0] == DegradeLevel.HOLDING.value
    assert levels[-1] == DegradeLevel.NORMAL.value
    assert fsm.recovering
    assert prev.point == pytest.approx((0.7, 0.0, 10.0))

    measured(
        f"恢复序列={levels}, 首帧 point={fmt_vec(prev.point)}"
    )


def test_guard_notes_are_exposed_on_result(measured):
    """场景：P2 输出点被安全护栏裁剪。

    操作：TRACK 目标高度 1 米（P2 已按 5 米安全高度抬过，护栏复核）。
    输入：target=(10,0,1)。
    预期：point 的 z=5；如果护栏真的改了值，notes 里应有记录；
          否则至少 clipped=True 表示被 P2/护栏处理过。
    """
    params = PlanParams()
    fsm = DegradeFSM(params)
    result = tick(
        fsm,
        params,
        1.0,
        target=make_target(1.0, position=(10.0, 0.0, 1.0)),
    )

    assert result.point == pytest.approx((10.0, 0.0, 5.0))
    assert result.clipped

    measured(
        f"point={fmt_vec(result.point)}, clipped={result.clipped}, "
        f"notes={result.guard_notes}"
    )


def test_previous_result_is_used_for_goal_jump_limit(measured):
    """场景：调用方显式传入上一帧结果。

    操作：两帧 TRACK，第二个点距第一个点很远。
    输入：max_step=1.0；prev_result 显式传入。
    预期：第二个点相对上一目标点最多 1 米。
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

    assert isinstance(first, PlanResult)
    # 两帧目标点相距 sqrt(2)≈1.414 米，超过 max_step=1，按比例裁到 1 米。
    expected = (1.0 - 0.5 ** 0.5, 0.5 ** 0.5, 10.0)
    assert second.point == pytest.approx(expected)

    measured(
        f"first={fmt_vec(first.point)}, second={fmt_vec(second.point)}"
    )
