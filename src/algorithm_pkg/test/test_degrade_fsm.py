"""P3 降级状态机测试：四级迁移、滞回、时间戳倒退与恢复。

只测 ``degrade_fsm.py`` 的纯状态机，不涉及具体发点几何。

运行：
    PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
        src/algorithm_pkg/test
"""

import pytest

from algorithm_pkg.algorithm_core import TargetState
from algorithm_pkg.degrade_fsm import DegradeFSM, DegradeLevel
from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.validators import (
    ANOMALY_EGO_LOST,
    ANOMALY_INVALID,
    ANOMALY_TIME_REVERSED,
    TargetCode,
    validate_ego,
    validate_target,
)


def make_state(
    now,
    *,
    age=0.0,
    position=(10.0, 0.0, 10.0),
    velocity=(1.0, 0.0, 0.0),
):
    """构造一个目标状态；age 用来模拟“最后一次采样有多老”。"""
    return TargetState(
        position=position,
        velocity=velocity,
        stamp=now - age,
        age=age,
        valid=age <= 1.0,
        velocity_valid=True,
    )


def tick_fresh(fsm, now):
    """喂一帧新鲜目标 + 新鲜己方状态。"""
    state = make_state(now)
    target_status = validate_target(
        state,
        state.stamp,
        now,
        fsm.params,
        last_stamp=fsm.last_target_stamp,
    )
    ego_status = validate_ego(
        (0.0, 0.0, 10.0),
        None,
        now,
        now,
        fsm.params,
        last_stamp=fsm.last_ego_stamp,
    )
    return fsm.update(target_status, ego_status, now)


def tick_missing(fsm, now):
    """喂一帧“完全没有目标”的输入。"""
    target_status = validate_target(
        None,
        None,
        now,
        fsm.params,
        last_stamp=fsm.last_target_stamp,
    )
    ego_status = validate_ego(
        (0.0, 0.0, 10.0),
        None,
        now,
        now,
        fsm.params,
        last_stamp=fsm.last_ego_stamp,
    )
    return fsm.update(target_status, ego_status, now)


def tick_stale(fsm, now, age):
    """喂一帧“目标还在，但已经 age 秒没更新”的输入。"""
    state = make_state(now, age=age)
    target_status = validate_target(
        state,
        state.stamp,
        now,
        fsm.params,
        last_stamp=fsm.last_target_stamp,
    )
    ego_status = validate_ego(
        (0.0, 0.0, 10.0),
        None,
        now,
        now,
        fsm.params,
        last_stamp=fsm.last_ego_stamp,
    )
    return fsm.update(target_status, ego_status, now)


def tick_ego_lost(fsm, now):
    """喂一帧新鲜目标，但己方状态断流。"""
    state = make_state(now)
    target_status = validate_target(
        state,
        state.stamp,
        now,
        fsm.params,
        last_stamp=fsm.last_target_stamp,
    )
    ego_status = validate_ego(
        None,
        None,
        None,
        now,
        fsm.params,
        last_stamp=fsm.last_ego_stamp,
    )
    return fsm.update(target_status, ego_status, now)


def tick_invalid_target(fsm, now):
    """喂一帧位置为 NaN 的目标。"""
    state = TargetState(
        position=(float("nan"), 0.0, 10.0),
        velocity=(0.0, 0.0, 0.0),
        stamp=now,
        age=0.0,
        valid=True,
        velocity_valid=True,
    )
    target_status = validate_target(
        state,
        now,
        now,
        fsm.params,
        last_stamp=fsm.last_target_stamp,
    )
    ego_status = validate_ego(
        (0.0, 0.0, 10.0),
        None,
        now,
        now,
        fsm.params,
        last_stamp=fsm.last_ego_stamp,
    )
    return fsm.update(target_status, ego_status, now)


def test_normal_to_extrapolating_to_holding(measured):
    """场景：目标正常 5 帧后完全消失，先短暂丢失再超过外推上限。

    操作：前 5 帧新鲜；之后每帧 now 递增 0.05s 且没有目标。
    输入：max_pred_age=0.5，t_extrap_max=1.0，n_enter=2。
    预期：短暂丢失时进 L1；丢失超过 1.0s 后进 L2；不会直接 L3。
    """
    fsm = DegradeFSM(PlanParams())

    levels = [tick_fresh(fsm, index * 0.05) for index in range(5)]
    assert levels[-1] is DegradeLevel.NORMAL

    # 目标在 now=0.2 之后消失。
    lost_levels = [
        tick_missing(fsm, 0.25 + index * 0.05) for index in range(21)
    ]

    assert DegradeLevel.EXTRAPOLATING in lost_levels
    assert lost_levels[-1] is DegradeLevel.HOLDING

    measured(
        f"前 5 帧={levels[-1].value}, 丢失后首帧={lost_levels[0].value}, "
        f"丢失后末帧={lost_levels[-1].value}"
    )


def test_recovery_goes_through_holding_then_normal(measured):
    """场景：目标丢失到 L2 后重新出现。

    操作：先制造 L2，再连续喂新鲜目标。
    输入：n_recover=3。
    预期：异常消失先回 HOLDING 确认；连续 3 帧有效后回 NORMAL；
          不会从 L2 直接跳回 NORMAL。
    """
    fsm = DegradeFSM(PlanParams())

    for index in range(5):
        tick_fresh(fsm, index * 0.05)
    for index in range(30):
        tick_missing(fsm, 0.25 + index * 0.05)
    assert fsm.level is DegradeLevel.HOLDING

    recovery_levels = [
        tick_fresh(fsm, 1.75 + index * 0.05) for index in range(3)
    ]

    assert recovery_levels[0] is DegradeLevel.HOLDING
    assert recovery_levels[-1] is DegradeLevel.NORMAL
    assert fsm.recovering

    measured(
        f"恢复到 NORMAL 的序列={[level.value for level in recovery_levels]}, "
        f"recovering={fsm.recovering}"
    )


def test_single_frame_anomaly_does_not_flap(measured):
    """场景：目标时有时无，每次只丢 1 帧就恢复。

    操作：新鲜/丢失交替 10 轮。
    输入：n_enter=2。
    预期：级别始终 NORMAL；单帧丢失不触发 L1，输出不横跳。
    """
    fsm = DegradeFSM(PlanParams())
    levels = []

    for index in range(10):
        levels.append(tick_fresh(fsm, index * 0.1))
        levels.append(tick_missing(fsm, index * 0.1 + 0.05))

    assert all(level is DegradeLevel.NORMAL for level in levels)

    measured(f"交替 10 轮后级别集合={sorted({x.value for x in levels})}")


def test_time_reversed_target_does_not_change_level(measured):
    """场景：已进入 L1 时收到一帧时间戳倒退的目标。

    操作：先让 FSM 进 L1，再直接喂 TIME_REVERSED 状态。
    输入：上一帧 stamp=0.2，坏帧 stamp=0.1。
    预期：级别仍为 L1；bad_frames/good_frames 都不增加。
    """
    fsm = DegradeFSM(PlanParams())

    for index in range(5):
        tick_fresh(fsm, index * 0.05)
    for index in range(3):
        tick_missing(fsm, 0.3 + index * 0.05)
    assert fsm.level is DegradeLevel.EXTRAPOLATING

    bad_before = fsm.bad_frames
    good_before = fsm.good_frames
    target_status = validate_target(
        (10.0, 0.0, 10.0),
        0.1,
        1.0,
        fsm.params,
        last_stamp=fsm.last_target_stamp,
    )
    ego_status = validate_ego(
        (0.0, 0.0, 10.0),
        None,
        1.0,
        1.0,
        fsm.params,
        last_stamp=fsm.last_ego_stamp,
    )
    level = fsm.update(target_status, ego_status, 1.0)

    assert target_status.code is TargetCode.TIME_REVERSED
    assert level is DegradeLevel.EXTRAPOLATING
    assert fsm.bad_frames == bad_before
    assert fsm.good_frames == good_before
    assert fsm.last_event.anomaly == ANOMALY_TIME_REVERSED
    # 单帧调试入口：FSM 保留最近一帧检测结果，时间戳倒退的坏帧也要留下。
    assert fsm.last_target_status is target_status
    assert fsm.last_ego_status is ego_status

    measured(
        f"level={level.value}, bad_frames={fsm.bad_frames}, "
        f"anomaly={fsm.last_event.anomaly}"
    )


def test_ego_lost_enters_no_plan(measured):
    """场景：目标一直新鲜，但己方状态突然断流。

    操作：连续喂 own=None。
    输入：n_enter=2。
    预期：第 2 帧进入 NO_PLAN，anomaly=EGO_LOST；不会误判成目标问题。
    """
    fsm = DegradeFSM(PlanParams())

    for index in range(5):
        tick_fresh(fsm, index * 0.05)
    levels = [tick_ego_lost(fsm, 0.25 + index * 0.05) for index in range(4)]

    assert levels[0] is DegradeLevel.NORMAL
    assert levels[1] is DegradeLevel.NO_PLAN
    assert fsm.anomaly == ANOMALY_EGO_LOST

    measured(
        f"levels={[level.value for level in levels]}, "
        f"anomaly={fsm.anomaly}"
    )


def test_invalid_target_enters_no_plan(measured):
    """场景：目标位置混入 NaN，不是“过期”，不能继续外推。

    操作：连续喂 NaN 目标。
    输入：n_enter=2。
    预期：第 2 帧进入 NO_PLAN，anomaly=INVALID。
    """
    fsm = DegradeFSM(PlanParams())

    for index in range(5):
        tick_fresh(fsm, index * 0.05)
    levels = [
        tick_invalid_target(fsm, 0.25 + index * 0.05) for index in range(3)
    ]

    assert levels[0] is DegradeLevel.NORMAL
    assert levels[1] is DegradeLevel.NO_PLAN
    assert fsm.anomaly == ANOMALY_INVALID

    measured(
        f"levels={[level.value for level in levels]}, "
        f"anomaly={fsm.anomaly}"
    )


def test_no_plan_recovers_through_holding(measured):
    """场景：己方状态恢复后，FSM 要从 L3 经 L2 回到 L0。

    操作：先制造 L3，再连续喂新鲜目标和己方状态。
    输入：n_exit=1，n_recover=3。
    预期：第 1 帧回 HOLDING，连续 3 帧有效后回 NORMAL。
    """
    fsm = DegradeFSM(PlanParams())

    for index in range(5):
        tick_fresh(fsm, index * 0.05)
    for index in range(3):
        tick_ego_lost(fsm, 0.25 + index * 0.05)
    assert fsm.level is DegradeLevel.NO_PLAN

    levels = [tick_fresh(fsm, 0.5 + index * 0.05) for index in range(3)]

    assert levels[0] is DegradeLevel.HOLDING
    assert levels[-1] is DegradeLevel.NORMAL

    measured(f"L3 恢复序列={[level.value for level in levels]}")


def test_extrapolation_timeout_moves_to_holding(measured):
    """场景：目标持续以旧时间戳存在，从外推窗口内滑到窗口外。

    操作：先新鲜；再喂 age=0.6 的 STALE 目标两帧；最后喂 age=1.1。
    输入：t_extrap_max=1.0。
    预期：age=0.6 时进 L1；age=1.1 时进 L2。
    """
    fsm = DegradeFSM(PlanParams())

    for index in range(5):
        tick_fresh(fsm, index * 0.05)
    # 目标最后一次真实更新在 now=0.2；这里保持 stamp=0.2，只让 now 前进。
    tick_stale(fsm, 0.8, 0.6)
    tick_stale(fsm, 0.9, 0.7)
    assert fsm.level is DegradeLevel.EXTRAPOLATING

    level = tick_stale(fsm, 1.3, 1.1)

    assert level is DegradeLevel.HOLDING

    measured(f"age=0.6/0.7 -> L1；age=1.1 -> {level.value}")


def test_recovery_step_limit_tightens_then_releases(measured):
    """场景：从 L2 恢复到 NORMAL 后，第一帧要临时收紧 max_step。

    操作：构造 max_step=1.0、max_step_recover=0.2、n_recover_step=3，
          观察恢复期 recovery_step_limit() 的变化。
    输入：L2 -> HOLDING -> NORMAL。
    预期：第一帧 0.2，随后 0.6、1.0；恢复结束后回到 1.0。
    """
    params = PlanParams(
        max_step=1.0,
        max_step_recover=0.2,
        n_recover_step=3,
        max_pred_age=0.5,
        t_extrap_max=1.0,
    )
    fsm = DegradeFSM(params)

    for index in range(5):
        tick_fresh(fsm, index * 0.05)
    for index in range(30):
        tick_missing(fsm, 0.25 + index * 0.05)
    assert fsm.level is DegradeLevel.HOLDING

    # 1.75/1.80/1.85 三帧有效后才回到 NORMAL；恢复收紧从 1.85 开始。
    limits = []
    tick_fresh(fsm, 1.75)
    tick_fresh(fsm, 1.80)
    tick_fresh(fsm, 1.85)
    assert fsm.level is DegradeLevel.NORMAL
    limits.append(fsm.recovery_step_limit())
    tick_fresh(fsm, 1.90)
    limits.append(fsm.recovery_step_limit())
    tick_fresh(fsm, 1.95)
    limits.append(fsm.recovery_step_limit())
    tick_fresh(fsm, 2.00)
    limits.append(fsm.recovery_step_limit())

    assert limits[0] == pytest.approx(0.2)
    assert limits[1] == pytest.approx(0.6)
    assert limits[2] == pytest.approx(1.0)
    assert limits[-1] == pytest.approx(1.0)
    assert not fsm.recovering

    assert fsm.last_recovery is not None
    assert fsm.last_recovery.completed
    assert fsm.last_recovery.frames == 3

    measured(f"恢复期 max_step 序列={limits}")
