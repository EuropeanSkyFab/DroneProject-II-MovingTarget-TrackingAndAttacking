"""P3 恢复逻辑测试：连续帧确认、恢复期 max_step 收紧、恢复中止。

只测 ``degrade_fsm`` + ``degrade_policy`` 的纯逻辑。

运行：
    PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
        src/algorithm_pkg/test
"""

import pytest

from algorithm_pkg.algorithm_core import TargetState
from algorithm_pkg.degrade_fsm import DegradeFSM, DegradeLevel
from algorithm_pkg.degrade_policy import plan_with_degrade
from algorithm_pkg.plan_params import PlanParams


def make_target(now, *, position=(10.0, 0.0, 10.0), velocity=(0.0, 0.0, 0.0)):
    """构造一个新鲜目标。"""
    return TargetState(
        position=position,
        velocity=velocity,
        stamp=now,
        age=0.0,
        valid=True,
        velocity_valid=True,
    )


def tick(
    fsm,
    params,
    now,
    *,
    own=(0.0, 0.0, 10.0),
    target=None,
    prev_result=None,
):
    """调用一次 plan_with_degrade()。"""
    return plan_with_degrade(
        "TRACK",
        own,
        target,
        fsm,
        params,
        now,
        prev_result,
    )


def make_recovery_params():
    """构造“容易观察恢复收紧”的参数。"""
    return PlanParams(
        max_step=1.0,
        max_step_recover=0.2,
        n_recover=3,
        n_recover_step=3,
        max_pred_age=0.5,
        t_extrap_max=1.0,
    )


def drive_to_l2(fsm, params, *, own=(0.5, 0.0, 10.0)):
    """先建立上一目标点 (0.5,0,10)，再把目标丢到 L2。"""
    prev = None
    for index in range(5):
        prev = tick(
            fsm,
            params,
            index * 0.05,
            own=own,
            target=make_target(
                index * 0.05,
                position=(0.5, 0.0, 10.0),
            ),
            prev_result=prev,
        )

    for index in range(50):
        prev = tick(
            fsm,
            params,
            0.25 + index * 0.05,
            own=own,
            target=None,
            prev_result=prev,
        )
        if prev.level == DegradeLevel.HOLDING.value:
            break
    return prev


def test_recovery_first_frame_is_tightened_then_ramps(measured):
    """场景：L2 恢复后，第一帧目标点距上一目标点太远。

    操作：先到 L2；目标重新出现在 (5,0,10)，连续 3 帧有效。
    输入：max_step=1.0，max_step_recover=0.2，n_recover_step=3。
    预期：第一帧只走 0.2 米；之后 0.6、1.0 米逐步放开。
    """
    params = make_recovery_params()
    fsm = DegradeFSM(params)
    prev = drive_to_l2(fsm, params)
    assert prev.level == DegradeLevel.HOLDING.value

    points = []
    own = (0.5, 0.0, 10.0)
    for index in range(5):
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
        points.append(prev.point)
        # 己方跟随上一帧指令点；这与真实控制回路的“飞向上一目标点”一致。
        if prev.point is not None:
            own = prev.point

    # 前两帧仍在 HOLDING，第 3 帧回 NORMAL 并开始收紧。
    step_lengths = [
        points[index + 1][0] - points[index][0]
        for index in range(len(points) - 1)
    ]

    assert points[2] == pytest.approx((0.7, 0.0, 10.0))
    assert step_lengths[1] == pytest.approx(0.2)
    assert step_lengths[2] == pytest.approx(0.6)
    assert step_lengths[3] == pytest.approx(1.0)

    measured(
        f"恢复点 x 序列={[round(point[0], 3) for point in points]}, "
        f"帧间步长={[round(value, 3) for value in step_lengths]}"
    )


def test_recovery_requires_continuous_valid_frames(measured):
    """场景：L2 恢复过程中又丢了一帧，连续计数必须清零。

    操作：2 帧有效 → 1 帧丢失 → 2 帧有效 → 再 1 帧有效。
    输入：n_recover=3。
    预期：前 2 帧有效仍停在 HOLDING；丢失后重新计数；
          直到再次连续 3 帧有效才回 NORMAL。
    """
    params = make_recovery_params()
    fsm = DegradeFSM(params)
    prev = drive_to_l2(fsm, params)
    assert prev.level == DegradeLevel.HOLDING.value

    levels = []
    for index in range(2):
        now = 1.8 + index * 0.05
        prev = tick(
            fsm,
            params,
            now,
            own=(0.5, 0.0, 10.0),
            target=make_target(now, position=(0.5, 0.0, 10.0)),
            prev_result=prev,
        )
        levels.append(prev.level)

    # 中断一帧：坏帧应把连续有效帧计数清零。
    prev = tick(
        fsm,
        params,
        1.95,
        own=(0.5, 0.0, 10.0),
        target=None,
        prev_result=prev,
    )
    levels.append(prev.level)
    assert fsm.good_frames == 0

    for index in range(3):
        now = 2.0 + index * 0.05
        prev = tick(
            fsm,
            params,
            now,
            own=(0.5, 0.0, 10.0),
            target=make_target(now, position=(0.5, 0.0, 10.0)),
            prev_result=prev,
        )
        levels.append(prev.level)

    assert levels[-1] == DegradeLevel.NORMAL.value

    measured(f"恢复序列={levels}（中途丢失后重新计数）")


def test_recovery_aborted_by_new_anomaly(measured):
    """场景：刚回到 NORMAL、max_step 还在收紧期，目标又丢了。

    操作：回到 NORMAL 后立刻喂一帧缺失目标。
    输入：n_enter=2，n_recover=3。
    预期：recovering 变 False，last_recovery.completed=False；
          连续再丢一帧后回到 L1。
    """
    params = make_recovery_params()
    fsm = DegradeFSM(params)
    prev = drive_to_l2(fsm, params)

    for index in range(3):
        now = 1.8 + index * 0.05
        prev = tick(
            fsm,
            params,
            now,
            own=(0.5, 0.0, 10.0),
            target=make_target(now, position=(0.5, 0.0, 10.0)),
            prev_result=prev,
        )
    assert prev.level == DegradeLevel.NORMAL.value
    assert fsm.recovering

    first = tick(
        fsm,
        params,
        2.0,
        own=(0.5, 0.0, 10.0),
        target=None,
        prev_result=prev,
    )
    second = tick(
        fsm,
        params,
        2.05,
        own=(0.5, 0.0, 10.0),
        target=None,
        prev_result=first,
    )

    assert not fsm.recovering
    assert fsm.last_recovery is not None
    assert not fsm.last_recovery.completed
    assert second.level == DegradeLevel.EXTRAPOLATING.value

    measured(
        f"first.level={first.level}, second.level={second.level}, "
        f"recovery.completed={fsm.last_recovery.completed}"
    )


def test_recovery_records_frames_and_elapsed(measured):
    """场景：完整走完恢复期后要能回答“从哪来、用了多久、多少帧”。

    操作：L2 → 3 帧有效 → NORMAL，再 3 帧让收紧期结束。
    输入：n_recover_step=3。
    预期：last_recovery.completed=True，frames=3，elapsed≈0.15s。
    """
    params = make_recovery_params()
    fsm = DegradeFSM(params)
    prev = drive_to_l2(fsm, params)

    for index in range(6):
        now = 1.8 + index * 0.05
        prev = tick(
            fsm,
            params,
            now,
            own=(0.5, 0.0, 10.0),
            target=make_target(now, position=(0.5, 0.0, 10.0)),
            prev_result=prev,
        )

    assert fsm.last_recovery is not None
    assert fsm.last_recovery.completed
    assert fsm.last_recovery.from_level == DegradeLevel.HOLDING.value
    assert fsm.last_recovery.frames == 3
    assert fsm.last_recovery.elapsed == pytest.approx(0.15)

    measured(
        f"from={fsm.last_recovery.from_level}, "
        f"frames={fsm.last_recovery.frames}, "
        f"elapsed={fsm.last_recovery.elapsed:.3f}s"
    )


def test_recovery_does_not_skip_confirmation_on_single_frame(measured):
    """场景：L2 中只恢复 1 帧就再次丢失。

    操作：1 帧有效后立刻丢 2 帧。
    输入：n_recover=3。
    预期：不会因为一帧有效就回 NORMAL；级别仍停在 L1/L2。
    """
    params = make_recovery_params()
    fsm = DegradeFSM(params)
    prev = drive_to_l2(fsm, params)

    first = tick(
        fsm,
        params,
        1.8,
        own=(0.5, 0.0, 10.0),
        target=make_target(1.8, position=(0.5, 0.0, 10.0)),
        prev_result=prev,
    )
    assert first.level == DegradeLevel.HOLDING.value

    second = tick(
        fsm,
        params,
        1.85,
        own=(0.5, 0.0, 10.0),
        target=None,
        prev_result=first,
    )
    third = tick(
        fsm,
        params,
        1.9,
        own=(0.5, 0.0, 10.0),
        target=None,
        prev_result=second,
    )

    assert third.level in (
        DegradeLevel.EXTRAPOLATING.value,
        DegradeLevel.HOLDING.value,
    )
    assert third.level != DegradeLevel.NORMAL.value

    measured(
        f"1 帧有效后级别={first.level}, 再丢 2 帧后级别={third.level}"
    )
