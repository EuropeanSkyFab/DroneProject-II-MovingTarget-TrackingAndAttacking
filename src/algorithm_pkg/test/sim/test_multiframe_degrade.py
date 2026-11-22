"""多帧追逃仿真：先追 5 s、目标丢失、己方断流、重捕后追上。

这条用例带简化己方运动模型：每帧朝规划点最多飞 ``v_max*dt``，目标以
2 m/s 沿 +x 飞。它和单点行为测试分开收集；图表由 ``scenario_chart``
fixture 在用例结束后生成。
"""

from math import hypot, radians

import pytest

from algorithm_pkg.adapter_core import (
    DroneRuntime,
    TargetBuffer,
    plan_drone,
    should_publish,
)
from algorithm_pkg.plan_params import PlanParams

P5_PARAMS = PlanParams(
    v_max=4.50,
    v_max_intercept=7.60,
    v_max_attack=14.80,
    a_max=9.79,
    yaw_rate_max=radians(172.4),
    max_step=0.74,
)

DT = 0.05
T_START = 100.0
TARGET_START_X = 25.0
TARGET_SPEED = 2.0
LOSS_START = 5.0
LOSS_END = 6.5
EGO_LOSS_START = 8.0
EGO_LOSS_END = 8.5
RUN_SECONDS = 12.2


def _distance(a, b):
    return hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _step_toward(own, point, max_step):
    dx = point[0] - own[0]
    dy = point[1] - own[1]
    dz = point[2] - own[2]
    distance = hypot(dx, dy, dz)
    if distance <= 1e-9:
        return own
    step = min(max_step, distance)
    return (
        own[0] + dx / distance * step,
        own[1] + dy / distance * step,
        own[2] + dz / distance * step,
    )


def test_multiframe_chase_loss_dropout_reacquire(measured, scenario_chart):
    """场景：先追 5 s，目标消失 1.5 s，己方断流 0.5 s，重捕后追上。

    设定：目标从 (25,0,10) 以 2 m/s 沿 +x 飞；己方从 (0,0,10) 出发，
    速度上限 4.5 m/s，每帧最多 0.74 m。5.0～6.5 s 目标流消失；
    8.0～8.5 s 己方状态消失。目标流消失、己方断流分别用阶段底色 A/B 标出。

    预期：前 5 s 距离持续缩小；丢失后进 L1→L2，保持点不再跟着目标；
    己方断流进 L3，不发点；重捕后回 NORMAL，12 s 前距离 ≤0.5 m。
    """
    buffer = TargetBuffer()
    runtime = DroneRuntime.create(P5_PARAMS)
    own = (0.0, 0.0, 10.0)
    records = {}

    scenario_chart.title("多帧追逃：先追 → 目标丢失 → 己方断流 → 重捕追上")
    scenario_chart.note(
        "参数：P5 v_max=4.50、max_step=0.74、max_pred_age=0.5、"
        "t_extrap_max=1.0；action=TRACK。\n"
        "拐点 A：105.00 target_missing；105.05 L1；"
        "106.00 hold_last_valid（L2 点冻结）；"
        "106.50 目标回来；106.70 恢复 NORMAL。\n"
        "拐点 B：108.00 own_invalid；108.05 no_plan_ego_lost（L3 不发点）；"
        "108.50 恢复 HOLDING；108.60 NORMAL；112.00 重捕。"
    )
    scenario_chart.band(T_START + LOSS_START, T_START + LOSS_END,
                        "A 目标流消失", "#F2D7C8")
    scenario_chart.band(T_START + EGO_LOSS_START, T_START + EGO_LOSS_END,
                        "B 己方断流", "#D9D9D9")
    for event_t, label in (
        (T_START + 5.0, "目标丢失"),
        (T_START + 6.0, "进 L2"),
        (T_START + 6.5, "目标回来"),
        (T_START + 8.05, "进 L3"),
        (T_START + 8.5, "己方恢复"),
        (T_START + 12.0, "重捕"),
    ):
        scenario_chart.event(event_t, label)

    total_frames = int(round(RUN_SECONDS / DT)) + 1
    for index in range(total_frames):
        now = T_START + DT * index
        t = now - T_START
        target_truth = (
            TARGET_START_X + TARGET_SPEED * t,
            0.0,
            10.0,
        )
        target_visible = not (LOSS_START <= t < LOSS_END)
        if target_visible:
            buffer.offer(now, target_truth, now)
        target = (
            buffer.predict(now, max_age=1.0, max_speed=15.0)
            if target_visible
            else None
        )

        ego_lost = EGO_LOSS_START <= t < EGO_LOSS_END
        own_arg = None if ego_lost else own
        ego_stamp = None if ego_lost else now
        result = plan_drone(
            runtime, "TRACK", own_arg, target, ego_stamp, now, P5_PARAMS
        )
        distance = _distance(own, target_truth)
        records[round(t, 2)] = (result, own, target_truth, distance)
        scenario_chart.add(
            now, own=own_arg, target=target.position if target else None,
            truth=target_truth, point=result.point, level=result.level,
            reason=result.reason, series="chase",
        )

        if result.point is not None and not ego_lost:
            own = _step_toward(own, result.point, P5_PARAMS.v_max * DT)

    # 前 5 s：追近，但还没命中。
    assert records[5.0][3] < 15.0
    assert records[5.0][3] > 0.5
    assert records[5.0][0].level == "NORMAL"

    # 5.0 s：目标流刚断，第 1 帧不降级，不发点。
    first_missing = records[5.0][0]
    assert first_missing.level == "NORMAL"
    assert first_missing.reason == "target_missing"
    assert not should_publish(first_missing)

    # 5.05 s：第 2 帧异常，进 L1，继续外推。
    extrapolating = records[5.05][0]
    assert extrapolating.level == "EXTRAPOLATING"
    assert should_publish(extrapolating)

    # 6.0 s：丢失超过 1.0 s，进 L2；保持点不再跟目标。
    holding = records[6.0][0]
    assert holding.level == "HOLDING"
    assert holding.reason == "hold_last_valid"
    assert holding.point is not None
    assert holding.point[0] < records[6.0][2][0] - 1.0

    # 6.5 s：目标回来，但还在 HOLDING 确认；6.7 s 已回到 NORMAL。
    assert records[6.5][0].level == "HOLDING"
    assert records[6.7][0].level == "NORMAL"

    # 8.0 s：己方断流第 1 帧，仍 NORMAL，但不发点。
    ego_first = records[8.0][0]
    assert ego_first.level == "NORMAL"
    assert ego_first.reason == "own_invalid"
    assert not should_publish(ego_first)

    # 8.05 s：第 2 帧异常，进 L3，不发点。
    no_plan = records[8.05][0]
    assert no_plan.level == "NO_PLAN"
    assert no_plan.reason == "no_plan_ego_lost"
    assert not should_publish(no_plan)

    # 8.5 s 恢复：先 HOLDING，8.6 s 回 NORMAL。
    assert records[8.5][0].level == "HOLDING"
    assert records[8.6][0].level == "NORMAL"

    # 重捕后重新追近，最终命中门限 0.5 m。
    after_recovery = [
        distance for t, (_, _, _, distance) in records.items() if t >= 8.5
    ]
    assert min(after_recovery) <= 0.5
    assert records[12.0][3] <= 0.5
    assert records[12.0][0].level == "NORMAL"

    measured(
        f"5 s 追到 {records[5.0][3]:.2f} m；目标丢失 1.5 s 进 L2，"
        f"保持点 {records[6.0][0].point[0]:.2f} m 不跟目标 "
        f"{records[6.0][2][0]:.2f} m；断流 0.5 s 进 L3；"
        f"重捕后 12 s 距离 {records[12.0][3]:.2f} m，"
        f"最终最小距离 {min(after_recovery):.2f} m"
    )
