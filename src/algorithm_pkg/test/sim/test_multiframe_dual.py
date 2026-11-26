"""双机多帧：主攻 + 警戒悬停，同一目标并存。

目标固定在 (20,0,10)。主攻机从 (0,0,5) 用 ATTACK 接近并命中；
警戒机从 (40,0,5) 用 HOVER 保持原位。两机初始到目标距离相同，
用于验证同目标下“一架主攻、一架不介入”可以并存。

三机联调由中枢负责，本用例只做两机。
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

BASELINE_PARAMS = PlanParams(
    v_max=4.50,
    v_max_intercept=7.60,
    v_max_attack=14.80,
    a_max=9.79,
    yaw_rate_max=radians(172.4),
    max_step=0.74,
)

DT = 0.05
T_START = 100.0
DURATION = 4.0
TARGET = (20.0, 0.0, 10.0)
PRIMARY_START = (0.0, 0.0, 5.0)
SECONDARY_START = (40.0, 0.0, 5.0)


def _distance(a, b) -> float:
    return hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _step_toward(own, point, limit: float):
    delta = (point[0] - own[0], point[1] - own[1], point[2] - own[2])
    length = hypot(delta[0], delta[1], delta[2])
    if length <= 1e-9:
        return own
    step = min(limit, length)
    return (
        own[0] + delta[0] / length * step,
        own[1] + delta[1] / length * step,
        own[2] + delta[2] / length * step,
    )


def test_multiframe_dual_primary_attack_secondary_hold(measured, dual_chart):
    """场景：目标 (20,0,10)；主攻机 (0,0,5) 用 ATTACK，警戒机 (40,0,5) HOVER 4 s。

    预期：主攻压到 1 m 内（安全护栏在 0.5 m 边界前停）；警戒机位置不变；两机始终分开；
    两机初始到目标距离相同，但只让主攻进入目标区。
    """
    buffer = TargetBuffer()
    primary_runtime = DroneRuntime.create(BASELINE_PARAMS)
    secondary_runtime = DroneRuntime.create(BASELINE_PARAMS)
    primary = PRIMARY_START
    secondary = SECONDARY_START
    primary_min = _distance(primary, TARGET)
    separation_min = _distance(primary, secondary)
    primary_end = primary
    secondary_end = secondary
    primary_point = None
    secondary_point = None

    initial_primary = _distance(primary, TARGET)
    initial_secondary = _distance(secondary, TARGET)
    assert initial_primary == pytest.approx(initial_secondary)

    dual_chart.configure(
        headline="双机同目标：主攻 ATTACK + 警戒 HOVER，两机并存不互撞",
        subtitle="主攻 (0,0,5) 追踪 (20,0,10)；警戒 (40,0,5) 保持原位；"
                 "两机初始到目标距离相同。",
    )

    total = int(round(DURATION / DT)) + 1
    for index in range(total):
        now = T_START + DT * index
        t = now - T_START
        buffer.offer(now, TARGET, now)
        target = buffer.predict(now, max_age=1.0, max_speed=15.0)

        primary_result = plan_drone(
            primary_runtime, "ATTACK", primary, target, now, now,
            BASELINE_PARAMS,
        )
        secondary_result = plan_drone(
            secondary_runtime, "HOVER", secondary, target, now, now,
            BASELINE_PARAMS,
        )
        assert should_publish(primary_result)
        assert should_publish(secondary_result)
        assert secondary_result.point is not None
        primary_point = primary_result.point
        secondary_point = secondary_result.point

        dual_chart.add(
            now,
            primary=primary,
            secondary=secondary,
            target=TARGET,
            primary_point=primary_point,
            secondary_point=secondary_point,
            primary_level=primary_result.level,
            secondary_level=secondary_result.level,
        )

        primary_min = min(primary_min, _distance(primary, TARGET))
        separation_min = min(separation_min, _distance(primary, secondary))
        primary_end = primary
        secondary_end = secondary
        if primary_point is not None:
            primary = _step_toward(
                primary, primary_point, BASELINE_PARAMS.v_max_attack * DT
            )
        if secondary_point is not None:
            secondary = _step_toward(
                secondary, secondary_point, BASELINE_PARAMS.v_max * DT
            )

    primary_dist = _distance(primary_end, TARGET)
    secondary_dist = _distance(secondary_end, TARGET)
    separation = _distance(primary_end, secondary_end)
    assert primary_min <= 1.0
    assert secondary_dist == pytest.approx(initial_secondary, abs=1e-6)
    assert separation_min > 6.0
    assert separation > 6.0

    measured(
        f"主攻最近 {primary_min:.2f} m（终点 {primary_dist:.2f} m）；"
        f"警戒保持 {secondary_dist:.2f} m；"
        f"最小间隔 {separation_min:.1f} m"
    )
