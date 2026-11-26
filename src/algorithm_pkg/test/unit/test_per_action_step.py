"""按动作分档步长：TRACK/INTERCEPT/ATTACK 的发点跳变不超过各自上限。"""

from math import hypot

import pytest

from algorithm_pkg.adapter_core import DroneRuntime, plan_drone
from algorithm_pkg.algorithm_core import TargetState
from algorithm_pkg.plan_params import PlanParams

PARAMS = PlanParams(
    v_max=4.50,
    v_max_intercept=7.60,
    v_max_attack=12.40,
    a_max=9.79,
    yaw_rate_max=172.4 * 3.141592653589793 / 180.0,
    max_step=0.30,
    max_step_track=0.14,
    max_step_intercept=0.38,
    max_step_attack=0.62,
)
OWN = (0.0, 0.0, 10.0)


def _target(now: float) -> TargetState:
    return TargetState(
        position=(40.0, 0.0, 10.0),
        velocity=(0.0, 0.0, 0.0),
        stamp=now,
        age=0.0,
        valid=True,
        velocity_valid=True,
    )


def _jump(a, b) -> float:
    return hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])


@pytest.mark.parametrize(
    "action,limit",
    (("TRACK", 0.14), ("INTERCEPT", 0.38), ("ATTACK", 0.62)),
)
def test_action_plan_jump_respects_step(action, limit, measured):
    """场景：目标在 40 m 外，连续两帧规划同一动作。

    预期：第一帧点到己方、第二帧点相对第一帧的跳变都不超过该动作上限。
    """
    runtime = DroneRuntime.create(PARAMS)
    now = 100.0
    first = plan_drone(runtime, action, OWN, _target(now), now, now, PARAMS)
    second = plan_drone(
        runtime, action, OWN, _target(now + 0.05), now + 0.05, now + 0.05, PARAMS
    )

    assert first.point is not None
    assert second.point is not None
    assert _jump(OWN, first.point) <= limit + 1e-9
    assert _jump(first.point, second.point) <= limit + 1e-9

    measured(
        f"{action}: 首帧 {_jump(OWN, first.point):.3f} m，"
        f"第二帧跳变 {_jump(first.point, second.point):.3f} m，上限 {limit}"
    )
