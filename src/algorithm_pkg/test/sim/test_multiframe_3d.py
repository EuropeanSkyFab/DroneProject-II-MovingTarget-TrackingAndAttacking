"""3D 多帧：目标爬升，TRACK 跟随预测点。

目标从 5 m 高度以 2 m/s 爬升，同时向东 2 m/s 移动；己方从 5 m 起飞，
TRACK 跟随三维预测点。z 变化超过 3D 阈值，过程图自动使用 3D 面板。

图表由 ``scenario_chart`` fixture 在用例结束后生成。
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
DURATION = 6.0
OWN_START = (0.0, 0.0, 5.0)


def _target(t: float) -> tuple[float, float, float]:
    """目标向东 2 m/s，同时从 5 m 爬升到 17 m。"""
    return (20.0 + 2.0 * t, 0.0, 5.0 + 2.0 * t)


def _distance(a, b) -> float:
    return hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _step_toward(own, point, limit: float):
    """朝发点走，最多 ``limit`` 米。"""
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


def test_multiframe_3d_target_climb(measured, scenario_chart):
    """场景：目标 (20,0,5) 向东 2 m/s、向上 2 m/s，己方 (0,0,5)，TRACK 6 s。

    预期：级别保持 NORMAL；己方和发点的 z 都上升，z 跨度超过 0.5 m；
    发点跟预测点，最终距离缩小。
    """
    buffer = TargetBuffer()
    runtime = DroneRuntime.create(BASELINE_PARAMS)
    own = OWN_START
    frames = []
    distances = []

    scenario_chart.configure(
        headline="3D 爬升目标：TRACK 跟预测点，高度从 5 m 升到 17 m",
        subtitle="目标 (20,0,5) 向东 2 m/s、向上 2 m/s；己方从 (0,0,5) 起飞；"
                 "参数：固定基线 v_max=4.50 m/s，max_step=0.74 m/帧。",
    )

    total = int(round(DURATION / DT)) + 1
    for index in range(total):
        now = T_START + DT * index
        t = now - T_START
        truth = _target(t)
        buffer.offer(now, truth, now)
        target = buffer.predict(now, max_age=1.0, max_speed=15.0)
        result = plan_drone(
            runtime, "TRACK", own, target, now, now, BASELINE_PARAMS
        )
        assert should_publish(result)
        assert result.level == "NORMAL"
        distances.append(_distance(own, truth))
        frames.append((t, own, truth, result))
        scenario_chart.add(
            now,
            own=own,
            target=target.position if target is not None else None,
            truth=truth,
            point=result.point,
            level=result.level,
            reason=result.reason,
            series="3d_climb",
            own_truth=own,
        )
        if result.point is not None:
            own = _step_toward(own, result.point, BASELINE_PARAMS.v_max * DT)

    own_z = [own[2] for _t, own, _truth, _result in frames]
    point_z = [
        result.point[2]
        for _t, _own, _truth, result in frames
        if result.point is not None
    ]
    assert max(own_z) - min(own_z) > 5.0
    assert max(point_z) - min(point_z) > 5.0
    assert frames[-1][0] == pytest.approx(DURATION)
    assert distances[-1] < distances[0]

    measured(
        f"己方 z {own_z[0]:.1f}→{own_z[-1]:.1f} m；"
        f"发点 z {point_z[0]:.1f}→{point_z[-1]:.1f} m；"
        f"距离 {distances[0]:.1f}→{distances[-1]:.1f} m"
    )
