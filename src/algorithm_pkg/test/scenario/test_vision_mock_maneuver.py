"""视觉 mock 的转弯目标：延迟 0.3 s 不到门限，级别不变，但预测会滞后。

这条用例只记录当前门限下的误差上界，不当作真实视觉精度验收。
"""

from math import hypot, radians

from algorithm_pkg.adapter_core import (
    DroneRuntime,
    TargetBuffer,
    plan_drone,
)
from algorithm_pkg.plan_params import PlanParams
from support.vision_mock import MockVision

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
RUN_SECONDS = 5.0
TURN_AT = 1.5
SPEED = 2.0
TARGET_START_X = 20.0
OWN = (0.0, 0.0, 5.0)


def _truth(t: float) -> tuple[float, float, float]:
    """目标先沿 +x，1.5 s 后折向 +y。"""
    elapsed = t - T_START
    if elapsed < TURN_AT:
        return (TARGET_START_X + SPEED * elapsed, 0.0, 5.0)
    return (
        TARGET_START_X + SPEED * TURN_AT,
        SPEED * (elapsed - TURN_AT),
        5.0,
    )


def _error(position, truth) -> float:
    return hypot(
        position[0] - truth[0],
        position[1] - truth[1],
        position[2] - truth[2],
    )


def test_vision_mock_maneuver_keeps_normal_but_lags(measured, scenario_chart):
    """场景：TRACK，视觉延迟 0.3 s、噪声 0.05 m；目标 1.5 s 时 90° 转弯。

    预期：延迟 0.3 s 小于 max_pred_age 0.5 s，级别一直是 NORMAL；
    转弯时匀速模型滞后，误差会明显增大。本用例只记录上界 2 m。
    """
    mock = MockVision(delay=0.3, noise_sigma=0.05)
    buffer = TargetBuffer(history_sec=3.0)
    runtime = DroneRuntime.create(BASELINE_PARAMS)
    records = []

    scenario_chart.title("视觉 mock 转弯目标：延迟 0.3 s、噪声 0.05 m")
    scenario_chart.configure(
        headline="转弯目标：0.3 s 延迟、0.05 m 噪声，级别保持 NORMAL 但位置滞后",
        subtitle="目标按转弯轨迹运动；延迟未超过 0.5 s，门禁不降级。",
    )
    scenario_chart.band(
        T_START + TURN_AT,
        T_START + TURN_AT + 0.5,
        "目标转弯",
        "#D9D9D9",
    )
    scenario_chart.note(
        "参数：固定基线 v_max=4.50、max_step=0.74、max_pred_age=0.5；"
        "action=TRACK。\n"
        "目标先沿 +x，1.5 s 后折向 +y；延迟 0.3 s 未超过门限，"
        "级别保持 NORMAL，转弯段误差增大。"
    )
    scenario_chart.event(T_START + TURN_AT, "目标转弯")

    prefill = int(round(0.3 / DT)) + 3
    total = int(round(RUN_SECONDS / DT)) + 1
    for index in range(-prefill, total):
        now = T_START + DT * index
        sample = mock.sample(now, _truth)
        if sample is not None:
            stamp, point = sample
            buffer.offer(stamp, point, now)
        if now < T_START:
            continue

        truth = _truth(now)
        target = buffer.predict(now, max_age=1.0, max_speed=15.0)
        result = plan_drone(
            runtime,
            "TRACK",
            OWN,
            target,
            ego_stamp=now,
            now=now,
            params=BASELINE_PARAMS,
        )
        records.append(
            {
                "t": now - T_START,
                "level": result.level,
                "point": result.point,
                "target": target,
                "truth": truth,
            }
        )
        scenario_chart.add(
            now,
            own=OWN,
            target=target.position if target is not None else None,
            truth=truth,
            point=result.point,
            level=result.level,
            reason=result.reason,
            series="turn",
        )

    assert all(record["level"] == "NORMAL" for record in records)
    assert all(record["point"] is not None for record in records)

    errors = [
        _error(record["target"].position, record["truth"])
        for record in records
        if record["target"] is not None
    ]
    assert max(errors) <= 2.0

    measured(
        f"级别全是 NORMAL；转弯段最大预测误差 {max(errors):.2f} m；"
        "延迟不到 0.5 s，门禁不会降级"
    )
