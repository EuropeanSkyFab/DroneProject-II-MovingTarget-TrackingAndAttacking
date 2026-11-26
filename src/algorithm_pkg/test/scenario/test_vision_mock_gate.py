"""真实视觉 mock 下的预测门禁：延迟 0.3 s、噪声 0.1 m、丢帧 0.4 s。

真实相机中继未交付前，用 mock 输入覆盖“视觉会抖、会丢”的情况。本用例
只证明算法门限和预测行为，不证明真实视觉测距正确。
"""

from collections import Counter
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
RUN_SECONDS = 5.5
WARMUP = 0.5
TARGET_START_X = 20.0
TARGET_SPEED = 2.0
OWN = (0.0, 0.0, 5.0)
DROP_START = 2.0
DROP_END = 2.4


def _truth(t: float) -> tuple[float, float, float]:
    return (
        TARGET_START_X + TARGET_SPEED * (t - T_START),
        0.0,
        5.0,
    )


def _error(position, truth) -> float:
    return hypot(
        position[0] - truth[0],
        position[1] - truth[1],
        position[2] - truth[2],
    )


def test_vision_mock_noise_dropout_gate(measured, scenario_chart):
    """场景：TRACK，视觉延迟 0.3 s、位置噪声 0.1 m，2.0～2.4 s 丢帧。

    预期：正常段 NORMAL；丢帧后目标年龄 >0.5 s，进 L1 外推，不出现
    NO_PLAN；丢帧结束后回 NORMAL；预测误差 <=0.5 m。L0→L1 的第一帧
    受 n_enter=2 滞回影响，可能不发点，最多 1 帧。
    """
    mock = MockVision(
        delay=0.3,
        noise_sigma=0.1,
        dropout=((T_START + DROP_START, T_START + DROP_END),),
    )
    buffer = TargetBuffer(history_sec=3.0)
    runtime = DroneRuntime.create(BASELINE_PARAMS)
    records = []

    scenario_chart.title("真实视觉 mock：延迟 0.3 s、噪声 0.1 m、丢帧 0.4 s")
    scenario_chart.configure(
        headline="真实视觉 mock：0.3 s 延迟 + 0.1 m 噪声 + 0.4 s 丢帧，"
                 "级别在 NORMAL 与 L1 之间",
        subtitle="模拟链路延迟、噪声和丢帧；mock 参数是假设值，"
                 "不代表真实视觉已通过。",
    )
    scenario_chart.band(
        T_START + DROP_START,
        T_START + DROP_END,
        "视觉丢帧",
        "#D9D9D9",
    )
    scenario_chart.note(
        "参数：固定基线 v_max=4.50、max_step=0.74、max_pred_age=0.5、"
        "t_extrap_max=1.0；action=TRACK。\n"
        "mock：延迟 0.3 s + 高斯噪声 σ=0.1 m + 丢帧 0.4 s；"
        "真实视觉未交付，这是假设输入。"
    )
    for event_t, label in (
        (T_START + DROP_START, "开始丢帧"),
        (T_START + DROP_END + 0.3, "丢帧结束送达"),
    ):
        scenario_chart.event(event_t, label)

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
                "reason": result.reason,
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
            series="mock_vision",
        )

    post = [record for record in records if record["t"] >= WARMUP]
    levels = Counter(record["level"] for record in post)

    assert post[0]["level"] == "NORMAL"
    dropout_peak = next(
        record for record in post if record["t"] >= 2.6
    )
    assert dropout_peak["level"] == "EXTRAPOLATING"
    assert post[-1]["level"] == "NORMAL"
    assert "NO_PLAN" not in levels
    missing = [record for record in post if record["point"] is None]
    assert len(missing) <= 1
    assert all(
        record["level"] == "NORMAL" and record["reason"] == "target_too_old"
        for record in missing
    )

    errors = [
        _error(record["target"].position, record["truth"])
        for record in post
        if record["target"] is not None
    ]
    assert errors
    assert max(errors) <= 0.5

    measured(
        f"级别 {dict(levels)}；丢帧中 2.60 s 为 {dropout_peak['level']}；"
        f"最大预测误差 {max(errors):.3f} m；末帧 {post[-1]['level']}"
    )
