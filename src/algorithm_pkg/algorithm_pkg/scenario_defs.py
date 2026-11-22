"""P5/P6 场景注册表：场景参数、目标轨迹、判定标准。

本模块刻意不依赖 ROS，宿主机可以直接 pytest。ROS 执行器在
``scenario_suite.py``，它只负责把这里的定义跑起来、写 CSV。

一个场景由三段数据描述：

1. **目标轨迹**：分段匀速直线，段与段之间位置连续；
2. **动作时序**：起飞后切哪个模式、是否按距离再切一次、跑多久；
3. **判定标准**：期望结果（HIT / MISS / GATE_FIRED）与阈值。

新增场景只改 ``SCENARIOS``，不改执行器。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, sqrt
from typing import Optional, Sequence

from algorithm_pkg.algorithm_core import TargetState, solve_intercept

# 尾随点距目标多远（与 PlanParams.d_tail 默认值一致，米）。
D_TAIL = 4.0
# 判定“疑似尾随点”的容差，米。
TAIL_TOLERANCE = 0.6


@dataclass(frozen=True)
class Segment:
    """一段匀速直线运动。

    Attributes:
        start_t: 该段开始的相对时刻，秒。
        start: 该段起点 ENU 坐标，米。
        velocity: 该段速度，米/秒；``(0,0,0)`` 表示静止悬停。
    """

    start_t: float
    start: tuple[float, float, float]
    velocity: tuple[float, float, float]


@dataclass(frozen=True)
class Scenario:
    """一个可执行的场景定义。

    Attributes:
        id: 场景编号，格式 ``SC-YYYY-MM-DD-NN`` 的简化版；执行器用它做文件名。
        title: 一句话标题。
        purpose: 这个场景要回答什么问题。
        track: 目标轨迹分段，按 ``start_t`` 升序。
        mode: 起飞后进入的战术模式，``INTERCEPT`` 或 ``ATTACK``。
        switch_mode: 可选，按距离触发的二次模式切换。
        switch_dist: 可选，触发 ``switch_mode`` 的水平距离，米。
        run_sec: 进入首模式后再跑多久，秒。
        expect: 期望结果，``HIT`` / ``MISS`` / ``GATE_FIRED``。
        max_min_dist: ``HIT`` 的最近距离阈值，米。
        note: 补充说明，写进汇总表。
    """

    id: str
    title: str
    purpose: str
    track: tuple[Segment, ...]
    mode: str
    run_sec: float
    expect: str
    max_min_dist: float = 0.5
    switch_mode: Optional[str] = None
    switch_dist: Optional[float] = None
    note: str = ""
    drone: str = "drone_1"

    def target_at(self, t: float) -> tuple[float, float, float]:
        """返回相对时刻 ``t`` 的目标位置，米。"""
        if not self.track:
            raise ValueError("scenario has empty track")
        chosen = self.track[0]
        for segment in self.track:
            if t >= segment.start_t:
                chosen = segment
            else:
                break
        dt = t - chosen.start_t
        return (
            chosen.start[0] + chosen.velocity[0] * dt,
            chosen.start[1] + chosen.velocity[1] * dt,
            chosen.start[2] + chosen.velocity[2] * dt,
        )

    def target_velocity(self, t: float) -> tuple[float, float, float]:
        """返回相对时刻 ``t`` 的目标速度，米/秒。"""
        chosen = self.track[0]
        for segment in self.track:
            if t >= segment.start_t:
                chosen = segment
            else:
                break
        return chosen.velocity


def _crossing_track(speed: float) -> tuple[Segment, ...]:
    """目标先在 (20,0,8) 悬停 6 秒，之后沿 +y 匀速横穿。"""
    return (
        Segment(0.0, (20.0, 0.0, 8.0), (0.0, 0.0, 0.0)),
        Segment(6.0, (20.0, 0.0, 8.0), (0.0, speed, 0.0)),
    )


def _fly_past_track() -> tuple[Segment, ...]:
    """目标从 (20,0,8) 沿 -x 迎面飞过己方，制造一次掉头需求。

    己方在原点附近追上去之后，目标已经跑到身后并继续远离，
    此时若还去追，需要接近 180° 的转向——这正是 P6 要拦的情况。
    """
    return (
        Segment(0.0, (20.0, 0.0, 8.0), (0.0, 0.0, 0.0)),
        Segment(6.0, (20.0, 0.0, 8.0), (-4.0, 0.0, 0.0)),
    )


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        id="intercept_cross_4",
        title="4 m/s 横穿，纯 INTERCEPT",
        purpose="P6 动力学判定在正常拦截中不能误拒；对照 P5 的 0.030 m 命中",
        track=_crossing_track(4.0),
        mode="INTERCEPT",
        run_sec=55.0,
        expect="HIT",
        note="P5 8.3.3 同口径",
    ),
    Scenario(
        id="cross_attack_4",
        title="4 m/s 横穿 + 10 m 切 ATTACK",
        purpose="复现 P5 第 10 章基线；ATTACK 未接入动力学判定，行为应保持",
        track=_crossing_track(4.0),
        mode="INTERCEPT",
        switch_mode="ATTACK",
        switch_dist=10.0,
        run_sec=55.0,
        expect="HIT",
        note="P5 第 10 章同口径（0.054 m）",
    ),
    Scenario(
        id="intercept_uturn",
        title="目标迎面掠过，逼出掉头",
        purpose="闭环里验证 dynamic_infeasible_tail 真的会触发，而不只是单测",
        track=_fly_past_track(),
        mode="INTERCEPT",
        run_sec=45.0,
        expect="GATE_FIRED",
        note="若未触发，说明闭环几何没逼出不可达，需要改轨迹",
    ),
    Scenario(
        id="intercept_cross_6",
        title="6 m/s 横穿（已知追不上）",
        purpose="记录 P5-02 支持边界；本轮修复不包含它",
        track=_crossing_track(6.0),
        mode="INTERCEPT",
        run_sec=45.0,
        expect="MISS",
        note="P5-02 未修，期望 MISS",
    ),
)


def get_scenario(scenario_id: str) -> Scenario:
    """按 id 取场景；不存在时抛 ``KeyError``。"""
    for scenario in SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    raise KeyError(f"unknown scenario: {scenario_id}")


@dataclass
class Metrics:
    """一次运行的统计结果。

    Attributes:
        min_dist: 首模式之后的最近距离，米。
        min_dist_t: 最近距离出现的相对时刻，秒。
        overshoot: 最近点后 3 秒内最大远离量，米。
        max_speed: 拦截段最大水平速度，米/秒。
        gate_frames: 离线复算中判定为 ``dynamic_infeasible_tail`` 的帧数。
        tail_frames: 计划点疑似为尾随点（位于目标后方约 d_tail）的帧数。
        samples: 参与统计的状态帧数。
    """

    min_dist: float = float("inf")
    min_dist_t: float = 0.0
    overshoot: float = 0.0
    max_speed: float = 0.0
    gate_frames: int = 0
    tail_frames: int = 0
    samples: int = 0


StateRow = tuple[float, float, float, float, float]
PlanRow = tuple[float, float, float, float, str]


def _distance(a, b) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _ego_velocity(rows: Sequence[StateRow], index: int):
    """用相邻帧中心差分估己方速度，米/秒。"""
    if len(rows) < 2:
        return (0.0, 0.0, 0.0)
    left = max(0, index - 1)
    right = min(len(rows) - 1, index + 1)
    span = rows[right][0] - rows[left][0]
    if span <= 1e-6:
        return (0.0, 0.0, 0.0)
    return (
        (rows[right][1] - rows[left][1]) / span,
        (rows[right][2] - rows[left][2]) / span,
        (rows[right][3] - rows[left][3]) / span,
    )


def gate_mask(
    scenario: Scenario,
    rows: Sequence[StateRow],
    *,
    v_max: float,
    a_max: float = 9.79,
    omega_max: float = 3.009324816594666,
) -> list[bool]:
    """逐帧复算：这一帧是否会被动力学判定拒掉。

    这是唯一不需要节点日志就能看到 ``dynamic_infeasible_tail`` 的办法；
    接口 C 的 JSON 不带 ``reason``。画图需要逐帧结果，统计只要总数，
    所以返回列表，由 ``replay_gate`` 求和。
    """
    if len(rows) < 2:
        return [False] * len(rows)
    mask: list[bool] = []
    for index, row in enumerate(rows):
        t = row[0]
        own = (row[1], row[2], row[3])
        velocity = _ego_velocity(rows, index)
        target = TargetState(
            position=scenario.target_at(t),
            velocity=scenario.target_velocity(t),
            stamp=t,
            age=0.0,
            valid=True,
            velocity_valid=True,
        )
        solution = solve_intercept(
            own,
            target,
            v_max,
            mode="INTERCEPT",
            own_velocity=velocity,
            a_max=a_max,
            omega_max=omega_max,
        )
        mask.append(solution.reason == "dynamic_infeasible_tail")
    return mask


def replay_gate(
    scenario: Scenario,
    rows: Sequence[StateRow],
    *,
    v_max: float,
    a_max: float = 9.79,
    omega_max: float = 3.009324816594666,
) -> int:
    """离线复算：这些状态帧里有多少帧会被动力学判定拒掉。"""
    return sum(
        gate_mask(
            scenario, rows, v_max=v_max, a_max=a_max, omega_max=omega_max
        )
    )


def count_tail_points(
    scenario: Scenario,
    plan_rows: Sequence[PlanRow],
) -> int:
    """统计疑似尾随点的计划帧数。

    接口 C 不带 ``reason``，只能看几何：计划点到目标约 ``d_tail``，
    且位于目标运动方向的反方向。
    """
    count = 0
    for row in plan_rows:
        t, x, y, z, _action = row
        target = scenario.target_at(t)
        point = (x, y, z)
        if abs(_distance(point, target) - D_TAIL) > TAIL_TOLERANCE:
            continue
        velocity = scenario.target_velocity(t)
        speed = hypot(velocity[0], velocity[1])
        if speed <= 1e-6:
            continue
        direction = (velocity[0] / speed, velocity[1] / speed)
        offset = (point[0] - target[0], point[1] - target[1])
        if offset[0] * direction[0] + offset[1] * direction[1] < 0.0:
            count += 1
    return count


def evaluate(
    scenario: Scenario,
    rows: Sequence[StateRow],
    plan_rows: Sequence[PlanRow],
    *,
    start_t: float = 0.0,
    v_max: float = 7.6,
) -> Metrics:
    """按状态序列算指标，不判定 PASS/FAIL。

    Args:
        scenario: 场景定义，用来取目标位置。
        rows: ``(t, x, y, z, speed)`` 状态序列。
        plan_rows: ``(t, x, y, z, action)`` 计划点序列。
        start_t: 统计起点，通常取切模式时刻。
        v_max: 离线复算用的速度上限。
    """
    metrics = Metrics()
    segment = [row for row in rows if row[0] >= start_t]
    metrics.samples = len(segment)
    if not segment:
        return metrics

    for row in segment:
        t = row[0]
        dist = _distance((row[1], row[2], row[3]), scenario.target_at(t))
        if dist < metrics.min_dist:
            metrics.min_dist = dist
            metrics.min_dist_t = t
        metrics.max_speed = max(metrics.max_speed, row[4])

    best = None
    for row in segment:
        if row[0] >= metrics.min_dist_t:
            best = row
            break
    if best is not None:
        base = _distance((best[1], best[2], best[3]), scenario.target_at(best[0]))
        for row in segment:
            if row[0] < metrics.min_dist_t:
                continue
            if row[0] > metrics.min_dist_t + 3.0:
                break
            dist = _distance((row[1], row[2], row[3]), scenario.target_at(row[0]))
            metrics.overshoot = max(metrics.overshoot, dist - base)

    metrics.gate_frames = replay_gate(scenario, segment, v_max=v_max)
    metrics.tail_frames = count_tail_points(scenario, plan_rows)
    return metrics


def verdict(scenario: Scenario, metrics: Metrics) -> tuple[bool, str]:
    """按场景期望判定 PASS/FAIL，返回 ``(passed, 一句话说明)``。"""
    if metrics.samples == 0:
        return False, "没有状态帧"
    if scenario.expect == "HIT":
        ok = metrics.min_dist < scenario.max_min_dist
        return ok, f"最近 {metrics.min_dist:.3f} m"
    if scenario.expect == "MISS":
        ok = metrics.min_dist >= scenario.max_min_dist
        return ok, f"最近 {metrics.min_dist:.3f} m"
    if scenario.expect == "GATE_FIRED":
        ok = metrics.gate_frames > 0 or metrics.tail_frames > 0
        return ok, f"复算触发 {metrics.gate_frames} 帧，疑似尾随点 {metrics.tail_frames} 帧"
    return False, f"未知期望 {scenario.expect}"
