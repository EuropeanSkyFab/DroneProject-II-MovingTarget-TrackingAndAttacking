"""适配层纯逻辑：把 ROS 回调里确定的事实变成 P3 的输入与发布决策。

本模块不 import ``rclpy``，也不自己读时钟：所有 ``now`` 都由节点在
发布时刻取 ROS clock 后传入。这样 ``test/test_node_adapter.py`` 可以在
不启动 ROS 的情况下验证：

- 目标/己方时间戳的零、未来、倒退帧如何被拦住，且不污染降级计数；
- 每架飞机复用同一个 ``DegradeFSM``，并把 ``prev_result`` 串联起来；
- 哪些 ``PlanResult`` 应该发布接口 C，payload 的字段与 stamp 语义；
- ATTACK 无解时补“当前位置保持点”（P2/P3 文档既定兜底）。

适配层只做搬运：不在这里做“该不该裁剪/该不该降级”的判断，那些都在
P1/P2/P3 的纯函数里。
"""

# ! 裁决 2/3 的实现边界（P4，2026-11-18，实验数据见 文档/P4联调记录.md）：
# ! 1) 接口 C 的 action 只是标签：中枢 plan_callback 不解析它，模式只由
# !    /strategy/cmd 驱动。仍按 7 个枚举发，不假设它能改变中枢限幅。
# ! 2) L3/no_plan 时本模块不发布接口 C，也不伪造“清空”消息。中枢会保留
# !    上一帧点最多 2.0s（2026-11-18 实测 1.98s），盲飞距离由 max_step
# !    限制（默认 0.62 m/帧，ATTACK 场景实测盲飞 0.00m）。
# ! 3) max_step 用单值 0.62（12.4 m/s × 0.05s）。扫描实验：全局 0.30/0.38
# !    会把 ATTACK 压到 6/7.6 m/s，最小目标距离恶化到 50.4/44.8m；按模式
# !    取 0.30/0.38/0.62 在 ATTACK 场景与单值 0.62 完全相同，收益不足，
# !    故不采用“按模式 max_step”。

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Deque, Optional

from algorithm_pkg.algorithm_core import (
    TargetSample,
    TargetState,
    Vec3,
    predict_target,
)
from algorithm_pkg.degrade_fsm import DegradeFSM
from algorithm_pkg.degrade_policy import plan_with_degrade
from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.plan_types import PlanResult
from algorithm_pkg.safety_guard import apply_safety_guard

# ROS 时间戳比较容差：秒。与 P3 validators 的 1e-6 保持一致。
TIME_EPS = 1e-6

# ATTACK 在 NORMAL 级别解不出点时，适配层补的保持点原因。
# L1/L2 的保持点由 P3 自己产生，reason 以 hold_ 开头。
ATTACK_HOLD_REASON = "hold_attack_no_solution"

STAMP_ZERO = "stamp_zero"
STAMP_FUTURE = "stamp_future"
STAMP_REVERSED = "stamp_reversed"
STAMP_NON_FINITE = "stamp_non_finite"
POINT_NON_FINITE = "point_non_finite"


def stamp_problem(
    stamp: Optional[float],
    now: float,
    last_stamp: Optional[float] = None,
) -> Optional[str]:
    """判断一个源时间戳能不能进入规划。

    Args:
        stamp: 消息头时间戳，秒；``None``/非有限数视为非法。
        now: 当前 ROS 时钟，秒。
        last_stamp: 上一次接受的时间戳，秒；用于检测倒退。

    Returns:
        合法返回 ``None``，否则返回机器可读原因字符串：

        - ``stamp_zero``：时间戳为 0 或负数；
        - ``stamp_future``：时间戳超前于 ``now``（含 1e-6 容差）；
        - ``stamp_reversed``：小于上一次接受的时间戳；
        - ``stamp_non_finite``：NaN/Inf/None。

    Note:
        零时间戳、未来时间戳和倒退帧都直接丢弃，不进入 P1 采样窗口，
        因此也不会改变 P3 的降级计数。这与 P3 策略表“倒退帧不参与
        计数”的语义一致。
    """
    if stamp is None or not isfinite(stamp):
        return STAMP_NON_FINITE
    if stamp <= 0.0:
        return STAMP_ZERO
    if stamp > now + TIME_EPS:
        return STAMP_FUTURE
    if last_stamp is not None and stamp < last_stamp - TIME_EPS:
        return STAMP_REVERSED
    return None


@dataclass
class TargetBuffer:
    """一个目标源（视觉或外部引导）的采样窗口。

    Attributes:
        history_sec: 采样保留时长，秒；与 P1 的 ``max_age`` 区分，后者是
            “目标过旧”的判定。
        max_samples: 采样上限，防止异常高频源撑爆内存。
        samples: 按时间递增的采样；非法时间戳不会进入这里。
        last_stamp: 上一次接受的时间戳，秒。
        last_rx: 上一次接受到合法采样的本地时刻，秒；用于视觉/外部源
            的切换判断。
    """

    history_sec: float = 3.0
    max_samples: int = 200
    samples: Deque[TargetSample] = field(default_factory=deque)
    last_stamp: Optional[float] = None
    last_rx: Optional[float] = None

    def offer(
        self,
        stamp: Optional[float],
        point: Vec3,
        now: float,
    ) -> Optional[str]:
        """尝试接收一帧目标位置。

        Args:
            stamp: 消息头时间戳，秒。
            point: ENU 位置，米。
            now: 当前 ROS 时钟，秒。

        Returns:
            接受返回 ``None``，否则返回 ``stamp_problem`` 的原因或
            ``point_non_finite``。
        """
        problem = stamp_problem(stamp, now, self.last_stamp)
        if problem is not None:
            return problem
        if any(not isfinite(value) for value in point):
            return POINT_NON_FINITE

        self.samples.append(
            TargetSample(stamp=stamp, x=point[0], y=point[1], z=point[2])
        )
        while self.samples and now - self.samples[0].stamp > self.history_sec:
            self.samples.popleft()
        while len(self.samples) > self.max_samples:
            self.samples.popleft()

        self.last_stamp = stamp
        self.last_rx = now
        return None

    def is_fresh(self, now: float, timeout: float) -> bool:
        """最近的合法采样是否还在 ``timeout`` 内。"""
        return self.last_rx is not None and now - self.last_rx <= timeout

    def predict(
        self,
        now: float,
        max_age: float,
        max_speed: float,
    ) -> Optional[TargetState]:
        """调用 P1 预测目标状态；没有采样时返回 ``None``。"""
        if not self.samples:
            return None
        return predict_target(
            list(self.samples),
            now,
            max_age=max_age,
            max_speed=max_speed,
        )


def choose_target(
    vision: TargetBuffer,
    external: TargetBuffer,
    now: float,
    target_timeout: float,
    max_age: float,
    max_speed: float,
) -> Optional[TargetState]:
    """按“视觉优先、外部兜底”的接口约定选目标源。

    视觉采样在 ``target_timeout`` 内优先；视觉不新鲜时用外部引导；
    两者都不新鲜时仍尝试视觉（让 P3 按年龄降级，而不是在这里变成
    “无目标”）。主源预测失败（例如速度越界）时再试另一个源。
    """
    vision_ready = bool(vision.samples) and vision.is_fresh(now, target_timeout)
    external_ready = bool(external.samples) and external.is_fresh(
        now, target_timeout
    )

    if vision_ready:
        order = (vision, external)
    elif external_ready:
        order = (external, vision)
    else:
        order = (vision, external)

    for buffer in order:
        state = buffer.predict(now, max_age, max_speed)
        if state is not None:
            return state
    return None


@dataclass
class DroneRuntime:
    """单架飞机跨帧需要复用的状态。

    Attributes:
        fsm: P3 降级状态机；同一架飞机必须始终复用同一个实例，否则
            滞回、恢复计数和 last_valid_point 都会丢。
        prev_result: 上一帧的 ``PlanResult``，供 L2 保持点和护栏跳变使用。
        last_level: 上一帧的级别字符串，用于“级别变化”日志。
    """

    fsm: DegradeFSM
    prev_result: Optional[PlanResult] = None
    last_level: str = "NORMAL"

    @classmethod
    def create(cls, params: PlanParams) -> "DroneRuntime":
        """按参数创建运行时；FSM 与参数必须同源。"""
        return cls(fsm=DegradeFSM(params))


def normalize_action(action: object) -> str:
    """动作归一化，与 P2/P3 的 ``strip().upper()`` 保持一致。"""
    return str(action).strip().upper()


def attack_hold(
    runtime: DroneRuntime,
    result: PlanResult,
    own: Vec3,
    now: float,
    params: PlanParams,
) -> Optional[PlanResult]:
    """ATTACK 在 NORMAL 解不出点时，补“当前位置保持点”。

    P2 对 ATTACK 无解只返回 ``point=None``；P3 的 L1/L2 会自己补保持点，
    但 NORMAL 不会。适配层按 P2/P3 文档的既定语义补一个己方当前位置的
    保持点，高度按 ATTACK 的下限（``min_alt``）收边，避免中枢继续拿
    上一帧攻击点全速飞。

    Args:
        runtime: 该机的运行时（提供 FSM 的上一有效点/stamp）。
        result: P2/P3 返回的无点结果；沿用它的级别、anomaly 和动作标签。
        own: 己方 ENU 位置，米。
        now: 当前时刻，秒。
        params: 安全护栏参数。

    Returns:
        可发布的保持点结果；护栏拒绝时返回 ``None``。
    """
    guard = apply_safety_guard(
        own,
        None,
        "ATTACK",
        params,
        now,
        runtime.fsm.last_valid_stamp,
        target=None,
    )
    if not guard.accepted or guard.point is None:
        return None

    runtime.fsm.remember_point(guard.point, guard.stamp)
    return replace(
        result,
        point=guard.point,
        reachable=False,
        reason=ATTACK_HOLD_REASON,
        clipped=guard.clipped,
        guard_notes=guard.notes,
        output_stamp=guard.stamp,
    )


def plan_drone(
    runtime: DroneRuntime,
    action: object,
    own: Optional[Vec3],
    target: Optional[TargetState],
    ego_stamp: Optional[float],
    now: float,
    params: PlanParams,
) -> PlanResult:
    """对一架飞机执行一帧 P3 规划。

    Args:
        runtime: 该机的 FSM/prev_result 容器。
        action: ``/strategy/cmd`` 下发的动作；大小写/空格会归一化。
        own: 己方 ENU 位置，米；``None`` 表示状态还没到。
        target: P1 预测结果；``None`` 表示没有可用目标。
        ego_stamp: 己方状态源时间戳，秒；``None`` 表示不检查新鲜度。
        now: 当前时刻，秒。
        params: 规划参数。

    Returns:
        ``PlanResult``；``point is None`` 时本帧不发布。
    """
    result = plan_with_degrade(
        action,
        own,
        target,
        runtime.fsm,
        params,
        now,
        runtime.prev_result,
        ego_stamp=ego_stamp,
    )

    # ! ATTACK 的 NORMAL 无解需要适配层补保持点；L1/L2 已由 P3 补过，
    # ! 不能重复补，否则会把 P3 的 hold_* 原因覆盖掉。
    if (
        result.point is None
        and normalize_action(action) == "ATTACK"
        and own is not None
        and result.level != "NO_PLAN"
    ):
        held = attack_hold(runtime, result, own, now, params)
        if held is not None:
            result = held

    runtime.prev_result = result
    return result


def should_publish(result: PlanResult) -> bool:
    """本帧是否应该发布接口 C；与 P2 的 ``point is not None`` 一致。"""
    return result.point is not None


def payload(drone_id: str, now: float, result: PlanResult) -> dict:
    """把 ``PlanResult`` 变成接口 C 的 JSON 字典。

    ``stamp`` 固定用调用方传入的发布时刻 ``now``，不是目标消息的
    ``header.stamp``；目标时间戳只用于 P1/P3 的年龄判断。

    Args:
        drone_id: 例如 ``"drone_1"``。
        now: 发布时刻的 ROS 时钟，秒。
        result: 已通过护栏、``point`` 非空的结果。

    Returns:
        可直接 ``json.dumps`` 的字典。
    """
    point = result.point
    if point is None:
        raise ValueError("payload() 需要 result.point 非空")

    return {
        "drone_id": drone_id,
        "stamp": float(now),
        "x": float(point[0]),
        "y": float(point[1]),
        "z": float(point[2]),
        "action": result.action,
    }


def level_transition(previous: str, result: PlanResult) -> Optional[str]:
    """返回级别变化日志文本；级别没变时返回 ``None``。"""
    current = result.level
    if current == previous:
        return None
    anomaly = result.anomaly if result.anomaly is not None else "-"
    return f"{previous}->{current} reason={result.reason} anomaly={anomaly}"
