"""适配层纯逻辑：把 ROS 回调里确定的事实变成 P3 的输入与发布决策。

本模块不 import ``rclpy``，也不自己读时钟：所有 ``now`` 都由节点在
发布时刻取 ROS clock 后传入。这样 ``test/unit/test_node_adapter.py`` 可以在
不启动 ROS 的情况下验证：

- 目标/己方时间戳的零、未来、倒退帧如何被拦住，且不污染降级计数；
- 每架飞机复用同一个 ``DegradeFSM``，并把 ``prev_result`` 串联起来；
- 哪些 ``PlanResult`` 应该发布接口 C，payload 的字段与 stamp 语义；
- ATTACK 无解时补“当前位置保持点”（P2/P3 文档既定兜底）；
- L3 且己方状态仍有效时补固定保持点，覆盖中枢保留的上一帧点。

适配层只做搬运：不在这里做“该不该裁剪/该不该降级”的判断，那些都在
P1/P2/P3 的纯函数里。
"""

# ! 实现边界（数据见 src/algorithm_pkg/README.md 的“已知限制与未决项”）：
# ! 1) 接口 C 的 action 只是标签：中枢只在 HOVER 时用它筛规划点，
# !    模式仍由 /strategy/cmd 驱动，限幅也不看 action。仍按 7 个枚举发。
# ! 2) L3/no_plan 时，若己方状态仍有效且异常来自目标，本模块补固定
# !    保持点（第一次进 L3 时的己方位置），覆盖中枢保留的上一帧点；己方
# !    状态失效时不发布，也不伪造“清空”消息。中枢仍会保留上一帧点
# !    最多 2.0s（实测 1.98s）。
# ! 3) max_step 默认 0.30 = 宪章 V_max 6.0 m/s × 0.05s。飞控步长 0.62
# !    对应 12.4 m/s，与《内部分工细节》的 V_max 冲突，因此默认取 0.30。
# !    若宪章更新，要同步改 launch、节点默认值和测试断言。

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
from algorithm_pkg.plan_policy import hover_in_envelope
from algorithm_pkg.plan_types import PLAN_ACTIONS, PlanResult
from algorithm_pkg.safety_guard import apply_safety_guard
from algorithm_pkg.validators import ANOMALY_INVALID, ANOMALY_STALE

# ROS 时间戳比较容差：秒。与 P3 validators 的 1e-6 保持一致。
TIME_EPS = 1e-6

# ATTACK 在 NORMAL 级别解不出点时，适配层补的保持点原因。
# L1/L2 的保持点由 P3 自己产生，reason 以 hold_ 开头。
ATTACK_HOLD_REASON = "hold_attack_no_solution"

# L3 且己方状态新鲜时，适配层补的“当前位置保持点”原因前缀。
# 后缀取异常分类（invalid/stale），日志能区分目标坏在哪一类。
NO_PLAN_HOLD_PREFIX = "hold_no_plan_"

# L2 固定锚点：进入 HOLDING 的第一帧捕获己方位置，之后不再跟目标。
L2_HOLD_REASON = "hold_fixed_anchor"

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
        hover_anchor: HOVER 固定锚点，米；切出 HOVER 时清空。
        l2_anchor: L2 固定锚点，米；进入 HOLDING 时捕获。
        l3_anchor: L3 固定锚点，米；第一次补保持点时捕获，恢复后清空。
    """

    fsm: DegradeFSM
    prev_result: Optional[PlanResult] = None
    last_level: str = "NORMAL"
    hover_anchor: Optional[Vec3] = None
    l2_anchor: Optional[Vec3] = None
    l3_anchor: Optional[Vec3] = None

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


def l2_hold(
    runtime: DroneRuntime,
    result: PlanResult,
    own: Optional[Vec3],
    now: float,
    params: PlanParams,
) -> Optional[PlanResult]:
    """L2 用进入 HOLDING 时捕获的己方位置作固定锚点。

    P3 默认保持上一有效目标点，可能是目标前方的拦截点。目标丢失后继续
    飞那个点，飞机不会停在原地。适配层在 L2 第一帧捕获己方位置，之后
    每帧发同一个点；中枢 main 已改为 HOVER 优先读规划点。

    Args:
        runtime: 该机运行时；持有 L2 锚点。
        result: P3 返回的 HOLDING 结果。
        own: 己方 ENU 位置，米。
        now: 当前时刻，秒。
        params: 安全护栏参数。

    Returns:
        带固定锚点的结果；护栏拒绝时返回 ``None``，沿用 P3 的保持点。
    """
    anchor = runtime.l2_anchor
    if anchor is None:
        if own is None:
            return None
        anchor = own

    guard = apply_safety_guard(
        anchor,
        None,
        result.action,
        params,
        now,
        runtime.fsm.last_valid_stamp,
        target=None,
    )
    if not guard.accepted or guard.point is None:
        return None

    if runtime.l2_anchor is None:
        runtime.l2_anchor = guard.point
    runtime.fsm.remember_point(guard.point, guard.stamp)
    return replace(
        result,
        point=guard.point,
        reachable=False,
        reason=L2_HOLD_REASON,
        clipped=guard.clipped,
        guard_notes=guard.notes,
        output_stamp=guard.stamp,
    )


def no_plan_hold(
    runtime: DroneRuntime,
    result: PlanResult,
    own: Vec3,
    now: float,
    params: PlanParams,
) -> Optional[PlanResult]:
    """L3 且己方状态仍有效时，补固定保持点。

    中枢收不到新规划点时会保留上一帧点最多 2.0s。L3 若由目标非法触发，
    己方位置仍然可信，此时发固定点可以覆盖上一帧攻击/拦截点，让飞机停在
    第一次进 L3 的位置等恢复。己方状态失效时不能补，否则会用过期位置
    继续下指令。

    Args:
        runtime: 该机运行时；读取 FSM 的己方状态和上一输出点。
        result: P3 返回的 L3 结果；沿用它的 action、level 和 anomaly。
        own: 己方 ENU 位置，米。
        now: 当前时刻，秒。
        params: 安全护栏参数。

    Returns:
        可发布的保持结果；己方状态失效、异常不是目标问题，或护栏拒绝
        时返回 ``None``。返回值的 ``level`` 仍是 ``NO_PLAN``，表示输入
        仍然异常；是否发点看 ``point``。
    """
    ego_status = runtime.fsm.last_ego_status
    if ego_status is None or not ego_status.is_fresh:
        runtime.l3_anchor = None
        return None
    if result.anomaly not in (ANOMALY_INVALID, ANOMALY_STALE):
        return None

    anchor = runtime.l3_anchor
    if anchor is None:
        anchor = own

    guard = apply_safety_guard(
        anchor,
        None,
        result.action,
        params,
        now,
        runtime.fsm.last_valid_stamp,
        target=None,
    )
    if not guard.accepted or guard.point is None:
        return None

    if runtime.l3_anchor is None:
        runtime.l3_anchor = guard.point
    runtime.fsm.remember_point(guard.point, guard.stamp)
    return replace(
        result,
        point=guard.point,
        reachable=False,
        reason=f"{NO_PLAN_HOLD_PREFIX}{str(result.anomaly).lower()}",
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
    *,
    ego_velocity: Optional[Vec3] = None,
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
        ego_velocity: 己方 ENU 速度，米/秒；来自接口 E 的 ``twist.linear``。
            ``None`` 表示没提供，P1 跳过动力学可达性判定。

    Returns:
        ``PlanResult``；``point is None`` 时本帧不发布。
    """
    action_name = normalize_action(action)

    # HOVER 的锚点只用当前合法位置捕获；己方位置无效时本帧不发点，
    # 锚点保留，等状态恢复后继续发同一个点。
    plan_own = own
    if action_name == "HOVER":
        # HOVER 锚点的高度按 3 m 下限 / 50 m 上限收边：飞机短暂低于 3 m
        # 时也捕获一个固定水平点，不能因为 z 越界就退回“跟着自己走”。
        anchor_point = None
        if own is not None and all(isfinite(value) for value in own):
            if abs(own[0]) <= params.max_xy and abs(own[1]) <= params.max_xy:
                anchor_point = (
                    own[0],
                    own[1],
                    max(params.min_alt, min(params.max_alt, own[2])),
                )
        if (
            runtime.hover_anchor is None
            and anchor_point is not None
            and hover_in_envelope(anchor_point, params)
        ):
            runtime.hover_anchor = anchor_point
        if (
            runtime.hover_anchor is not None
            and own is not None
            and hover_in_envelope(anchor_point, params)
        ):
            plan_own = runtime.hover_anchor
    else:
        runtime.hover_anchor = None

    result = plan_with_degrade(
        action,
        plan_own,
        target,
        runtime.fsm,
        params,
        now,
        runtime.prev_result,
        ego_stamp=ego_stamp,
        ego_velocity=ego_velocity,
    )

    # ! L2 的固定锚点在进入 HOLDING 的第一帧捕获；目标恢复后清空。
    if result.level == "HOLDING":
        held = l2_hold(runtime, result, own, now, params)
        if held is not None:
            result = held
    else:
        runtime.l2_anchor = None

    # ! ATTACK 的 NORMAL 无解需要适配层补保持点；L1/L2 已由 P3 补过，
    # ! 不能重复补，否则会把 P3 的 hold_* 原因覆盖掉。
    if (
        result.point is None
        and action_name == "ATTACK"
        and own is not None
        and result.level != "NO_PLAN"
    ):
        held = attack_hold(runtime, result, own, now, params)
        if held is not None:
            result = held

    # ! L3 若只是目标坏了、己方状态仍有效，补固定保持点，覆盖中枢
    # ! 保留的上一帧攻击点。己方也失效时不补，避免拿过期位置继续下指令。
    if (
        result.point is None
        and result.level == "NO_PLAN"
        and result.action in PLAN_ACTIONS
        and own is not None
    ):
        held = no_plan_hold(runtime, result, own, now, params)
        if held is not None:
            result = held
    elif result.level != "NO_PLAN":
        runtime.l3_anchor = None

    result = _attach_source_stamp(
        result, action, target, ego_stamp, runtime.prev_result
    )
    runtime.prev_result = result
    return result


def _attach_source_stamp(
    result: PlanResult,
    action: object,
    target: Optional[TargetState],
    ego_stamp: Optional[float],
    prev_result: Optional[PlanResult],
) -> PlanResult:
    """给要发布的点补上“这帧规划用的源头时刻”。

    规则与接口语义对应：

    - 正常跟踪/拦截/攻击：用目标采样的 ``stamp``；
    - L1 外推：仍用参与外推的最新目标采样 ``stamp``；
    - L2 固定点：用己方状态时刻；
    - L1 几何回退：沿用上一有效点的 ``source_stamp``；
    - L3 固定点：用己方状态时刻；
    - HOVER / ATTACK 无解保持点：用己方状态时刻。

    没有可用来源时保持 ``None``，由 ``payload`` 回落到发布时刻。
    """
    if result.point is None:
        return result

    action_name = normalize_action(action)
    reason = result.reason
    source: Optional[float] = None

    hold_reason = reason.startswith("hold_") or result.level == "HOLDING"
    l3_hold_reason = reason.startswith(NO_PLAN_HOLD_PREFIX)
    if reason in (ATTACK_HOLD_REASON, L2_HOLD_REASON):
        source = ego_stamp
    elif l3_hold_reason:
        # L3 固定点是己方位置，stamp 必须用当前己方状态时刻，
        # 不能沿用上一帧目标点，否则中枢会按数据延迟把它丢掉。
        source = ego_stamp
    elif hold_reason:
        if reason.startswith("hold_current_ego"):
            source = ego_stamp
        elif prev_result is not None and prev_result.source_stamp is not None:
            source = prev_result.source_stamp
        elif target is not None:
            source = target.stamp
        else:
            source = ego_stamp
    elif action_name in PLAN_ACTIONS and target is not None:
        source = target.stamp
    else:
        source = ego_stamp

    if source is None or not isfinite(source) or source <= 0.0:
        return result
    return replace(result, source_stamp=float(source))


def should_publish(result: PlanResult) -> bool:
    """本帧是否应该发布接口 C；与 P2 的 ``point is not None`` 一致。"""
    return result.point is not None


def payload(drone_id: str, now: float, result: PlanResult) -> dict:
    """把 ``PlanResult`` 变成接口 C 的 JSON 字典。

    ``stamp`` 优先用 ``result.source_stamp``（规范第 5 节要求的源头
    时间戳）；结果没有来源时刻时回落到发布时刻 ``now``，兼容直接调用。

    Args:
        drone_id: 例如 ``"drone_1"``。
        now: 发布时刻的 ROS 时钟，秒；只在没有源头时刻时使用。
        result: 已通过护栏、``point`` 非空的结果。

    Returns:
        可直接 ``json.dumps`` 的字典。
    """
    point = result.point
    if point is None:
        raise ValueError("payload() 需要 result.point 非空")

    stamp = result.source_stamp if result.source_stamp is not None else now
    return {
        "drone_id": drone_id,
        "stamp": float(stamp),
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
