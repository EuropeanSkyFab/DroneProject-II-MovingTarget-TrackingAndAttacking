"""P3 分级降级输出：在 P2 的 ``plan()`` 外面包一层。

P2 回答“这个动作这一帧该发什么点”；P3 回答“输入已经异常时，这个动作
还能不能按原语义发点，发什么替代点”。两者分工：

- NORMAL：直接调用 P2 的 ``plan()``，然后用安全护栏收一次；
- EXTRAPOLATING（L1）：用 P1 的外推位置构造一个“短时快照”，仍然调用
  P2 的 ``plan()``，因此七动作语义不变；
- HOLDING（L2）：输出保持点（上一有效点或己方当前位置），动作标签保留，
  reason 明确写 ``hold_*``；
- NO_PLAN（L3）：不出点，reason 明确写 ``no_plan_*``。

关键约束：

- L1/L2 不能绕过 P2 的七动作语义。INTERCEPT 可以退尾随点，ATTACK
  不能默默输出尾随点；ATTACK 在 L1 解不出拦截点时改用“显式保持点”，
  而不是伪造一个 ATTACK 尾随点。
- 时间戳倒退的帧丢弃，不改变 FSM 级别；本帧不出点。
- 结果仍是 ``PlanResult``，只增加了 ``level`` / ``anomaly`` /
  ``guard_notes`` / ``output_stamp`` 等字段，P2 的 58 项测试不受影响。

已知问题（不在 P3 处理）：

- 中枢对 ``/planning/target_pose`` 没有显式 no_plan 消息语义，收到不到
  新点时会保留上一帧最多 2 秒；P3 只能把 no_plan/hold 语义表达清楚，
  由 P4 的适配层决定是否补发保持点。
- E/F 同名 topic 自回灌会影响己方状态新鲜度判断；P3 在 validators 里
  记录风险，不在这里修。
"""

from __future__ import annotations

from math import isfinite
from typing import Optional

from algorithm_pkg.algorithm_core import TargetState, Vec3
from algorithm_pkg.degrade_fsm import DegradeFSM, DegradeLevel
from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.plan_policy import plan as base_plan
from algorithm_pkg.plan_types import (
    ALL_ACTIONS,
    CENTER_ACTIONS,
    DEFERRED_ACTIONS,
    PlanResult,
)
from algorithm_pkg.safety_guard import GuardResult, apply_safety_guard
from algorithm_pkg.validators import (
    ANOMALY_CMD_INVALID,
    ANOMALY_EGO_LOST,
    ANOMALY_INVALID,
    ANOMALY_STALE,
    ANOMALY_TIME_REVERSED,
    EgoCode,
    EgoStatus,
    TargetCode,
    TargetStatus,
    validate_ego,
    validate_target,
)

# L3 的机器可读原因；适配层日志直接打印这些字符串。
NO_PLAN_EGO_LOST = "no_plan_ego_lost"
NO_PLAN_TARGET_INVALID = "no_plan_target_invalid"
NO_PLAN_TARGET_STALE = "no_plan_target_stale"
NO_PLAN_UNKNOWN = "no_plan"
NO_PLAN_HOLD_UNAVAILABLE = "no_plan_hold_unavailable"

# 时间戳倒退帧的丢弃原因。
DISCARDED_TIME_REVERSED = "frame_discarded_time_reversed"


def _finite_vec(value: Optional[Vec3]) -> bool:
    """判断三维点存在、长度为 3 且每个分量都是有限数。"""
    if value is None:
        return False
    try:
        if len(value) != 3:
            return False
    except TypeError:
        return False
    return all(isfinite(component) for component in value)


def _recovery_fields(
    fsm: DegradeFSM,
) -> tuple[Optional[str], Optional[int]]:
    """取恢复期信息，供 PlanResult 记录。"""
    if not fsm.recovering:
        return None, None
    return fsm.recovering_from, fsm.recover_frames


def _no_plan_result(
    action: str,
    raw_action: str,
    reason: str,
    level: DegradeLevel,
    anomaly: Optional[str],
    *,
    guard_notes: tuple[str, ...] = (),
    recovered_from: Optional[str] = None,
    recovery_frames: Optional[int] = None,
) -> PlanResult:
    """构造“本帧不出点”的结果。"""
    return PlanResult(
        action=action,
        raw_action=raw_action,
        point=None,
        reachable=False,
        t_go=None,
        reason=reason,
        level=level.value,
        anomaly=anomaly,
        guard_notes=guard_notes,
        recovered_from=recovered_from,
        recovery_frames=recovery_frames,
    )


def _guard_and_wrap(
    base: PlanResult,
    guard: GuardResult,
    level: DegradeLevel,
    anomaly: Optional[str],
    fsm: DegradeFSM,
) -> PlanResult:
    """把护栏结果包成 ``PlanResult``，并在接受时记录最后一个有效点。"""
    recovered_from, recovery_frames = _recovery_fields(fsm)

    if not guard.accepted:
        reason = guard.rejected_reason or "guard_rejected"
        # ! 非有限点不是“裁剪问题”，而是输入非法：本帧按 L3 处理，
        # ! 让日志能区分“限制幅”和“数据坏了”。
        if reason == "guard_non_finite":
            result_level = DegradeLevel.NO_PLAN
            result_anomaly = ANOMALY_INVALID
        else:
            result_level = level
            result_anomaly = anomaly
        return PlanResult(
            action=base.action,
            raw_action=base.raw_action,
            point=None,
            reachable=False,
            t_go=None,
            reason=reason,
            clipped=base.clipped,
            level=result_level.value,
            anomaly=result_anomaly,
            guard_notes=guard.notes,
            recovered_from=recovered_from,
            recovery_frames=recovery_frames,
        )

    fsm.remember_point(guard.point, guard.stamp)
    return PlanResult(
        action=base.action,
        raw_action=base.raw_action,
        point=guard.point,
        reachable=base.reachable,
        t_go=base.t_go,
        reason=base.reason,
        clipped=base.clipped or guard.clipped,
        level=level.value,
        anomaly=anomaly,
        guard_notes=guard.notes,
        output_stamp=guard.stamp,
        recovered_from=recovered_from,
        recovery_frames=recovery_frames,
    )


def _l1_snapshot(
    target: Optional[TargetState],
    fsm: DegradeFSM,
    now: float,
    params: PlanParams,
) -> Optional[TargetState]:
    """为 L1 构造一个“位置已经外推到 now”的短时快照。

    优先用当前 ``TargetState``：P1 的 ``predict_target`` 已经把位置外推到
    调用时刻，即使 ``valid=False``，位置也仍然是 now 时刻的预测值。
    只有目标完全消失（``target=None``）时，才用 FSM 里最后一次可用目标
    再加一段 CV 外推。

    Args:
        target: 当前目标状态；可以为 ``None``。
        fsm: 降级状态机；读取最后一次可用目标。
        now: 当前时刻，秒。
        params: 参数对象；用 ``t_extrap_max`` 限制外推时长。

    Returns:
        可以交给 P2 ``plan()`` 的 ``TargetState``；无法构造时返回 ``None``。
        返回值的 ``age`` 置 0、``valid`` 置 True，表示“这是 P3 特意合成
        的快照，不是新鲜的原始数据”；这样 P2 的新鲜度门限不会再把 L1
        拒掉，但动作语义仍然由 P2 决定。
    """
    if target is not None and isinstance(target, TargetState):
        position = target.position
        velocity = target.velocity
        velocity_valid = target.velocity_valid
        if not _finite_vec(position) or not _finite_vec(velocity):
            return None
        return TargetState(
            position=position,
            velocity=velocity,
            stamp=now,
            age=0.0,
            valid=True,
            velocity_valid=velocity_valid,
        )

    last = fsm.last_valid_target
    if last is None or last.position is None:
        return None

    dt = now - last.observed_at
    if not isfinite(dt) or dt < 0.0:
        dt = 0.0
    dt = min(dt, params.t_extrap_max)

    velocity = last.velocity if last.velocity is not None else (0.0, 0.0, 0.0)
    if not _finite_vec(velocity):
        return None
    position = (
        last.position[0] + velocity[0] * dt,
        last.position[1] + velocity[1] * dt,
        last.position[2] + velocity[2] * dt,
    )
    if not _finite_vec(position):
        return None

    return TargetState(
        position=position,
        velocity=velocity,
        stamp=now,
        age=0.0,
        valid=True,
        velocity_valid=last.velocity_valid,
    )


def _hold_point(
    own: Optional[Vec3],
    prev_result: Optional[PlanResult],
    fsm: DegradeFSM,
    params: PlanParams,
) -> tuple[Optional[Vec3], str]:
    """按 ``hold_point_source`` 选择 L2 保持点。

    Returns:
        ``(point, reason)``。point 为 ``None`` 表示既没有上一有效点，
        也没有可用己方位置，调用方应返回 no_plan。
    """
    prev_point: Optional[Vec3] = None
    if prev_result is not None and prev_result.point is not None:
        prev_point = prev_result.point
    if prev_point is None:
        prev_point = fsm.last_valid_point

    own_valid = _finite_vec(own)
    prev_valid = _finite_vec(prev_point)

    if params.hold_point_source == "current_ego":
        if own_valid:
            return own, "hold_current_ego"
        if prev_valid:
            return prev_point, "hold_last_valid"
    else:
        if prev_valid:
            return prev_point, "hold_last_valid"
        if own_valid:
            return own, "hold_current_ego"

    return None, "hold_unavailable"


def _hold_result(
    action: str,
    raw_action: str,
    own: Optional[Vec3],
    prev_result: Optional[PlanResult],
    fsm: DegradeFSM,
    params: PlanParams,
    now: float,
    level: DegradeLevel,
    anomaly: Optional[str],
    *,
    reason: Optional[str] = None,
) -> PlanResult:
    """输出 L1/L2 的保持点。

    ``reason`` 为 ``None`` 时用 ``hold_last_valid`` / ``hold_current_ego``；
    L1 因几何无解而回退到保持时，传入 ``hold_fallback_<p2_reason>``，
    日志里能看出“本来想外推，但 ATTACK 解不出来”。
    """
    point, source = _hold_point(own, prev_result, fsm, params)
    reason_text = reason if reason is not None else source

    if point is None:
        return _no_plan_result(
            action,
            raw_action,
            NO_PLAN_HOLD_UNAVAILABLE,
            DegradeLevel.NO_PLAN,
            ANOMALY_EGO_LOST,
        )

    base = PlanResult(
        action=action,
        raw_action=raw_action,
        point=None,
        reachable=False,
        t_go=None,
        reason=reason_text,
    )
    # 保持点不做“目标点跳变”限制：它输出的是上一有效点或当前位置，
    # 本身不是一个新的跳变指令；但仍要过 XY/高度/时间戳护栏。
    guard = apply_safety_guard(
        point,
        None,
        action,
        params,
        now,
        _last_published_stamp(fsm, prev_result),
        target=None,
    )
    return _guard_and_wrap(base, guard, level, anomaly, fsm)


def _last_published_stamp(
    fsm: DegradeFSM,
    prev_result: Optional[PlanResult],
) -> Optional[float]:
    """取上一帧真正发出去的时间戳。

    只统计“发过点”的帧：no_plan 帧没有发布 stamp，不应把时间戳推进。
    """
    if fsm.last_valid_stamp is not None:
        return fsm.last_valid_stamp
    if prev_result is not None and prev_result.point is not None:
        return prev_result.output_stamp
    return None


def _previous_goal(
    fsm: DegradeFSM,
    prev_result: Optional[PlanResult],
) -> Optional[Vec3]:
    """取上一帧真正发出去的目标点。"""
    if fsm.last_valid_point is not None:
        return fsm.last_valid_point
    if prev_result is not None and prev_result.point is not None:
        return prev_result.point
    return None


def _no_plan_reason(anomaly: Optional[str]) -> str:
    """把 FSM 异常分类翻译成 L3 的 reason。"""
    if anomaly == ANOMALY_EGO_LOST:
        return NO_PLAN_EGO_LOST
    if anomaly == ANOMALY_INVALID:
        return NO_PLAN_TARGET_INVALID
    if anomaly == ANOMALY_STALE:
        return NO_PLAN_TARGET_STALE
    return NO_PLAN_UNKNOWN


def plan_with_degrade(
    raw_action: object,
    own: Optional[Vec3],
    target: Optional[TargetState],
    fsm: DegradeFSM,
    params: Optional[PlanParams] = None,
    now: float = 0.0,
    prev_result: Optional[PlanResult] = None,
    *,
    ego_stamp: Optional[float] = None,
    ego_status: Optional[EgoStatus] = None,
    ego_velocity: Optional[Vec3] = None,
) -> PlanResult:
    """在 P2 ``plan()`` 外面加一层异常降级与安全护栏。

    Args:
        raw_action: 上游动作；大小写/空格会归一化。
        own: 己方 ENU 位置，米；``None`` 表示状态还没到。
        target: P1 ``predict_target`` 的输出，或 ``None``。
        fsm: 调用方持有的 ``DegradeFSM``；同一架飞机必须复用同一个实例，
            否则滞回和恢复计数会丢。
        params: 参数对象；``None`` 时使用默认 ``PlanParams()``。应与
            构造 FSM 时使用同一份参数。
        now: 当前时刻，秒。
        prev_result: 上一帧的 ``PlanResult``；用于取“上一目标点”和
            “上一发布 stamp”。可以不传，FSM 会自己记住最后一个有效点。
        ego_stamp: 己方状态源时间戳，秒；``None`` 表示调用方没提供，
            此时 ``validate_ego`` 返回 ``ego_freshness_unchecked``。
        ego_status: 已经检测好的己方状态；传了就不再重复检测。
        ego_velocity: 己方当前速度，ENU 米/秒；``None`` 表示调用方没提供，
            此时 P2 不做动力学可达性判定。只在 NORMAL 级别下传：
            L1/L2 的输入已经被外推或保持，用它们的点再判转弯半径没有意义，
            反而会让保持点逐帧变成尾随点。

    Returns:
        ``PlanResult``。``point is None`` 时本帧不要发布。

    Note:
        - 未知动作返回 L3 + ``unknown_action``，不改 FSM 状态。
        - TAKEOFF/HOVER/RETURN/RESUME_PATROL 是中枢动作，原样透传 P2
          结果，不驱动降级状态机。
        - 时间戳倒退帧返回 ``frame_discarded_time_reversed``，级别不变。
    """
    params = PlanParams() if params is None else params

    raw = str(raw_action)
    action = raw.strip().upper()

    # 1. 未知动作：和 P2 一样不猜，但 P3 把级别明确标成 L3。
    if action not in ALL_ACTIONS:
        return _no_plan_result(
            action,
            raw,
            "unknown_action",
            DegradeLevel.NO_PLAN,
            ANOMALY_CMD_INVALID,
        )

    # 2. 中枢动作：原样透传，不参与降级状态机。
    if action in CENTER_ACTIONS or action in DEFERRED_ACTIONS:
        result = base_plan(action, own, target, params)
        return PlanResult(
            action=result.action,
            raw_action=result.raw_action,
            point=result.point,
            reachable=result.reachable,
            t_go=result.t_go,
            reason=result.reason,
            clipped=result.clipped,
            level=DegradeLevel.NORMAL.value,
            anomaly=None,
        )

    # 3. 检测目标/己方，更新 FSM。
    target_stamp = target.stamp if isinstance(target, TargetState) else None
    freq_anomaly = (
        fsm.observe_target_frequency(target_stamp)
        if target_stamp is not None
        else False
    )
    target_status = validate_target(
        target,
        target_stamp,
        now,
        params,
        last_stamp=fsm.last_target_stamp,
        freq_anomaly=freq_anomaly,
    )
    if ego_status is None:
        ego_status = validate_ego(
            own,
            None,
            ego_stamp,
            now,
            params,
            last_stamp=fsm.last_ego_stamp,
        )

    level = fsm.update(target_status, ego_status, now)

    # 4. 时间戳倒退：丢弃本帧，不参与降级计数，也不发布基于坏数据的点。
    if (
        target_status.code is TargetCode.TIME_REVERSED
        or ego_status.code is EgoCode.TIME_REVERSED
    ):
        return _no_plan_result(
            action,
            raw,
            DISCARDED_TIME_REVERSED,
            level,
            ANOMALY_TIME_REVERSED,
        )

    # 5. 按级别分发。
    if level is DegradeLevel.NORMAL:
        return _plan_normal(
            action, raw, own, target, target_status, fsm, params, now,
            prev_result, ego_velocity,
        )
    if level is DegradeLevel.EXTRAPOLATING:
        # ! L1 不传 ego_velocity：目标已被外推，再用转弯半径判一次会
        # ! 把本可外推的点判成尾随点，属于降级期行为突变。
        return _plan_extrapolating(
            action, raw, own, target, fsm, params, now, prev_result,
        )
    if level is DegradeLevel.HOLDING:
        return _hold_result(
            action,
            raw,
            own,
            prev_result,
            fsm,
            params,
            now,
            level,
            fsm.anomaly,
        )

    return _no_plan_result(
        action,
        raw,
        _no_plan_reason(fsm.anomaly),
        level,
        fsm.anomaly,
    )


def _plan_normal(
    action: str,
    raw: str,
    own: Optional[Vec3],
    target: Optional[TargetState],
    target_status: TargetStatus,
    fsm: DegradeFSM,
    params: PlanParams,
    now: float,
    prev_result: Optional[PlanResult],
    ego_velocity: Optional[Vec3] = None,
) -> PlanResult:
    """NORMAL：调用 P2，再收一次护栏。"""
    # ! ego_velocity 只在 NORMAL 传：这是唯一“用新鲜目标正经算点”的级别。
    result = base_plan(action, own, target, params, own_velocity=ego_velocity)

    if result.point is None:
        # P2 自己决定不发点时，P3 不伪造点；级别仍为 NORMAL，但把
        # 输入异常分类带出去，方便日志看到“虽然还没降级，但目标已过期”。
        return PlanResult(
            action=result.action,
            raw_action=result.raw_action,
            point=None,
            reachable=result.reachable,
            t_go=result.t_go,
            reason=result.reason,
            clipped=result.clipped,
            level=DegradeLevel.NORMAL.value,
            anomaly=target_status.anomaly,
        )

    guard = apply_safety_guard(
        result.point,
        _previous_goal(fsm, prev_result),
        action,
        params,
        now,
        _last_published_stamp(fsm, prev_result),
        target=target,
        recovering=False,
        step_limit=fsm.recovery_step_limit(),
    )
    return _guard_and_wrap(
        result,
        guard,
        DegradeLevel.NORMAL,
        target_status.anomaly,
        fsm,
    )


def _plan_extrapolating(
    action: str,
    raw: str,
    own: Optional[Vec3],
    target: Optional[TargetState],
    fsm: DegradeFSM,
    params: PlanParams,
    now: float,
    prev_result: Optional[PlanResult],
) -> PlanResult:
    """L1：用 CV 外推快照继续走 P2 的动作语义。"""
    snapshot = _l1_snapshot(target, fsm, now, params)
    if snapshot is None:
        # 连“最后一次可用目标”都没有，外推不出来；只能退保持点。
        return _hold_result(
            action,
            raw,
            own,
            prev_result,
            fsm,
            params,
            now,
            DegradeLevel.EXTRAPOLATING,
            ANOMALY_STALE,
            reason="hold_fallback_no_extrapolation",
        )

    result = base_plan(action, own, snapshot, params)

    if result.point is None:
        # ! ATTACK 在 L1 解不出拦截点时，绝不能退化成尾随点；
        # ! 用显式保持点代替，和适配层现有的 ATTACK 兜底保持一致。
        return _hold_result(
            action,
            raw,
            own,
            prev_result,
            fsm,
            params,
            now,
            DegradeLevel.EXTRAPOLATING,
            ANOMALY_STALE,
            reason=f"hold_fallback_{result.reason}",
        )

    guard = apply_safety_guard(
        result.point,
        _previous_goal(fsm, prev_result),
        action,
        params,
        now,
        _last_published_stamp(fsm, prev_result),
        target=snapshot,
        recovering=False,
        step_limit=fsm.recovery_step_limit(),
    )
    return _guard_and_wrap(
        result,
        guard,
        DegradeLevel.EXTRAPOLATING,
        ANOMALY_STALE,
        fsm,
    )
