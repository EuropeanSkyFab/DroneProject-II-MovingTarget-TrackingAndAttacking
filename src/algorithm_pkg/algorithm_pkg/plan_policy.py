"""P2 模式策略：把“动作”翻译成“这一帧发不发点、发哪个点”。

调用顺序故意固定，目的是让错误原因可复现：

1. 归一化动作；未知动作直接拒绝；
2. HOVER 发适配层保存的固定锚点；TAKEOFF/RETURN 由中枢负责；
   RESUME_PATROL 在节点层保持原模式；
3. 检查目标是否存在、是否有效、是否太旧；
4. 检查己方位置是否有效；
5. INTERCEPT/ATTACK 还要求目标速度有效；
6. 调 P1 的 ``solve_intercept``；
7. 调 P1 的 ``apply_constraints``。

P2 不发布、不打日志、不生成 ``stamp``。这些是 ROS 适配层的事。
"""

from __future__ import annotations

from math import isfinite
from typing import Optional

from algorithm_pkg.algorithm_core import (
    TargetState,
    Vec3,
    apply_constraints,
    solve_intercept,
)
from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.plan_types import (
    ALL_ACTIONS,
    CENTER_ACTIONS,
    DEFERRED_ACTIONS,
    PlanResult,
)


def _no_plan(action: str, raw_action: str, reason: str) -> PlanResult:
    """构造一个“本帧不发布”的结果。"""
    return PlanResult(
        action=action,
        raw_action=raw_action,
        point=None,
        reachable=False,
        t_go=None,
        reason=reason,
    )


def _finite_vec(value: Optional[Vec3]) -> bool:
    """判断三维点是否存在且每个分量都是有限数。"""
    if value is None:
        return False
    return all(isfinite(component) for component in value)


def hover_in_envelope(own: Optional[Vec3], params: PlanParams) -> bool:
    """己方位置是否能作为 HOVER 点发给中枢。

    中枢 ``plan_callback`` 只接受 ``|x|/|y| <= max_xy``、
    ``min_alt <= z <= max_alt`` 的点；越界会在中枢被丢弃。
    """
    if not _finite_vec(own):
        return False
    return not (
        abs(own[0]) > params.max_xy
        or abs(own[1]) > params.max_xy
        or own[2] < params.min_alt
        or own[2] > params.max_alt
    )


def plan(
    raw_action: object,
    own: Optional[Vec3],
    target: Optional[TargetState],
    params: Optional[PlanParams] = None,
    *,
    own_velocity: Optional[Vec3] = None,
) -> PlanResult:
    """根据战术动作生成一帧规划结果。

    Args:
        raw_action: 上游传来的动作，可能是 ``"track"``、``" TRACK "``
            甚至 ``None``；函数会先归一化，再决定是否接受。
        own: 己方当前位置，ENU 米；``None`` 表示状态还没到。
        target: ``predict_target`` 的输出；``None`` 表示还没有目标。
        params: 策略参数；``None`` 时使用 ``PlanParams()`` 的默认值。
        own_velocity: 己方当前速度，ENU 米/秒；``None`` 表示调用方没提供，
            此时不做动力学可达性判定。

    Returns:
        ``PlanResult``。调用方只应在 ``result.point is not None`` 时发布。

    Note:
        这里不把未知动作“修正”为 HOVER。未知动作必须显式拒绝，
        否则上游拼错一个字母就可能让飞机继续执行旧模式。
    """
    if params is None:
        params = PlanParams()

    raw = str(raw_action)
    action = raw.strip().upper()

    # 1. 未知动作：不要猜，直接拒绝。
    if action not in ALL_ACTIONS:
        return _no_plan(action, raw, "unknown_action")

    # 2. 中枢自己生成坐标的动作：规划节点不参与。
    #    HOVER 例外：适配层把固定锚点当作 own 传进来，这里只检查包线。
    #    锚点的保存和清除在 adapter_core，P2 保持纯函数。
    if action == "HOVER":
        if not _finite_vec(own):
            return _no_plan(action, raw, "own_invalid")
        # ! 地面/越界时不发：中枢 plan_callback 只接受
        # ! z ∈ [min_alt, max_alt]、|x|/|y| <= max_xy，越界会被丢弃并打 warn。
        if not hover_in_envelope(own, params):
            return _no_plan(action, raw, "hover_out_of_envelope")
        return PlanResult(
            action=action,
            raw_action=raw,
            point=own,
            reachable=True,
            t_go=None,
            reason="hover_hold",
        )

    if action in CENTER_ACTIONS:
        return _no_plan(action, raw, f"{action.lower()}_center_generated")

    # 3. 接口里有名字，但当前没有行为定义。
    if action in DEFERRED_ACTIONS:
        return _no_plan(action, raw, "resume_patrol_unspecified")

    # 4. 目标不存在 / 已过期 / 比 P2 允许的年龄更老。
    if target is None:
        return _no_plan(action, raw, "target_missing")
    if not target.valid:
        return _no_plan(action, raw, "target_invalid")
    if not isfinite(target.age) or target.age > params.max_pred_age:
        return _no_plan(action, raw, "target_too_old")

    # 5. 没有己方位置就无法算拦截/跟踪点。
    if not _finite_vec(own):
        return _no_plan(action, raw, "own_invalid")

    # 6. 单帧目标的速度是“按 0 处理”，不是真的静止。
    #    TRACK 只需要位置，可以继续；INTERCEPT/ATTACK 不允许。
    if action in ("INTERCEPT", "ATTACK") and not target.velocity_valid:
        return _no_plan(action, raw, "velocity_unavailable")

    # 7. 交给 P1 算几何点；v_max 按动作选。
    #    动力学参数取 PlanParams：a_max 与 yaw_rate_max 是既有参数出口，
    #    这里第一次真正参与求解（规范 §4）。
    solution = solve_intercept(
        own,
        target,
        params.v_max_for(action),
        mode=action,
        t_go_max=params.t_go_max,
        min_distance=params.min_distance,
        d_tail=params.d_tail,
        own_velocity=own_velocity,
        a_max=params.a_max,
        omega_max=params.yaw_rate_max,
    )
    if solution.point is None:
        # ATTACK 无解时这里就是 None；不会伪造尾随点。
        return PlanResult(
            action=action,
            raw_action=raw,
            point=None,
            reachable=False,
            t_go=solution.t_go,
            reason=solution.reason,
        )

    # 8. 安全边界；裁剪后可能又超出 max_step，apply_constraints 会二次裁剪。
    point = apply_constraints(
        solution.point,
        action,
        own=own,
        min_alt=params.min_alt,
        safe_alt=params.safe_alt,
        max_alt=params.max_alt,
        max_xy=params.max_xy,
        max_step=params.max_step_for(action),
    )
    if point is None:
        return PlanResult(
            action=action,
            raw_action=raw,
            point=None,
            reachable=False,
            t_go=None,
            reason="constraint_rejected",
        )

    return PlanResult(
        action=action,
        raw_action=raw,
        point=point,
        reachable=solution.reachable,
        t_go=solution.t_go,
        reason=solution.reason,
        clipped=point != solution.point,
    )
