"""P3 输出安全护栏：发点之前的最后一道裁剪。

护栏只做“裁剪/拒绝”，不改变动作语义，也不替 P2 重新规划。顺序是
固定的，不能换来换去：

1. NaN/Inf 检查：非法点直接拒绝，不再往下算；
2. XY 边界裁剪：``|x|, |y| <= max_xy``；
3. 高度按模式裁剪：TRACK/INTERCEPT 用 ``safe_alt``，ATTACK 用
   ``min_alt``，上限 ``max_alt``；
4. 跳变限制：``|p_goal - p_prev_goal| <= max_step``；恢复期用
   ``max_step_recover`` 起步；
5. 与目标最小距离检查：太近拒绝，避免把“拦截/攻击”变成撞机；
6. 时间戳单调性：输出 stamp 必须严格递增，不能重复使用旧值。

为什么顺序不能反：

- 先做跳变限制再裁高度，可能把点拉回包线外；
- 先做最小距离检查再裁剪，可能把点裁到目标身上；
- 时间戳检查放最后，前面每一步都不应该改变“这一帧是否有效”。

本模块不 import rclpy，不发布、不打日志；裁剪原值/裁剪值/原因都放在
``GuardResult.notes`` 里，由适配层转成 ROS 日志。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Optional, Union

from algorithm_pkg.algorithm_core import TargetState, Vec3
from algorithm_pkg.plan_params import PlanParams

_EPS = 1e-9

# 时间戳被时钟抖动卡住时，往前推的最小步长，单位秒。
_STAMP_STEP = 1e-6

# 最小距离检查只对“要撞上去”的动作生效；TRACK 按 P2 契约只发目标点，
# 近距离环绕由中枢做，不能在这里把 TRACK 点拒掉。
_MIN_DISTANCE_ACTIONS = ("INTERCEPT", "ATTACK")


@dataclass(frozen=True)
class GuardResult:
    """护栏的处理结果。

    Attributes:
        point: 裁剪后的 ENU 点；``None`` 表示本帧拒绝输出。
        stamp: 单调递增后的输出时刻；拒绝时为 ``None``。
        clipped: 点是否被 XY/高度/跳变裁剪过；时间戳调整不算裁剪。
        notes: 处理记录，例如 ``"xy_clamped:..."``、``"z_clamped:..."``。
        rejected_reason: 拒绝原因；接受时为 ``None``。
    """

    point: Optional[Vec3]
    stamp: Optional[float]
    clipped: bool
    notes: tuple[str, ...]
    rejected_reason: Optional[str]

    @property
    def accepted(self) -> bool:
        """护栏是否接受这个点。"""
        return self.point is not None and self.rejected_reason is None


def _clamp(value: float, low: float, high: float) -> float:
    """把 value 限制在 [low, high] 内。"""
    return max(low, min(high, value))


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


def _norm(value: Vec3) -> float:
    """三维向量长度。"""
    return sqrt(
        value[0] * value[0]
        + value[1] * value[1]
        + value[2] * value[2]
    )


def _target_position(
    target: Optional[Union[TargetState, Vec3]],
) -> Optional[Vec3]:
    """从 ``TargetState`` 或裸三维点里取目标位置。"""
    if target is None:
        return None
    if isinstance(target, TargetState):
        return target.position
    return target


def _rejected(reason: str, note: str) -> GuardResult:
    """构造一个拒绝结果。"""
    return GuardResult(
        point=None,
        stamp=None,
        clipped=False,
        notes=(note,),
        rejected_reason=reason,
    )


def apply_safety_guard(
    p_goal: Optional[Vec3],
    p_prev: Optional[Vec3],
    mode: str,
    params: Optional[PlanParams] = None,
    now: float = 0.0,
    last_stamp: Optional[float] = None,
    target: Optional[Union[TargetState, Vec3]] = None,
    *,
    recovering: bool = False,
    step_limit: Optional[float] = None,
) -> GuardResult:
    """按固定顺序裁剪一帧规划点。

    Args:
        p_goal: 待输出的 ENU 点，米；``None`` 表示本帧没有点。
        p_prev: 上一帧真正发出去的点（不是己方当前位置），米；用于
            跳变限制。``None`` 表示没有历史点，跳过第 4 步。
        mode: 当前动作；只接受 TRACK/INTERCEPT/ATTACK。
        params: 参数对象；``None`` 时使用默认 ``PlanParams()``。
        now: 本帧发布时刻，秒。
        last_stamp: 上一帧真正发出去的时间戳，秒；``None`` 表示没有历史。
        target: 当前目标状态或位置，米；用于最小距离检查。
        recovering: 是否处于恢复期；True 时默认使用
            ``params.max_step_recover``。
        step_limit: 直接覆盖第 4 步的跳变上限；``None`` 时按
            ``recovering`` 和 ``params`` 选择。

    Returns:
        ``GuardResult``。``accepted`` 为 False 时不要发布点。

    Raises:
        ValueError: 参数对象内部的边界矛盾（构造 PlanParams 时已校验）。

    Note:
        - 最小距离检查只对 INTERCEPT/ATTACK 生效；TRACK 仍按 P2 契约
          发目标预测点，近距环绕由中枢负责。
        - ``p_prev`` 假定是上一帧已经过护栏的点；如果调用方传入包线外
          的点，护栏优先保证本次输出在包线内。
    """
    params = PlanParams() if params is None else params
    action = str(mode).strip().upper()

    if step_limit is not None:
        if not isfinite(step_limit) or step_limit <= 0.0:
            raise ValueError("step_limit must be positive when provided")

    # 1. 非规划动作 / 空点 / 非有限数：直接拒绝，不进入裁剪。
    if action not in ("TRACK", "INTERCEPT", "ATTACK"):
        return _rejected(
            "guard_unsupported_mode",
            f"mode={action}",
        )
    if p_goal is None:
        return _rejected("guard_no_point", "point=None")
    if not _finite_vec(p_goal):
        return _rejected(
            "guard_non_finite",
            f"point=({p_goal[0]!r}, {p_goal[1]!r}, {p_goal[2]!r})",
        )

    notes: list[str] = []
    clipped = False
    x, y, z = float(p_goal[0]), float(p_goal[1]), float(p_goal[2])

    # 2. XY 边界裁剪。
    clamped_x = _clamp(x, -params.max_xy, params.max_xy)
    clamped_y = _clamp(y, -params.max_xy, params.max_xy)
    if clamped_x != x:
        notes.append(f"xy_clamped:x:{x:.3f}->{clamped_x:.3f}")
        clipped = True
    if clamped_y != y:
        notes.append(f"xy_clamped:y:{y:.3f}->{clamped_y:.3f}")
        clipped = True
    x, y = clamped_x, clamped_y

    # 3. 高度按模式裁剪：ATTACK 允许打到 3 米，TRACK/INTERCEPT 保持 5 米。
    z_min = params.min_alt if action == "ATTACK" else max(
        params.min_alt, params.safe_alt
    )
    clamped_z = _clamp(z, z_min, params.max_alt)
    if clamped_z != z:
        notes.append(f"z_clamped:{z:.3f}->{clamped_z:.3f}")
        clipped = True
    z = clamped_z
    point: Vec3 = (x, y, z)

    # 4. 跳变限制：限制的是“这一帧目标点相对上一帧目标点”的跳变，
    #    不是己方当前位置到目标点的距离。
    if step_limit is not None:
        limit = step_limit
    elif recovering:
        limit = params.max_step_recover
    else:
        limit = params.max_step

    if p_prev is not None and _finite_vec(p_prev) and limit is not None:
        delta = (
            point[0] - p_prev[0],
            point[1] - p_prev[1],
            point[2] - p_prev[2],
        )
        length = _norm(delta)
        if length > limit and length > _EPS:
            scale = limit / length
            point = (
                p_prev[0] + delta[0] * scale,
                p_prev[1] + delta[1] * scale,
                p_prev[2] + delta[2] * scale,
            )
            notes.append(f"step_clamped:{length:.3f}->{limit:.3f}")
            clipped = True

            # ! 缩短之后 x/y/z 可能又滑出安全边界，必须再裁一次；
            # ! 否则会出现“裁剪后反而越界”。
            bounded = (
                _clamp(point[0], -params.max_xy, params.max_xy),
                _clamp(point[1], -params.max_xy, params.max_xy),
                _clamp(point[2], z_min, params.max_alt),
            )
            if bounded != point:
                notes.append("step_reclamped_to_bounds")
                clipped = True
            point = bounded

    # 5. 与目标最小距离检查：只对 INTERCEPT/ATTACK 生效。
    target_position = _target_position(target)
    if (
        target_position is not None
        and _finite_vec(target_position)
        and action in _MIN_DISTANCE_ACTIONS
    ):
        distance = _norm(
            (
                point[0] - target_position[0],
                point[1] - target_position[1],
                point[2] - target_position[2],
            )
        )
        if distance <= params.min_distance:
            return GuardResult(
                point=None,
                stamp=None,
                clipped=False,
                notes=tuple(notes)
                + (f"min_distance:{distance:.3f}<{params.min_distance:.3f}",),
                rejected_reason="guard_target_too_close",
            )

    # 6. 时间戳单调性：输出 stamp 必须严格大于上一帧，不能刷新旧值。
    if not isfinite(now):
        return GuardResult(
            point=None,
            stamp=None,
            clipped=False,
            notes=tuple(notes) + ("stamp_non_finite",),
            rejected_reason="guard_stamp_non_finite",
        )

    stamp = float(now)
    if last_stamp is not None and isfinite(last_stamp):
        if stamp <= last_stamp + _EPS:
            stamp = last_stamp + _STAMP_STEP
            notes.append(
                f"stamp_adjusted:{now:.6f}->{stamp:.6f}"
            )

    return GuardResult(
        point=point,
        stamp=stamp,
        clipped=clipped,
        notes=tuple(notes),
        rejected_reason=None,
    )
