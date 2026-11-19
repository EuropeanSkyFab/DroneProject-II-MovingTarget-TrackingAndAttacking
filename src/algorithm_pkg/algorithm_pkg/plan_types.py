"""P2 模式策略的公共类型。

P2 只负责回答一个问题：

    给定一个战术动作、己方位置和目标状态，这一帧该不该发点？
    如果要发，发哪个 ENU 点？

它不碰 ROS 消息、不发布、不打日志。ROS 适配层拿到 ``PlanResult`` 后，
只在 ``point is not None`` 时发布 ``/planning/target_pose``。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from algorithm_pkg.algorithm_core import Vec3

# 只有这三个动作需要规划节点发点。
PLAN_ACTIONS = ("TRACK", "INTERCEPT", "ATTACK")

# 中枢动作。TAKEOFF/RETURN 的坐标由中枢生成；HOVER 在 P2 里
# 单独处理：发己方当前位置（宪章语义：原地保持不动）。
CENTER_ACTIONS = ("TAKEOFF", "HOVER", "RETURN")

# 接口里提到、但当前没有定义行为的动作。
DEFERRED_ACTIONS = ("RESUME_PATROL",)

ALL_ACTIONS = CENTER_ACTIONS + PLAN_ACTIONS + DEFERRED_ACTIONS

# P3 降级级别。P2 直接调用 plan() 时结果固定为 NORMAL；
# P3 的 wrapper 才会填 EXTRAPOLATING / HOLDING / NO_PLAN。
DEGRADE_LEVELS = ("NORMAL", "EXTRAPOLATING", "HOLDING", "NO_PLAN")


@dataclass(frozen=True)
class PlanResult:
    """一帧规划的完整结果。

    不要用裸 ``None`` 表示“不发点”，因为那样分不清是：
    - 动作本来就不该发点（TAKEOFF/RETURN）；
    - 目标过期；
    - 拦截无解；
    - 未知动作。
    这些情况对适配层和日志的含义完全不同。

    Attributes:
        action: 归一化后的动作，例如 ``"TRACK"``；未知动作保留原样。
        raw_action: 调用方传入的原始动作，便于排查大小写/空格问题。
        point: 要发布的 ENU 点；``None`` 表示本帧不发布。
        reachable: 是否真正解出了可达点（TRACK 也记为 True）。
        t_go: 预计拦截时间，单位秒；TRACK 或尾随点为 ``None``。
        reason: 机器可读的原因，例如 ``"target_too_old"``。
        clipped: 点是否被安全边界裁剪过。
        level: P3 降级级别（``DEGRADE_LEVELS`` 之一）；P2 直接调用时为 NORMAL。
        anomaly: P3 异常分类，例如 ``"STALE"``、``"INVALID"``、``"EGO_LOST"``；
            ``None`` 表示本帧没有异常。它是给日志和降级策略表用的，
            不改变 P2 的动作语义。
        guard_notes: 安全护栏的裁剪记录，例如 ``"z_clamped:1.000->5.000"``；
            保持为元组，保证结果不可变、可安全跨线程传递。
        output_stamp: 护栏确认后的发布时刻；P3 不生成 ROS 时间，
            只是把 ``now`` 和上一帧 stamp 比较后给出单调递增的值。
        recovered_from: 本帧仍处于恢复期时，记录从哪个降级级别恢复。
        recovery_frames: 恢复期已经过的帧数；非恢复期为 ``None``。
    """

    action: str
    raw_action: str
    point: Optional[Vec3]
    reachable: bool
    t_go: Optional[float]
    reason: str
    clipped: bool = False
    level: str = "NORMAL"
    anomaly: Optional[str] = None
    guard_notes: tuple[str, ...] = ()
    output_stamp: Optional[float] = None
    recovered_from: Optional[str] = None
    recovery_frames: Optional[int] = None

    @property
    def should_publish(self) -> bool:
        """是否应该把 point 发布到 ``/planning/target_pose``。"""
        return self.point is not None
