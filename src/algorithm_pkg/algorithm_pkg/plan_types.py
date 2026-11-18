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

# 这三个动作的坐标由中枢自己生成，规划节点不应抢控制。
CENTER_ACTIONS = ("TAKEOFF", "HOVER", "RETURN")

# 接口里提到、但当前没有定义行为的动作。
DEFERRED_ACTIONS = ("RESUME_PATROL",)

ALL_ACTIONS = CENTER_ACTIONS + PLAN_ACTIONS + DEFERRED_ACTIONS


@dataclass(frozen=True)
class PlanResult:
    """一帧规划的完整结果。

    不要用裸 ``None`` 表示“不发点”，因为那样分不清是：
    - 动作本来就不该发点（HOVER）；
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
    """

    action: str
    raw_action: str
    point: Optional[Vec3]
    reachable: bool
    t_go: Optional[float]
    reason: str
    clipped: bool = False

    @property
    def should_publish(self) -> bool:
        """是否应该把 point 发布到 ``/planning/target_pose``。"""
        return self.point is not None
