"""HOVER 固定点选择。

``center_node`` 的 HOVER 分支原来只用己方当前位置。算法包接入固定点后，
需要优先用规划点。这里把“用不用规划点”的判断抽成纯函数，方便不启动
ROS 也能测。
"""

from __future__ import annotations

from math import isfinite
from typing import Optional

Vec3 = tuple[float, float, float]


def _finite(point: Optional[Vec3]) -> bool:
    """三维点存在且三个分量都是有限数。"""
    if point is None:
        return False
    try:
        return len(point) == 3 and all(isfinite(value) for value in point)
    except TypeError:
        return False


def resolve_hover_point(
    own: Vec3,
    plan: Optional[Vec3],
    plan_action: str,
    plan_age: float,
    max_plan_age: float,
) -> tuple[Vec3, str]:
    """选择 HOVER 这一帧使用的基准点。

    Args:
        own: 己方当前位置，ENU 米。
        plan: 最近一次规划点，ENU 米；没有时为 ``None``。
        plan_action: 该规划点对应的动作标签。
        plan_age: 规划点从收到到现在的时间，秒。
        max_plan_age: 允许的最大时间，秒。

    Returns:
        ``(point, source)``。用规划点时 ``source="plan"``，回退当前位置时
        ``source="own"``。

    Note:
        动作标签只作为筛选条件，不驱动模式。TRACK/INTERCEPT/ATTACK 的
        旧点不能拿来悬停，否则刚切 HOVER 时会先朝旧目标冲一下。
    """
    if (
        _finite(plan)
        and str(plan_action).strip().upper() == "HOVER"
        and isfinite(plan_age)
        and 0.0 <= plan_age <= max_plan_age
    ):
        return plan, "plan"
    return own, "own"
