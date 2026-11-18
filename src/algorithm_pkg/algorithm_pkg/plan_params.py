"""P2 策略参数：把“不同模式用不同速度”集中在一个地方。

P1 的 ``solve_intercept`` 只接受一个 ``v_max``。到了 P2，动作有
TRACK / INTERCEPT / ATTACK 三种，飞控给它们的步长上限也不同，
所以不能在调用处随手传一个数字。

这三个默认值来自上游飞控的步长：

- INTERCEPT: 0.38 m / 控制周期 × 20 Hz = 7.6 m/s
- ATTACK:    0.62 m / 控制周期 × 20 Hz = 12.4 m/s
- v_max:     6.0 m/s，只作为兜底

注意：7.6/12.4 是“飞控控制周期”的换算，不是规划节点的发布频率。
如果飞控周期变了，应改这里的参数，而不是在策略函数里写死。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional


@dataclass(frozen=True)
class PlanParams:
    """P2 一帧规划需要的全部参数。

    Attributes:
        v_max: 兜底速度上限，米/秒。
        v_max_intercept: INTERCEPT 速度上限，米/秒。
        v_max_attack: ATTACK 速度上限，米/秒。
        t_go_max: 最长允许拦截时间，秒；超过认为追不上。
        min_distance: 目标太近时不出点，米。
        d_tail: INTERCEPT 追不上时，尾随点距目标多远，米。
        min_alt: ATTACK 最低高度，米。
        safe_alt: TRACK/INTERCEPT 最低高度，米。
        max_alt: 最高高度，米。
        max_xy: 水平边界，x/y 限制在 ±max_xy，米。
        max_step: 单帧最大位移，米；None 表示不限制。
        max_pred_age: 目标预测允许的最大年龄，秒。
            即使 ``TargetState.valid`` 为 True，超过这个年龄也不规划。
    """

    v_max: float = 6.0
    v_max_intercept: float = 7.6
    v_max_attack: float = 12.4
    t_go_max: float = 8.0
    min_distance: float = 0.5
    d_tail: float = 4.0
    min_alt: float = 3.0
    safe_alt: float = 5.0
    max_alt: float = 50.0
    max_xy: float = 1000.0
    max_step: Optional[float] = None
    max_pred_age: float = 0.5

    def __post_init__(self) -> None:
        """在创建参数对象时就把配置错误暴露出来。"""
        if not isfinite(self.v_max) or self.v_max <= 0.0:
            raise ValueError("v_max must be positive")
        if not isfinite(self.v_max_intercept) or self.v_max_intercept <= 0.0:
            raise ValueError("v_max_intercept must be positive")
        if not isfinite(self.v_max_attack) or self.v_max_attack <= 0.0:
            raise ValueError("v_max_attack must be positive")
        if not isfinite(self.t_go_max) or self.t_go_max <= 0.0:
            raise ValueError("t_go_max must be positive")
        if not isfinite(self.min_distance) or self.min_distance < 0.0:
            raise ValueError("min_distance must be non-negative")
        if not isfinite(self.d_tail) or self.d_tail < 0.0:
            raise ValueError("d_tail must be non-negative")
        if not isfinite(self.min_alt) or self.min_alt < 0.0:
            raise ValueError("min_alt must be non-negative")
        if not isfinite(self.safe_alt) or self.safe_alt < 0.0:
            raise ValueError("safe_alt must be non-negative")
        if not isfinite(self.max_alt) or self.max_alt < 0.0:
            raise ValueError("max_alt must be non-negative")
        if self.max_alt < max(self.min_alt, self.safe_alt):
            raise ValueError("max_alt must be at least min_alt and safe_alt")
        if not isfinite(self.max_xy) or self.max_xy <= 0.0:
            raise ValueError("max_xy must be positive")
        if self.max_step is not None:
            if not isfinite(self.max_step) or self.max_step <= 0.0:
                raise ValueError("max_step must be positive when provided")
        if not isfinite(self.max_pred_age) or self.max_pred_age <= 0.0:
            raise ValueError("max_pred_age must be positive")

    def v_max_for(self, action: str) -> float:
        """按动作选速度上限；未知动作退回 ``v_max``。

        ``plan()`` 会先拒绝未知动作，所以这里的兜底只防调用方直接
        拿这个方法来选速度。
        """
        normalized = str(action).strip().upper()
        if normalized == "INTERCEPT":
            return self.v_max_intercept
        if normalized == "ATTACK":
            return self.v_max_attack
        return self.v_max
