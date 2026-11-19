"""P2 策略参数：集中所有可调参数，启动时校验。

速度口径以《内部分工细节》(V2.0) 为准：最大平飞速度 ``V_max = 6.0 m/s``。
INTERCEPT/ATTACK 的“全速”也是 6.0，不再使用 P4 联调时的 7.6/12.4。
原因：那两档是飞控步长 0.38/0.62 m × 20 Hz 推出来的，和宪章的 V_max 冲突；
宪章是团队最高准则，先按 6.0 统一。

如果团队确认飞控确实允许更大步长，改这三个字段即可，不要在策略函数里写死。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Optional


@dataclass(frozen=True)
class PlanParams:
    """P2 一帧规划需要的全部参数。

    Attributes:
        v_max: 兜底速度上限，米/秒，按宪章 V_max=6.0。
        v_max_intercept: INTERCEPT 速度上限，米/秒；宪章下同为 6.0。
        v_max_attack: ATTACK 速度上限，米/秒；宪章下同为 6.0。
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
        t_extrap_max: P3 允许用 CV 外推继续追的时间上限，秒。
            目标年龄在 ``(max_pred_age, t_extrap_max]`` 内进 L1，
            超过进 L2；默认 1.0 s，待联调确认。
        n_enter: 连续多少帧异常才进入更差的降级级别；默认 2 帧，待确认。
        n_exit: 从 L3 恢复到 L2 所需的连续有效帧数；默认 1 帧，待确认。
        n_recover: 从降级状态回到 L0 所需的连续有效帧数；默认 3 帧
            （20Hz 下 0.15 s），待确认。
        max_step_recover: 恢复期第一帧临时收紧的单帧位移，米；
            None 表示不额外收紧。默认 0.2 m/帧（20Hz 下约 4 m/s），待确认。
        n_recover_step: 恢复期逐步放开 ``max_step`` 所用的帧数；
            默认 5 帧，待确认。
        freq_anomaly_enabled: 是否启用目标丢帧率检测；默认关闭。
        freq_anomaly_threshold: 丢帧率阈值，0～1；默认 0.5，待确认。
        freq_window_size: 丢帧率统计窗口（帧数）；默认 20。
        hold_point_source: L2 保持点来源，``"last_valid"`` 或
            ``"current_ego"``；默认 ``"last_valid"``，待确认。
        ego_timeout: 己方状态允许的最大年龄，秒；默认 1.0，与适配层一致。
        target_min_alt: 目标输入的最低高度，米；默认 0.0（地面）。
            低于这个值按输入越界处理；目标贴地 1 米仍合法，是否抬高由
            输出护栏按模式决定。
    """

    v_max: float = 6.0
    v_max_intercept: float = 6.0
    v_max_attack: float = 6.0
    t_go_max: float = 8.0
    min_distance: float = 0.5
    d_tail: float = 4.0
    min_alt: float = 3.0
    safe_alt: float = 5.0
    max_alt: float = 50.0
    max_xy: float = 1000.0
    max_step: Optional[float] = None
    max_pred_age: float = 0.5

    # ------------------------- P3 异常降级参数 -------------------------
    # 以下是“可配置、可标定”的初值，联调后按 src/algorithm_pkg/README.md
    # 的“降级生存法则”调整；
    # 不要在降级代码里再写死这些数字。
    t_extrap_max: float = 1.0
    n_enter: int = 2
    n_exit: int = 1
    n_recover: int = 3
    max_step_recover: Optional[float] = 0.2
    n_recover_step: int = 5
    freq_anomaly_enabled: bool = False
    freq_anomaly_threshold: float = 0.5
    freq_window_size: int = 20
    hold_point_source: str = "last_valid"
    ego_timeout: float = 1.0
    target_min_alt: float = 0.0

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

        # P3 参数的校验：错误配置在节点启动时即暴露，而不是在降级过程中才失败。
        if not isfinite(self.t_extrap_max) or self.t_extrap_max <= 0.0:
            raise ValueError("t_extrap_max must be positive")
        for name, value in (
            ("n_enter", self.n_enter),
            ("n_exit", self.n_exit),
            ("n_recover", self.n_recover),
            ("n_recover_step", self.n_recover_step),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.max_step_recover is not None:
            if (
                not isfinite(self.max_step_recover)
                or self.max_step_recover <= 0.0
            ):
                raise ValueError(
                    "max_step_recover must be positive when provided"
                )
        if not isfinite(self.freq_anomaly_threshold):
            raise ValueError("freq_anomaly_threshold must be finite")
        if not 0.0 <= self.freq_anomaly_threshold <= 1.0:
            raise ValueError("freq_anomaly_threshold must be within [0, 1]")
        if not isinstance(self.freq_window_size, int) or isinstance(
            self.freq_window_size, bool
        ):
            raise ValueError("freq_window_size must be an integer")
        if self.freq_window_size < 2:
            raise ValueError("freq_window_size must be at least 2")
        if self.hold_point_source not in ("last_valid", "current_ego"):
            raise ValueError(
                "hold_point_source must be 'last_valid' or 'current_ego'"
            )
        if not isfinite(self.ego_timeout) or self.ego_timeout <= 0.0:
            raise ValueError("ego_timeout must be positive")
        if not isfinite(self.target_min_alt) or self.target_min_alt < 0.0:
            raise ValueError("target_min_alt must be non-negative")
        if self.target_min_alt > self.max_alt:
            raise ValueError("target_min_alt must not exceed max_alt")

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
