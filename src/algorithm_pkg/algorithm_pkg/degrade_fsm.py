"""P3 降级状态机：把“输入异常”变成带滞回的 L0/L1/L2/L3。

四级含义：

============  ============================================================
级别          含义
============  ============================================================
NORMAL        输入新鲜合法，正常规划（L0）。
EXTRAPOLATING 目标短暂丢失，仍在 ``t_extrap_max`` 内，用 CV 外推继续追（L1）。
HOLDING       外推超时，目标真丢了；发保持点，原地等（L2）。
NO_PLAN       己方断流、输入非法或无法降级；本帧不出点，交中枢接管（L3）。
============  ============================================================

为什么要状态机而不是每帧独立判断：

- 目标在“刚好过期”附近抖动时，独立判断会让输出在“有点/没点”之间横跳；
- 进入更差级别要连续 ``n_enter`` 帧确认，回到 NORMAL 要连续
  ``n_recover`` 帧确认；
- 时间戳倒退的帧不参与计数，只丢弃，避免一帧坏时间戳把状态打乱。

恢复时先回 HOLDING 确认，再回 NORMAL；回到 NORMAL 后的第一帧由
``recovery_step_limit`` 临时收紧 ``max_step``，再由外层逐步放开。

本模块不 import rclpy，不发布、不打日志；状态迁移和原因通过
``last_event`` / ``last_recovery`` 暴露给适配层转日志。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Optional

from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.validators import (
    ANOMALY_EGO_LOST,
    ANOMALY_INVALID,
    ANOMALY_STALE,
    ANOMALY_TIME_REVERSED,
    EgoCode,
    EgoStatus,
    FrequencyMonitor,
    TargetCode,
    TargetStatus,
)

# 时间戳比较容差，与 validators 保持一致。
_TIME_EPS = 1e-6


class DegradeLevel(str, Enum):
    """P3 降级级别。"""

    NORMAL = "NORMAL"
    EXTRAPOLATING = "EXTRAPOLATING"
    HOLDING = "HOLDING"
    NO_PLAN = "NO_PLAN"


@dataclass(frozen=True)
class DegradeEvent:
    """一次 FSM ``update`` 的迁移信息。

    Attributes:
        previous: 更新前的级别。
        current: 更新后的级别。
        reason: 机器可读原因，例如 ``"enter_extrapolating"``。
        anomaly: 本帧异常分类；``None`` 表示没有异常。
        changed: 级别是否真的发生变化。
        recover_frames: 本帧仍处于恢复期时的已恢复帧数。
        recover_elapsed: 本帧仍处于恢复期时的已恢复用时，秒。
    """

    previous: str
    current: str
    reason: str
    anomaly: Optional[str]
    changed: bool
    recover_frames: Optional[int] = None
    recover_elapsed: Optional[float] = None


@dataclass(frozen=True)
class RecoveryInfo:
    """一次恢复尝试的完整记录。

    Attributes:
        from_level: 从哪个降级级别开始恢复。
        frames: 实际经过的恢复帧数。
        elapsed: 从开始恢复到完成/中止的用时，秒。
        completed: 是否完整走完恢复期；中途再次异常时为 False。
        reason: 完成或中止原因。
    """

    from_level: str
    frames: int
    elapsed: float
    completed: bool
    reason: str


class DegradeFSM:
    """带滞回的四级降级状态机。

    典型用法：

    .. code-block:: python

        fsm = DegradeFSM(params)
        target_status = validate_target(target, stamp, now, params,
                                        last_stamp=fsm.last_target_stamp)
        ego_status = validate_ego(own, None, ego_stamp, now, params,
                                  last_stamp=fsm.last_ego_stamp)
        level = fsm.update(target_status, ego_status, now)

    ``update`` 只负责级别和计数；发什么点由 ``degrade_policy`` 决定。
    """

    def __init__(self, params: Optional[PlanParams] = None) -> None:
        """构造状态机。

        Args:
            params: 参数对象；``None`` 时使用默认 ``PlanParams()``。
        """
        self.params = PlanParams() if params is None else params

        self.level = DegradeLevel.NORMAL
        self.anomaly: Optional[str] = None
        self.last_event = DegradeEvent(
            previous=DegradeLevel.NORMAL.value,
            current=DegradeLevel.NORMAL.value,
            reason="init",
            anomaly=None,
            changed=False,
        )
        self.last_recovery: Optional[RecoveryInfo] = None

        # 最近一帧的检测结果；供单帧调试/日志读取，不参与状态迁移。
        self._last_target_status: Optional[TargetStatus] = None
        self._last_ego_status: Optional[EgoStatus] = None

        # 滞回计数。
        self._bad_frames = 0
        self._good_frames = 0

        # 最近一次“目标可用”的时刻，用于目标完全消失后的 L1/L2 判断。
        self._last_fresh_time: Optional[float] = None
        self._last_valid_target: Optional[TargetStatus] = None
        self._last_valid_ego: Optional[EgoStatus] = None

        # 最近一次接受过的源时间戳；用于识别时间戳倒退。
        self._last_target_stamp: Optional[float] = None
        self._last_ego_stamp: Optional[float] = None

        # 恢复期状态。
        self._recovering = False
        self._recover_frames = 0
        self._recover_start: Optional[float] = None
        self._recovering_from: Optional[str] = None

        # 最后一个真正发出去的点，作为 L2 的“最后有效点”保持来源。
        self.last_valid_point: Optional[tuple[float, float, float]] = None
        self.last_valid_stamp: Optional[float] = None

        # 频率检测只统计“新目标帧”；默认关闭。
        self._freq_monitor = FrequencyMonitor(
            window_size=self.params.freq_window_size,
            drop_rate_threshold=self.params.freq_anomaly_threshold,
            enabled=self.params.freq_anomaly_enabled,
        )
        self._freq_last_stamp: Optional[float] = None

    # ------------------------------ 只读属性 ------------------------------
    @property
    def last_target_status(self) -> Optional[TargetStatus]:
        """最近一帧传入 FSM 的目标检测结果；供调试/日志读取。"""
        return self._last_target_status

    @property
    def last_ego_status(self) -> Optional[EgoStatus]:
        """最近一帧传入 FSM 的己方检测结果；供调试/日志读取。"""
        return self._last_ego_status

    @property
    def last_target_stamp(self) -> Optional[float]:
        """最近一次接受过的目标时间戳；给下一帧判断倒退用。"""
        return self._last_target_stamp

    @property
    def last_ego_stamp(self) -> Optional[float]:
        """最近一次接受过的己方时间戳。"""
        return self._last_ego_stamp

    @property
    def last_valid_target(self) -> Optional[TargetStatus]:
        """最近一帧可用的目标状态；L1 外推的起点。"""
        return self._last_valid_target

    @property
    def last_valid_ego(self) -> Optional[EgoStatus]:
        """最近一帧可用的己方状态。"""
        return self._last_valid_ego

    @property
    def bad_frames(self) -> int:
        """当前连续异常帧数。"""
        return self._bad_frames

    @property
    def good_frames(self) -> int:
        """当前连续有效帧数。"""
        return self._good_frames

    @property
    def recovering(self) -> bool:
        """是否处于回到 NORMAL 后的 ``max_step`` 收紧期。"""
        return self._recovering

    @property
    def recover_frames(self) -> int:
        """恢复期已经过的帧数；非恢复期为 0。"""
        return self._recover_frames

    @property
    def recovering_from(self) -> Optional[str]:
        """本次恢复的起始降级级别。"""
        return self._recovering_from

    # ------------------------------ 外部调用 ------------------------------
    def observe_target_frequency(self, stamp: Optional[float]) -> bool:
        """在新目标帧到达时更新丢帧率统计。

        只应在“收到新目标采样”时调用；时间戳未变化或倒退时直接忽略，
        避免把主循环频率当成目标频率。

        Args:
            stamp: 目标源时间戳，秒。

        Returns:
            True 表示本帧命中丢帧率异常；检测未启用时恒为 False。
        """
        if stamp is None or not isfinite(stamp):
            return False
        if (
            self._freq_last_stamp is not None
            and stamp <= self._freq_last_stamp + _TIME_EPS
        ):
            return False
        self._freq_last_stamp = stamp
        return self._freq_monitor.observe(stamp)

    def remember_point(
        self,
        point: Optional[tuple[float, float, float]],
        stamp: Optional[float] = None,
    ) -> None:
        """记录最后一个真正输出的点，供 L2 的 ``last_valid`` 保持来源使用。

        Args:
            point: 已通过安全护栏的 ENU 点；``None`` 不覆盖旧值。
            stamp: 该点的发布时刻；只增不减。
        """
        if point is None:
            return
        if not all(isfinite(component) for component in point):
            return
        self.last_valid_point = point
        if stamp is not None and isfinite(stamp):
            if self.last_valid_stamp is None or stamp > self.last_valid_stamp:
                self.last_valid_stamp = stamp

    def recovery_step_limit(self, action: Optional[str] = None) -> Optional[float]:
        """返回当前帧应使用的 ``max_step``。

        非恢复期直接返回该动作的 ``max_step``；恢复期第一帧收紧到
        ``max_step_recover``，之后按 ``n_recover_step`` 线性放开到该动作的
        ``max_step``。传 ``action`` 时用按动作分档的值；不传时用兜底值。

        Returns:
            本帧的跳变上限；``None`` 表示不限制。
        """
        max_step = (
            self.params.max_step_for(action)
            if action is not None
            else self.params.max_step
        )
        max_step_recover = self.params.max_step_recover
        if not self._recovering or max_step_recover is None:
            return max_step
        if self._recover_frames <= 0:
            return max_step_recover
        if max_step is None:
            return None

        span = max(1, self.params.n_recover_step - 1)
        fraction = min(1.0, self._recover_frames / span)
        return max_step_recover + (max_step - max_step_recover) * fraction

    def reset(self) -> None:
        """清空所有状态，恢复到初始 NORMAL。"""
        self.level = DegradeLevel.NORMAL
        self.anomaly = None
        self.last_event = DegradeEvent(
            previous=DegradeLevel.NORMAL.value,
            current=DegradeLevel.NORMAL.value,
            reason="reset",
            anomaly=None,
            changed=False,
        )
        self.last_recovery = None
        self._last_target_status = None
        self._last_ego_status = None
        self._bad_frames = 0
        self._good_frames = 0
        self._last_fresh_time = None
        self._last_valid_target = None
        self._last_valid_ego = None
        self._last_target_stamp = None
        self._last_ego_stamp = None
        self._recovering = False
        self._recover_frames = 0
        self._recover_start = None
        self._recovering_from = None
        self.last_valid_point = None
        self.last_valid_stamp = None
        self._freq_monitor.reset()
        self._freq_last_stamp = None

    def update(
        self,
        target_status: TargetStatus,
        ego_status: EgoStatus,
        now: float,
    ) -> DegradeLevel:
        """更新降级级别。

        Args:
            target_status: ``validate_target`` 的结果。
            ego_status: ``validate_ego`` 的结果。
            now: 当前时刻，秒。

        Returns:
            更新后的 ``DegradeLevel``。迁移原因在 ``last_event`` 里。

        Note:
            - 目标或己方时间戳倒退时，本帧不参与任何计数、不改变级别；
              由 ``degrade_policy`` 丢弃该帧的输出。
            - 己方异常优先于目标异常：没有己方位置时直接往 L3 走。
        """
        self._last_target_status = target_status
        self._last_ego_status = ego_status

        if not isfinite(now):
            self._enter(
                DegradeLevel.NO_PLAN,
                reason="enter_no_plan_now_invalid",
                anomaly=ANOMALY_INVALID,
            )
            return self.level

        # ! 时间戳倒退帧只丢弃，不参与降级计数。
        # ! 如果己方时间戳倒退，也不能让它挡住己方断流的 L3。
        if (
            target_status.code is TargetCode.TIME_REVERSED
            or ego_status.code is EgoCode.TIME_REVERSED
        ):
            self.last_event = DegradeEvent(
                previous=self.level.value,
                current=self.level.value,
                reason="frame_discarded_time_reversed",
                anomaly=ANOMALY_TIME_REVERSED,
                changed=False,
            )
            return self.level

        self._remember_stamps(target_status, ego_status)

        target_usable = target_status.is_usable
        ego_usable = ego_status.is_fresh
        usable = target_usable and ego_usable

        if usable:
            self._good_frames += 1
            self._bad_frames = 0
            self._last_fresh_time = now
            if target_status.position is not None:
                self._last_valid_target = target_status
            self._last_valid_ego = ego_status
        else:
            self._good_frames = 0
            self._bad_frames += 1
            self._abort_recovery(now, "recovery_aborted")

        if usable:
            self._on_usable_frame(target_status, now)
        else:
            self._on_anomaly_frame(target_status, ego_status, now)

        return self.level

    # ------------------------------ 内部分支 ------------------------------
    def _remember_stamps(
        self,
        target_status: TargetStatus,
        ego_status: EgoStatus,
    ) -> None:
        """记录已经接受过的源时间戳，防止倒退帧污染下一次比较。"""
        if (
            target_status.stamp is not None
            and isfinite(target_status.stamp)
            and target_status.code is not TargetCode.TIME_REVERSED
        ):
            if (
                self._last_target_stamp is None
                or target_status.stamp > self._last_target_stamp
            ):
                self._last_target_stamp = target_status.stamp
        if (
            ego_status.stamp is not None
            and isfinite(ego_status.stamp)
            and ego_status.code is not EgoCode.TIME_REVERSED
        ):
            if (
                self._last_ego_stamp is None
                or ego_status.stamp > self._last_ego_stamp
            ):
                self._last_ego_stamp = ego_status.stamp

    def _on_usable_frame(
        self,
        target_status: TargetStatus,
        now: float,
    ) -> None:
        """处理“目标和己方都可用”的帧。"""
        anomaly = target_status.anomaly

        if self.level is DegradeLevel.NORMAL:
            self.anomaly = anomaly
            # 恢复期的后续帧：只有连续有效帧才推进恢复计时。
            if self._recovering:
                self._recover_frames += 1
                if self._recover_frames >= self.params.n_recover_step:
                    self._finish_recovery(
                        now, completed=True, reason="recovered"
                    )
            self.last_event = DegradeEvent(
                previous=self.level.value,
                current=self.level.value,
                reason="normal",
                anomaly=anomaly,
                changed=False,
                recover_frames=(
                    self._recover_frames if self._recovering else None
                ),
                recover_elapsed=(
                    self._recover_elapsed(now)
                    if self._recovering
                    else None
                ),
            )
            return

        if self.level is DegradeLevel.EXTRAPOLATING:
            # 异常刚消失：先回 HOLDING 确认，不要直接跳 NORMAL。
            self._enter(
                DegradeLevel.HOLDING,
                reason="recovering_from_extrapolating",
                anomaly=None,
            )
            self.last_event = DegradeEvent(
                previous=DegradeLevel.EXTRAPOLATING.value,
                current=DegradeLevel.HOLDING.value,
                reason="recovering_from_extrapolating",
                anomaly=None,
                changed=True,
                recover_frames=self._good_frames,
                recover_elapsed=None,
            )
            return

        if self.level is DegradeLevel.HOLDING:
            if self._good_frames >= self.params.n_recover:
                self._begin_recovery(now)
            else:
                self.anomaly = anomaly
                self.last_event = DegradeEvent(
                    previous=DegradeLevel.HOLDING.value,
                    current=DegradeLevel.HOLDING.value,
                    reason="recovering_wait_confirm",
                    anomaly=anomaly,
                    changed=False,
                    recover_frames=self._good_frames,
                    recover_elapsed=None,
                )
            return

        # 只剩 NO_PLAN：先回 HOLDING 确认，再按 n_recover 回 NORMAL。
        if self._good_frames >= self.params.n_exit:
            self._enter(
                DegradeLevel.HOLDING,
                reason="recovered_from_no_plan",
                anomaly=None,
            )
            self.last_event = DegradeEvent(
                previous=DegradeLevel.NO_PLAN.value,
                current=DegradeLevel.HOLDING.value,
                reason="recovered_from_no_plan",
                anomaly=None,
                changed=True,
                recover_frames=self._good_frames,
                recover_elapsed=None,
            )
        else:
            self.last_event = DegradeEvent(
                previous=DegradeLevel.NO_PLAN.value,
                current=DegradeLevel.NO_PLAN.value,
                reason="waiting_recover",
                anomaly=self.anomaly,
                changed=False,
            )

    def _on_anomaly_frame(
        self,
        target_status: TargetStatus,
        ego_status: EgoStatus,
        now: float,
    ) -> None:
        """处理“目标或己方不可用”的帧。"""
        anomaly, reason = self._classify_anomaly(target_status, ego_status)

        if self.level is DegradeLevel.NORMAL:
            if self._bad_frames < self.params.n_enter:
                # 滞回：单帧异常不立刻改级别，只记录诊断信息。
                self.anomaly = anomaly
                self.last_event = DegradeEvent(
                    previous=self.level.value,
                    current=self.level.value,
                    reason="waiting_enter",
                    anomaly=anomaly,
                    changed=False,
                )
                return

            if anomaly == ANOMALY_EGO_LOST:
                self._enter(
                    DegradeLevel.NO_PLAN,
                    reason="enter_no_plan_ego_lost",
                    anomaly=anomaly,
                )
            elif anomaly == ANOMALY_INVALID:
                self._enter(
                    DegradeLevel.NO_PLAN,
                    reason="enter_no_plan_target_invalid",
                    anomaly=anomaly,
                )
            else:
                age = self._stale_age(target_status, now)
                if age <= self.params.t_extrap_max:
                    self._enter(
                        DegradeLevel.EXTRAPOLATING,
                        reason="enter_extrapolating",
                        anomaly=anomaly,
                    )
                else:
                    self._enter(
                        DegradeLevel.HOLDING,
                        reason="enter_holding",
                        anomaly=anomaly,
                    )
            return

        if self.level is DegradeLevel.EXTRAPOLATING:
            # 外推期间，己方断流或目标非法都必须立刻收手；
            # 只有“仍然只是过期”才允许继续外推。
            if anomaly == ANOMALY_EGO_LOST:
                self._enter(
                    DegradeLevel.NO_PLAN,
                    reason="enter_no_plan_ego_lost",
                    anomaly=anomaly,
                )
            elif anomaly == ANOMALY_INVALID:
                self._enter(
                    DegradeLevel.NO_PLAN,
                    reason="enter_no_plan_target_invalid",
                    anomaly=anomaly,
                )
            elif (
                self._stale_age(target_status, now)
                > self.params.t_extrap_max
            ):
                self._enter(
                    DegradeLevel.HOLDING,
                    reason="extrapolation_timeout",
                    anomaly=anomaly,
                )
            else:
                self.anomaly = anomaly
                self.last_event = DegradeEvent(
                    previous=self.level.value,
                    current=self.level.value,
                    reason="extrapolating",
                    anomaly=anomaly,
                    changed=False,
                )
            return

        if self.level is DegradeLevel.HOLDING:
            if anomaly == ANOMALY_EGO_LOST:
                self._enter(
                    DegradeLevel.NO_PLAN,
                    reason="enter_no_plan_ego_lost",
                    anomaly=anomaly,
                )
            elif anomaly == ANOMALY_INVALID:
                self._enter(
                    DegradeLevel.NO_PLAN,
                    reason="enter_no_plan_target_invalid",
                    anomaly=anomaly,
                )
            else:
                self.anomaly = anomaly
                self.last_event = DegradeEvent(
                    previous=self.level.value,
                    current=self.level.value,
                    reason="holding",
                    anomaly=anomaly,
                    changed=False,
                )
            return

        # NO_PLAN：保持，但更新最近一次异常原因，便于日志看到“为什么还没恢复”。
        self.anomaly = anomaly
        self.last_event = DegradeEvent(
            previous=self.level.value,
            current=self.level.value,
            reason=reason,
            anomaly=anomaly,
            changed=False,
        )

    def _classify_anomaly(
        self,
        target_status: TargetStatus,
        ego_status: EgoStatus,
    ) -> tuple[str, str]:
        """按“己方优先”给出本帧异常分类和原因。"""
        if not ego_status.is_fresh:
            return ANOMALY_EGO_LOST, "ego_unusable"
        if target_status.code in (
            TargetCode.INVALID,
            TargetCode.OUT_OF_BOUNDS,
        ):
            return ANOMALY_INVALID, "target_invalid"
        return ANOMALY_STALE, "target_stale"

    def _stale_age(self, target_status: TargetStatus, now: float) -> float:
        """返回用于判断 L1/L2 的“丢失时长”。

        目标还有一个旧时间戳时直接用 ``age``；目标完全消失时用
        “距离最后一次可用目标的时间”。两者都没有时返回 inf。
        """
        if target_status.code is TargetCode.MISSING or not isfinite(
            target_status.age
        ):
            if self._last_fresh_time is None:
                return float("inf")
            return max(0.0, now - self._last_fresh_time)
        return max(0.0, target_status.age)

    def _enter(
        self,
        level: DegradeLevel,
        reason: str,
        anomaly: Optional[str],
    ) -> None:
        """切换到指定级别并记录事件。"""
        previous = self.level
        self.level = level
        self.anomaly = anomaly
        self.last_event = DegradeEvent(
            previous=previous.value,
            current=level.value,
            reason=reason,
            anomaly=anomaly,
            changed=previous is not level,
        )

    def _begin_recovery(self, now: float) -> None:
        """进入回到 NORMAL 后的 max_step 收紧期。"""
        previous = self.level.value
        self.level = DegradeLevel.NORMAL
        self.anomaly = None
        self._recovering = True
        self._recover_frames = 0
        self._recover_start = now
        self._recovering_from = previous
        self.last_event = DegradeEvent(
            previous=previous,
            current=DegradeLevel.NORMAL.value,
            reason="recovered_to_normal",
            anomaly=None,
            changed=previous != DegradeLevel.NORMAL.value,
            recover_frames=0,
            recover_elapsed=0.0,
        )

    def _finish_recovery(
        self,
        now: float,
        *,
        completed: bool,
        reason: str,
    ) -> None:
        """结束恢复期并保存恢复用时/帧数。"""
        if self._recovering_from is None:
            elapsed = 0.0
        elif self._recover_start is None:
            elapsed = 0.0
        else:
            elapsed = max(0.0, now - self._recover_start)
        self.last_recovery = RecoveryInfo(
            from_level=self._recovering_from,
            frames=self._recover_frames,
            elapsed=elapsed,
            completed=completed,
            reason=reason,
        )
        self._recovering = False
        self._recover_frames = 0
        self._recover_start = None
        self._recovering_from = None

    def _abort_recovery(self, now: float, reason: str) -> None:
        """恢复中途再次异常时中止收紧期。"""
        if not self._recovering:
            return
        self._finish_recovery(now, completed=False, reason=reason)

    def _recover_elapsed(self, now: float) -> Optional[float]:
        """返回恢复期已经过的用时；非恢复期为 None。"""
        if not self._recovering or self._recover_start is None:
            return None
        return max(0.0, now - self._recover_start)
