"""P3 异常检测：把“这一帧输入还能不能用”变成可分类的结果。

P3 的降级状态机不直接猜“目标是不是丢了”，而是先看这里的分类：

- 目标：FRESH / STALE / INVALID / OUT_OF_BOUNDS / TIME_REVERSED /
  FREQ_ANOMALY / MISSING；
- 己方：FRESH / STALE / INVALID / TIME_REVERSED / MISSING；
- 战术指令：JSON 能不能解析、动作是否在已知枚举里。

本模块只做分类，不做“降级到哪一级”的决策；决策在 ``degrade_fsm.py``。
也不 import rclpy、不订阅/发布、不打日志，保持 pytest 可直接测。

异常分类原则：

- **时间戳倒退**（含未来时间戳）单独标 ``TIME_REVERSED``，由 FSM 丢弃
  该帧、维持当前级别；不要混进 ``STALE``，否则一帧坏时间戳会把整个
  降级状态打乱。
- **NaN/Inf** 标 ``INVALID``。
- **越界**标 ``OUT_OF_BOUNDS``；目标是输入侧越界，不是输出护栏。
- 超过 ``max_pred_age`` 标 ``STALE``；外推多久由 FSM 用 ``t_extrap_max``
  决定，检测器不替它做决定。
- **丢帧率异常**标 ``FREQ_ANOMALY``，但位置仍然可用；默认只做诊断标记，
  是否降级由配置和 FSM 决定。

已知限制：
- 规划侧假设 ``/<drone_id>/state`` 的来源是桥接原始 odom。若中枢把同一
  topic 回灌给规划（接口 E/F 同名），己方状态的新鲜度判断不可靠；P3 只在
  文档记录这个风险，不处理自回灌（属于适配层接口核对）。
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from statistics import median
from typing import Deque, Optional

from algorithm_pkg.algorithm_core import TargetState, Vec3
from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.plan_types import ALL_ACTIONS

# 时间戳比较容差，单位秒；ROS 时间戳和 ROS 时钟在同一台机器上只有
# 微秒级抖动，1e-6 足够区分“同帧重复”和“倒退”。
_TIME_EPS = 1e-6

# P3 对外输出的异常原因分类；日志和降级策略表按这几个字符串对齐。
ANOMALY_STALE = "STALE"
ANOMALY_INVALID = "INVALID"
ANOMALY_TIME_REVERSED = "TIME_REVERSED"
ANOMALY_EGO_LOST = "EGO_LOST"
ANOMALY_FREQ = "FREQ_ANOMALY"
ANOMALY_CMD_INVALID = "CMD_INVALID"


class TargetCode(str, Enum):
    """目标输入的分类结果。"""

    FRESH = "FRESH"
    STALE = "STALE"
    INVALID = "INVALID"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    TIME_REVERSED = "TIME_REVERSED"
    FREQ_ANOMALY = "FREQ_ANOMALY"
    MISSING = "MISSING"


class EgoCode(str, Enum):
    """己方状态的分类结果。"""

    FRESH = "FRESH"
    STALE = "STALE"
    INVALID = "INVALID"
    TIME_REVERSED = "TIME_REVERSED"
    MISSING = "MISSING"


class CmdCode(str, Enum):
    """战术指令的分类结果。"""

    FRESH = "FRESH"
    INVALID = "INVALID"


@dataclass(frozen=True)
class TargetStatus:
    """一帧目标的检测结果。

    Attributes:
        code: 分类结果；调用方用 ``is_usable`` 判断能否继续规划。
        position: 目标 ENU 位置；``MISSING`` 时为 ``None``。
        velocity: 目标速度；检测器没有速度信息时为 ``None``。
        velocity_valid: 速度是否可用（与 P1 的 ``velocity_valid`` 对齐）。
        stamp: 这一帧使用的源时间戳，秒。
        age: ``now - stamp``，秒；``MISSING`` 时为 ``inf``。
        reason: 机器可读原因，例如 ``"target_stale"``。
        observed_at: 调用检测器时的 ``now``；FSM 用它计算丢失后的外推时长。
        freq_anomaly: 是否同时命中丢帧率异常标记。
    """

    code: TargetCode
    position: Optional[Vec3]
    velocity: Optional[Vec3]
    velocity_valid: bool
    stamp: Optional[float]
    age: float
    reason: str
    observed_at: float
    freq_anomaly: bool = False

    @property
    def is_fresh(self) -> bool:
        """是否新鲜且没有任何异常。"""
        return self.code is TargetCode.FRESH

    @property
    def is_usable(self) -> bool:
        """是否仍可用来规划；``FREQ_ANOMALY`` 的位置仍然可用。"""
        return self.code in (TargetCode.FRESH, TargetCode.FREQ_ANOMALY)

    @property
    def anomaly(self) -> Optional[str]:
        """把分类结果映射成 P3 的异常分类；``None`` 表示无异常。"""
        if self.code is TargetCode.FRESH:
            return None
        if self.code is TargetCode.TIME_REVERSED:
            return ANOMALY_TIME_REVERSED
        if self.code is TargetCode.FREQ_ANOMALY:
            return ANOMALY_FREQ
        if self.code in (TargetCode.STALE, TargetCode.MISSING):
            return ANOMALY_STALE
        return ANOMALY_INVALID


@dataclass(frozen=True)
class EgoStatus:
    """一帧己方状态的检测结果。

    Attributes:
        code: 分类结果。
        position: 己方 ENU 位置；``MISSING`` 时为 ``None``。
        velocity: 己方速度；调用方没提供时为 ``None``。
        stamp: 这一帧使用的源时间戳，秒；``None`` 表示没有时间戳。
        age: ``now - stamp``，秒；未知时为 ``inf``。
        reason: 机器可读原因，例如 ``"ego_stale"``。
        observed_at: 调用检测器时的 ``now``。
    """

    code: EgoCode
    position: Optional[Vec3]
    velocity: Optional[Vec3]
    stamp: Optional[float]
    age: float
    reason: str
    observed_at: float

    @property
    def is_fresh(self) -> bool:
        """是否新鲜且可用。"""
        return self.code is EgoCode.FRESH

    @property
    def anomaly(self) -> Optional[str]:
        """把分类结果映射成 P3 的异常分类。"""
        if self.code is EgoCode.FRESH:
            return None
        if self.code is EgoCode.TIME_REVERSED:
            return ANOMALY_TIME_REVERSED
        return ANOMALY_EGO_LOST


@dataclass(frozen=True)
class CmdStatus:
    """一条 ``/strategy/cmd`` JSON 的检测结果。

    Attributes:
        code: ``FRESH`` 表示可执行，``INVALID`` 表示必须拒绝。
        drone_id: 指令里的无人机 ID；缺失时为 ``None``。
        action: 归一化后的动作（大写、去空格）；无法归一化时保留空串。
        raw_action: 原始动作值，便于排查大小写/空格/类型问题。
        reason: 机器可读原因，例如 ``"cmd_json_invalid"``。
    """

    code: CmdCode
    drone_id: Optional[str]
    action: str
    raw_action: object
    reason: str

    @property
    def is_valid(self) -> bool:
        """指令是否可以执行。"""
        return self.code is CmdCode.FRESH


class FrequencyMonitor:
    """目标丢帧率统计。

    只在“接收到新目标帧”时调用 ``observe(stamp)``；不要在每个控制周期
    都调，否则测到的是主循环频率，不是目标频率。

    估算方式：用最近 ``window_size - 1`` 个到达间隔的中位数作为“期望
    周期”，当前间隔比期望大得越多，丢帧率越高：

        drop_rate = 1 - expected_period / actual_period

    例如目标从 10Hz 掉到 1Hz：期望 0.1s、实际 1.0s，丢帧率 0.9，
    超过默认阈值 0.5，命中 ``FREQ_ANOMALY``。窗口滑过一半后中位数
    变成 1.0s，异常自动消失，避免长期误报。

    Note:
        该检测默认关闭（``PlanParams.freq_anomaly_enabled=False``）。
        P3 先保证“可配置、行为可解释”，阈值等联调标定。
    """

    def __init__(
        self,
        *,
        window_size: int = 20,
        drop_rate_threshold: float = 0.5,
        enabled: bool = False,
    ) -> None:
        """构造检测器。

        Args:
            window_size: 保留多少个到达时刻用于估计期望周期；至少 2。
            drop_rate_threshold: 丢帧率阈值，0～1。
            enabled: 是否启用；关闭时 ``observe`` 始终返回 False。

        Raises:
            ValueError: 窗口、阈值或开关类型非法。
        """
        if not isinstance(window_size, int) or isinstance(window_size, bool):
            raise ValueError("window_size must be an integer")
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        if not isfinite(drop_rate_threshold):
            raise ValueError("drop_rate_threshold must be finite")
        if not 0.0 <= drop_rate_threshold <= 1.0:
            raise ValueError("drop_rate_threshold must be within [0, 1]")
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a bool")

        self.window_size = window_size
        self.drop_rate_threshold = drop_rate_threshold
        self.enabled = enabled

        # 到达时刻用 deque 保存；长度达到窗口后自动丢掉最旧的一帧。
        self._stamps: Deque[float] = deque(maxlen=window_size)

    def reset(self) -> None:
        """清空历史，用于模式切换或测试隔离。"""
        self._stamps.clear()

    def observe(self, stamp: float) -> bool:
        """记录一帧到达时刻并返回本帧是否命中丢帧率异常。

        Args:
            stamp: 目标源时间戳，秒；必须与之前几帧使用同一时钟。

        Returns:
            True 表示本帧丢帧率超过阈值；关闭、时间戳非法、时间戳不递增
            或历史不足两帧时返回 False。
        """
        if not self.enabled:
            return False
        if not isfinite(stamp):
            return False

        # ! 同一时刻重复帧或时间戳倒退不参与频率统计；
        # ! 倒退帧由 validate_target 单独标 TIME_REVERSED 并丢弃。
        if self._stamps and stamp <= self._stamps[-1] + _TIME_EPS:
            return False

        anomaly = False
        if len(self._stamps) >= 2:
            deltas = [
                later - earlier
                for earlier, later in zip(self._stamps, list(self._stamps)[1:])
            ]
            expected = median(deltas)
            actual = stamp - self._stamps[-1]
            if expected > 0.0 and actual > 0.0:
                drop_rate = max(0.0, 1.0 - expected / actual)
                anomaly = drop_rate >= self.drop_rate_threshold

        self._stamps.append(stamp)
        return anomaly


def _params_or_default(params: Optional[PlanParams]) -> PlanParams:
    """允许调用方省略参数对象。"""
    return PlanParams() if params is None else params


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


def _position_tuple(value: object) -> Optional[Vec3]:
    """把长度为 3 的位置序列转成 ``Vec3``；不合法时返回 ``None``。"""
    try:
        sequence = tuple(value)  # type: ignore[arg-type]
        if len(sequence) != 3:
            return None
        return (float(sequence[0]), float(sequence[1]), float(sequence[2]))
    except (TypeError, ValueError):
        return None


def _target_fields(
    p_t: object,
    stamp: Optional[float],
) -> tuple[Optional[Vec3], Optional[Vec3], bool, Optional[float]]:
    """从 ``TargetState`` 或裸三维点里取出位置/速度/时间戳。"""
    if isinstance(p_t, TargetState):
        source_stamp = p_t.stamp if stamp is None else stamp
        return p_t.position, p_t.velocity, p_t.velocity_valid, source_stamp

    position = _position_tuple(p_t)
    return position, None, False, stamp


def validate_target(
    p_t: object,
    stamp: Optional[float],
    now: float,
    params: Optional[PlanParams] = None,
    *,
    last_stamp: Optional[float] = None,
    freq_anomaly: bool = False,
) -> TargetStatus:
    """检测一帧目标输入。

    Args:
        p_t: 目标位置；可以是 P1 的 ``TargetState``，也可以是裸三维点
            ``(x, y, z)`` 或 ``None``（表示这一帧没有目标）。
        stamp: 目标源时间戳，秒；``None`` 时若 ``p_t`` 是 ``TargetState``
            则用它的 ``stamp``。
        now: 当前时刻，秒；必须和 ``stamp`` 使用同一时钟。
        params: 参数对象；``None`` 时使用默认 ``PlanParams()``。
        last_stamp: 上一帧已接受的目标时间戳；用于识别时间戳倒退。
        freq_anomaly: 丢帧率检测器的结果；True 时标 ``FREQ_ANOMALY``。

    Returns:
        ``TargetStatus``。``TIME_REVERSED`` 表示这一帧应丢弃、但不改变
        FSM 当前降级级别；``STALE`` 只是“超过 max_pred_age”，外推多久
        由 FSM 决定。

    Note:
        位置越界按输入侧边界判断：``|x|/|y| > max_xy``、``z > max_alt``
        或 ``z < 0`` 都算 ``OUT_OF_BOUNDS``。目标贴地到 1 米仍然合法，
        是否抬高由输出护栏按模式处理。
    """
    params = _params_or_default(params)

    if not isfinite(now):
        return TargetStatus(
            code=TargetCode.INVALID,
            position=None,
            velocity=None,
            velocity_valid=False,
            stamp=stamp,
            age=0.0,
            reason="now_non_finite",
            observed_at=now,
        )

    if p_t is None:
        return TargetStatus(
            code=TargetCode.MISSING,
            position=None,
            velocity=None,
            velocity_valid=False,
            stamp=None,
            age=float("inf"),
            reason="target_missing",
            observed_at=now,
        )

    position, velocity, velocity_valid, source_stamp = _target_fields(
        p_t, stamp
    )

    if source_stamp is None:
        return TargetStatus(
            code=TargetCode.INVALID,
            position=position,
            velocity=velocity,
            velocity_valid=velocity_valid,
            stamp=None,
            age=0.0,
            reason="stamp_missing",
            observed_at=now,
        )
    if not isfinite(source_stamp):
        return TargetStatus(
            code=TargetCode.INVALID,
            position=position,
            velocity=velocity,
            velocity_valid=velocity_valid,
            stamp=source_stamp,
            age=0.0,
            reason="stamp_non_finite",
            observed_at=now,
        )

    # ! 时间戳倒退和未来时间戳都先不参与降级计数，交给 FSM 丢弃该帧。
    # ! 这两类混进 STALE 会让一帧坏时间戳直接改变降级级别。
    if source_stamp > now + _TIME_EPS:
        return TargetStatus(
            code=TargetCode.TIME_REVERSED,
            position=position,
            velocity=velocity,
            velocity_valid=velocity_valid,
            stamp=source_stamp,
            age=0.0,
            reason="stamp_in_future",
            observed_at=now,
        )
    if last_stamp is not None and source_stamp < last_stamp - _TIME_EPS:
        return TargetStatus(
            code=TargetCode.TIME_REVERSED,
            position=position,
            velocity=velocity,
            velocity_valid=velocity_valid,
            stamp=source_stamp,
            age=0.0,
            reason="stamp_reversed",
            observed_at=now,
        )

    if not _finite_vec(position):
        return TargetStatus(
            code=TargetCode.INVALID,
            position=position,
            velocity=velocity,
            velocity_valid=velocity_valid,
            stamp=source_stamp,
            age=0.0,
            reason="target_position_non_finite",
            observed_at=now,
        )

    if velocity is not None and not _finite_vec(velocity):
        return TargetStatus(
            code=TargetCode.INVALID,
            position=position,
            velocity=velocity,
            velocity_valid=False,
            stamp=source_stamp,
            age=0.0,
            reason="target_velocity_non_finite",
            observed_at=now,
        )

    x, y, z = position
    if (
        abs(x) > params.max_xy
        or abs(y) > params.max_xy
        or z > params.max_alt
        or z < params.target_min_alt
    ):
        return TargetStatus(
            code=TargetCode.OUT_OF_BOUNDS,
            position=position,
            velocity=velocity,
            velocity_valid=velocity_valid,
            stamp=source_stamp,
            age=max(0.0, now - source_stamp),
            reason="target_out_of_bounds",
            observed_at=now,
        )

    age = max(0.0, now - source_stamp)

    # ? 正常情况下 P1 只在 age > max_age 时把 valid 置 False；如果调用方
    # ? 在数据还新鲜时就把 valid 置 False，说明它明确判定这一帧不可用，
    # ? 不能靠 P3 外推把“已判无效”的数据救回来，按 INVALID 进 L3。
    marked_invalid = isinstance(p_t, TargetState) and not p_t.valid
    if marked_invalid and age <= params.max_pred_age:
        return TargetStatus(
            code=TargetCode.INVALID,
            position=position,
            velocity=velocity,
            velocity_valid=velocity_valid,
            stamp=source_stamp,
            age=age,
            reason="target_marked_invalid",
            observed_at=now,
        )

    # ? STALE 只表示“超过 P2 的 0.5 秒发点门限”，不是“目标已经不可用”：
    # ? FSM 还会用 t_extrap_max 判断能不能继续 CV 外推。
    if age > params.max_pred_age:
        return TargetStatus(
            code=TargetCode.STALE,
            position=position,
            velocity=velocity,
            velocity_valid=velocity_valid,
            stamp=source_stamp,
            age=age,
            reason="target_stale",
            observed_at=now,
            freq_anomaly=freq_anomaly,
        )

    if freq_anomaly:
        return TargetStatus(
            code=TargetCode.FREQ_ANOMALY,
            position=position,
            velocity=velocity,
            velocity_valid=velocity_valid,
            stamp=source_stamp,
            age=age,
            reason="target_freq_anomaly",
            observed_at=now,
            freq_anomaly=True,
        )

    return TargetStatus(
        code=TargetCode.FRESH,
        position=position,
        velocity=velocity,
        velocity_valid=velocity_valid,
        stamp=source_stamp,
        age=age,
        reason="target_fresh",
        observed_at=now,
    )


def validate_ego(
    p_o: Optional[Vec3],
    v_o: Optional[Vec3],
    stamp: Optional[float],
    now: float,
    params: Optional[PlanParams] = None,
    *,
    last_stamp: Optional[float] = None,
) -> EgoStatus:
    """检测一帧己方状态。

    Args:
        p_o: 己方 ENU 位置；``None`` 表示这一帧没有状态。
        v_o: 己方速度；接口 E 建议使用 ``twist.linear``。``None`` 表示
            调用方没有提供速度（P1 的拦截几何不依赖己方速度，仍可用）。
        stamp: 己方状态源时间戳，秒；``None`` 表示没带时间戳。
        now: 当前时刻，秒。
        params: 参数对象；``None`` 时使用默认 ``PlanParams()``。
        last_stamp: 上一帧已接受的己方时间戳；用于识别时间戳倒退。

    Returns:
        ``EgoStatus``。``STALE`` 表示超过 ``ego_timeout``；``MISSING``
        表示没有己方位置，FSM 会据此进 L3。

    Note:
        - 速度 NaN/Inf 单独标 ``INVALID``，不会让拦截点带着 NaN 继续算。
        - ``stamp`` 为 ``None`` 时按“新鲜度未校验”处理，返回 FRESH 且
          ``reason="ego_freshness_unchecked"``；适配层应优先传入
          ``header.stamp``，否则桥接断流时这里看不出来。
        - 已知问题：若中枢把 ``/drone_<id>/state`` 回灌给规划，时间戳
          可能仍是旧的，新鲜度判断不可靠；P3 不处理 E/F 自回灌。
    """
    params = _params_or_default(params)

    if not isfinite(now):
        return EgoStatus(
            code=EgoCode.INVALID,
            position=p_o,
            velocity=v_o,
            stamp=stamp,
            age=0.0,
            reason="now_non_finite",
            observed_at=now,
        )

    if p_o is None:
        return EgoStatus(
            code=EgoCode.MISSING,
            position=None,
            velocity=v_o,
            stamp=stamp,
            age=float("inf"),
            reason="ego_missing",
            observed_at=now,
        )

    if not _finite_vec(p_o):
        return EgoStatus(
            code=EgoCode.INVALID,
            position=p_o,
            velocity=v_o,
            stamp=stamp,
            age=0.0,
            reason="ego_position_non_finite",
            observed_at=now,
        )

    if v_o is not None and not _finite_vec(v_o):
        return EgoStatus(
            code=EgoCode.INVALID,
            position=p_o,
            velocity=v_o,
            stamp=stamp,
            age=0.0,
            reason="ego_velocity_non_finite",
            observed_at=now,
        )

    # ? 没带时间戳时不假装断流，也不假装新鲜：返回 FRESH + 明确 reason，
    # ? 让适配层日志里能看到“这一帧没做时效判断”。
    if stamp is None:
        return EgoStatus(
            code=EgoCode.FRESH,
            position=p_o,
            velocity=v_o,
            stamp=None,
            age=0.0,
            reason="ego_freshness_unchecked",
            observed_at=now,
        )

    if not isfinite(stamp):
        return EgoStatus(
            code=EgoCode.INVALID,
            position=p_o,
            velocity=v_o,
            stamp=stamp,
            age=0.0,
            reason="ego_stamp_non_finite",
            observed_at=now,
        )

    if stamp > now + _TIME_EPS:
        return EgoStatus(
            code=EgoCode.TIME_REVERSED,
            position=p_o,
            velocity=v_o,
            stamp=stamp,
            age=0.0,
            reason="ego_stamp_in_future",
            observed_at=now,
        )
    if last_stamp is not None and stamp < last_stamp - _TIME_EPS:
        return EgoStatus(
            code=EgoCode.TIME_REVERSED,
            position=p_o,
            velocity=v_o,
            stamp=stamp,
            age=0.0,
            reason="ego_stamp_reversed",
            observed_at=now,
        )

    age = max(0.0, now - stamp)
    if age > params.ego_timeout:
        return EgoStatus(
            code=EgoCode.STALE,
            position=p_o,
            velocity=v_o,
            stamp=stamp,
            age=age,
            reason="ego_stale",
            observed_at=now,
        )

    return EgoStatus(
        code=EgoCode.FRESH,
        position=p_o,
        velocity=v_o,
        stamp=stamp,
        age=age,
        reason="ego_fresh",
        observed_at=now,
    )


def validate_cmd(json_str: object) -> CmdStatus:
    """检测一条 ``/strategy/cmd`` JSON。

    Args:
        json_str: ``std_msgs/String.data`` 的原始内容。

    Returns:
        ``CmdStatus``。``INVALID`` 时 ``reason`` 说明是解析失败、缺字段
        还是动作不在枚举里；调用方应返回 L3 no_plan，而不是默认 HOVER。

    Note:
        这里不把缺失/拼错的 ``action`` 修正为 HOVER。理由和 P2 的
        ``unknown_action`` 一致：一个拼写错误不能让飞机继续按旧模式飞。
    """
    # 判断顺序固定：类型 -> 空串 -> JSON -> 对象 -> 字段 -> 动作枚举。
    if not isinstance(json_str, str):
        return CmdStatus(
            code=CmdCode.INVALID,
            drone_id=None,
            action="",
            raw_action=json_str,
            reason="cmd_not_string",
        )

    text = json_str.strip()
    if not text:
        return CmdStatus(
            code=CmdCode.INVALID,
            drone_id=None,
            action="",
            raw_action=json_str,
            reason="cmd_empty",
        )

    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return CmdStatus(
            code=CmdCode.INVALID,
            drone_id=None,
            action="",
            raw_action=json_str,
            reason="cmd_json_invalid",
        )

    if not isinstance(data, dict):
        return CmdStatus(
            code=CmdCode.INVALID,
            drone_id=None,
            action="",
            raw_action=json_str,
            reason="cmd_not_object",
        )

    drone_id = data.get("drone_id")
    if drone_id is None or not str(drone_id).strip():
        return CmdStatus(
            code=CmdCode.INVALID,
            drone_id=None,
            action="",
            raw_action=json_str,
            reason="cmd_missing_drone_id",
        )

    if "action" not in data or data.get("action") is None:
        return CmdStatus(
            code=CmdCode.INVALID,
            drone_id=str(drone_id).strip(),
            action="",
            raw_action=None,
            reason="cmd_missing_action",
        )

    raw_action = data.get("action")
    action = str(raw_action).strip().upper()
    if action not in ALL_ACTIONS:
        return CmdStatus(
            code=CmdCode.INVALID,
            drone_id=str(drone_id).strip(),
            action=action,
            raw_action=raw_action,
            reason="cmd_unknown_action",
        )

    return CmdStatus(
        code=CmdCode.FRESH,
        drone_id=str(drone_id).strip(),
        action=action,
        raw_action=raw_action,
        reason="cmd_ok",
    )


def command_anomaly(status: CmdStatus) -> Optional[str]:
    """把指令分类映射成 P3 异常分类；``None`` 表示无异常。"""
    return None if status.is_valid else ANOMALY_CMD_INVALID
