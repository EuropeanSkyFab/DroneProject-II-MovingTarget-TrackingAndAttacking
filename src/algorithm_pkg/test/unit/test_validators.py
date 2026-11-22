"""P3 异常检测测试：目标、己方、指令的分类与边界。

每个用例只测 ``validators.py`` 的纯函数，不启动 ROS、不读配置。

运行：
    PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
        src/algorithm_pkg/test
"""

import math

import pytest

from algorithm_pkg.algorithm_core import TargetState
from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.validators import (
    ANOMALY_CMD_INVALID,
    ANOMALY_EGO_LOST,
    ANOMALY_FREQ,
    ANOMALY_INVALID,
    ANOMALY_STALE,
    ANOMALY_TIME_REVERSED,
    CmdCode,
    EgoCode,
    FrequencyMonitor,
    TargetCode,
    command_anomaly,
    validate_cmd,
    validate_ego,
    validate_target,
)


def make_target(
    position=(10.0, 0.0, 10.0),
    velocity=(1.0, 0.0, 0.0),
    *,
    stamp=1.0,
    velocity_valid=True,
):
    """构造 P1 的 ``TargetState``，便于检测器直接读取。"""
    return TargetState(
        position=position,
        velocity=velocity,
        stamp=stamp,
        age=0.0,
        valid=True,
        velocity_valid=velocity_valid,
    )


def fmt_vec(value):
    """把三维向量格式化成便于摘要阅读的字符串。"""
    if value is None:
        return "None"
    return f"({value[0]:.3f}, {value[1]:.3f}, {value[2]:.3f})"


# ============================================================
# 第一组：目标输入
# ============================================================


def test_fresh_target_is_usable(measured):
    """场景：目标时间戳 0.1 秒前，位置和速度都正常。

    操作：调用 validate_target()，now=1.0，stamp=0.9。
    输入：TargetState(position=(10,0,10), velocity=(1,0,0))。
    预期：code=FRESH，age=0.1，position/velocity 原样保留，无异常。
    """
    status = validate_target(
        make_target(stamp=0.9),
        0.9,
        1.0,
        PlanParams(),
    )

    assert status.code is TargetCode.FRESH
    assert status.is_fresh
    assert status.is_usable
    assert status.anomaly is None
    assert status.age == pytest.approx(0.1)
    assert status.position == pytest.approx((10.0, 0.0, 10.0))
    assert status.velocity == pytest.approx((1.0, 0.0, 0.0))

    measured(
        f"code={status.code.value}, age={status.age:.3f}s, "
        f"p={fmt_vec(status.position)}"
    )


def test_missing_target_is_classified(measured):
    """场景：系统还没收到任何目标，或这一帧没有目标。

    操作：调用 validate_target(None, None, now)。
    输入：p_t=None。
    预期：code=MISSING，anomaly=STALE，age=inf，不抛异常。
    """
    status = validate_target(None, None, 1.0, PlanParams())

    assert status.code is TargetCode.MISSING
    assert not status.is_usable
    assert status.anomaly == ANOMALY_STALE
    assert math.isinf(status.age)

    measured(
        f"code={status.code.value}, anomaly={status.anomaly}, "
        f"age={status.age}"
    )


def test_nan_target_position_is_invalid(measured):
    """场景：目标位置混入 NaN。

    操作：调用 validate_target()，position=(NaN,0,10)。
    输入：含 NaN 的位置。
    预期：code=INVALID，anomaly=INVALID，位置不可用。
    """
    status = validate_target(
        (math.nan, 0.0, 10.0),
        0.9,
        1.0,
        PlanParams(),
    )

    assert status.code is TargetCode.INVALID
    assert status.anomaly == ANOMALY_INVALID
    assert not status.is_usable

    measured(f"code={status.code.value}, reason={status.reason}")


def test_nan_target_velocity_is_invalid(measured):
    """场景：目标位置正常，但估出来的速度是 NaN。

    操作：调用 validate_target()，velocity=(NaN,0,0)。
    输入：TargetState 带 NaN 速度。
    预期：code=INVALID，reason=target_velocity_non_finite。
    """
    target = make_target(velocity=(math.nan, 0.0, 0.0))
    status = validate_target(target, 0.9, 1.0, PlanParams())

    assert status.code is TargetCode.INVALID
    assert status.reason == "target_velocity_non_finite"

    measured(f"code={status.code.value}, reason={status.reason}")


def test_target_marked_invalid_while_fresh_is_invalid(measured):
    """场景：调用方在数据还新鲜时就把 TargetState.valid 置 False。

    操作：调用 validate_target()，stamp=now，valid=False。
    输入：age=0 但显式标记无效。
    预期：code=INVALID，reason=target_marked_invalid；
          P3 不能用外推把“已判无效”的数据救回来。
    """
    target = TargetState(
        position=(10.0, 0.0, 10.0),
        velocity=(1.0, 0.0, 0.0),
        stamp=1.0,
        age=0.0,
        valid=False,
        velocity_valid=True,
    )
    status = validate_target(target, 1.0, 1.0, PlanParams())

    assert status.code is TargetCode.INVALID
    assert status.reason == "target_marked_invalid"

    measured(f"code={status.code.value}, reason={status.reason}")


def test_target_age_boundary_matches_max_pred_age(measured):
    """场景：目标年龄刚好 0.5 秒，等于 P2 的 max_pred_age。

    操作：调用 validate_target() 两次，now=0.5 和 now=0.5001。
    输入：同一帧 stamp=0。
    预期：边界取等号算 FRESH；超过后算 STALE。
    """
    at_boundary = validate_target((10.0, 0.0, 10.0), 0.0, 0.5)
    over = validate_target((10.0, 0.0, 10.0), 0.0, 0.5001)

    assert at_boundary.code is TargetCode.FRESH
    assert over.code is TargetCode.STALE
    assert over.anomaly == ANOMALY_STALE

    measured(
        f"age=0.500 -> {at_boundary.code.value}, "
        f"age=0.5001 -> {over.code.value}"
    )


def test_reversed_target_stamp_is_discarded_not_stale(measured):
    """场景：目标时间戳从 2.0 倒退到 1.9。

    操作：调用 validate_target()，last_stamp=2.0，stamp=1.9。
    输入：倒退的源时间戳。
    预期：code=TIME_REVERSED，reason=stamp_reversed；
          不能判成 STALE，FSM 会丢弃这一帧但不降级。
    """
    status = validate_target(
        (10.0, 0.0, 10.0),
        1.9,
        2.0,
        PlanParams(),
        last_stamp=2.0,
    )

    assert status.code is TargetCode.TIME_REVERSED
    assert status.anomaly == ANOMALY_TIME_REVERSED
    assert status.reason == "stamp_reversed"

    measured(f"code={status.code.value}, reason={status.reason}")


def test_future_target_stamp_is_time_reversed(measured):
    """场景：目标时间戳比 now 还超前（时钟不同步或坏帧）。

    操作：调用 validate_target()，stamp=1.2，now=1.0。
    输入：未来时间戳。
    预期：code=TIME_REVERSED，reason=stamp_in_future。
    """
    status = validate_target((10.0, 0.0, 10.0), 1.2, 1.0, PlanParams())

    assert status.code is TargetCode.TIME_REVERSED
    assert status.reason == "stamp_in_future"

    measured(f"code={status.code.value}, reason={status.reason}")


@pytest.mark.parametrize(
    "position",
    [
        (1001.0, 0.0, 10.0),
        (0.0, -1001.0, 10.0),
        (0.0, 0.0, 51.0),
        (0.0, 0.0, -1.0),
    ],
)
def test_target_out_of_bounds_is_classified(position, measured):
    """场景：目标位置超出输入侧边界（XY ±1000、高度 [0,50]）。

    操作：调用 validate_target() 四次。
    输入：x/y 越界、z 过高、z 低于地面。
    预期：四次都返回 OUT_OF_BOUNDS，anomaly=INVALID。
    """
    status = validate_target(position, 0.9, 1.0, PlanParams())

    assert status.code is TargetCode.OUT_OF_BOUNDS
    assert status.anomaly == ANOMALY_INVALID

    measured(f"position={position} -> {status.code.value}")


def test_freq_anomaly_keeps_position_usable(measured):
    """场景：丢帧率检测命中，但目标位置本身仍新鲜。

    操作：调用 validate_target(..., freq_anomaly=True)。
    输入：新鲜目标 + 频率异常标记。
    预期：code=FREQ_ANOMALY，is_usable=True，anomaly=FREQ_ANOMALY。
    """
    status = validate_target(
        (10.0, 0.0, 10.0),
        0.9,
        1.0,
        PlanParams(),
        freq_anomaly=True,
    )

    assert status.code is TargetCode.FREQ_ANOMALY
    assert status.is_usable
    assert status.anomaly == ANOMALY_FREQ

    measured(
        f"code={status.code.value}, usable={status.is_usable}, "
        f"anomaly={status.anomaly}"
    )


def test_frequency_monitor_detects_10hz_to_1hz_and_recovers(measured):
    """场景：目标前 10 帧 10Hz，之后掉到 1Hz。

    操作：用 FrequencyMonitor(window=10, threshold=0.5) 逐帧 observe。
    输入：stamp=0,0.1,...,0.9，然后 1.9,2.9,...,9.9。
    预期：掉频第一帧命中；窗口滑过一半后不再命中；全程不抛异常。
    """
    monitor = FrequencyMonitor(
        window_size=10,
        drop_rate_threshold=0.5,
        enabled=True,
    )

    for index in range(10):
        assert monitor.observe(index * 0.1) is False

    results = [monitor.observe(1.9 + index * 1.0) for index in range(9)]

    assert results[0] is True
    assert results[-1] is False

    measured(
        f"1Hz 帧命中序列={results}（首帧异常、窗口滑过后恢复）"
    )


def test_frequency_monitor_disabled_never_flags(measured):
    """场景：频率检测默认关闭，不能因为目标掉频就自行降级。

    操作：构造 enabled=False 的检测器，喂 10Hz→1Hz 序列。
    输入：与上一个用例相同的掉频序列。
    预期：observe 始终返回 False。
    """
    monitor = FrequencyMonitor(enabled=False)

    for index in range(10):
        monitor.observe(index * 0.1)
    results = [monitor.observe(1.9 + index * 1.0) for index in range(5)]

    assert results == [False] * 5

    measured(f"disabled -> results={results}")


# ============================================================
# 第二组：己方状态
# ============================================================


def test_fresh_ego_is_usable(measured):
    """场景：己方状态时间戳 0.05 秒前，位置和速度正常。

    操作：调用 validate_ego()，now=1.0，stamp=0.95。
    输入：position=(0,0,10)，velocity=(0,0,0)。
    预期：code=FRESH，age=0.05，无异常。
    """
    status = validate_ego(
        (0.0, 0.0, 10.0),
        (0.0, 0.0, 0.0),
        0.95,
        1.0,
        PlanParams(),
    )

    assert status.code is EgoCode.FRESH
    assert status.is_fresh
    assert status.anomaly is None
    assert status.age == pytest.approx(0.05)

    measured(f"code={status.code.value}, age={status.age:.3f}s")


def test_missing_ego_is_lost(measured):
    """场景：己方状态断流，own=None。

    操作：调用 validate_ego(None, None, None, now)。
    输入：没有位置。
    预期：code=MISSING，anomaly=EGO_LOST，age=inf。
    """
    status = validate_ego(None, None, None, 1.0, PlanParams())

    assert status.code is EgoCode.MISSING
    assert status.anomaly == ANOMALY_EGO_LOST
    assert math.isinf(status.age)

    measured(f"code={status.code.value}, anomaly={status.anomaly}")


def test_ego_velocity_nan_is_invalid(measured):
    """场景：己方位置正常，但 twist.linear 里混入 NaN。

    操作：调用 validate_ego()，v_o=(NaN,0,0)。
    输入：含 NaN 的己方速度。
    预期：code=INVALID，reason=ego_velocity_non_finite。
    """
    status = validate_ego(
        (0.0, 0.0, 10.0),
        (math.nan, 0.0, 0.0),
        0.95,
        1.0,
        PlanParams(),
    )

    assert status.code is EgoCode.INVALID
    assert status.reason == "ego_velocity_non_finite"

    measured(f"code={status.code.value}, reason={status.reason}")


def test_ego_position_must_be_three_dimensional(measured):
    """场景：己方状态里只有二维坐标（长度 2）。

    操作：调用 validate_ego()，p_o=(0,10)。
    输入：长度不对的位置。
    预期：code=INVALID，reason=ego_position_non_finite；
          不能让二维坐标进入拦截几何后触发 IndexError。
    """
    status = validate_ego(
        (0.0, 10.0),  # type: ignore[arg-type]
        None,
        0.95,
        1.0,
        PlanParams(),
    )

    assert status.code is EgoCode.INVALID
    assert status.reason == "ego_position_non_finite"

    measured(f"code={status.code.value}, reason={status.reason}")


def test_stale_ego_exceeds_timeout(measured):
    """场景：己方状态 1.5 秒没更新，超过 ego_timeout=1.0。

    操作：调用 validate_ego()，stamp=0，now=1.5。
    输入：过期的己方状态。
    预期：code=STALE，anomaly=EGO_LOST。
    """
    status = validate_ego(
        (0.0, 0.0, 10.0),
        (0.0, 0.0, 0.0),
        0.0,
        1.5,
        PlanParams(),
    )

    assert status.code is EgoCode.STALE
    assert status.anomaly == ANOMALY_EGO_LOST
    assert status.age == pytest.approx(1.5)

    measured(f"code={status.code.value}, age={status.age:.3f}s")


def test_reversed_ego_stamp_is_time_reversed(measured):
    """场景：己方状态时间戳倒退（可能是中枢回灌旧帧）。

    操作：调用 validate_ego()，last_stamp=2.0，stamp=1.9。
    输入：倒退的己方时间戳。
    预期：code=TIME_REVERSED，该帧由 FSM 丢弃。
    """
    status = validate_ego(
        (0.0, 0.0, 10.0),
        None,
        1.9,
        2.0,
        PlanParams(),
        last_stamp=2.0,
    )

    assert status.code is EgoCode.TIME_REVERSED
    assert status.reason == "ego_stamp_reversed"

    measured(f"code={status.code.value}, reason={status.reason}")


def test_ego_without_stamp_is_explicitly_unchecked(measured):
    """场景：桥接状态没带 header.stamp，调用方只能退回接收时刻。

    操作：调用 validate_ego()，stamp=None。
    输入：位置有效，但没有时间戳。
    预期：code=FRESH，reason=ego_freshness_unchecked；
          不能假装做了时效判断。
    """
    status = validate_ego(
        (0.0, 0.0, 10.0),
        None,
        None,
        1.0,
        PlanParams(),
    )

    assert status.code is EgoCode.FRESH
    assert status.reason == "ego_freshness_unchecked"

    measured(f"code={status.code.value}, reason={status.reason}")


# ============================================================
# 第三组：战术指令 JSON
# ============================================================


def test_cmd_normalizes_action(measured):
    """场景：课题三下发小写、带空格的合法动作。

    操作：调用 validate_cmd()。
    输入：{"drone_id":"drone_1","action":" track "}。
    预期：code=FRESH，action=TRACK，raw_action 保留原值。
    """
    status = validate_cmd('{"drone_id":"drone_1","action":" track "}')

    assert status.code is CmdCode.FRESH
    assert status.is_valid
    assert status.drone_id == "drone_1"
    assert status.action == "TRACK"
    assert status.raw_action == " track "
    assert command_anomaly(status) is None

    measured(
        f"code={status.code.value}, action={status.action}, "
        f"raw={status.raw_action!r}"
    )


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (None, "cmd_not_string"),
        (123, "cmd_not_string"),
        ("", "cmd_empty"),
        ("   ", "cmd_empty"),
        ("{not json", "cmd_json_invalid"),
        ("[1,2,3]", "cmd_not_object"),
        ('{"action":"TRACK"}', "cmd_missing_drone_id"),
        ('{"drone_id":"drone_1"}', "cmd_missing_action"),
        (
            '{"drone_id":"drone_1","action":"TRACKK"}',
            "cmd_unknown_action",
        ),
        (
            '{"drone_id":"drone_1","action":null}',
            "cmd_missing_action",
        ),
    ],
)
def test_invalid_cmd_is_rejected_with_reason(payload, reason, measured):
    """场景：指令 JSON 解析失败、缺字段或动作拼错。

    操作：调用 validate_cmd()。
    输入：None、非字符串、空串、坏 JSON、数组、缺 drone_id、
          缺 action、未知动作。
    预期：code=INVALID，reason 精确说明是哪一类；不能默认 HOVER。
    """
    status = validate_cmd(payload)

    assert status.code is CmdCode.INVALID
    assert not status.is_valid
    assert status.reason == reason
    assert command_anomaly(status) == ANOMALY_CMD_INVALID

    measured(f"payload={payload!r} -> {status.reason}")


def test_cmd_requires_known_action_enum(measured):
    """场景：动作不在 7 个已知枚举里，但 JSON 本身合法。

    操作：调用 validate_cmd()。
    输入：{"drone_id":"drone_2","action":"ATTACKK"}。
    预期：code=INVALID，reason=cmd_unknown_action，保留原始动作。
    """
    status = validate_cmd(
        '{"drone_id":"drone_2","action":"ATTACKK"}'
    )

    assert status.code is CmdCode.INVALID
    assert status.reason == "cmd_unknown_action"
    assert status.action == "ATTACKK"
    assert status.drone_id == "drone_2"

    measured(
        f"code={status.code.value}, action={status.action}, "
        f"reason={status.reason}"
    )
