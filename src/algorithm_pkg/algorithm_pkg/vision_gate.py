"""算法侧视觉输入门：协方差、测距模式和话题回退。"""

from __future__ import annotations

from math import isfinite
from typing import Optional, Sequence

Vec3 = tuple[float, float, float]

# 视觉包约定的两种测距模式：1.0=地面投影，0.5=尺寸估距。
VISION_MODES = (1.0, 0.5)


def _finite_vec(point: Optional[Vec3]) -> bool:
    if point is None:
        return False
    try:
        return len(point) == 3 and all(isfinite(value) for value in point)
    except TypeError:
        return False


def validate_vision_sample(
    position: Optional[Vec3],
    covariance: Optional[Sequence[float]],
    mode: float,
    max_var: float = 100.0,
    min_z: float = 0.0,
    max_z: float = 50.0,
) -> tuple[bool, str]:
    """检查一帧带协方差的视觉消息能不能进目标缓存。

    Args:
        position: 目标 ENU 位置，米。
        covariance: 36 元素行优先协方差；只检查 x/y/z 三个对角元。
        mode: ``orientation.w``，1.0=地面投影，0.5=尺寸估距。
        max_var: 允许的最大位置方差，米²。
        min_z: 允许的最低目标高度，米；地面目标可为 0。
        max_z: 允许的最高目标高度，米。

    Returns:
        ``(ok, reason)``；``ok=False`` 时不要写进 ``TargetBuffer``。
    """
    if not _finite_vec(position):
        return False, "position_non_finite"

    if not any(abs(float(mode) - item) <= 1e-6 for item in VISION_MODES):
        return False, "unknown_distance_mode"
    if not isfinite(max_var) or max_var <= 0.0:
        return False, "bad_max_var"
    if not isfinite(min_z) or not isfinite(max_z) or min_z > max_z:
        return False, "bad_z_range"

    z = position[2]
    if z < min_z or z > max_z:
        return False, "z_out_of_range"

    if covariance is None:
        return False, "covariance_missing"
    try:
        if len(covariance) != 36:
            return False, "covariance_size"
        diag = (float(covariance[0]), float(covariance[7]), float(covariance[14]))
    except (TypeError, ValueError, IndexError):
        return False, "covariance_invalid"

    if not all(isfinite(value) and value >= 0.0 for value in diag):
        return False, "covariance_invalid"
    if max(diag) > max_var:
        return False, "covariance_too_large"

    return True, "ok"


def legacy_vision_allowed(
    now: float,
    last_cov_seen: Optional[float],
    timeout: float,
) -> bool:
    """无协方差 ``PoseStamped`` 视觉帧现在能不能用。

    带协方差话题只要还在发（包括被协方差门拒绝的帧），无协方差话题就
    不覆盖；带协方差话题停了超过 ``timeout`` 才回退无协方差话题。
    """
    if last_cov_seen is None:
        return True
    elapsed = now - last_cov_seen
    # 时钟回跳时不要把无协方差话题永久锁死。
    return elapsed > timeout or elapsed < 0.0
