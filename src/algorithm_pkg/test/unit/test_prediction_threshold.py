"""预测阈值：0.45 s 边界下，0.45 内正常、0.45 外不发点。"""

from algorithm_pkg.algorithm_core import TargetState
from algorithm_pkg.plan_params import PlanParams
from algorithm_pkg.plan_policy import plan
from algorithm_pkg.validators import TargetCode, validate_target

PARAMS = PlanParams(max_pred_age=0.45)
NOW = 100.0


def _target(stamp: float) -> TargetState:
    return TargetState(
        position=(30.0, 0.0, 10.0),
        velocity=(0.0, 0.0, 0.0),
        stamp=stamp,
        age=NOW - stamp,
        valid=True,
        velocity_valid=True,
    )


def test_age_below_045_is_fresh(measured):
    """场景：目标年龄 0.44 s，低于 0.45 s 门限。

    预期：validate_target 判 FRESH；TRACK 仍发点。
    """
    status = validate_target(_target(NOW - 0.44), NOW - 0.44, NOW, PARAMS)
    assert status.code is TargetCode.FRESH

    result = plan("TRACK", (0.0, 0.0, 10.0), _target(NOW - 0.44), PARAMS)
    assert result.point is not None

    measured("0.440 s -> FRESH，TRACK 发点")


def test_age_just_over_045_is_stale_and_not_published(measured):
    """场景：目标年龄 0.46 s。

    预期：validate_target 判 STALE；TRACK 以 target_too_old 不发点。
    """
    status = validate_target(_target(NOW - 0.46), NOW - 0.46, NOW, PARAMS)
    assert status.code is TargetCode.STALE

    result = plan("TRACK", (0.0, 0.0, 10.0), _target(NOW - 0.46), PARAMS)
    assert result.point is None
    assert result.reason == "target_too_old"

    measured("0.460 s -> STALE，TRACK target_too_old 不发点")
