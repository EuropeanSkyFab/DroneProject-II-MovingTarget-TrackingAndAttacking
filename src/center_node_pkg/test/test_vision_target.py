"""新视觉消息筛选规则测试；不启动 ROS。"""

import pytest

from center_node_pkg.vision_target import validate_vision_sample

POSITION = (10.0, 20.0, 5.0)


def _cov(x=1.0, y=1.0, z=1.0):
    cov = [0.0] * 36
    cov[0] = x
    cov[7] = y
    cov[14] = z
    return cov


def test_ground_mode_low_covariance_is_accepted():
    assert validate_vision_sample(POSITION, _cov(), 1.0) == (True, "ok")


def test_size_mode_low_covariance_is_accepted():
    assert validate_vision_sample(POSITION, _cov(2.0, 2.0, 4.0), 0.5) == (True, "ok")


def test_ground_target_z_zero_is_accepted():
    """地面目标 z=0 可以进中枢；发送前由 send_single_target 统一抬到 3 m。"""
    ok, reason = validate_vision_sample(
        (10.0, 20.0, 0.0), _cov(), 1.0, min_z=0.0
    )
    assert ok, reason


def test_high_covariance_is_rejected():
    ok, reason = validate_vision_sample(
        POSITION, _cov(1e6, 1e6, 1e6), 1.0, max_var=100.0
    )
    assert not ok
    assert reason == "covariance_too_large"


def test_unknown_mode_is_rejected():
    ok, reason = validate_vision_sample(POSITION, _cov(), 0.7)
    assert not ok
    assert reason == "unknown_distance_mode"


def test_bad_covariance_and_z_range_are_rejected():
    assert validate_vision_sample(POSITION, None, 1.0)[0] is False
    assert validate_vision_sample(POSITION, [0.0] * 35, 1.0)[0] is False
    assert validate_vision_sample(POSITION, _cov(-1.0, 1.0, 1.0), 1.0)[1] == "covariance_invalid"
    assert validate_vision_sample(POSITION, _cov(), 1.0, min_z=6.0)[1] == "z_out_of_range"
