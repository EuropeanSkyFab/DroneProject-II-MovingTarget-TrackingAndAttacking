"""算法侧视觉门测试：不启动 ROS。"""

from algorithm_pkg.vision_gate import legacy_vision_allowed, validate_vision_sample

POSITION = (10.0, 20.0, 5.0)


def _cov(x=1.0, y=1.0, z=1.0):
    cov = [0.0] * 36
    cov[0] = x
    cov[7] = y
    cov[14] = z
    return cov


def test_ground_and_size_modes_are_accepted():
    assert validate_vision_sample(POSITION, _cov(), 1.0) == (True, "ok")
    assert validate_vision_sample(POSITION, _cov(2.0, 2.0, 4.0), 0.5) == (True, "ok")


def test_ground_target_z_zero_is_accepted():
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


def test_bad_mode_and_bad_covariance_are_rejected():
    assert validate_vision_sample(POSITION, _cov(), 0.7)[1] == "unknown_distance_mode"
    assert validate_vision_sample(POSITION, None, 1.0)[1] == "covariance_missing"
    assert validate_vision_sample(POSITION, [0.0] * 35, 1.0)[1] == "covariance_size"
    assert validate_vision_sample(POSITION, _cov(-1.0, 1.0, 1.0), 1.0)[1] == "covariance_invalid"


def test_position_and_z_range_are_rejected():
    assert validate_vision_sample(None, _cov(), 1.0)[0] is False
    assert validate_vision_sample((float("nan"), 0.0, 5.0), _cov(), 1.0)[0] is False
    assert validate_vision_sample(POSITION, _cov(), 1.0, min_z=6.0)[1] == "z_out_of_range"


def test_legacy_fallback_waits_for_new_topic_timeout():
    assert legacy_vision_allowed(10.0, None, 0.5)
    assert not legacy_vision_allowed(10.2, 10.0, 0.5)
    assert legacy_vision_allowed(10.6, 10.0, 0.5)
    # 时钟回跳：不要让无协方差话题永久锁死。
    assert legacy_vision_allowed(9.0, 10.0, 0.5)
