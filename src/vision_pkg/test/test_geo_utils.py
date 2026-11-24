"""视觉几何纯测试：不启动 ROS、不需要相机。"""

import math

import numpy as np
import pytest

from vision_pkg.geo_utils import (
    camera_mount_matrix,
    pixel_size_to_world,
    pixel_to_world,
    pixel_to_world_cov,
)

K = [100.0, 0.0, 320.0, 0.0, 100.0, 240.0, 0.0, 0.0, 1.0]
POS = (0.0, 0.0, 10.0)
QUAT = (0.0, 0.0, 0.0, 1.0)
R_BC = camera_mount_matrix(0.0, 0.0, 0.0)


def test_ground_projection_returns_ground_point_and_covariance():
    point, cov, valid = pixel_to_world_cov(
        320.0, 240.0, K, POS, QUAT, R_BC, ground_z=0.0
    )
    assert valid
    assert point == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert cov[0, 0] == pytest.approx(cov[1, 1])
    assert cov[0, 0] > 0.0
    assert cov[2, 2] == pytest.approx(0.01)


def test_ray_up_is_invalid_and_high_covariance():
    roll_pi = (1.0, 0.0, 0.0, 0.0)  # 绕 x 转 180°，光轴朝上
    point, cov, valid = pixel_to_world_cov(
        320.0, 240.0, K, POS, roll_pi, R_BC, ground_z=0.0
    )
    assert not valid
    assert point == pytest.approx(POS)
    assert cov[0, 0] == pytest.approx(1e6)
    assert cov[1, 1] == pytest.approx(1e6)


def test_ground_plane_behind_camera_is_invalid():
    point, _cov, valid = pixel_to_world_cov(
        320.0, 240.0, K, POS, QUAT, R_BC, ground_z=20.0
    )
    assert not valid
    assert point == pytest.approx(POS)


def test_invalid_intrinsics_is_invalid():
    bad_k = [0.0, 0.0, 320.0, 0.0, 0.0, 240.0, 0.0, 0.0, 1.0]
    _point, _cov, valid = pixel_to_world_cov(
        320.0, 240.0, bad_k, POS, QUAT, R_BC, ground_z=0.0
    )
    assert not valid


def test_size_estimation_uses_known_width():
    point = pixel_size_to_world(
        320.0, 240.0, 20.0, K, POS, QUAT, R_BC, target_real_width=2.0
    )
    assert point == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_size_estimation_rejects_bad_width_or_pixels():
    assert pixel_size_to_world(
        320.0, 240.0, 0.0, K, POS, QUAT, R_BC, 2.0
    ) is None
    assert pixel_size_to_world(
        320.0, 240.0, 20.0, K, POS, QUAT, R_BC, 0.0
    ) is None


def test_legacy_pixel_to_world_still_works():
    point = pixel_to_world(320.0, 240.0, K, POS, QUAT, R_BC, ground_z=0.0)
    assert point == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    assert pixel_to_world(320.0, 240.0, [0, 0, 0, 0, 0, 0, 0, 0, 1],
                          POS, QUAT, R_BC) is None
