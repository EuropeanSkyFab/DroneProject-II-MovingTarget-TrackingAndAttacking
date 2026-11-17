# -*- coding: utf-8 -*-
"""vision_pkg 几何反投影、多机三维射线三角测量与抗强光色块检测单元测试。"""

import math
import numpy as np

from vision_pkg.detector import ColorBlobDetector
from vision_pkg.geo_utils import (
    camera_mount_matrix,
    pixel_to_ray_world,
    pixel_to_world,
    quat_to_matrix,
    triangulate_rays,
)


def _project_world_to_pixel(target_enu, drone_pos, drone_quat, R_bc, K, cam_offset=(0.12, 0.0, -0.04)):
    R_wb = quat_to_matrix(*drone_quat)
    origin = np.asarray(drone_pos, dtype=float) + R_wb @ np.asarray(cam_offset, dtype=float)
    rel_world = np.asarray(target_enu, dtype=float) - origin
    d_cam = R_bc.T @ (R_wb.T @ rel_world)
    assert d_cam[2] > 0.1
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    u = fx * (d_cam[0] / d_cam[2]) + cx
    v = fy * (d_cam[1] / d_cam[2]) + cy
    return u, v


def test_single_view_and_3d_stereo_triangulation_accuracy():
    K = [400.1, 0.0, 320.0, 0.0, 400.1, 240.0, 0.0, 0.0, 1.0]
    R_bc = camera_mount_matrix(math.radians(55.0), 0.0, -math.pi / 2.0)

    # 真实空中三维机动目标位于 (18.4, 4.2, 7.35) —— 注意高度不是默认的 5.5m！
    true_target = np.array([18.4, 4.2, 7.35], dtype=float)

    # 3 架无人机从不同方位与高度俯视目标
    drones = [
        ((12.0, 4.2, 11.2), (0.0, 0.0, 0.0, 1.0)),  # 朝正东 (yaw=0)
        ((18.4, -2.0, 11.5), (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))),  # 朝正北 (yaw=90°)
        ((24.5, 4.2, 10.8), (0.0, 0.0, 1.0, 0.0)),  # 朝正西 (yaw=180°)
    ]

    rays = []
    for pos, quat in drones:
        u, v = _project_world_to_pixel(true_target, pos, quat, R_bc, K)
        # 叠加亚像素量化误差
        ray = pixel_to_ray_world(u + 0.35, v - 0.25, K, pos, quat, R_bc, cam_offset_body=(0.12, 0.0, -0.04))
        assert ray is not None
        rays.append(ray)

    # 1. 多机三维立体三角测量：无需知道 z=7.35，误差必须 < 0.05m！
    tri = triangulate_rays(rays, min_parallax_deg=3.0, max_skew_rms_m=1.0)
    assert tri is not None
    tx, ty, tz, rms = tri
    err_stereo = float(np.linalg.norm(np.array([tx, ty, tz]) - true_target))
    assert err_stereo < 0.05
    assert rms < 0.03

    # 2. 对比如果单目错误假设 ground_z=5.5m 时的高度投影偏差 (>2.0m)，证明立体三角化的巨大收益
    u0, v0 = _project_world_to_pixel(true_target, drones[0][0], drones[0][1], R_bc, K)
    mono_wrong_z = pixel_to_world(
        u0, v0, K, drones[0][0], drones[0][1], R_bc, ground_z=5.5, cam_offset_body=(0.12, 0.0, -0.04)
    )
    assert mono_wrong_z is not None
    err_mono_wrong = float(np.linalg.norm(np.array(mono_wrong_z) - true_target))
    assert err_mono_wrong > 2.0
    assert err_stereo < err_mono_wrong * 0.02

    # 3. 共线退化保护测试 (<2.5° 视差必须安全返回 None)
    deg_res = triangulate_rays([rays[0], (rays[0][0] + np.array([0.05, 0.0, 0.0]), rays[0][1])])
    assert deg_res is None


def test_color_blob_detector_glare_suppression_and_long_range():
    det = ColorBlobDetector(min_area=24, enable_glare_suppression=True)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # 注入大面积暖白阳光高光干扰块 (BGR = [205, 215, 250], R-G = 35 以内) -> 必须被抑制！
    img[50:150, 50:180] = (220, 225, 250)
    assert det.detect(img) is None

    # 注入远处 7x7 (=49 px^2) 真实绯红靶机色块 (BGR = [15, 20, 240]) -> 必须精准检出！
    img[260:268, 410:418] = (15, 20, 240)
    hit = det.detect_with_meta(img)
    assert hit is not None
    u, v, area, span = hit
    assert abs(u - 413.5) < 1.5
    assert abs(v - 263.5) < 1.5
    assert area >= 24.0
    assert span >= 6.0
