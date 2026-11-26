# -*- coding: utf-8 -*-
"""F5 多目标全配对极线三角测量与 3D 障碍物视线遮挡 (LoS) 单元测试。"""

import math
import time

import numpy as np

from vision_pkg.geo_utils import (
    Obstacle3D,
    RayObservation,
    camera_mount_matrix,
    has_line_of_sight,
    load_obstacles_from_json,
    load_obstacles_from_yaml,
    pixel_to_ray_world,
    quat_to_matrix,
    segment_intersects_obstacle,
    triangulate_multi_target,
    triangulate_rays,
)


def _project_world_to_pixel(target_enu, drone_pos, drone_quat, R_bc, K, cam_offset=(0.12, 0.0, -0.04)):
    R_wb = quat_to_matrix(*drone_quat)
    origin = np.asarray(drone_pos, dtype=float) + R_wb @ np.asarray(cam_offset, dtype=float)
    rel_world = np.asarray(target_enu, dtype=float) - origin
    d_cam = R_bc.T @ (R_wb.T @ rel_world)
    assert d_cam[2] > 0.1
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    return fx * (d_cam[0] / d_cam[2]) + cx, fy * (d_cam[1] / d_cam[2]) + cy


def _yaw_quat(yaw_rad: float):
    return (0.0, 0.0, math.sin(0.5 * yaw_rad), math.cos(0.5 * yaw_rad))


def test_six_drones_three_coplanar_targets_zero_ghosts():
    K = [400.1, 0.0, 320.0, 0.0, 400.1, 240.0, 0.0, 0.0, 1.0]
    R_bc = camera_mount_matrix(math.radians(55.0), 0.0, -math.pi / 2.0)

    # 3 个处于同一高度平面 (z=7.0m)、相同类别标签的共面靶标，极易产生交叉极线鬼影
    gt_targets = [
        np.array([12.0, -6.0, 7.0]),
        np.array([20.0, 0.0, 7.0]),
        np.array([14.0, 8.0, 7.0]),
    ]

    # 6 架围绕目标布阵的无人机
    drones = [
        ('drone_1', (4.0, -10.0, 14.0), (16.0, -1.0, 7.0)),
        ('drone_2', (28.0, -12.0, 14.5), (16.0, -1.0, 7.0)),
        ('drone_3', (30.0, 2.0, 15.0), (16.0, 1.0, 7.0)),
        ('drone_4', (24.0, 16.0, 14.2), (16.0, 3.0, 7.0)),
        ('drone_5', (4.0, 14.0, 14.8), (15.0, 2.0, 7.0)),
        ('drone_6', (2.0, 1.0, 15.2), (15.0, 0.0, 7.0)),
    ]

    rays = []
    rng = np.random.default_rng(42)
    for d_id, pos, look_at in drones:
        yaw = math.atan2(look_at[1] - pos[1], look_at[0] - pos[0])
        quat = _yaw_quat(yaw)
        for t_idx, tgt in enumerate(gt_targets):
            u, v = _project_world_to_pixel(tgt, pos, quat, R_bc, K)
            u += float(rng.normal(0.0, 0.2))
            v += float(rng.normal(0.0, 0.2))
            ray = pixel_to_ray_world(u, v, K, pos, quat, R_bc, cam_offset_body=(0.12, 0.0, -0.04))
            assert ray is not None
            rays.append(
                RayObservation(
                    drone_id=d_id,
                    origin=ray[0],
                    direction=ray[1],
                    class_id='crimson_uav',  # 全同类别，严苛考验零鬼影集合打包
                    conf=0.92,
                    timestamp=100.0,
                    det_id=t_idx,
                )
            )

    t0 = time.perf_counter()
    res = triangulate_multi_target(
        rays,
        min_parallax_deg=6.0,
        max_skew_rms_m=0.55,
        min_baseline_m=3.0,
        min_alt_m=1.5,
        max_alt_m=30.0,
        cluster_radius_m=1.5,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    assert len(res) == 3, f'Expected 3 true targets and 0 ghosts, got {len(res)}: {res}'
    assert dt_ms < 25.0
    for tgt_est in res:
        assert tgt_est.stereo is True
        assert tgt_est.rays_used >= 4
        dists = [float(np.linalg.norm(np.array([tgt_est.x, tgt_est.y, tgt_est.z]) - gt)) for gt in gt_targets]
        assert min(dists) < 0.10


def test_heterogeneous_classes_never_cross_pair_and_degenerate_rejection():
    # 两条完全相交于 (10, 0, 8) 的射线但属于不同目标类别 -> 绝对不可跨类配对
    r_crimson = RayObservation(
        drone_id='drone_1',
        origin=np.array([0.0, 0.0, 12.0]),
        direction=np.array([10.0, 0.0, -4.0]) / math.sqrt(116.0),
        class_id='crimson_uav',
        timestamp=10.0,
    )
    r_amber = RayObservation(
        drone_id='drone_2',
        origin=np.array([20.0, 0.0, 12.0]),
        direction=np.array([-10.0, 0.0, -4.0]) / math.sqrt(116.0),
        class_id='amber_uav',
        timestamp=10.0,
    )
    assert triangulate_multi_target([r_crimson, r_amber], min_parallax_deg=5.0) == []

    # 179 度反向共线对射拒绝 + 1 度同向共线拒绝 + 超时不同步拒绝
    r_head1 = RayObservation(
        drone_id='drone_1',
        origin=np.array([0.0, 0.0, 8.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        class_id='crimson_uav',
        timestamp=10.0,
    )
    r_head2 = RayObservation(
        drone_id='drone_2',
        origin=np.array([20.0, 0.0, 8.0]),
        direction=np.array([-1.0, 0.01, 0.0]),
        class_id='crimson_uav',
        timestamp=10.0,
    )
    assert triangulate_multi_target([r_head1, r_head2], min_parallax_deg=8.0) == []

    # 同机多条射线不可自配对
    r_same_drone = RayObservation(
        drone_id='drone_1',
        origin=np.array([10.0, -10.0, 12.0]),
        direction=np.array([0.0, 10.0, -4.0]),
        class_id='crimson_uav',
        timestamp=10.0,
    )
    assert triangulate_multi_target([r_crimson, r_same_drone], min_parallax_deg=5.0) == []


def test_obstacle_loaders_and_exact_3d_los_geometry(tmp_path):
    # 缺失文件安全返回 ([], None)
    assert load_obstacles_from_yaml('/tmp/nonexistent_obstacles_m2.yaml') == ([], None)
    assert load_obstacles_from_json('') == ([], None)

    yaml_content = """
arena_bounds:
  x_min: -60.0
  x_max: 60.0
  y_min: -60.0
  y_max: 60.0
  z_min: 0.0
  z_max: 40.0
obstacles:
  - id: tower_cyl
    type: cylinder
    cx: 10.0
    cy: 0.0
    z_min: 0.0
    z_max: 14.0
    radius: 2.5
    safety_margin_m: 1.5
  - id: bldg_box
    type: box
    cx: 25.0
    cy: 10.0
    z_min: 0.0
    z_max: 12.0
    sx: 6.0
    sy: 4.0
    yaw: 0.78539816
no_fly_zones:
  - id: nfz_1
    cx: 0.0
    cy: 0.0
    z_min: 0.0
    z_max: 30.0
    radius: 5.0
"""
    yml = tmp_path / 'obstacles.yaml'
    yml.write_text(yaml_content, encoding='utf-8')
    obs_list, bounds = load_obstacles_from_yaml(str(yml))
    assert bounds is not None and bounds['x_max'] == 60.0
    assert len(obs_list) == 2  # no_fly_zones 不作为物理光学遮挡

    cyl, box = obs_list[0], obs_list[1]
    # 穿过圆柱主体 -> 遮挡
    assert segment_intersects_obstacle([0.0, 0.0, 6.0], [20.0, 0.0, 6.0], cyl) is True
    # 从圆柱顶上方飞越 -> 畅通
    assert segment_intersects_obstacle([0.0, 0.0, 16.0], [20.0, 0.0, 16.0], cyl) is False
    # 从圆柱侧方绕过 -> 畅通
    assert segment_intersects_obstacle([0.0, 4.0, 6.0], [20.0, 4.0, 6.0], cyl) is False
    # 垂直向下命中圆柱顶面 -> 遮挡
    assert segment_intersects_obstacle([10.0, 0.0, 20.0], [10.0, 0.0, 5.0], cyl) is True

    # 旋转 OBB 穿透 vs 侧旁绕过
    assert segment_intersects_obstacle([18.0, 10.0, 6.0], [32.0, 10.0, 6.0], box) is True
    assert segment_intersects_obstacle([18.0, 17.0, 6.0], [32.0, 17.0, 6.0], box) is False
    # 紧贴建筑物前表面（end_clearance_m=0.05）不自遮挡
    assert has_line_of_sight([0.0, 0.0, 6.0], [7.49, 0.0, 6.0], [cyl], end_clearance_m=0.05) is True


def test_partial_building_occlusion_still_triangulates_from_unobstructed_rays():
    # 目标在 (20, 0, 6)；建筑物 box 挡住位于 x < 14 的 4 架无人机的直视视线，仅 2 架右侧无人机可视
    wall = Obstacle3D(
        obs_id='wall_box',
        obs_type='box',
        cx=14.0,
        cy=0.0,
        z_min=0.0,
        z_max=20.0,
        sx=2.0,
        sy=20.0,
    )
    target = np.array([20.0, 0.0, 6.0])
    cam_positions = [
        ('drone_1', np.array([6.0, -4.0, 12.0])),   # occluded
        ('drone_2', np.array([6.0, 4.0, 12.0])),    # occluded
        ('drone_3', np.array([8.0, -2.0, 14.0])),   # occluded
        ('drone_4', np.array([8.0, 2.0, 14.0])),    # occluded
        ('drone_5', np.array([26.0, -6.0, 13.0])),  # clear
        ('drone_6', np.array([26.0, 6.0, 13.0])),   # clear
    ]
    rays = []
    for d_id, pos in cam_positions:
        d = target - pos
        d = d / np.linalg.norm(d)
        rays.append(
            RayObservation(
                drone_id=d_id,
                origin=pos,
                direction=d,
                class_id='amber_uav',
                conf=0.90,
                timestamp=50.0,
            )
        )

    res = triangulate_multi_target(
        rays,
        obstacles=[wall],
        min_parallax_deg=8.0,
        max_skew_rms_m=0.4,
        min_baseline_m=3.0,
    )
    assert len(res) == 1
    assert res[0].rays_used == 2
    assert set(res[0].drone_ids) == {'drone_5', 'drone_6'}
    assert np.linalg.norm(np.array([res[0].x, res[0].y, res[0].z]) - target) < 0.05


def test_ten_drones_five_coplanar_targets_zero_ghosts_and_latency():
    K = [400.1, 0.0, 320.0, 0.0, 400.1, 240.0, 0.0, 0.0, 1.0]
    R_bc = camera_mount_matrix(math.radians(55.0), 0.0, -math.pi / 2.0)
    gt_targets = [
        np.array([10.0, -6.0, 7.0]),
        np.array([18.0, -4.0, 7.0]),
        np.array([15.0, 2.0, 7.0]),
        np.array([22.0, 5.0, 7.0]),
        np.array([11.0, 7.0, 7.0]),
    ]
    drones = []
    for idx in range(10):
        ang = (2.0 * math.pi * idx) / 10.0
        pos = (16.0 + 13.0 * math.cos(ang), 1.0 + 13.0 * math.sin(ang), 14.5 + 0.3 * (idx % 3))
        yaw = math.atan2(1.0 - pos[1], 16.0 - pos[0])
        drones.append((f'drone_{idx + 1}', pos, _yaw_quat(yaw)))

    rng = np.random.default_rng(123)
    dt_list = []
    for trial in range(10):
        rays = []
        for d_id, pos, quat in drones:
            for t_idx, tgt in enumerate(gt_targets):
                u, v = _project_world_to_pixel(tgt, pos, quat, R_bc, K)
                u += float(rng.normal(0.0, 1.0))
                v += float(rng.normal(0.0, 1.0))
                ray = pixel_to_ray_world(u, v, K, pos, quat, R_bc, cam_offset_body=(0.12, 0.0, -0.04))
                assert ray is not None
                rays.append(
                    RayObservation(
                        drone_id=d_id,
                        origin=ray[0],
                        direction=ray[1],
                        class_id='crimson_uav',
                        conf=0.90,
                        timestamp=200.0 + trial * 0.1,
                        det_id=t_idx,
                    )
                )
        t0 = time.perf_counter()
        res = triangulate_multi_target(
            rays,
            min_parallax_deg=8.0,
            max_skew_rms_m=0.55,
            min_baseline_m=3.4,
            min_alt_m=1.5,
            max_alt_m=35.0,
            cluster_radius_m=1.5,
        )
        dt_list.append((time.perf_counter() - t0) * 1000.0)
        assert len(res) == 5, f'Trial {trial}: expected 5 targets (0 ghosts), got {len(res)}'
        for est in res:
            dists = [float(np.linalg.norm(np.array([est.x, est.y, est.z]) - gt)) for gt in gt_targets]
            assert min(dists) < 0.25

    mean_ms = sum(dt_list) / len(dt_list)
    assert mean_ms < 10.0, f'N=10, M=5 mean latency {mean_ms:.2f}ms >= 10.0ms'


def test_three_way_ghost_trap_parsimonious_set_packing():
    # 3 个真实 2-ray 目标 T1={d1,d2}, T2={d3,d4}, T3={d5,d6}，其中 {d1,d3,d5} 恰好共交于 3-ray 鬼影 G
    G = np.array([15.0, 0.0, 10.0])
    d1 = np.array([5.0, -10.0, 15.0])
    d3 = np.array([25.0, 0.0, 15.0])
    d5 = np.array([5.0, 10.0, 15.0])
    T1 = d1 + 0.55 * (G - d1)
    T2 = d3 + 0.55 * (G - d3)
    T3 = d5 + 0.55 * (G - d5)

    d2 = T1 + np.array([-4.0, 5.0, 6.0])
    d4 = T2 + np.array([4.0, 6.0, 6.0])
    d6 = T3 + np.array([-5.0, -4.0, 6.0])

    pairs = [
        ('drone_1', d1, T1),
        ('drone_2', d2, T1),
        ('drone_3', d3, T2),
        ('drone_4', d4, T2),
        ('drone_5', d5, T3),
        ('drone_6', d6, T3),
    ]
    rays = []
    for d_id, orig, tgt in pairs:
        vec = tgt - orig
        rays.append(
            RayObservation(
                drone_id=d_id,
                origin=orig,
                direction=vec / np.linalg.norm(vec),
                class_id='crimson_uav',
                conf=0.90,
                timestamp=10.0,
            )
        )

    res = triangulate_multi_target(
        rays,
        min_parallax_deg=8.0,
        max_skew_rms_m=0.55,
        min_baseline_m=3.4,
        min_alt_m=1.5,
        max_alt_m=35.0,
    )
    assert len(res) == 3, f'Expected 3 true targets and 0 ghosts, got {len(res)}: {res}'
    gt_list = [T1, T2, T3]
    matched = set()
    for est in res:
        dists = [float(np.linalg.norm(np.array([est.x, est.y, est.z]) - gt)) for gt in gt_list]
        best_i = int(np.argmin(dists))
        assert dists[best_i] < 0.05
        matched.add(best_i)
    assert matched == {0, 1, 2}


def test_triangulate_rays_antiparallel_rejection_and_short_segment_los():
    ang2 = math.radians(2.0)
    r1 = (np.array([0.0, 0.0, 10.0]), np.array([1.0, 0.0, 0.0]))
    r2 = (np.array([20.0, 0.0, 10.0]), np.array([-math.cos(ang2), math.sin(ang2), 0.0]))
    assert triangulate_rays([r1, r2], min_parallax_deg=5.0) is None

    box = Obstacle3D(
        obs_id='box_x10_sx2',
        obs_type='box',
        cx=10.0,
        cy=0.0,
        z_min=0.0,
        z_max=10.0,
        sx=2.0,
        sy=2.0,
    )
    assert segment_intersects_obstacle([8.99, 0.0, 5.0], [9.03, 0.0, 5.0], box, end_clearance_m=0.05) is True
    assert segment_intersects_obstacle([8.90, 0.0, 5.0], [8.94, 0.0, 5.0], box, end_clearance_m=0.05) is False
    assert segment_intersects_obstacle([10.0, 0.0, 5.0], [10.0, 0.0, 5.0], box, end_clearance_m=0.05) is True
    assert has_line_of_sight([0.0, 0.0, 5.0], [8.98, 0.0, 5.0], [box], end_clearance_m=0.05) is True
