# -*- coding: utf-8 -*-
"""target_sim_node 轨迹边界、无障碍走廊约束与雷达可复现噪声单元测试。"""

import rclpy
from flight_ctrl_pkg.target_sim_node import TargetSimNode


def test_target_sim_profiles_stay_within_obstacle_free_corridor():
    rclpy.init()
    node = TargetSimNode()
    try:
        for profile in ('evasive_weave', 'figure8', 'circle', 'hover'):
            node.profile = profile
            for k in range(60):
                t = k * 0.5
                p_ref, v_ref = node.reference_trajectory(t)
                # 严格验证位于中央无建筑净空走廊 x in [9, 33], y in [-5, 10], z in [4.5, 7.0]
                assert 9.0 <= p_ref[0] <= 33.0
                assert -5.0 <= p_ref[1] <= 10.0
                assert 4.5 <= p_ref[2] <= 7.0
                assert (v_ref[0] ** 2 + v_ref[1] ** 2 + v_ref[2] ** 2) ** 0.5 < 12.0
    finally:
        node.destroy_node()
        rclpy.shutdown()
