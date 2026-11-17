# -*- coding: utf-8 -*-
"""planning_pkg.guidance 单元测试：Apollonius 最优可达时间、超速兜底、120° 锁相合围与 3D APN。"""

import math
import numpy as np

from planning_pkg.estimator import ManeuveringTargetEstimator
from planning_pkg.guidance import (
    CooperativeGuidancePlanner,
    DynamicsProfile,
    assign_encirclement_slots,
    solve_intercept_time,
)


def test_apollonius_intercept_solver_and_unreachable_fallback():
    # 目标从 (20, 0, 6) 以 (0, 5, 0) 向北匀速飞行；我机在 (0, 0, 6) 极速 15 m/s
    def pred_fn(tau: float):
        return np.array([20.0, 5.0 * tau, 6.0]), np.array([0.0, 5.0, 0.0])

    t_go, reachable = solve_intercept_time(
        ego_pos=np.array([0.0, 0.0, 6.0]),
        ego_vel=np.array([0.0, 0.0, 0.0]),
        ego_speed_cap=15.0,
        a_max=12.0,
        predict_target_fn=pred_fn,
        t_max=8.0,
    )
    assert reachable
    p_hit, _ = pred_fn(t_go)
    # 验证交汇点在正北偏东前方 (y > 6m) 且时间合理 (~1.5-2.0s)
    assert 1.2 < t_go < 2.4
    assert p_hit[1] > 6.0

    # 超速目标不可达测试 (目标 30 m/s 远离，我机限速 6 m/s)
    def fast_away(tau: float):
        return np.array([25.0 + 30.0 * tau, 0.0, 6.0]), np.array([30.0, 0.0, 0.0])

    _, reach_fast = solve_intercept_time(
        ego_pos=np.array([0.0, 0.0, 6.0]),
        ego_vel=np.array([0.0, 0.0, 0.0]),
        ego_speed_cap=6.0,
        a_max=8.0,
        predict_target_fn=fast_away,
        t_max=6.0,
    )
    assert not reach_fast


def test_encirclement_120deg_and_multi_mode_planner():
    drones = ["drone_1", "drone_2", "drone_3"]
    pos_map = {
        "drone_1": np.array([16.0, 0.0, 9.0]),
        "drone_2": np.array([7.0, 5.0, 9.0]),
        "drone_3": np.array([7.0, -5.0, 9.0]),
    }
    slots = assign_encirclement_slots(drones, pos_map, (10.0, 0.0), radius=6.0, base_phase_rad=0.0)
    assert len(slots) == 3
    # 验证三机分配到互不相同的 3 个 120° 槽位，两两水平间距 = 6 * sqrt(3) ≈ 10.39m
    pts = [np.array([slots[d][0], slots[d][1]]) for d in drones]
    for i in range(3):
        for j in range(i + 1, 3):
            dist = float(np.linalg.norm(pts[i] - pts[j]))
            assert abs(dist - 6.0 * math.sqrt(3.0)) < 1e-3

    # 验证完整 Planner 在 TRACK / INTERCEPT / ATTACK / RESUME_PATROL 下输出合法坐标
    est = ManeuveringTargetEstimator()
    for k in range(15):
        est.update((15.0 + 0.2 * k, 3.0, 5.5), k * 0.05, source="vision")

    planner = CooperativeGuidancePlanner(DynamicsProfile())
    states = {d: (pos_map[d], np.zeros(3)) for d in drones}
    modes = {"drone_1": "ATTACK", "drone_2": "INTERCEPT", "drone_3": "RESUME_PATROL"}
    t_now = 0.70
    t_pos, t_vel, t_acc = est.get_kinematics_at(t_now)
    cmds = planner.compute_swarm_targets(
        now_sec=t_now,
        swarm_ids=drones,
        drone_modes=modes,
        drone_states=states,
        target_valid=True,
        target_pos=t_pos,
        target_vel=t_vel,
        target_acc=t_acc,
        predict_fn=lambda tau: est.predict_pv_ahead(tau),
    )
    assert set(cmds.keys()) == {"drone_1", "drone_2", "drone_3"}
    assert cmds["drone_1"]["action"] == "ATTACK"
    assert cmds["drone_2"]["action"] == "INTERCEPT"
    assert cmds["drone_3"]["action"] == "TRACK"
    for d in drones:
        assert cmds[d]["z"] >= 3.05
