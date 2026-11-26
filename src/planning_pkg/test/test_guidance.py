# -*- coding: utf-8 -*-
"""planning_pkg.guidance 单元测试：Apollonius 拦截、广义合围/钳形封堵、WTA、3D APN 与三维避障。"""

from __future__ import annotations

import math
import time
import warnings

import numpy as np

from planning_pkg.estimator import ManeuveringTargetEstimator
from planning_pkg.guidance import (
    CooperativeGuidancePlanner,
    DynamicsProfile,
    TargetTrackState,
    _obstacle_surface_info,
    assign_encirclement_slots,
    compute_wta_assignments,
    solve_intercept_time,
)
from planning_pkg.obstacles import ArenaBounds, ObstacleMap, ObstaclePrimitive


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


def _segments_intersect_2d(a1, a2, b1, b2) -> bool:
    def ccw(p1, p2, p3):
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) > (p2[1] - p1[1]) * (p3[0] - p1[0])

    return ccw(a1, b1, b2) != ccw(a2, b1, b2) and ccw(a1, a2, b1) != ccw(a1, a2, b2)


def test_generalized_encirclement_and_pincer_k6_k8_non_crossing():
    for k_cnt in (6, 8):
        drones = [f"drone_{i + 1}" for i in range(k_cnt)]
        pos_map = {
            d: np.array([15.0 + 8.0 * math.cos(2.0 * math.pi * i / k_cnt + 0.2), 8.0 * math.sin(2.0 * math.pi * i / k_cnt + 0.2), 9.0])
            for i, d in enumerate(drones)
        }
        t0 = time.perf_counter()
        slots = assign_encirclement_slots(drones, pos_map, (15.0, 0.0), radius=6.0, base_phase_rad=0.0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        assert elapsed_ms < 2.0
        assert len({v[2] for v in slots.values()}) == k_cnt

    # 验证 K=6 INTERCEPT 1-Wasserstein 钳形槽位匹配在交叉初始布局下无平面航迹交叉
    planner = CooperativeGuidancePlanner(DynamicsProfile())
    drones6 = [f"drone_{i + 1}" for i in range(6)]
    states6 = {
        "drone_1": (np.array([0.0, -10.0, 8.0]), np.zeros(3)),
        "drone_2": (np.array([2.0, 10.0, 8.0]), np.zeros(3)),
        "drone_3": (np.array([4.0, -6.0, 8.0]), np.zeros(3)),
        "drone_4": (np.array([6.0, 6.0, 8.0]), np.zeros(3)),
        "drone_5": (np.array([8.0, -2.0, 8.0]), np.zeros(3)),
        "drone_6": (np.array([10.0, 2.0, 8.0]), np.zeros(3)),
    }
    modes6 = {d: "INTERCEPT" for d in drones6}
    t_pos = np.array([25.0, 0.0, 6.0])
    t_vel = np.array([4.0, 0.0, 0.0])
    cmds = planner.compute_swarm_targets(
        now_sec=1.0,
        swarm_ids=drones6,
        drone_modes=modes6,
        drone_states=states6,
        target_valid=True,
        target_pos=t_pos,
        target_vel=t_vel,
        target_acc=np.zeros(3),
        predict_fn=lambda tau: (t_pos + t_vel * tau, t_vel),
    )
    segs = [
        (states6[d][0][0:2], np.array([cmds[d]["x"], cmds[d]["y"]]))
        for d in drones6
    ]
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            assert not _segments_intersect_2d(segs[i][0], segs[i][1], segs[j][0], segs[j][1])


def _make_target_state(tid: str, pos, vel, priority: float = 1.0) -> TargetTrackState:
    p = np.asarray(pos, dtype=float)
    v = np.asarray(vel, dtype=float)
    return TargetTrackState(
        target_id=tid,
        pos=p,
        vel=v,
        acc=np.zeros(3, dtype=float),
        predict_fn=lambda tau, p0=p, v0=v: (p0 + v0 * tau, v0),
        imm_probs={"CV": 0.7, "CTRV": 0.2, "JINK": 0.1},
        priority=priority,
    )


def test_dynamic_wta_pack_splitting_soc_hysteresis_and_kill_retasking():
    prof = DynamicsProfile()

    # 1. N=6 vs M=3 -> [2, 2, 2] 均衡分群
    drones6 = [f"drone_{i + 1}" for i in range(6)]
    states6 = {d: (np.array([0.0, i * 4.0 - 10.0, 8.0]), np.zeros(3)) for i, d in enumerate(drones6)}
    targets3 = [
        _make_target_state("tA", [25.0, -8.0, 6.0], [-2.0, 0.0, 0.0]),
        _make_target_state("tB", [25.0, 0.0, 6.0], [-2.0, 0.0, 0.0]),
        _make_target_state("tC", [25.0, 8.0, 6.0], [-2.0, 0.0, 0.0]),
    ]
    wta6 = compute_wta_assignments(drones6, states6, targets3, prof)
    counts6 = {tid: list(wta6.values()).count(tid) for tid in ("tA", "tB", "tC")}
    assert counts6 == {"tA": 2, "tB": 2, "tC": 2}

    # 2. N=7 vs M=3 -> [3, 2, 2]（最高威胁目标得 3 架）
    drones7 = drones6 + ["drone_7"]
    states7 = dict(states6)
    states7["drone_7"] = (np.array([0.0, 0.5, 8.0]), np.zeros(3))
    targets3_threat = [
        _make_target_state("tA", [40.0, -10.0, 6.0], [0.0, 0.0, 0.0]),
        _make_target_state("tB", [12.0, 0.0, 6.0], [-4.0, 0.0, 0.0], priority=2.0),
        _make_target_state("tC", [40.0, 10.0, 6.0], [0.0, 0.0, 0.0]),
    ]
    wta7 = compute_wta_assignments(drones7, states7, targets3_threat, prof)
    assert list(wta7.values()).count("tB") == 3
    assert list(wta7.values()).count("tA") == 2
    assert list(wta7.values()).count("tC") == 2

    # 3. N=2 vs M=4 -> 锁定威胁最高的前 2 个目标
    drones2 = ["drone_1", "drone_2"]
    states2 = {"drone_1": (np.array([0.0, 0.0, 8.0]), np.zeros(3)), "drone_2": (np.array([0.0, 3.0, 8.0]), np.zeros(3))}
    targets4 = [
        _make_target_state("t1", [60.0, -10.0, 6.0], [0.0, 0.0, 0.0], priority=0.5),
        _make_target_state("t2", [14.0, 0.0, 6.0], [-4.0, 0.0, 0.0], priority=2.0),
        _make_target_state("t3", [16.0, 3.0, 6.0], [-4.0, 0.0, 0.0], priority=2.0),
        _make_target_state("t4", [65.0, 10.0, 6.0], [0.0, 0.0, 0.0], priority=0.5),
    ]
    wta2 = compute_wta_assignments(drones2, states2, targets4, prof)
    assert set(wta2.values()) == {"t2", "t3"}

    # 4. 电池 SoC 偏置：高电量机接战远目标，低电量机接战近目标
    states_eq = {"drone_1": (np.array([0.0, 0.0, 8.0]), np.zeros(3)), "drone_2": (np.array([0.0, 0.0, 8.0]), np.zeros(3))}
    targets_nf = [
        _make_target_state("t_near", [12.0, 0.0, 6.0], [0.0, 0.0, 0.0]),
        _make_target_state("t_far", [48.0, 0.0, 6.0], [0.0, 0.0, 0.0]),
    ]
    wta_soc = compute_wta_assignments(
        ["drone_1", "drone_2"],
        states_eq,
        targets_nf,
        prof,
        drone_batteries={"drone_1": 0.20, "drone_2": 0.95},
    )
    assert wta_soc["drone_1"] == "t_near"
    assert wta_soc["drone_2"] == "t_far"

    # 5. 滞回防抖与击毁后 1-tick 即时重分配
    planner = CooperativeGuidancePlanner(prof)
    modes6 = {d: "ATTACK" for d in drones6}
    cmds_before = planner.compute_multi_target_swarm_targets(1.0, drones6, modes6, states6, targets3)
    assert set(cmds_before.keys()) == set(drones6)
    planner.notify_target_killed(["tB"], external_target_id="tB")
    assert planner.re_tasking_events >= 1
    cmds_after = planner.compute_multi_target_swarm_targets(1.05, drones6, modes6, states6, targets3)
    assert set(cmds_after.keys()) == set(drones6)
    assert all(cmds_after[d]["target_id"] in ("tA", "tC") for d in drones6)


def test_3d_apn_terminal_miss_reduction_vs_pure_pursuit():
    def run_sim(use_apn: bool) -> float:
        prof = DynamicsProfile(v_attack=14.0, a_max=12.0, min_alt=3.2)
        planner = CooperativeGuidancePlanner(prof)
        p_d = np.array([0.0, 0.0, 8.0])
        v_d = np.array([6.0, 0.0, 0.0])
        dt = 0.02
        min_miss = 1e9
        for step in range(300):
            t = step * dt
            p_t = np.array([28.0 + 4.5 * t, 4.5 * math.sin(1.8 * t), 8.0 + 1.2 * math.cos(1.4 * t)])
            v_t = np.array([4.5, 4.5 * 1.8 * math.cos(1.8 * t), -1.2 * 1.4 * math.sin(1.4 * t)])
            a_t = np.array([0.0, -4.5 * (1.8 ** 2) * math.sin(1.8 * t), -1.2 * (1.4 ** 2) * math.cos(1.4 * t)])
            d_now = float(np.linalg.norm(p_t - p_d))
            min_miss = min(min_miss, d_now)
            if d_now < 0.12:
                break

            def pred(tau, pt=p_t, vt=v_t, at=a_t):
                return pt + vt * tau + 0.5 * at * (tau ** 2), vt + at * tau

            if use_apn:
                tgt = TargetTrackState("t1", p_t, v_t, a_t, pred, imm_probs={"CV": 0.05, "CTRV": 0.15, "JINK": 0.80})
                res = planner.compute_multi_target_swarm_targets(
                    t, ["d1"], {"d1": "ATTACK"}, {"d1": (p_d, v_d)}, [tgt]
                )["d1"]
                a_cmd = np.array([res["ax"], res["ay"], res["az"]])
            else:
                u = (p_t - p_d) / max(d_now, 1e-6)
                a_cmd = (prof.v_attack * u - v_d) * 3.0
                nrm = float(np.linalg.norm(a_cmd))
                if nrm > prof.a_max:
                    a_cmd *= prof.a_max / nrm
            v_d = v_d + a_cmd * dt
            spd = float(np.linalg.norm(v_d))
            if spd > prof.v_attack:
                v_d *= prof.v_attack / spd
            p_d = p_d + v_d * dt
        return min_miss

    apn_miss = run_sim(True)
    pp_miss = run_sim(False)
    assert apn_miss < 0.5
    assert apn_miss < 0.5 * pp_miss


def test_obstacle_map_detour_wide_box_cylinder_nfz_and_overflight():
    cfg = {
        "arena_bounds": {"x_min": -50, "x_max": 100, "y_min": -50, "y_max": 50, "z_min": 0, "z_max": 50},
        "obstacles": [
            {"id": "wall_1", "type": "box", "cx": 15.0, "cy": 0.0, "z_min": 0.0, "z_max": 25.0, "sx": 2.5, "sy": 18.0, "safety_margin_m": 1.2},
            {"id": "cyl_1", "type": "cylinder", "cx": 28.0, "cy": 8.0, "z_min": 0.0, "z_max": 25.0, "radius": 3.5, "safety_margin_m": 1.2},
            {"id": "low_bldg", "type": "box", "cx": 15.0, "cy": -25.0, "z_min": 0.0, "z_max": 5.0, "sx": 4.0, "sy": 20.0, "safety_margin_m": 1.0},
        ],
        "no_fly_zones": [
            {"id": "nfz_1", "cx": 25.0, "cy": -5.0, "z_min": 0.0, "z_max": 50.0, "radius": 4.0, "safety_margin_m": 1.5},
        ],
    }
    om = ObstacleMap.from_dict(cfg)
    s = np.array([0.0, 0.0, 8.0])
    g = np.array([40.0, 0.0, 8.0])
    # 预热缓存后测量查询耗时
    om.plan_path_3d(s, g, max_alt=35.0, drone_id="drone_1")
    t0 = time.perf_counter()
    for _ in range(50):
        _, _, active, poly = om.compute_detour_waypoint(s, g, max_alt=35.0, drone_id="drone_1")
    mean_ms = (time.perf_counter() - t0) * 1000.0 / 50.0
    assert mean_ms < 1.0
    assert active
    assert all(om.segment_clear_3d(poly[i], poly[i + 1]) for i in range(len(poly) - 1))

    # 闭环引导仿真穿过复杂障碍场，验证无局部极小值卡死且零碰撞
    p = s.copy()
    for _ in range(180):
        wp, u, _, _ = om.compute_detour_waypoint(p, g, max_alt=35.0, drone_id="drone_1")
        step_dir = wp - p
        dist_step = float(np.linalg.norm(step_dir))
        if dist_step < 0.4 and float(np.linalg.norm(g - p)) < 0.8:
            break
        p_next = p + (step_dir / max(dist_step, 1e-6)) * min(0.5, dist_step)
        assert om.segment_clear_3d(p, p_next)
        p = p_next
    assert float(np.linalg.norm(p - g)) < 1.0

    # 低矮宽墙触发越顶绕飞
    poly_over = om.plan_path_3d(np.array([5.0, -25.0, 4.0]), np.array([25.0, -25.0, 4.0]), max_alt=35.0)
    assert max(float(pt[2]) for pt in poly_over) > 6.0
    assert all(om.segment_clear_3d(poly_over[i], poly_over[i + 1]) for i in range(len(poly_over) - 1))


def test_obstacle_map_3d_endpoints_above_footprint_and_boundary_projection():
    om_yaml, src = ObstacleMap.load_default()
    assert "obstacles.yaml" in src
    assert len(om_yaml.primitives) >= 15

    # 1. 端点位于低矮障碍物/建筑二维投影上方时，不得退化为穿障 [s, g]
    endpoint_pairs = [
        (np.array([50.7231, -5.2457, 8.9257]), np.array([32.2917, 9.1159, 16.0786])),
        (np.array([36.0, 10.5, 16.2]), np.array([20.0, 0.0, 8.0])),
        (np.array([21.0, 19.5, 9.5]), np.array([49.0, 1.5, 8.0])),
        (np.array([36.0, -9.5, 15.0]), np.array([16.0, -16.5, 22.0])),
    ]
    for s_pt, g_pt in endpoint_pairs:
        poly = om_yaml.plan_path_3d(s_pt, g_pt, max_alt=35.0)
        assert all(om_yaml.segment_clear_3d(poly[k], poly[k + 1]) for k in range(len(poly) - 1))

    # 2. 200 对随机起点/终点路径全量无碰撞
    rng = np.random.default_rng(123)
    for _ in range(200):
        s_r = np.array([rng.uniform(-12.0, 90.0), rng.uniform(-30.0, 30.0), rng.uniform(3.5, 22.0)])
        g_r = np.array([rng.uniform(-12.0, 90.0), rng.uniform(-30.0, 30.0), rng.uniform(3.5, 22.0)])
        poly = om_yaml.plan_path_3d(s_r, g_r, max_alt=35.0)
        assert all(om_yaml.segment_clear_3d(poly[k], poly[k + 1]) for k in range(len(poly) - 1))

    # 3. 端点位于低矮宽墙上方 + 高空禁飞区阻挡越顶 -> 阶梯侧向 A* 仍无碰撞
    om_over_vs_nfz = ObstacleMap.from_dict(
        {
            "arena_bounds": {"x_min": -50, "x_max": 100, "y_min": -50, "y_max": 50, "z_min": 0, "z_max": 50},
            "obstacles": [
                {"id": "low_wide", "type": "box", "cx": 18.0, "cy": 0.0, "z_min": 0.0, "z_max": 5.0, "sx": 4.0, "sy": 32.0, "safety_margin_m": 1.0},
            ],
            "no_fly_zones": [
                {"id": "tall_nfz", "cx": 28.0, "cy": 0.0, "z_min": 0.0, "z_max": 50.0, "radius": 4.5, "safety_margin_m": 1.5},
            ],
        }
    )
    for s_pt, g_pt in (
        (np.array([5.0, 0.0, 4.0]), np.array([42.0, 0.0, 4.0])),
        (np.array([18.0, 0.0, 7.0]), np.array([42.0, 0.0, 7.0])),
    ):
        poly_step = om_over_vs_nfz.plan_path_3d(s_pt, g_pt, max_alt=35.0)
        assert all(om_over_vs_nfz.segment_clear_3d(poly_step[k], poly_step[k + 1]) for k in range(len(poly_step) - 1))

    # 4. 贴近赛场边界的 box 与 cylinder 在 project_outside 后必须既安全又在边界内
    om_edge = ObstacleMap.from_dict(
        {
            "arena_bounds": {"x_min": -10, "x_max": 50, "y_min": -20, "y_max": 20, "z_min": 0, "z_max": 45},
            "obstacles": [
                {"id": "edge_wall", "type": "box", "cx": 48.0, "cy": 0.0, "z_min": 0.0, "z_max": 35.0, "sx": 5.0, "sy": 10.0, "safety_margin_m": 1.2},
                {"id": "edge_cyl", "type": "cylinder", "cx": -8.5, "cy": 18.5, "z_min": 0.0, "z_max": 35.0, "radius": 2.0, "safety_margin_m": 1.2},
            ],
        }
    )
    for pt_in in (np.array([48.0, 0.0, 10.0]), np.array([-8.5, 18.5, 10.0])):
        proj = om_edge.project_outside(pt_in)
        assert om_edge.point_clear_3d(proj)
        assert om_edge.bounds.x_min <= proj[0] <= om_edge.bounds.x_max
        assert om_edge.bounds.y_min <= proj[1] <= om_edge.bounds.y_max
        assert om_edge.bounds.z_min <= proj[2] <= om_edge.bounds.z_max


def test_obstacle_map_3d_tight_footprint_overflight_and_stepped_altitudes():
    om_yaml, _ = ObstacleMap.load_default()

    # 1. 全量 obstacles.yaml 障碍物上方紧贴净空 dz in (0, 0.5] 不得退化为穿障 [s, g]
    for obs in om_yaml.primitives:
        _, z_hi = om_yaml._effective_z_interval(obs)
        if z_hi + 2.5 >= 34.0:
            continue
        for dz in (0.01, 0.20, 0.45, 0.48, 0.50):
            z_s = z_hi + dz
            s = np.array([obs.cx, obs.cy, z_s], dtype=float)
            if not om_yaml.point_clear_3d(s):
                continue
            for g in (
                np.array([obs.cx + 22.0, obs.cy + 6.0, 4.0], dtype=float),
                np.array([46.0, 10.5, z_s], dtype=float),
            ):
                if (
                    g[0] < om_yaml.bounds.x_min + 1.0
                    or g[0] > om_yaml.bounds.x_max - 1.0
                    or g[1] < om_yaml.bounds.y_min + 1.0
                    or g[1] > om_yaml.bounds.y_max - 1.0
                    or not om_yaml.point_clear_3d(g)
                ):
                    continue
                for max_a in (35.0, max(z_s + 1.5, 11.0)):
                    poly = om_yaml.plan_path_3d(s, g, max_alt=max_a)
                    assert len(poly) >= 2
                    assert all(om_yaml.segment_clear_3d(poly[k], poly[k + 1]) for k in range(len(poly) - 1))

    # 2. 低矮障碍物上方 +0.48m 遇全高禁飞区 + 起终点分处不同障碍物上方 + 紧限高
    om_multi = ObstacleMap.from_dict(
        {
            "arena_bounds": {"x_min": -20, "x_max": 80, "y_min": -30, "y_max": 30, "z_min": 0, "z_max": 45},
            "obstacles": [
                {"id": "low_a", "type": "box", "cx": 5.0, "cy": 0.0, "z_min": 0.0, "z_max": 5.0, "sx": 6.0, "sy": 6.0, "safety_margin_m": 1.2},
                {"id": "mid_b", "type": "box", "cx": 45.0, "cy": 0.0, "z_min": 0.0, "z_max": 10.0, "sx": 6.0, "sy": 6.0, "safety_margin_m": 1.2},
            ],
            "no_fly_zones": [
                {"id": "tall_nfz", "cx": 25.0, "cy": 0.0, "z_min": 0.0, "radius": 4.0},
            ],
        }
    )
    for s_pt, g_pt, max_a in (
        (np.array([5.0, 0.0, 6.48]), np.array([35.0, 0.0, 6.48]), 35.0),
        (np.array([5.0, 0.0, 6.20]), np.array([35.0, 0.0, 6.20]), 6.8),
        (np.array([5.0, 0.0, 6.48]), np.array([45.0, 0.0, 11.48]), 35.0),
        (np.array([5.0, 0.0, 6.05]), np.array([45.0, 0.0, 11.05]), 11.6),
        (np.array([5.0, 0.0, 6.48]), np.array([35.0, 0.0, 3.0]), 35.0),
    ):
        poly = om_multi.plan_path_3d(s_pt, g_pt, max_alt=max_a)
        assert len(poly) >= 2
        assert all(om_multi.segment_clear_3d(poly[k], poly[k + 1]) for k in range(len(poly) - 1))


def _assert_clear_and_in_bounds(om: ObstacleMap, proj: np.ndarray) -> None:
    assert om.point_clear_3d(proj)
    assert om.bounds.x_min <= proj[0] <= om.bounds.x_max
    assert om.bounds.y_min <= proj[1] <= om.bounds.y_max
    assert om.bounds.z_min <= proj[2] <= om.bounds.z_max


def test_project_outside_overlapping_nfz_traps_bounds_and_detour():
    # 1. 重叠 NFZ 径向脱困陷阱（含 Challenger R2 Defect #2 复现点）与同心不等半径 NFZ
    om_nfz = ObstacleMap(
        bounds=ArenaBounds(-20.0, 50.0, -20.0, 50.0, 0.0, 30.0),
        primitives=[
            ObstaclePrimitive("nfz_1", "nfz", -5.0, 25.0, 0.0, 30.0, radius=4.0, safety_margin_m=1.5),
            ObstaclePrimitive("nfz_2", "nfz", 1.5, 25.0, 0.0, 30.0, radius=4.0, safety_margin_m=1.5),
            ObstaclePrimitive("nfz_small", "nfz", 0.0, 0.0, 0.0, 30.0, radius=2.0, safety_margin_m=1.5),
            ObstaclePrimitive("nfz_big", "nfz", 1.0, 0.0, 0.0, 30.0, radius=5.0, safety_margin_m=1.5),
        ],
    )
    for pt in (
        np.array([-3.0, 25.0, 10.0]),
        np.array([-1.75, 25.0, 10.0]),
        np.array([0.1, 0.0, 10.0]),
    ):
        _assert_clear_and_in_bounds(om_nfz, om_nfz.project_outside(pt))

    wp, _, active, _ = om_nfz.compute_detour_waypoint(
        np.array([-3.0, 25.0, 10.0]), np.array([10.0, 35.0, 10.0]), max_alt=28.0
    )
    assert active
    _assert_clear_and_in_bounds(om_nfz, wp)

    # 2. 长方体正交面投影与角点均被邻近 NFZ/边界遮挡时，边采样仍能脱困
    om_box_nfz = ObstacleMap(
        bounds=ArenaBounds(-10.0, 10.0, -10.0, 10.0, 0.0, 30.0),
        primitives=[
            ObstaclePrimitive("long_box", "box", 0.0, 0.0, 0.0, 30.0, sx=2.0, sy=20.0, safety_margin_m=1.0),
            ObstaclePrimitive("nfz_r", "nfz", 3.0, 0.0, 0.0, 30.0, radius=2.0, safety_margin_m=1.0),
            ObstaclePrimitive("nfz_l", "nfz", -3.0, 0.0, 0.0, 30.0, radius=2.0, safety_margin_m=1.0),
        ],
    )
    _assert_clear_and_in_bounds(om_box_nfz, om_box_nfz.project_outside(np.array([0.0, 0.0, 10.0])))

    # 3. 多基元围堵走廊乒乓陷阱（触发 Stage 2 全局安全投影回退）
    om_corridor_trap = ObstacleMap(
        bounds=ArenaBounds(-4.0, 9.0, -30.0, 30.0, 0.0, 30.0),
        primitives=[
            ObstaclePrimitive("nfz_1", "nfz", 0.0, 0.0, 0.0, 30.0, radius=3.0, safety_margin_m=1.5),
            ObstaclePrimitive("nfz_2", "nfz", 5.0, 0.0, 0.0, 30.0, radius=3.0, safety_margin_m=1.5),
            ObstaclePrimitive("wall_n", "box", 2.5, 11.0, 0.0, 30.0, sx=20.0, sy=14.0, safety_margin_m=1.5),
            ObstaclePrimitive("wall_s", "box", 2.5, -11.0, 0.0, 30.0, sx=20.0, sy=14.0, safety_margin_m=1.5),
        ],
    )
    _assert_clear_and_in_bounds(om_corridor_trap, om_corridor_trap.project_outside(np.array([1.0, 0.0, 10.0])))

    # 4. 越顶退出受 max_alt 约束且越界 z 被安全钳位
    om_tall = ObstacleMap(
        bounds=ArenaBounds(-50.0, 100.0, -50.0, 50.0, 0.0, 50.0),
        primitives=[
            ObstaclePrimitive("tall_thick_box", "box", 0.0, 0.0, 0.0, 38.0, sx=20.0, sy=20.0, safety_margin_m=1.2),
        ],
    )
    proj_maxalt = om_tall.project_outside(np.array([0.0, 0.0, 34.0]), max_alt=35.0)
    _assert_clear_and_in_bounds(om_tall, proj_maxalt)
    assert proj_maxalt[2] <= 35.0
    for z_oob in (-1.0, 55.0):
        _assert_clear_and_in_bounds(om_tall, om_tall.project_outside(np.array([0.0, 0.0, z_oob]), max_alt=35.0))

    # 5. obstacles.yaml 重叠障碍物与 NFZ 中心点
    om_yaml, _ = ObstacleMap.load_default()
    for pt in (
        np.array([-7.0, 16.5, 10.0]),
        np.array([58.0, 15.5, 10.0]),
        np.array([16.0, -16.5, 10.0]),
    ):
        _assert_clear_and_in_bounds(om_yaml, om_yaml.project_outside(pt))


def _strict_cross_2d(a1, a2, b1, b2) -> bool:
    def ccw(p1, p2, p3):
        return (p3[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    d1 = ccw(b1, b2, a1)
    d2 = ccw(b1, b2, a2)
    d3 = ccw(a1, a2, b1)
    d4 = ccw(a1, a2, b2)
    return ((d1 > 1e-9 and d2 < -1e-9) or (d1 < -1e-9 and d2 > 1e-9)) and (
        (d3 > 1e-9 and d4 < -1e-9) or (d3 < -1e-9 and d4 > 1e-9)
    )


def test_non_crossing_1wasserstein_encirclement_pincer_and_overwatch():
    rng = np.random.default_rng(42)
    prof = DynamicsProfile()
    for k_cnt in (2, 3, 5, 6, 8, 10):
        drones = [f"drone_{i + 1}" for i in range(k_cnt)]
        for trial in range(10):
            pts = {
                d: np.array([rng.uniform(-15.0, 25.0), rng.uniform(-15.0, 15.0), rng.uniform(4.0, 12.0)])
                for d in drones
            }
            # TRACK fresh (1-Wasserstein)
            sl = assign_encirclement_slots(drones, pts, (20.0, 0.0), radius=6.0, base_phase_rad=trial * 0.1)
            segs_t = [(pts[d][0:2], np.array([sl[d][0], sl[d][1]])) for d in drones]
            for i in range(k_cnt):
                for j in range(i + 1, k_cnt):
                    assert not _strict_cross_2d(segs_t[i][0], segs_t[i][1], segs_t[j][0], segs_t[j][1])

            # TRACK hysteresis + adversarial drone swap
            planner = CooperativeGuidancePlanner(prof)
            modes_t = {d: "TRACK" for d in drones}
            st_t = {d: (pts[d], np.zeros(3)) for d in drones}
            t_pos = np.array([25.0, 0.0, 6.0])
            t_vel = np.array([1.0, 0.0, 0.0])
            planner.compute_swarm_targets(
                0.0, drones, modes_t, st_t, True, t_pos, t_vel, np.zeros(3), lambda tau: (t_pos + t_vel * tau, t_vel)
            )
            pts_swapped = dict(pts)
            pts_swapped[drones[0]], pts_swapped[drones[1]] = pts[drones[1]].copy(), pts[drones[0]].copy()
            st_sw = {d: (pts_swapped[d], np.zeros(3)) for d in drones}
            cmds_h = planner.compute_swarm_targets(
                0.05, drones, modes_t, st_sw, True, t_pos, t_vel, np.zeros(3), lambda tau: (t_pos + t_vel * tau, t_vel)
            )
            segs_h = [(pts_swapped[d][0:2], np.array([cmds_h[d]["x"], cmds_h[d]["y"]])) for d in drones]
            for i in range(k_cnt):
                for j in range(i + 1, k_cnt):
                    assert not _strict_cross_2d(segs_h[i][0], segs_h[i][1], segs_h[j][0], segs_h[j][1])

            # INTERCEPT slow & fast staggered
            for t_v in (np.array([4.0, 0.0, 0.0]), np.array([8.5, 1.5, 0.0])):
                planner_i = CooperativeGuidancePlanner(prof)
                pts_i = {
                    d: np.array([rng.uniform(0.0, 22.0), rng.uniform(-12.0, 12.0), 8.0])
                    for d in drones
                }
                st_i = {d: (pts_i[d], np.zeros(3)) for d in drones}
                cmds_i = planner_i.compute_swarm_targets(
                    1.0,
                    drones,
                    {d: "INTERCEPT" for d in drones},
                    st_i,
                    True,
                    t_pos,
                    t_v,
                    np.zeros(3),
                    lambda tau, p=t_pos, v=t_v: (p + v * tau, v),
                )
                segs_i = [(pts_i[d][0:2], np.array([cmds_i[d]["x"], cmds_i[d]["y"]])) for d in drones]
                for i in range(k_cnt):
                    for j in range(i + 1, k_cnt):
                        assert not _strict_cross_2d(segs_i[i][0], segs_i[i][1], segs_i[j][0], segs_i[j][1])

            # ATTACK overwatch ring non-crossing
            if k_cnt >= 3:
                planner_a = CooperativeGuidancePlanner(prof)
                pts_a = {
                    d: np.array([rng.uniform(0.0, 20.0), rng.uniform(-12.0, 12.0), 8.0])
                    for d in drones
                }
                st_a = {d: (pts_a[d], np.zeros(3)) for d in drones}
                cmds_a = planner_a.compute_swarm_targets(
                    1.0,
                    drones,
                    {d: "ATTACK" for d in drones},
                    st_a,
                    True,
                    t_pos,
                    t_vel,
                    np.zeros(3),
                    lambda tau: (t_pos + t_vel * tau, t_vel),
                )
                striker = planner_a._active_striker_id
                wing_drones = [d for d in drones if d != striker]
                segs_w = [(pts_a[d][0:2], np.array([cmds_a[d]["x"], cmds_a[d]["y"]])) for d in wing_drones]
                for i in range(len(segs_w)):
                    for j in range(i + 1, len(segs_w)):
                        assert not _strict_cross_2d(segs_w[i][0], segs_w[i][1], segs_w[j][0], segs_w[j][1])


def test_apn_strike_early_obstacle_detour_high_speed():
    prof = DynamicsProfile(v_attack=13.5, a_max=10.5, min_alt=3.2, max_alt=35.0)
    om = ObstacleMap.from_dict(
        {
            "arena_bounds": {"x_min": -50, "x_max": 120, "y_min": -50, "y_max": 50, "z_min": 0, "z_max": 50},
            "obstacles": [
                {"id": "pillar", "type": "cylinder", "cx": 16.0, "cy": 0.0, "z_min": 0.0, "z_max": 25.0, "radius": 3.0, "safety_margin_m": 1.2}
            ],
        }
    )
    planner = CooperativeGuidancePlanner(prof, obstacle_map=om)
    p_d = np.array([0.0, 0.0, 8.0])
    v_d = np.array([6.0, 0.0, 0.0])
    dt = 0.02
    min_miss = 1e9
    collisions = 0
    for step in range(350):
        t = step * dt
        p_t = np.array([32.0 + 2.0 * t, 0.0, 8.0])
        v_t = np.array([2.0, 0.0, 0.0])
        a_t = np.zeros(3)
        d_now = float(np.linalg.norm(p_t - p_d))
        min_miss = min(min_miss, d_now)
        if d_now < 0.08:
            break

        def pred(tau, pt=p_t, vt=v_t):
            return pt + vt * tau, vt

        tgt = TargetTrackState("t1", p_t, v_t, a_t, pred, imm_probs={"CV": 0.05, "CTRV": 0.25, "JINK": 0.70})
        res = planner.compute_multi_target_swarm_targets(
            t, ["d1"], {"d1": "ATTACK"}, {"d1": (p_d, v_d)}, [tgt]
        )["d1"]
        p_cmd = np.array([res["x"], res["y"], res["z"]])
        v_cmd = np.array([res["vx"], res["vy"], res["vz"]])
        a_cmd = np.array([res["ax"], res["ay"], res["az"]])

        a_eff = 2.5 * (p_cmd - p_d) + 3.2 * (v_cmd - v_d) + 0.6 * a_cmd
        an = float(np.linalg.norm(a_eff))
        if an > prof.a_max:
            a_eff *= prof.a_max / an
        v_next = v_d + a_eff * dt
        spd = float(np.linalg.norm(v_next))
        if spd > prof.v_attack:
            v_next *= prof.v_attack / spd
        p_next = p_d + v_next * dt
        if not om.segment_clear_3d(p_d, p_next):
            collisions += 1
        p_d, v_d = p_next, v_next

    assert collisions == 0
    assert min_miss < 0.25


def test_apn_strike_high_speed_headon_and_yaml_canyon_pd_dynamics():
    profile = DynamicsProfile(a_max=10.5, v_attack=13.5, v_intercept=10.5, min_alt=3.2, max_alt=35.0)
    om_yaml, _ = ObstacleMap.load_default()

    scenarios = [
        (
            "single_pillar_headon",
            ObstacleMap(
                bounds=ArenaBounds(-20.0, 80.0, -30.0, 30.0, 0.0, 35.0),
                primitives=[
                    ObstaclePrimitive("pillar", "cylinder", 22.0, 0.0, 0.0, 22.0, radius=2.5, safety_margin_m=1.2)
                ],
            ),
            np.array([8.0, 0.0, 9.0]),
            np.array([12.0, 0.0, 0.0]),
            lambda t: (
                np.array([42.0 + 2.5 * math.sin(0.8 * t), 2.0 * math.cos(0.9 * t), 9.0]),
                np.array([2.0 * math.cos(0.8 * t), -1.8 * math.sin(0.9 * t), 0.0]),
                np.array([-1.6 * math.sin(0.8 * t), -1.62 * math.cos(0.9 * t), 0.0]),
            ),
        ),
        (
            "double_staggered_canyon",
            ObstacleMap(
                bounds=ArenaBounds(-20.0, 80.0, -30.0, 30.0, 0.0, 35.0),
                primitives=[
                    ObstaclePrimitive("b1", "box", 20.0, 2.0, 0.0, 20.0, sx=4.0, sy=5.0, safety_margin_m=1.2),
                    ObstaclePrimitive("c2", "cylinder", 34.0, -2.0, 0.0, 20.0, radius=2.2, safety_margin_m=1.2),
                ],
            ),
            np.array([5.0, 0.0, 9.0]),
            np.array([10.0, 1.0, 0.0]),
            lambda t: (
                np.array([46.0 + 3.0 * t, 1.5 * math.sin(1.2 * t), 9.5 + 0.5 * math.cos(t)]),
                np.array([3.0, 1.8 * math.cos(1.2 * t), -0.5 * math.sin(t)]),
                np.array([0.0, -2.16 * math.sin(1.2 * t), -0.5 * math.cos(t)]),
            ),
        ),
        (
            "yaml_canyon_corridor",
            om_yaml,
            np.array([24.0, 1.0, 8.0]),
            np.array([8.0, 0.0, 0.0]),
            lambda t: (
                np.array([58.0 + 2.5 * math.cos(0.6 * t), 1.5 + 3.0 * math.sin(0.9 * t), 8.5]),
                np.array([-1.5 * math.sin(0.6 * t), 2.7 * math.cos(0.9 * t), 0.0]),
                np.array([-0.9 * math.cos(0.6 * t), -2.43 * math.sin(0.9 * t), 0.0]),
            ),
        ),
        (
            "low_wall_overflight",
            ObstacleMap(
                bounds=ArenaBounds(-20.0, 80.0, -30.0, 30.0, 0.0, 35.0),
                primitives=[
                    ObstaclePrimitive("low_wall", "box", 20.0, 0.0, 0.0, 5.0, sx=4.0, sy=26.0, safety_margin_m=1.2)
                ],
            ),
            np.array([5.0, 0.0, 4.5]),
            np.array([10.0, 0.0, 0.0]),
            lambda t: (
                np.array([40.0 + 2.0 * t, 1.0 * math.sin(0.8 * t), 4.5]),
                np.array([2.0, 0.8 * math.cos(0.8 * t), 0.0]),
                np.array([0.0, -0.64 * math.sin(0.8 * t), 0.0]),
            ),
        ),
    ]

    dt = 0.02
    for _, cmap, p0, v0, tgt_fn in scenarios:
        for dyn in ("acc", "pd_pos_vel"):
            planner = CooperativeGuidancePlanner(profile=profile, obstacle_map=cmap)
            p = p0.copy()
            v = v0.copy()
            min_miss = 1e9
            collisions = 0
            for step in range(300):
                t = step * dt
                tp, tv, ta = tgt_fn(t)
                p_cmd, v_cmd, a_cmd, _, _ = planner._compute_apn_strike(
                    "drone_1",
                    p,
                    v,
                    tp,
                    tv,
                    ta,
                    predict_fn=lambda tau, tt=t, fn=tgt_fn: fn(tt + tau)[:2],
                    imm_probs={"CV": 0.1, "CTRV": 0.3, "JINK": 0.6},
                )
                if dyn == "acc":
                    a_eff = a_cmd
                else:
                    a_eff = 2.5 * (p_cmd - p) + 3.2 * (v_cmd - v) + 0.6 * a_cmd
                an = float(np.linalg.norm(a_eff))
                if an > profile.a_max:
                    a_eff = a_eff * (profile.a_max / an)
                v = v + a_eff * dt
                vn = float(np.linalg.norm(v))
                if vn > profile.v_attack:
                    v = v * (profile.v_attack / vn)
                p_next = p + v * dt
                if not cmap.point_clear_3d(p_next) or not cmap.segment_clear_3d(p, p_next):
                    collisions += 1
                p = p_next
                miss = float(np.linalg.norm(tp - p))
                min_miss = min(min_miss, miss)
                if miss < 0.25:
                    break
            assert collisions == 0
            assert min_miss < 0.5

    # 开阔空域高机动目标在 pd_pos_vel 动力学下 APN 仍显著优于 Pure Pursuit
    misses = {}
    for mode in ("APN", "PP"):
        planner = CooperativeGuidancePlanner(profile=profile)
        p = np.array([0.0, 0.0, 8.0])
        v = np.array([6.0, 0.0, 0.0])
        min_miss = 1e9
        for step in range(350):
            t = step * dt
            tp = np.array([35.0 + 4.5 * t, 6.0 * math.sin(1.4 * t), 9.0 + 2.0 * math.cos(1.1 * t)])
            tv = np.array([4.5, 8.4 * math.cos(1.4 * t), -2.2 * math.sin(1.1 * t)])
            ta = np.array([0.0, -11.76 * math.sin(1.4 * t), -2.42 * math.cos(1.1 * t)])
            if mode == "APN":
                p_cmd, v_cmd, a_cmd, _, _ = planner._compute_apn_strike(
                    "drone_1",
                    p,
                    v,
                    tp,
                    tv,
                    ta,
                    predict_fn=lambda tau, tt=t: (
                        np.array([35.0 + 4.5 * (tt + tau), 6.0 * math.sin(1.4 * (tt + tau)), 9.0 + 2.0 * math.cos(1.1 * (tt + tau))]),
                        np.array([4.5, 8.4 * math.cos(1.4 * (tt + tau)), -2.2 * math.sin(1.1 * (tt + tau))]),
                    ),
                    imm_probs={"CV": 0.05, "CTRV": 0.25, "JINK": 0.70},
                )
            else:
                d_vec = tp - p
                dn = max(float(np.linalg.norm(d_vec)), 1e-6)
                v_cmd = (d_vec / dn) * profile.v_attack
                p_cmd = p + v_cmd * 0.4
                a_cmd = (v_cmd - v) * 3.0
                an = float(np.linalg.norm(a_cmd))
                if an > profile.a_max:
                    a_cmd = a_cmd * (profile.a_max / an)

            a_eff = 2.5 * (p_cmd - p) + 3.2 * (v_cmd - v) + 0.6 * a_cmd
            an = float(np.linalg.norm(a_eff))
            if an > profile.a_max:
                a_eff = a_eff * (profile.a_max / an)
            v = v + a_eff * dt
            vn = float(np.linalg.norm(v))
            if vn > profile.v_attack:
                v = v * (profile.v_attack / vn)
            p = p + v * dt
            min_miss = min(min_miss, float(np.linalg.norm(tp - p)))
            if min_miss < 0.08:
                break
        misses[mode] = min_miss
    assert misses["APN"] < 0.5
    assert misses["APN"] < misses["PP"]


def test_apn_strike_zigzag_rotated45_and_normal_priority_braking():
    profile = DynamicsProfile(v_attack=13.5, v_intercept=10.5, a_max=10.5, min_alt=3.2, max_alt=35.0)
    bounds = ArenaBounds(-30.0, 110.0, -40.0, 40.0, 0.0, 40.0)
    cmap = ObstacleMap(
        bounds=bounds,
        primitives=[
            ObstaclePrimitive("g1_box", "box", 18.0, 3.0, 0.0, 24.0, sx=4.0, sy=8.0, yaw=math.pi / 4.0, safety_margin_m=1.2),
            ObstaclePrimitive("g2_cyl", "cylinder", 34.0, -3.5, 0.0, 24.0, radius=3.2, safety_margin_m=1.2),
            ObstaclePrimitive("g3_nfz", "nfz", 50.0, 3.5, 0.0, 35.0, radius=3.5, safety_margin_m=1.5),
            ObstaclePrimitive("g4_box", "box", 65.0, -2.5, 0.0, 24.0, sx=6.0, sy=4.0, yaw=-math.pi / 4.0, safety_margin_m=1.2),
        ],
    )

    planner_probe = CooperativeGuidancePlanner(profile=profile, obstacle_map=cmap)
    p_probe = np.array([5.5, 0.5, 10.0])
    v_probe = np.array([11.5, 3.1, 0.0])
    near = _obstacle_surface_info(cmap, p_probe)
    d_surf, n_out, _, _ = min(near, key=lambda item: item[0])
    assert d_surf > 6.5
    _, _, a_probe, _, _ = planner_probe._compute_apn_strike(
        "drone_1",
        p_probe,
        v_probe,
        np.array([82.0, 0.0, 10.0]),
        np.zeros(3),
        np.zeros(3),
        predict_fn=lambda tau: (np.array([82.0, 0.0, 10.0]), np.zeros(3)),
    )
    assert float(np.dot(a_probe, n_out)) >= 0.48 * profile.a_max - 1e-6
    assert float(np.linalg.norm(a_probe)) <= profile.a_max + 1e-6

    def tgt_fn(t: float):
        return (
            np.array([82.0 + 2.5 * math.sin(0.8 * t), 3.0 * math.sin(1.2 * t), 10.0 + 1.2 * math.cos(0.9 * t)]),
            np.array([2.0 * math.cos(0.8 * t), 3.6 * math.cos(1.2 * t), -1.08 * math.sin(0.9 * t)]),
            np.array([-1.6 * math.sin(0.8 * t), -4.32 * math.sin(1.2 * t), -0.972 * math.cos(0.9 * t)]),
        )

    for dyn in ("acc", "pd_pos_vel"):
        planner = CooperativeGuidancePlanner(profile=profile, obstacle_map=cmap)
        p = np.array([2.0, 0.0, 10.0])
        v = np.array([12.0, 0.0, 0.0])
        dt = 0.02
        min_miss = 1e9
        collisions = 0
        for step in range(600):
            t = step * dt
            tp, tv, ta = tgt_fn(t)
            p_cmd, v_cmd, a_cmd, _, _ = planner._compute_apn_strike(
                "drone_1",
                p,
                v,
                tp,
                tv,
                ta,
                predict_fn=lambda tau, tt=t: tgt_fn(tt + tau)[:2],
                imm_probs={"CV": 0.1, "CTRV": 0.3, "JINK": 0.6},
            )
            assert np.all(np.isfinite(p_cmd)) and np.all(np.isfinite(v_cmd)) and np.all(np.isfinite(a_cmd))
            a_eff = a_cmd.copy() if dyn == "acc" else 2.5 * (p_cmd - p) + 3.2 * (v_cmd - v) + 0.6 * a_cmd
            an = float(np.linalg.norm(a_eff))
            if an > profile.a_max:
                a_eff = a_eff * (profile.a_max / an)
            v_next = v + a_eff * dt
            vn = float(np.linalg.norm(v_next))
            if vn > profile.v_attack:
                v_next = v_next * (profile.v_attack / vn)
            p_next = p + v_next * dt
            if not cmap.point_clear_3d(p_next) or not cmap.segment_clear_3d(p, p_next):
                collisions += 1
            p, v = p_next, v_next
            min_miss = min(min_miss, float(np.linalg.norm(tp - p)))
            if min_miss < 0.25:
                break
        assert collisions == 0
        assert min_miss < 0.5


def test_apn_strike_simultaneous_multi_obstacle_corner_and_gate():
    profile = DynamicsProfile(v_attack=13.5, v_intercept=10.5, a_max=10.5, min_alt=3.2, max_alt=35.0)
    bounds = ArenaBounds(-30.0, 110.0, -40.0, 40.0, 0.0, 40.0)
    v_corner_map = ObstacleMap(
        bounds=bounds,
        primitives=[
            ObstaclePrimitive("b_top", "box", 20.0, 4.5, 0.0, 24.0, sx=8.0, sy=3.0, yaw=math.pi / 4.0, safety_margin_m=1.2),
            ObstaclePrimitive("b_bot", "box", 20.0, -4.5, 0.0, 24.0, sx=8.0, sy=3.0, yaw=-math.pi / 4.0, safety_margin_m=1.2),
        ],
    )
    gate_map = ObstacleMap(
        bounds=bounds,
        primitives=[
            ObstaclePrimitive("c_up", "cylinder", 22.0, 5.2, 0.0, 24.0, radius=3.0, safety_margin_m=1.2),
            ObstaclePrimitive("c_dn", "cylinder", 22.0, -5.2, 0.0, 24.0, radius=3.0, safety_margin_m=1.2),
        ],
    )
    for cmap, p0, v0, tgt_fn in (
        (
            v_corner_map,
            np.array([2.0, 0.2, 10.0]),
            np.array([13.5, 0.0, 0.0]),
            lambda t: (
                np.array([45.0, 2.0 * math.sin(0.8 * t), 10.0]),
                np.array([0.0, 1.6 * math.cos(0.8 * t), 0.0]),
                np.array([0.0, -1.28 * math.sin(0.8 * t), 0.0]),
            ),
        ),
        (
            gate_map,
            np.array([4.0, 2.8, 10.0]),
            np.array([13.0, -3.5, 0.0]),
            lambda t: (np.array([46.0, -2.0, 10.0]), np.zeros(3), np.zeros(3)),
        ),
    ):
        for dyn in ("acc", "pd_pos_vel"):
            planner = CooperativeGuidancePlanner(profile=profile, obstacle_map=cmap)
            p, v = p0.copy(), v0.copy()
            dt = 0.02
            min_miss = 1e9
            collisions = 0
            for step in range(500):
                t = step * dt
                tp, tv, ta = tgt_fn(t)
                p_cmd, v_cmd, a_cmd, _, _ = planner._compute_apn_strike(
                    "drone_1",
                    p,
                    v,
                    tp,
                    tv,
                    ta,
                    predict_fn=lambda tau, tt=t, fn=tgt_fn: fn(tt + tau)[:2],
                )
                assert np.all(np.isfinite(p_cmd)) and np.all(np.isfinite(v_cmd)) and np.all(np.isfinite(a_cmd))
                a_eff = a_cmd.copy() if dyn == "acc" else 2.5 * (p_cmd - p) + 3.2 * (v_cmd - v) + 0.6 * a_cmd
                an = float(np.linalg.norm(a_eff))
                if an > profile.a_max:
                    a_eff = a_eff * (profile.a_max / an)
                v_next = v + a_eff * dt
                vn = float(np.linalg.norm(v_next))
                if vn > profile.v_attack:
                    v_next = v_next * (profile.v_attack / vn)
                p_next = p + v_next * dt
                if not cmap.point_clear_3d(p_next) or not cmap.segment_clear_3d(p, p_next):
                    collisions += 1
                p, v = p_next, v_next
                min_miss = min(min_miss, float(np.linalg.norm(tp - p)))
                if min_miss < 0.25:
                    break
            assert collisions == 0
            assert min_miss < 0.5


def test_apn_strike_roof_overflight_and_underpass_zero_horizontal_stall():
    profile = DynamicsProfile(a_max=10.5, v_attack=13.5, v_intercept=10.5, min_alt=3.2, max_alt=35.0)
    bounds = ArenaBounds(-20.0, 80.0, -30.0, 30.0, 0.0, 40.0)
    empty_map = ObstacleMap(bounds=bounds, primitives=[])

    roof_obs = ObstaclePrimitive("roof_box", "box", 20.0, 0.0, 0.0, 6.0, sx=8.0, sy=8.0, yaw=math.pi / 4.0, safety_margin_m=1.2)
    roof_map = ObstacleMap(bounds=bounds, primitives=[roof_obs])
    _, z_hi_roof = roof_map._effective_z_interval(roof_obs)

    under_obs = ObstaclePrimitive("elevated_bridge", "box", 20.0, 0.0, 10.0, 18.0, sx=8.0, sy=10.0, yaw=0.0, safety_margin_m=1.2)
    under_map = ObstacleMap(bounds=bounds, primitives=[under_obs])
    z_lo_under, _ = under_map._effective_z_interval(under_obs)

    p_empty = CooperativeGuidancePlanner(profile=profile, obstacle_map=empty_map)
    p_roof = CooperativeGuidancePlanner(profile=profile, obstacle_map=roof_map)
    p_under = CooperativeGuidancePlanner(profile=profile, obstacle_map=under_map)

    for pl, z_val in [
        (p_roof, z_hi_roof + 0.10),
        (p_roof, z_hi_roof + 0.02),
        (p_under, z_lo_under - 0.10),
        (p_under, z_lo_under - 0.02),
    ]:
        for x_cur in (10.0, 15.0, 20.0):
            p_cur = np.array([x_cur, 0.0, z_val])
            v_cur = np.array([13.0, 0.0, 0.0])
            tp = np.array([45.0, 0.0, z_val])
            pred = lambda tau, p=tp: (p, np.zeros(3))
            _, v0, a0, _, _ = p_empty._compute_apn_strike("d1", p_cur, v_cur, tp, np.zeros(3), np.zeros(3), pred)
            _, v1, a1, _, _ = pl._compute_apn_strike("d1", p_cur, v_cur, tp, np.zeros(3), np.zeros(3), pred)
            assert float(v1[0] - v0[0]) == 0.0
            assert np.allclose(v1, v0, atol=1e-9)
            assert np.allclose(a1, a0, atol=1e-9)

    om_yaml, _ = ObstacleMap.load_default()
    low_wall_map = ObstacleMap(
        bounds=bounds,
        primitives=[ObstaclePrimitive("low_wall", "box", 20.0, 0.0, 0.0, 5.0, sx=6.0, sy=30.0, safety_margin_m=1.2)],
    )
    dt = 0.02
    cases = [
        ("obs_hangar_north", om_yaml, np.array([5.0, 19.5, 7.30]), np.array([38.0, 19.5, 7.30])),
        ("low_wall", low_wall_map, np.array([5.0, 0.0, 6.10]), np.array([45.0, 0.0, 6.10])),
        ("closed_roof_overflight_+0.10", roof_map, np.array([2.0, 0.0, z_hi_roof + 0.10]), np.array([44.0, 0.0, z_hi_roof + 0.10])),
        ("closed_underpass_-0.10", under_map, np.array([2.0, 0.0, z_lo_under - 0.10]), np.array([44.0, 0.0, z_lo_under - 0.10])),
        ("roof_descend_after_clear", roof_map, np.array([10.0, 0.0, z_hi_roof + 0.12]), np.array([45.0, 0.0, 5.0])),
        ("under_ascend_after_clear", under_map, np.array([10.0, 0.0, z_lo_under - 0.13]), np.array([45.0, 0.0, 13.0])),
    ]
    for name, cmap, p0, tp_base in cases:
        for dyn in ("acc", "pd_pos_vel"):
            planner = CooperativeGuidancePlanner(profile=profile, obstacle_map=cmap)
            p = p0.copy()
            v = np.array([13.0, 0.0, 0.0])
            min_miss = 1e9
            min_speed = float(np.linalg.norm(v))
            collisions = 0
            for step in range(450):
                t = step * dt
                tp = tp_base + np.array([1.5 * math.sin(0.8 * t), 1.2 * math.cos(0.9 * t), 0.0])
                tv = np.array([1.2 * math.cos(0.8 * t), -1.08 * math.sin(0.9 * t), 0.0])
                ta = np.array([-0.96 * math.sin(0.8 * t), -0.972 * math.cos(0.9 * t), 0.0])
                p_cmd, v_cmd, a_cmd, _, _ = planner._compute_apn_strike(
                    "drone_1",
                    p,
                    v,
                    tp,
                    tv,
                    ta,
                    predict_fn=lambda tau, tt=t: (
                        tp_base + np.array([1.5 * math.sin(0.8 * (tt + tau)), 1.2 * math.cos(0.9 * (tt + tau)), 0.0]),
                        np.array([1.2 * math.cos(0.8 * (tt + tau)), -1.08 * math.sin(0.9 * (tt + tau)), 0.0]),
                    ),
                )
                a_eff = a_cmd if dyn == "acc" else 2.5 * (p_cmd - p) + 3.2 * (v_cmd - v) + 0.6 * a_cmd
                an = float(np.linalg.norm(a_eff))
                if an > profile.a_max:
                    a_eff = a_eff * (profile.a_max / an)
                v = v + a_eff * dt
                vn = float(np.linalg.norm(v))
                if vn > profile.v_attack:
                    v = v * (profile.v_attack / vn)
                p_next = p + v * dt
                if not cmap.point_clear_3d(p_next) or not cmap.segment_clear_3d(p, p_next):
                    collisions += 1
                p = p_next
                min_speed = min(min_speed, float(np.linalg.norm(v)))
                miss = float(np.linalg.norm(tp - p))
                min_miss = min(min_miss, miss)
                if miss < 0.25:
                    break
            assert collisions == 0, f"{name} ({dyn}) collisions={collisions}"
            assert min_miss < 0.35, f"{name} ({dyn}) min_miss={min_miss}"
            if name not in ("roof_descend_after_clear", "under_ascend_after_clear"):
                assert min_speed >= 10.0, f"{name} ({dyn}) min_speed={min_speed}"


def test_obstacle_map_nonfinite_sanitization_and_detour():
    om_yaml, _ = ObstacleMap.load_default()
    om_empty = ObstacleMap.empty()
    bad_pts = [
        np.array([math.inf, -math.inf, 10.0]),
        np.array([math.nan, 0.0, 5.0]),
        np.array([0.0, 0.0, math.nan]),
        np.array([-math.inf, math.inf, math.inf]),
    ]
    good_pt = np.array([40.0, 0.0, 8.0])
    for om in (om_yaml, om_empty):
        for bad in bad_pts:
            assert not om.point_clear_3d(bad)
            assert not om.segment_clear_3d(bad, good_pt)
            assert not om.segment_clear_3d(good_pt, bad)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for s_in, g_in in ((bad, good_pt), (good_pt, bad), (bad, bad)):
                    wp, u, _, poly = om.compute_detour_waypoint(s_in, g_in)
                    assert np.all(np.isfinite(wp))
                    assert np.all(np.isfinite(u))
                    assert all(np.all(np.isfinite(pt)) for pt in poly)
                    assert om.point_clear_3d(wp)
            assert len(caught) == 0


def test_apn_strike_sharp_reversal_clear_los_no_detour_hijack():
    prof = DynamicsProfile(a_max=10.5, v_attack=13.5, v_intercept=10.5, min_alt=3.2, max_alt=35.0)
    bounds = ArenaBounds(-20.0, 80.0, -30.0, 30.0, 0.0, 35.0)
    om_pillar = ObstacleMap(
        bounds=bounds,
        primitives=[ObstaclePrimitive("pillar", "cylinder", 22.0, 0.0, 0.0, 22.0, radius=2.5, safety_margin_m=1.2)],
    )
    om_wall = ObstacleMap(
        bounds=bounds,
        primitives=[ObstaclePrimitive("wall", "box", 20.5, 0.0, 0.0, 22.0, sx=2.0, sy=30.0, safety_margin_m=1.2)],
    )
    tp = np.array([2.0, 8.0, 9.0])
    v0 = np.array([12.0, 0.0, 0.0])

    for cmap in (om_pillar, om_wall):
        planner = CooperativeGuidancePlanner(profile=prof, obstacle_map=cmap)
        _, v_cmd0, a_cmd0, _, _ = planner._compute_apn_strike(
            "drone_1", np.array([10.0, 0.0, 9.0]), v0, tp, np.zeros(3), np.zeros(3), lambda tau: (tp, np.zeros(3))
        )
        assert float(a_cmd0[0]) < -5.0
        assert float(v_cmd0[1]) > 0.1

    cases = [
        (om_pillar, 10.0, tp),
        (om_pillar, 11.0, tp),
        (om_wall, 10.0, tp),
        (om_wall, 11.0, tp),
        (om_pillar, 10.5, np.array([7.5, 2.0, 9.0])),
    ]
    dt = 0.02
    for cmap, p0x, tgt in cases:
        for dyn in ("acc", "pd_pos_vel"):
            planner = CooperativeGuidancePlanner(profile=prof, obstacle_map=cmap)
            p = np.array([p0x, 0.0, 9.0])
            v = v0.copy()
            collisions = 0
            min_miss = 1e9
            max_x = float(p[0])
            for _ in range(300):
                p_cmd, v_cmd, a_cmd, _, _ = planner._compute_apn_strike(
                    "drone_1", p, v, tgt, np.zeros(3), np.zeros(3), lambda tau, t=tgt: (t, np.zeros(3))
                )
                a_eff = a_cmd if dyn == "acc" else 2.5 * (p_cmd - p) + 3.2 * (v_cmd - v) + 0.6 * a_cmd
                an = float(np.linalg.norm(a_eff))
                if an > prof.a_max:
                    a_eff = a_eff * (prof.a_max / an)
                v = v + a_eff * dt
                vn = float(np.linalg.norm(v))
                if vn > prof.v_attack:
                    v = v * (prof.v_attack / vn)
                p_next = p + v * dt
                if not cmap.point_clear_3d(p_next) or not cmap.segment_clear_3d(p, p_next):
                    collisions += 1
                p = p_next
                max_x = max(max_x, float(p[0]))
                miss = float(np.linalg.norm(tgt - p))
                min_miss = min(min_miss, miss)
                if miss < 0.25:
                    break
            assert collisions == 0
            assert max_x < 17.85
            assert min_miss < 0.35
