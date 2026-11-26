# -*- coding: utf-8 -*-
"""planning_pkg.estimator 单元测试：收敛性、野值门控、匈牙利分配、3模型 IMM 与多目标航迹库。"""

from __future__ import annotations

import itertools
import math
import sys
import time

import numpy as np

from planning_pkg.estimator import (
    IMMTrack,
    IMMTrackBank,
    ManeuveringTargetEstimator,
    TrackObservation,
    _singer_blocks_param,
    hungarian_assignment,
)


def test_linear_and_curved_tracking_convergence():
    est = ManeuveringTargetEstimator(sigma_vision=0.15, sigma_external=0.5)
    # 圆周运动靶机: R=15m, omega=0.3 rad/s -> v = 4.5 m/s, z = 6.0m
    r = 15.0
    omega = 0.30
    dt = 0.05
    for k in range(80):
        t = k * dt
        x = 20.0 + r * math.cos(omega * t)
        y = 5.0 + r * math.sin(omega * t)
        z = 6.0 + 0.4 * math.sin(0.5 * t)
        assert est.update((x, y, z), t, source="vision")

    t_now = 79 * dt
    assert est.is_valid(t_now)
    # 验证速度估计精度
    vx_true = -r * omega * math.sin(omega * t_now)
    vy_true = r * omega * math.cos(omega * t_now)
    assert math.hypot(est.x[3] - vx_true, est.x[4] - vy_true) < 0.85
    assert abs(est.turn_rate_est - omega) < 0.15

    # 验证视觉丢失 1.2s 后的 CTRV 曲率预测优于零阶保持
    tau = 1.2
    t_fut = t_now + tau
    true_fut = np.array([
        20.0 + r * math.cos(omega * t_fut),
        5.0 + r * math.sin(omega * t_fut),
        6.0 + 0.4 * math.sin(0.5 * t_fut),
    ])
    pred_pos, _ = est.predict_pv_ahead(tau)
    err_pred = float(np.linalg.norm(pred_pos[0:2] - true_fut[0:2]))
    err_zoh = float(np.linalg.norm(est.x[0:2] - true_fut[0:2]))
    assert err_pred < 1.2
    assert err_pred < err_zoh * 0.35


def test_mahalanobis_outlier_rejection_and_nan_defense():
    est = ManeuveringTargetEstimator()
    for k in range(20):
        t = k * 0.05
        est.update((10.0 + 3.0 * t, 2.0, 5.5), t, source="vision")

    assert not est.update((float("nan"), 2.0, 5.5), 1.05, source="vision")
    assert not est.update((10.0, float("inf"), 5.5), 1.05, source="external")

    # 注入单帧 80m 极端野值跳变：必须被马氏门控拦截，不污染内部位置估计
    pos_before = est.x[0:3].copy()
    accepted = est.update((95.0, -70.0, 40.0), 1.05, source="vision")
    assert not accepted
    assert float(np.linalg.norm(est.x[0:3] - pos_before)) < 0.5


def _bruteforce_assignment(cost: np.ndarray):
    n_r, n_c = cost.shape
    best_card = -1
    best_cost = float("inf")
    if n_r <= n_c:
        for cols in itertools.permutations(range(n_c), n_r):
            valid_pairs = [(r, cols[r]) for r in range(n_r) if np.isfinite(cost[r, cols[r]])]
            card = len(valid_pairs)
            tot = sum(float(cost[r, c]) for r, c in valid_pairs)
            if card > best_card or (card == best_card and tot < best_cost):
                best_card = card
                best_cost = tot
    else:
        for rows in itertools.permutations(range(n_r), n_c):
            valid_pairs = [(rows[c], c) for c in range(n_c) if np.isfinite(cost[rows[c], c])]
            card = len(valid_pairs)
            tot = sum(float(cost[r, c]) for r, c in valid_pairs)
            if card > best_card or (card == best_card and tot < best_cost):
                best_card = card
                best_cost = tot
    return best_card, best_cost


def test_hungarian_assignment_optimality_edge_cases_and_speed():
    assert "scipy" not in sys.modules

    # 空矩阵与全 inf 边界
    r0, c0, tot0 = hungarian_assignment(np.empty((0, 3)))
    assert len(r0) == 0 and len(c0) == 0 and tot0 == 0.0

    r_inf, c_inf, tot_inf = hungarian_assignment(np.full((3, 3), np.inf))
    assert len(r_inf) == 0 and len(c_inf) == 0 and tot_inf == 0.0

    # max_cost 门控与部分不可行边
    mat_gate = np.array([
        [1.0, 100.0, 2.0],
        [100.0, 100.0, 100.0],
        [3.0, -2.5, 4.0],
    ])
    rg, cg, tg = hungarian_assignment(mat_gate, max_cost=10.0)
    assert len(rg) == 2
    assert abs(tg - (-1.5)) < 1e-9  # (0,0)->1.0 + (2,1)->-2.5 = -1.5

    # 随机矩形阵与暴力穷举对比
    rng = np.random.default_rng(12345)
    for _ in range(150):
        nr = int(rng.integers(1, 6))
        nc = int(rng.integers(1, 6))
        m = rng.uniform(-5.0, 20.0, size=(nr, nc))
        # 随机注入部分 +inf
        mask_inf = rng.random((nr, nc)) < 0.25
        m[mask_inf] = np.inf
        ri, ci, tot = hungarian_assignment(m)
        assert len(set(ri.tolist())) == len(ri)
        assert len(set(ci.tolist())) == len(ci)
        exp_card, exp_cost = _bruteforce_assignment(m)
        assert len(ri) == exp_card
        if exp_card > 0:
            assert abs(tot - exp_cost) < 1e-8

    # K=10 延迟测试 (< 2.0 ms)
    mats = [rng.uniform(0.0, 50.0, size=(10, 10)) for _ in range(100)]
    t0 = time.perf_counter()
    for m in mats:
        hungarian_assignment(m)
    mean_ms = (time.perf_counter() - t0) * 1000.0 / len(mats)
    assert mean_ms < 2.0, f"Hungarian K=10 took {mean_ms:.3f} ms"
    assert "scipy" not in sys.modules


def _run_single_cv_rmse(meas_seq, dt: float) -> tuple:
    x = np.array([meas_seq[0][0], meas_seq[0][1], meas_seq[0][2], 0.0, 0.0, 0.0], dtype=float)
    P = np.diag([1.0, 1.0, 1.0, 9.0, 9.0, 4.0]).astype(float)
    F = np.eye(6, dtype=float)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    q = 0.25
    Q = np.zeros((6, 6), dtype=float)
    for i in range(3):
        Q[i, i] = (q ** 2) * (dt ** 4) / 4.0
        Q[i, i + 3] = Q[i + 3, i] = (q ** 2) * (dt ** 3) / 2.0
        Q[i + 3, i + 3] = (q ** 2) * (dt ** 2)
    R = np.eye(3, dtype=float) * (0.28 ** 2)
    H = np.eye(3, 6, dtype=float)
    filt_pos = []
    pred_1s = []
    for z in meas_seq:
        x = F @ x
        P = F @ P @ F.T + Q
        S = P[0:3, 0:3] + R
        K = P[:, 0:3] @ np.linalg.inv(S)
        x = x + K @ (z - x[0:3])
        P = (np.eye(6) - K @ H) @ P
        filt_pos.append(x[0:3].copy())
        pred_1s.append((x[0:3] + x[3:6] * 1.0).copy())
    return np.array(filt_pos), np.array(pred_1s)


def _run_single_singer_rmse(meas_seq, dt: float) -> tuple:
    x = np.array([meas_seq[0][0], meas_seq[0][1], meas_seq[0][2], 0, 0, 0, 0, 0, 0], dtype=float)
    P = np.diag([1.0, 1.0, 1.0, 9.0, 9.0, 4.0, 12.0, 12.0, 6.0]).astype(float)
    F, Q = _singer_blocks_param(dt, 1.35, 3.6)
    F1, _ = _singer_blocks_param(1.0, 1.35, 3.6)
    R = np.eye(3, dtype=float) * (0.28 ** 2)
    H = np.eye(3, 9, dtype=float)
    filt_pos = []
    pred_1s = []
    for z in meas_seq:
        x = F @ x
        P = F @ P @ F.T + Q
        S = P[0:3, 0:3] + R
        K = P[:, 0:3] @ np.linalg.inv(S)
        x = x + K @ (z - x[0:3])
        P = (np.eye(9) - K @ H) @ P
        filt_pos.append(x[0:3].copy())
        pred_1s.append((F1 @ x)[0:3].copy())
    return np.array(filt_pos), np.array(pred_1s)


def test_imm_mode_convergence_and_step_turn_rmse():
    # 1. 直线 CV 段 -> CV 概率 > 0.70
    trk_cv = IMMTrack(sigma_vision=0.20)
    for k in range(60):
        t = k * 0.05
        assert trk_cv.update((5.0 + 6.0 * t, -2.0 + 1.5 * t, 6.0), t, source="vision")
    probs_cv = trk_cv.imm_probs
    assert abs(sum(probs_cv.values()) - 1.0) < 1e-6
    assert probs_cv["CV"] > 0.70

    # 2. 恒定角速度转弯 -> CTRV 概率 > 0.65
    trk_ctrv = IMMTrack(sigma_vision=0.15)
    for k in range(80):
        t = k * 0.05
        trk_ctrv.update((20.0 + 15.0 * math.cos(0.30 * t), 5.0 + 15.0 * math.sin(0.30 * t), 6.0), t)
    probs_ctrv = trk_ctrv.imm_probs
    assert abs(sum(probs_ctrv.values()) - 1.0) < 1e-6
    assert probs_ctrv["CTRV"] > 0.65

    # 3. 急剧加速机动 -> JINK 概率 > 0.45
    trk_jink = IMMTrack(sigma_vision=0.15)
    for k in range(40):
        t = k * 0.05
        trk_jink.update((5.0 * t, 0.0, 5.0), t)
    for k in range(40, 65):
        t = k * 0.05
        dt_j = t - 2.0
        trk_jink.update((10.0 + 5.0 * dt_j + 4.5 * dt_j * dt_j, -3.5 * dt_j * dt_j, 5.0 + 1.5 * dt_j * dt_j), t)
    assert trk_jink.imm_probs["JINK"] > 0.45

    # 4. 急转弯跃变 10 种子蒙特卡洛：0 -> +1.3 -> -1.4 rad/s
    dt = 0.05
    n_steps = 160
    lookahead_steps = int(round(1.0 / dt))
    truth = []
    px, py, pz, vx, vy = 0.0, 0.0, 6.0, 8.0, 0.0
    for k in range(n_steps + lookahead_steps):
        if k < 40:
            w = 0.0
        elif k < 100:
            w = 1.3
        else:
            w = -1.4
        truth.append(np.array([px, py, pz], dtype=float))
        c, s = math.cos(w * dt), math.sin(w * dt)
        vx, vy = c * vx - s * vy, s * vx + c * vy
        px += vx * dt
        py += vy * dt
    truth_arr = np.array(truth)

    imm_f_errs, imm_p_errs = [], []
    cv_f_errs, cv_p_errs = [], []
    ca_f_errs, ca_p_errs = [], []
    for seed in range(10):
        rng = np.random.default_rng(100 + seed)
        meas = truth_arr[:n_steps] + rng.normal(0.0, 0.28, size=(n_steps, 3))
        trk = IMMTrack(sigma_vision=0.28)
        f_imm, p_imm = [], []
        for k in range(n_steps):
            trk.update(tuple(meas[k]), k * dt, source="vision")
            f_imm.append(trk.x[0:3].copy())
            pp, _ = trk.predict_pv_ahead(1.0)
            p_imm.append(pp)
        f_imm = np.array(f_imm)
        p_imm = np.array(p_imm)
        f_cv, p_cv = _run_single_cv_rmse(meas, dt)
        f_ca, p_ca = _run_single_singer_rmse(meas, dt)

        gt_f = truth_arr[15:n_steps]
        gt_p = truth_arr[15 + lookahead_steps : n_steps + lookahead_steps]
        imm_f_errs.append(float(np.sqrt(np.mean(np.sum((f_imm[15:] - gt_f) ** 2, axis=1)))))
        cv_f_errs.append(float(np.sqrt(np.mean(np.sum((f_cv[15:] - gt_f) ** 2, axis=1)))))
        ca_f_errs.append(float(np.sqrt(np.mean(np.sum((f_ca[15:] - gt_f) ** 2, axis=1)))))

        imm_p_errs.append(float(np.sqrt(np.mean(np.sum((p_imm[15:] - gt_p) ** 2, axis=1)))))
        cv_p_errs.append(float(np.sqrt(np.mean(np.sum((p_cv[15:] - gt_p) ** 2, axis=1)))))
        ca_p_errs.append(float(np.sqrt(np.mean(np.sum((p_ca[15:] - gt_p) ** 2, axis=1)))))

    assert np.mean(imm_f_errs) < np.mean(cv_f_errs)
    assert np.mean(imm_f_errs) < np.mean(ca_f_errs)
    assert np.mean(imm_p_errs) < np.mean(cv_p_errs)
    assert np.mean(imm_p_errs) < np.mean(ca_p_errs)


def test_imm_track_bank_lifecycle_clutter_crossing_coast_and_kill():
    bank = IMMTrackBank(sigma_vision=0.25, confirm_m=3, confirm_n=5, max_coast_sec=3.0)
    dt = 0.05

    # 三条交叉飞行航迹 + 第 2 帧注入单帧杂波
    id_history = {0: [], 1: [], 2: []}
    for k in range(50):
        t = k * dt
        p0 = (0.0 + 6.0 * t, 0.0 + 3.0 * t, 6.0)
        p1 = (0.0 + 6.0 * t, 7.5 - 3.0 * t, 6.5)  # 与 p0 在 t=1.25s 处横向交叉
        p2 = (20.0 - 4.0 * t, -4.0, 7.0)
        obs = [
            TrackObservation(p0, class_id="crimson_uav"),
            TrackObservation(p1, class_id="amber_uav"),
            TrackObservation(p2, class_id="slate_uav"),
        ]
        if k == 2:
            obs.append(TrackObservation((50.0, 50.0, 12.0), class_id="clutter"))
        bank.update_batch(obs, t, source="vision")

        if k >= 4:
            confirmed = bank.get_confirmed_tracks(t)
            assert len(confirmed) == 3  # 杂波从未确认
            for idx, pt in enumerate((p0, p1, p2)):
                nearest = min(confirmed, key=lambda tr: float(np.linalg.norm(tr.x[0:3] - np.array(pt))))
                id_history[idx].append(nearest.track_id)

    # 验证交叉前后无 ID 跳变
    for idx in range(3):
        assert len(set(id_history[idx])) == 1

    # 遮挡滑行与恢复测试：遮挡目标 2 长达 1.2s，然后恢复观测
    t_last = 49 * dt
    t2_id = id_history[2][-1]
    for step in range(1, 25):  # 1.2s
        t = t_last + step * dt
        p0 = (0.0 + 6.0 * t, 0.0 + 3.0 * t, 6.0)
        p1 = (0.0 + 6.0 * t, 7.5 - 3.0 * t, 6.5)
        bank.update_batch(
            [TrackObservation(p0, class_id="crimson_uav"), TrackObservation(p1, class_id="amber_uav")],
            t,
            source="vision",
        )
    assert bank.tracks[t2_id].status == "COASTING"
    p_coast_std = float(np.trace(bank.tracks[t2_id].P[0:3, 0:3]))
    assert p_coast_std > 1.0

    # 恢复目标 2 观测
    t_rec = t_last + 25 * dt
    p0_r = (0.0 + 6.0 * t_rec, 0.0 + 3.0 * t_rec, 6.0)
    p1_r = (0.0 + 6.0 * t_rec, 7.5 - 3.0 * t_rec, 6.5)
    p2_r = (20.0 - 4.0 * t_rec, -4.0, 7.0)
    bank.update_batch(
        [
            TrackObservation(p0_r, class_id="crimson_uav"),
            TrackObservation(p1_r, class_id="amber_uav"),
            TrackObservation(p2_r, class_id="slate_uav"),
        ],
        t_rec,
        source="vision",
    )
    assert bank.tracks[t2_id].status == "CONFIRMED"

    # 击毁目标 0（按空间位置）并验证防残骸原地重生
    killed = bank.mark_killed(target_id="tgt_0_sim", pos_enu=p0_r, now_sec=t_rec)
    assert len(killed) == 1
    bank.update_batch(
        [
            TrackObservation(p0_r, class_id="crimson_uav"),  # 原地残骸回波
            TrackObservation(p1_r, class_id="amber_uav"),
            TrackObservation(p2_r, class_id="slate_uav"),
        ],
        t_rec + 0.05,
        source="vision",
    )
    assert len(bank.get_active_tracks(include_tentative=True)) == 2

    # 超过 3.2s 完全失锁后航迹被清除
    bank.step_maintenance(t_rec + 3.5)
    assert len(bank.get_confirmed_tracks(t_rec + 3.5)) == 0


def test_legacy_estimator_initialized_property_invalidation():
    est = ManeuveringTargetEstimator()
    assert not est.initialized
    assert est.update((10.0, 2.0, 5.0), 1.0, source="vision")
    assert est.initialized
    assert est.is_valid(1.1)

    # 外部击毁清零 initialized 后必须同步失活并拒绝前向传播复活
    est.initialized = False
    assert not est.initialized
    assert not est._imm.initialized
    assert not est.is_valid(1.1)
    est.propagate_to(1.2)
    assert not est.initialized
    assert not est.is_valid(1.2)

    # 后续新观测到来可正常重新初始化
    assert est.update((11.0, 2.0, 5.0), 1.3, source="vision")
    assert est.initialized
    assert est.is_valid(1.3)


def test_nonfinite_timestamps_conf_and_entry_guards():
    trk = IMMTrack()
    assert trk.update((1.0, 2.0, 3.0), 1.0)
    assert trk.update((2.0, 2.0, 3.0), 1.1)

    for bad_t in (float("nan"), float("inf"), float("-inf")):
        trk.propagate_to(bad_t)
        assert math.isfinite(trk.last_state_time) and abs(trk.last_state_time - 1.1) < 1e-12
        assert not trk.is_valid(bad_t)
        p_a, v_a = trk.predict_pv_ahead(bad_t)
        assert np.allclose(p_a, trk.x[0:3]) and np.allclose(v_a, trk.x[3:6])
        p_k, v_k, a_k = trk.get_kinematics_at(bad_t)
        assert np.allclose(p_k, trk.x[0:3]) and np.allclose(v_k, trk.x[3:6]) and np.allclose(a_k, trk.x[6:9])
        assert not trk.update((2.1, 2.0, 3.0), 1.15, conf=bad_t)
        assert np.all(np.isfinite(trk.x)) and np.all(np.isfinite(trk.P))

    bank = IMMTrackBank(confirm_m=1, confirm_n=1)
    bank.update_batch([TrackObservation((0.0, 0.0, 5.0), target_id="t1")], 1.0)
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        bank.update_batch([TrackObservation((1.0, 0.0, 5.0), target_id="t1", timestamp=bad_val)], 1.05)
        bank.update_batch([{"x": 1.0, "y": 0.0, "z": 5.0, "target_id": "t1", "stamp": bad_val}], 1.05)
        bank.update_batch([TrackObservation((1.0, 0.0, 5.0), target_id="t1", conf=bad_val)], 1.05)
        bank.update_batch([{"x": 1.0, "y": 0.0, "z": 5.0, "target_id": "t1", "conf": bad_val}], 1.05)
        bank.step_maintenance(bad_val)
    assert len(bank.tracks) == 1
    t1 = next(iter(bank.tracks.values()))
    assert t1.update_count == 1
    assert abs(t1.last_update_time - 1.0) < 1e-12
    assert math.isfinite(t1.last_state_time) and abs(t1.last_state_time - 1.05) < 1e-12
    assert bank.get_confirmed_tracks(float("nan")) == []
    assert bank.get_confirmed_tracks(float("-inf")) == []


def test_coast_covariance_psd_and_hungarian_large_scale():
    trk = IMMTrack(sigma_vision=0.25)
    for k in range(10):
        t = k * 0.05
        assert trk.update((5.0 * t, 0.0, 5.0), t, source="vision")
    # 增大速度方差以确保各模型位置方差在 2.8s 滑行后均触发 100.0 上限夹紧
    trk.P_cv[3, 3] = 30.0
    trk.P_ctrv[3, 3] = 30.0
    trk.P_jink[3, 3] = 30.0
    trk.propagate_to(9 * 0.05 + 2.8, coast_damping=True)
    for Pm in (trk.P_cv, trk.P_ctrv, trk.P_jink, trk.P):
        assert np.allclose(Pm, Pm.T, atol=1e-10)
        min_eig = float(np.min(np.linalg.eigvalsh(0.5 * (Pm + Pm.T))))
        assert min_eig > 0.0
    for Pm in (trk.P_cv, trk.P_ctrv, trk.P_jink):
        assert max(float(Pm[0, 0]), float(Pm[1, 1]), float(Pm[2, 2])) <= 100.0 + 1e-9

    # 极端量级 (>1e30) 成本矩阵不得受旧 inf_sentinel=1e30 限制
    mat_huge = np.array(
        [
            [0.0, 2e31, np.inf],
            [2e31, 0.0, np.inf],
            [np.inf, 2e31, 5e30],
        ],
        dtype=float,
    )
    ri, ci, tot = hungarian_assignment(mat_huge)
    assert list(ri) == [0, 1, 2]
    assert list(ci) == [0, 1, 2]
    assert abs(tot - 5e30) < 1e20


def test_kill_zone_radius_inferred_timestamp_and_no_collateral_kill():
    # 1. kill_radius_m 生效 + now_sec=None 从活跃航迹推断时间戳
    bank = IMMTrackBank(
        sigma_vision=0.25,
        confirm_m=2,
        confirm_n=3,
        kill_radius_m=3.5,
        kill_suppression_sec=2.5,
    )
    for k in range(4):
        t = 20.0 + k * 0.05
        bank.update_batch(
            [
                TrackObservation((10.0, 0.0, 6.0), target_id="tgt_a", class_id="crimson_uav"),
                TrackObservation((50.0, 0.0, 6.0), class_id="slate_uav"),
            ],
            t,
        )
    killed_a = bank.mark_killed(target_id="tgt_a", pos_enu=(10.0, 0.0, 6.0), now_sec=20.15)
    assert len(killed_a) == 1

    # 2.6m (< kill_radius_m=3.5m, > 旧硬编码 2.0m) 处纯视觉残骸不得重生
    for step in range(1, 5):
        bank.update_batch(
            [
                TrackObservation((12.6, 0.0, 6.0), class_id="crimson_uav"),
                TrackObservation((50.0, 0.0, 6.0), class_id="slate_uav"),
            ],
            20.15 + step * 0.05,
        )
    assert len(bank.get_active_tracks(include_tentative=True)) == 1

    # now_sec=None 时必须从活跃航迹推断 t=20.35（而非 0.0），从而在 t=20.40 抑制残骸重生
    killed_vis = bank.mark_killed(pos_enu=(50.0, 0.0, 6.0))
    assert len(killed_vis) == 1
    bank.update_batch([TrackObservation((50.2, 0.1, 6.0), class_id="slate_uav")], 20.40)
    assert len(bank.get_active_tracks(include_tentative=True)) == 0

    # 2. 显式指定 target_id 时不得因 pos_enu 邻近误杀具有不同 external_id 的邻近航迹
    bank2 = IMMTrackBank(confirm_m=2, confirm_n=3, kill_radius_m=3.5)
    for k in range(4):
        t = k * 0.05
        bank2.update_batch(
            [
                TrackObservation((20.0 + k * 0.1, -0.5 + k * 0.1, 6.0), target_id="tgt_alpha"),
                TrackObservation((20.0 + k * 0.1, 1.5 - k * 0.1, 6.0), target_id="tgt_beta"),
                TrackObservation((40.0, 0.0, 6.0)),  # 纯视觉无 external_id
            ],
            t,
        )
    killed_alpha = bank2.mark_killed(target_id="tgt_alpha", pos_enu=(20.3, -0.2, 6.0), now_sec=0.20)
    assert len(killed_alpha) == 1
    rem_ext_ids = sorted(tr.external_id for tr in bank2.get_confirmed_tracks(0.20))
    assert rem_ext_ids == ["", "tgt_beta"]

    # 无 external_id 的纯视觉航迹仍允许按 pos_enu 匹配击毁
    killed_vis2 = bank2.mark_killed(target_id="tgt_vis_sim", pos_enu=(40.0, 0.0, 6.0), now_sec=0.25)
    assert len(killed_vis2) == 1
    assert [tr.external_id for tr in bank2.get_confirmed_tracks(0.25)] == ["tgt_beta"]
