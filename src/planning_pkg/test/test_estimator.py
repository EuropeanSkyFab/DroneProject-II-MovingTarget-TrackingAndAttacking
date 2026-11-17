# -*- coding: utf-8 -*-
"""planning_pkg.estimator 单元测试：收敛性、野值门控、曲率外推与非法输入防御。"""

import math
import numpy as np

from planning_pkg.estimator import ManeuveringTargetEstimator


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
