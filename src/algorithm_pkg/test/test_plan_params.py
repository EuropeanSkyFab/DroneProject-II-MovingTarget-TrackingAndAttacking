"""P2 参数测试：确认“按动作选速度”和错误配置会被拦住。

运行：
    PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
        src/algorithm_pkg/test
"""

import math

import pytest

from algorithm_pkg.plan_params import PlanParams


def test_default_params_match_three_speed_levels(measured):
    """场景：没有传任何参数，使用 P2 默认值。

    操作：构造 PlanParams()，读取三个速度上限。
    输入：无。
    预期：兜底 6.0、INTERCEPT 7.6、ATTACK 12.4 m/s。
    """
    params = PlanParams()

    assert params.v_max == pytest.approx(6.0)
    assert params.v_max_intercept == pytest.approx(7.6)
    assert params.v_max_attack == pytest.approx(12.4)

    measured(
        f"v_max={params.v_max:.1f}, "
        f"v_intercept={params.v_max_intercept:.1f}, "
        f"v_attack={params.v_max_attack:.1f}"
    )


def test_v_max_for_selects_by_action(measured):
    """场景：同一个参数对象，不同动作要拿不同速度。

    操作：调用 v_max_for()，分别传入 TRACK/INTERCEPT/ATTACK/unknown。
    输入：默认 PlanParams。
    预期：TRACK=6.0，INTERCEPT=7.6，ATTACK=12.4，unknown=6.0 兜底。
    """
    params = PlanParams()

    assert params.v_max_for("TRACK") == pytest.approx(6.0)
    assert params.v_max_for("INTERCEPT") == pytest.approx(7.6)
    assert params.v_max_for("ATTACK") == pytest.approx(12.4)
    assert params.v_max_for("something") == pytest.approx(6.0)

    measured(
        f"TRACK={params.v_max_for('TRACK'):.1f}, "
        f"INTERCEPT={params.v_max_for('INTERCEPT'):.1f}, "
        f"ATTACK={params.v_max_for('ATTACK'):.1f}"
    )


def test_v_max_for_is_case_and_space_insensitive(measured):
    """场景：上游传了小写和空格的模式名。

    操作：调用 v_max_for(" intercept ")。
    输入：小写且带空格的 INTERCEPT。
    预期：仍返回 INTERCEPT 的 7.6 m/s。
    """
    params = PlanParams()

    assert params.v_max_for(" intercept ") == pytest.approx(7.6)

    measured(f"v_max_for(' intercept ')={params.v_max_for(' intercept '):.1f}")


def test_custom_params_are_kept(measured):
    """场景：联调时想临时改速度和最大预测年龄。

    操作：构造 PlanParams(v_max_intercept=9.0, max_pred_age=0.3)。
    输入：自定义参数。
    预期：读回 9.0 和 0.3。
    """
    params = PlanParams(v_max_intercept=9.0, max_pred_age=0.3)

    assert params.v_max_for("INTERCEPT") == pytest.approx(9.0)
    assert params.max_pred_age == pytest.approx(0.3)

    measured(
        f"v_intercept={params.v_max_for('INTERCEPT'):.1f}, "
        f"max_pred_age={params.max_pred_age:.2f}s"
    )


def test_p3_defaults_are_documented_initial_values(measured):
    """场景：P3 参数先用一套可跑、可解释的初值，联调后再标定。

    操作：构造 PlanParams()，读取 P3 字段。
    输入：默认参数。
    预期：t_extrap_max=1.0、n_enter=2、n_exit=1、n_recover=3、
          max_step_recover=0.2、n_recover_step=5、频率检测关闭、
          hold_point_source=last_valid、ego_timeout=1.0。
    """
    params = PlanParams()

    assert params.t_extrap_max == pytest.approx(1.0)
    assert params.n_enter == 2
    assert params.n_exit == 1
    assert params.n_recover == 3
    assert params.max_step_recover == pytest.approx(0.2)
    assert params.n_recover_step == 5
    assert params.freq_anomaly_enabled is False
    assert params.freq_anomaly_threshold == pytest.approx(0.5)
    assert params.freq_window_size == 20
    assert params.hold_point_source == "last_valid"
    assert params.ego_timeout == pytest.approx(1.0)
    assert params.target_min_alt == pytest.approx(0.0)

    measured(
        f"t_extrap_max={params.t_extrap_max:.1f}s, n_enter={params.n_enter}, "
        f"n_exit={params.n_exit}, n_recover={params.n_recover}, "
        f"max_step_recover={params.max_step_recover:.2f}, "
        f"freq_enabled={params.freq_anomaly_enabled}"
    )


def test_max_step_none_means_no_limit(measured):
    """场景：默认不限制单帧位移，交给飞控自己的步长限制。

    操作：构造 PlanParams()，读取 max_step。
    输入：默认参数。
    预期：max_step 是 None。
    """
    params = PlanParams()

    assert params.max_step is None

    measured(f"max_step={params.max_step}")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"v_max": 0.0},
        {"v_max_intercept": 0.0},
        {"v_max_attack": 0.0},
        {"t_go_max": 0.0},
        {"min_distance": -1.0},
        {"d_tail": -1.0},
        {"max_xy": 0.0},
        {"max_pred_age": 0.0},
        {"max_step": 0.0},
        {"max_alt": 4.0},
        {"t_extrap_max": 0.0},
        {"n_enter": 0},
        {"n_exit": 0},
        {"n_recover": 0},
        {"n_recover_step": 0},
        {"max_step_recover": 0.0},
        {"freq_anomaly_threshold": 1.5},
        {"freq_window_size": 1},
        {"hold_point_source": "somewhere"},
        {"ego_timeout": 0.0},
        {"target_min_alt": -1.0},
        {"target_min_alt": 60.0},
        {"v_max": math.nan},
        {"t_go_max": math.inf},
        {"min_alt": math.nan},
        {"max_xy": math.inf},
        {"max_pred_age": math.nan},
        {"t_extrap_max": math.inf},
        {"max_step_recover": math.inf},
        {"freq_anomaly_threshold": math.nan},
        {"n_enter": 2.5},
        {"freq_window_size": 3.5},
    ],
)
def test_invalid_params_raise_value_error(kwargs, measured):
    """场景：配置里出现 0、负数或高度矛盾，不能悄悄继续飞。

    操作：用非法参数构造 PlanParams()。
    输入：kwargs 指定一个非法字段。
    预期：抛出 ValueError。
    """
    with pytest.raises(ValueError):
        PlanParams(**kwargs)

    measured(f"kwargs={kwargs} -> ValueError")
