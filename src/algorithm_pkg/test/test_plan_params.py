"""P2 参数测试：确认“按动作选速度”和错误配置会被拦住。

运行：
    PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
        src/algorithm_pkg/test
"""

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
