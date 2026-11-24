"""HOVER 固定点选择测试；不启动 ROS。"""

import pytest

from center_node_pkg.hover_target import resolve_hover_point

OWN = (10.0, 20.0, 5.0)
PLAN = (30.0, 40.0, 6.0)


def test_fresh_hover_plan_is_used():
    """场景：规划点在 0.2 s 前收到，动作是 HOVER。

    预期：用规划点，source=plan。
    """
    point, source = resolve_hover_point(OWN, PLAN, "HOVER", 0.2, 0.8)
    assert point == pytest.approx(PLAN)
    assert source == "plan"



def test_stale_plan_falls_back_to_own():
    """场景：规划点已收到 1.0 s，超过 0.8 s。

    预期：回退己方当前位置，source=own。
    """
    point, source = resolve_hover_point(OWN, PLAN, "HOVER", 1.0, 0.8)
    assert point == pytest.approx(OWN)
    assert source == "own"



def test_non_hover_plan_is_ignored():
    """场景：最近的规划点动作是 TRACK。

    预期：不拿来悬停，回退当前位置。
    """
    point, source = resolve_hover_point(OWN, PLAN, "TRACK", 0.1, 0.8)
    assert point == pytest.approx(OWN)
    assert source == "own"



def test_missing_plan_falls_back_to_own():
    """场景：没有规划点。

    预期：回退当前位置。
    """
    point, source = resolve_hover_point(OWN, None, "", 0.0, 0.8)
    assert point == pytest.approx(OWN)
    assert source == "own"



def test_action_name_is_normalized():
    """场景：动作写成小写和带空格。

    预期：仍识别为 HOVER。
    """
    point, source = resolve_hover_point(OWN, PLAN, " hover ", 0.1, 0.8)
    assert point == pytest.approx(PLAN)
    assert source == "plan"



def test_non_finite_plan_and_negative_age_fall_back():
    """场景：规划点含 NaN，或收到时间为负。

    预期：都回退当前位置，不能把坏点发给飞控。
    """
    bad = (float("nan"), 0.0, 5.0)
    point, source = resolve_hover_point(OWN, bad, "HOVER", 0.1, 0.8)
    assert point == pytest.approx(OWN)
    assert source == "own"

    point, source = resolve_hover_point(OWN, PLAN, "HOVER", -0.1, 0.8)
    assert point == pytest.approx(OWN)
    assert source == "own"

