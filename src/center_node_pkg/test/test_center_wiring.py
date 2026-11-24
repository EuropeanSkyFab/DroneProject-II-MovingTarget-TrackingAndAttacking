"""中枢关键接线检查。

这一层不是行为测试，只防止 HOVER/视觉接线被整段回退。行为验证在
Gazebo 多帧场景里做，见 PR 说明。
"""

from pathlib import Path

CENTER = (
    Path(__file__).resolve().parents[1]
    / "center_node_pkg"
    / "center_node.py"
)


def test_hover_branch_uses_plan_anchor():
    """HOVER 必须读规划点动作标签并调用 resolve_hover_point。"""
    text = CENTER.read_text(encoding="utf-8")
    assert "resolve_hover_point(" in text
    assert "latest_plan_actions" in text
    assert 'current_mode not in ["TAKEOFF", "RETURN"]' in text


def test_center_subscribes_covariance_vision():
    """中枢必须订阅新视觉话题。"""
    text = CENTER.read_text(encoding="utf-8")
    assert "PoseWithCovarianceStamped" in text
    assert "/vision/target_pose_cov" in text
    assert "validate_vision_sample(" in text
