"""生产 launch 的关键参数检查：防止代码默认值和交付值再次分叉。"""

from pathlib import Path

LAUNCH = (
    Path(__file__).resolve().parents[2] / "launch" / "algorithm.launch.py"
)


def _text() -> str:
    return LAUNCH.read_text(encoding="utf-8")


def test_launch_uses_per_action_step_and_w11_speed(measured):
    """场景：读取生产 launch。

    预期：速度用 4.50/7.60/12.40；单帧位移按动作 0.14/0.38/0.62；
    兜底 0.30；不再用单值 0.74。
    """
    text = _text()
    for item in (
        "{'v_max': 4.50}",
        "{'v_max_intercept': 7.60}",
        "{'v_max_attack': 12.40}",
        "{'max_step': 0.30}",
        "{'max_step_track': 0.14}",
        "{'max_step_intercept': 0.38}",
        "{'max_step_attack': 0.62}",
    ):
        assert item in text, item

    measured("launch：4.50/7.60/12.40；步长 0.14/0.38/0.62；兜底 0.30")


def test_launch_prediction_threshold_is_045(measured):
    """场景：读取生产 launch 的预测阈值。

    预期：0.45 s；0.5 s 边界在 Gazebo 实测里会抖进 L1。
    """
    text = _text()
    assert "{'max_pred_age': 0.45}" in text

    measured("launch：max_pred_age=0.45 s")
