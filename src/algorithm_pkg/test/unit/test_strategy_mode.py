"""策略指令到模式的纯函数测试：RESUME_PATROL 不得让算法静默。"""

from algorithm_pkg.plan_types import mode_after_command


def test_resume_patrol_keeps_current_mode(measured):
    """场景：当前 TRACK，收到 RESUME_PATROL。

    预期：模式仍是 TRACK；中枢 main 对 RESUME_PATROL 也是保持原态，
    两边不能一边继续飞、一边停发点。
    """
    assert mode_after_command("TRACK", "RESUME_PATROL") == "TRACK"
    assert mode_after_command("ATTACK", " resume_patrol ") == "ATTACK"

    measured("TRACK + RESUME_PATROL -> TRACK；ATTACK -> ATTACK")


def test_normal_action_switches_mode(measured):
    """场景：当前 TRACK，收到 INTERCEPT 或小写 hover。

    预期：模式按归一化后的动作切换。
    """
    assert mode_after_command("TRACK", "INTERCEPT") == "INTERCEPT"
    assert mode_after_command("TRACK", " hover ") == "HOVER"

    measured("TRACK + INTERCEPT -> INTERCEPT；TRACK + hover -> HOVER")
