"""P3 场景回归测试：A/B/C 三个复杂场景的语义与指标边界。

把 ``simulate_p3.py`` 的场景固化进 pytest：失败时 pytest 返回非 0，
CI 可以直接用。指标分三个口径，图表不参与断言：

- ``p3_jump``：纯 P3 输出 ``PlanResult.point``；
- ``plan_jump``：适配层准备发布点（含 ATTACK 保持点）；
- ``published_jump``：中枢实际指令（TRACK 会叠加视觉/环绕）。

运行：
    PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
        src/algorithm_pkg/test
"""

import json

import pytest

from algorithm_pkg.plan_params import PlanParams

from simulate_p3 import (
    build_scenarios,
    check_scenario,
    main,
    simulate,
    summarize,
)

SCENARIOS = {scenario["name"]: scenario for scenario in build_scenarios()}


def run_scenario(name: str):
    """按场景标定参数跑一遍，返回 (scenario, records, summary)。"""
    scenario = SCENARIOS[name]
    params = PlanParams(max_step=scenario["max_step"])
    records = simulate(
        name=scenario["name"],
        action=scenario["action"],
        params=params,
        own_start=scenario["own_start"],
        duration=scenario["duration"],
        target_fn=scenario["target_fn"],
        visible_fn=scenario["visible_fn"],
        ego_dropout_fn=scenario["ego_dropout_fn"],
    )
    return scenario, records, summarize(name, records)


@pytest.mark.parametrize("name", ["A", "B", "C"])
def test_scenario_expectations(name):
    """场景 X：语义迁移、跳变上限与中枢保留时长符合预期。"""
    scenario, records, summary = run_scenario(name)
    failures = check_scenario(scenario, records, summary)
    assert not failures, "\n".join(failures)


def test_main_exit_code_and_json(tmp_path):
    """CLI：全 PASS 返回 0，JSON 含场景摘要与判定。"""
    out = tmp_path / "p3.json"
    code = main(["--scenario", "A", "--no-plot", "--json", str(out)])
    assert code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["scenarios"][0]["name"] == "A"
    assert payload["scenarios"][0]["failures"] == []


def test_default_params_are_observed_only():
    """默认 max_step=None 只观察：C 的纯 P3 跳变明显大于 0.62m。"""
    scenario = SCENARIOS["C"]
    records = simulate(
        name=scenario["name"],
        action=scenario["action"],
        params=PlanParams(),
        own_start=scenario["own_start"],
        duration=scenario["duration"],
        target_fn=scenario["target_fn"],
        visible_fn=scenario["visible_fn"],
        ego_dropout_fn=scenario["ego_dropout_fn"],
    )
    summary = summarize("C", records)
    assert summary["p3_jump"] > scenario["max_step"]
