"""pytest 公共 fixture：measured 摘要 + scenario_chart 证据图。

unit 测试只写终端摘要；scenario/sim 测试额外把逐帧记录交给
scenario_chart，用例结束后自动出 PNG，避免“测试完了但没有图”。
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

# 让 support 包在 pytest 直接运行时也能导入。
sys.path.insert(0, str(Path(__file__).parent))

# pytest 会在一个进程里跑完整个测试会话，这里用模块级列表收集摘要。
_RESULTS: list[tuple[str, str, str, str]] = []
_CHARTS: list[Path] = []


def pytest_sessionstart(session):
    """每轮测试开始时清空上一轮的结果。"""
    _RESULTS.clear()
    _CHARTS.clear()


_FIELD_PREFIXES = ("场景：", "操作：", "输入：", "预期：")


def _doc_block(doc: str, prefix: str) -> str:
    """取 docstring 中以 ``prefix`` 开头的段落（含换行续写）。"""
    lines = doc.splitlines()
    for index, line in enumerate(lines):
        if not line.strip().startswith(prefix):
            continue

        parts = [line.strip()]
        for following in lines[index + 1:]:
            text = following.strip()
            if not text:
                break
            # 下一个字段开始，说明当前段落结束。
            if any(text.startswith(item) for item in _FIELD_PREFIXES):
                break
            parts.append(text)
        return " ".join(parts)
    return ""


def pytest_addoption(parser):
    """给情景测试图指定输出目录；不传时用环境变量或 /tmp。"""
    parser.addoption(
        "--scenario-chart-dir",
        action="store",
        default=None,
        help="情景测试 PNG 输出目录（默认 /tmp/algorithm_scenario_charts）",
    )


@pytest.fixture
def scenario_chart(request):
    """记录情景帧，用例结束自动出图；没有记录则测试失败。"""
    from support.scenario_chart import ScenarioChart

    out_dir = request.config.getoption("--scenario-chart-dir")
    chart = ScenarioChart(request.node.name, out_dir)
    yield chart
    for path in chart.save():
        _CHARTS.append(path)


@pytest.fixture
def measured(request):
    """返回一个函数：测试成功时把“测出的值”写入摘要。"""

    def _measured(text: str) -> None:
        # 测试函数的 docstring 已经写清场景和预期，直接复用到终端摘要。
        function = getattr(request.node, "function", None)
        doc = inspect.getdoc(function) if function is not None else ""
        doc = doc or ""
        _RESULTS.append(
            (
                request.node.name,
                _doc_block(doc, "场景"),
                _doc_block(doc, "预期"),
                text,
            )
        )

    return _measured


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """所有测试结束后，按“场景 / 预期 / 实测”逐条打印。"""
    if _RESULTS:
        terminalreporter.write_sep("=", f"算法实测结果（{len(_RESULTS)} 项）")
        for name, scenario, expected, measured_text in _RESULTS:
            terminalreporter.write_line(f"- {name}")
            if scenario:
                terminalreporter.write_line(f"  {scenario}")
            if expected:
                terminalreporter.write_line(f"  {expected}")
            terminalreporter.write_line(f"  实测：{measured_text}")
    if _CHARTS:
        terminalreporter.write_sep("=", f"情景证据图（{len(_CHARTS)} 张）")
        for path in _CHARTS:
            terminalreporter.write_line(f"- {path}")
