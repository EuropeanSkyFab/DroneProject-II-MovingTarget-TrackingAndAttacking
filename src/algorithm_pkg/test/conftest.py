"""P1/P2 测试的终端摘要。

每个测试调用 measured() 记录“实际测出了什么”；全部跑完后，在终端末尾
打印“场景 / 预期 / 实测”三行，避免只有 “N passed” 而看不出每个测试
到底在验证什么。
"""

from __future__ import annotations

import inspect

import pytest

# pytest 会在一个进程里跑完整个测试会话，这里用模块级列表收集摘要。
_RESULTS: list[tuple[str, str, str, str]] = []


def pytest_sessionstart(session):
    """每轮测试开始时清空上一轮的结果。"""
    _RESULTS.clear()


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
    if not _RESULTS:
        return
    terminalreporter.write_sep("=", f"算法实测结果（{len(_RESULTS)} 项）")
    for name, scenario, expected, measured_text in _RESULTS:
        terminalreporter.write_line(f"- {name}")
        if scenario:
            terminalreporter.write_line(f"  {scenario}")
        if expected:
            terminalreporter.write_line(f"  {expected}")
        terminalreporter.write_line(f"  实测：{measured_text}")
