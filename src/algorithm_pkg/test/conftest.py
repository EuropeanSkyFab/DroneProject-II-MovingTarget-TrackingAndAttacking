"""P1 测试的终端摘要。

每个测试调用 measured() 记录“实际测出了什么”；
全部跑完后，在终端末尾统一打印，避免只有“20 passed”而看不出覆盖内容。
"""

from __future__ import annotations

import pytest

# pytest 会在一个进程里跑完整个测试会话，这里用模块级列表收集摘要。
_RESULTS: list[tuple[str, str]] = []


def pytest_sessionstart(session):
    """每轮测试开始时清空上一轮的结果。"""
    _RESULTS.clear()


@pytest.fixture
def measured(request):
    """返回一个函数：测试成功时把“测出的值”写入摘要。"""

    def _measured(text: str) -> None:
        # 只保留测试函数名，终端里比完整 nodeid 更好读。
        _RESULTS.append((request.node.name, text))

    return _measured


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """所有测试结束后，打印“P1 实测结果”表。"""
    if not _RESULTS:
        return
    terminalreporter.write_sep("=", f"P1 实测结果（{len(_RESULTS)} 项）")
    for nodeid, measured_text in _RESULTS:
        terminalreporter.write_line(f"- {nodeid}: {measured_text}")
