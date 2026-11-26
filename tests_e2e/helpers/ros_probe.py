"""GPU-guarded command execution, colcon test runner, and ROS 2 topic probe helpers."""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .hardware_git_guard import check_cuda_env_excludes_gpu3

DEFAULT_ROS2_WS = Path("/home/hkw/flying/ros2_ws")
ENV_SH_PATH = DEFAULT_ROS2_WS / "src" / "flight_ctrl_pkg" / "scripts" / "env.sh"
INSTALL_SETUP_PATH = DEFAULT_ROS2_WS / "install" / "setup.bash"


@dataclass
class CommandResult:
    """Result of an external command invocation."""

    returncode: int
    stdout: str
    stderr: str
    elapsed_s: float
    timed_out: bool = False


@dataclass
class ColconTestResult:
    """Structured summary of colcon test execution."""

    ok: bool
    total: int
    errors: int
    failures: int
    skipped: int
    raw_summary: str
    returncode: int = 0


def _build_guarded_env(extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if not check_cuda_env_excludes_gpu3(env):
        env["CUDA_VISIBLE_DEVICES"] = "2"
    if extra_env:
        env.update(extra_env)
    return env


def run_command_with_gpu_guard(
    cmd: Union[Sequence[str], str],
    cwd: Optional[Union[str, Path]] = None,
    timeout_s: float = 60.0,
    extra_env: Optional[Dict[str, str]] = None,
    shell: bool = False,
    source_env: bool = False,
) -> CommandResult:
    """Execute a subprocess with PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 and GPU-3 excluded."""
    env = _build_guarded_env(extra_env)
    work_dir = str(cwd) if cwd is not None else None

    if source_env:
        if isinstance(cmd, str):
            cmd_str = cmd
        else:
            cmd_str = shlex.join([str(c) for c in cmd])
        source_prefix = (
            f"source {shlex.quote(str(ENV_SH_PATH))} >/dev/null 2>&1 && "
            f"source {shlex.quote(str(INSTALL_SETUP_PATH))} >/dev/null 2>&1 && "
        )
        exec_cmd: Union[List[str], str] = ["bash", "-lc", source_prefix + cmd_str]
        use_shell = False
    else:
        exec_cmd = cmd if isinstance(cmd, str) else [str(c) for c in cmd]
        use_shell = shell or isinstance(cmd, str)

    t0 = time.monotonic()
    proc = subprocess.Popen(
        exec_cmd,
        cwd=work_dir,
        env=env,
        shell=use_shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        elapsed = time.monotonic() - t0
        return CommandResult(
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            elapsed_s=elapsed,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        elapsed = time.monotonic() - t0
        return CommandResult(
            returncode=-9,
            stdout=stdout or "",
            stderr=(stderr or "") + f"\nCommand timed out after {timeout_s}s",
            elapsed_s=elapsed,
            timed_out=True,
        )


def _parse_colcon_summary(text: str) -> tuple[int, int, int, int]:
    match = re.search(
        r"Summary:\s*(\d+)\s+tests?,\s*(\d+)\s+errors?,\s*(\d+)\s+failures?,\s*(\d+)\s+skipped",
        text,
    )
    if not match:
        return (0, 0, 0, 0)
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
    )


def run_colcon_test(
    ws_dir: Union[str, Path] = DEFAULT_ROS2_WS,
    packages: Optional[Sequence[str]] = None,
    timeout_s: float = 300.0,
    result_only: bool = False,
) -> ColconTestResult:
    """Run colcon test (or colcon test-result) under GPU-3 guard and parse summary."""
    ws = Path(ws_dir)
    if not result_only:
        test_cmd = ["colcon", "test"]
        if packages:
            test_cmd.extend(["--packages-select", *[str(p) for p in packages]])
        test_res = run_command_with_gpu_guard(
            test_cmd, cwd=ws, timeout_s=timeout_s, source_env=True
        )
        if test_res.timed_out:
            return ColconTestResult(
                ok=False,
                total=0,
                errors=1,
                failures=0,
                skipped=0,
                raw_summary=test_res.stderr,
                returncode=test_res.returncode,
            )

    res_proc = run_command_with_gpu_guard(
        ["colcon", "test-result", "--verbose"],
        cwd=ws,
        timeout_s=min(60.0, timeout_s),
        source_env=True,
    )
    combined = (res_proc.stdout + "\n" + res_proc.stderr).strip()
    total, errors, failures, skipped = _parse_colcon_summary(combined)
    ok = (
        res_proc.returncode == 0
        and total > 0
        and errors == 0
        and failures == 0
        and skipped == 0
    )
    return ColconTestResult(
        ok=ok,
        total=total,
        errors=errors,
        failures=failures,
        skipped=skipped,
        raw_summary=combined,
        returncode=res_proc.returncode,
    )


def probe_ros_topics(timeout_s: float = 5.0) -> List[Tuple[str, str]]:
    """Return list of (topic_name, msg_type) discovered via `ros2 topic list -t`."""
    res = run_command_with_gpu_guard(
        ["ros2", "topic", "list", "-t"],
        timeout_s=timeout_s,
        source_env=True,
    )
    if res.returncode != 0:
        return []
    topics: List[Tuple[str, str]] = []
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(/\S+)\s+\[([^\]]+)\]$", line)
        if m:
            topics.append((m.group(1), m.group(2)))
        elif line.startswith("/"):
            topics.append((line.split()[0], ""))
    return topics


def echo_topic_once(
    topic: str,
    msg_type: Optional[str] = None,
    timeout_s: float = 10.0,
) -> CommandResult:
    """Echo a single ROS 2 topic message with timeout and GPU-3 guard."""
    cmd = ["ros2", "topic", "echo", "--once", topic]
    if msg_type:
        cmd.append(msg_type)
    return run_command_with_gpu_guard(cmd, timeout_s=timeout_s, source_env=True)
