"""Hardware GPU-3 isolation and Git rollback anchor verification helpers."""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Set, Union

DEFAULT_REPO_DIR = Path("/home/hkw/flying/ros2_ws")
DEFAULT_EXPECTED_ANCHORS: Dict[str, str] = {
    "main": "ba401d5",
    "feat/flight-ctrl-pkg": "0157ecc",
}

DEFAULT_GPU3_BUS_SUFFIX = "d9:00.0"
DEFAULT_GPU3_UUID = "gpu-617ae266-eed3-ef41-ae0f-5f25c730f5d5"
KNOWN_NON_GPU3_UUIDS: Set[str] = {
    "gpu-b5e964a1-8266-9595-863d-e26136c268ca",
    "gpu-a0491ff7-3f45-ed3b-b84a-411630af9bde",
    "gpu-5643e739-f46d-feb6-e907-3aa2c1ad4963",
}
KNOWN_NON_GPU3_BUS_SUFFIXES: Set[str] = {"3b:00.0", "5e:00.0", "b1:00.0"}

SIM_RENDER_CMD_RE = re.compile(
    r"(gz|gazebo|ogre|python|ffmpeg|px4|rviz|ros)", re.IGNORECASE
)


@dataclass
class Gpu3Status:
    """Result of GPU-3 zero-occupancy check."""

    ok: bool
    gpu3_pids: List[int]
    gpu3_mem_mib: int
    raw_output: str
    external_pids: List[int] = field(default_factory=list)
    notes: str = ""


@dataclass
class OgreLogAudit:
    """Result of Ogre2 EGL log GPU-3 exclusion audit."""

    ok: bool
    detected_devices: List[str]
    violations: List[str]
    num_egl_devices: Optional[int] = None


@dataclass
class GitAnchorResult:
    """Result of Git rollback anchor verification."""

    ok: bool
    actual: Dict[str, str]
    errors: List[str]


def _normalize_bus_suffix(bus_id: str) -> str:
    cleaned = bus_id.strip().lower()
    parts = cleaned.split(":")
    if len(parts) >= 2:
        return ":".join(parts[-2:])
    return cleaned


def _parse_mem_mib(raw_mem: str) -> int:
    match = re.search(r"(\d+)", raw_mem)
    return int(match.group(1)) if match else 0


def _parse_gpu_map(gpu_map_output: str) -> tuple[Set[str], Set[str]]:
    bus_suffixes: Set[str] = set()
    uuids: Set[str] = set()
    for line in gpu_map_output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        if parts[0] != "3":
            continue
        for tok in parts[1:]:
            low = tok.lower()
            if low.startswith("gpu-"):
                uuids.add(low)
            elif ":" in low and "." in low:
                bus_suffixes.add(_normalize_bus_suffix(low))
    if not bus_suffixes:
        bus_suffixes.add(DEFAULT_GPU3_BUS_SUFFIX)
    if not uuids:
        uuids.add(DEFAULT_GPU3_UUID)
    return bus_suffixes, uuids


def _matches_gpu3(identifier: str, bus_suffixes: Set[str], uuids: Set[str]) -> bool:
    low = identifier.strip().lower()
    if not low:
        return False
    if low in uuids:
        return True
    if ":" in low and "." in low:
        return _normalize_bus_suffix(low) in bus_suffixes
    return False


def _parse_pmon_lines(pmon_output: str) -> List[tuple[int, int, str, str]]:
    """Return list of (gpu_idx, pid, ptype, command) from nvidia-smi pmon output."""
    entries: List[tuple[int, int, str, str]] = []
    cmd_col_idx: Optional[int] = None

    for raw_line in pmon_output.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            header_tokens = stripped.lstrip("#").split()
            if "gpu" in header_tokens and "command" in header_tokens:
                cmd_col_idx = header_tokens.index("command")
            continue

        parts = stripped.split()
        if len(parts) < 4:
            continue
        try:
            gpu_idx = int(parts[0])
            pid = int(parts[1])
        except ValueError:
            continue
        ptype = parts[2]

        if cmd_col_idx is not None and len(parts) > cmd_col_idx:
            command = " ".join(parts[cmd_col_idx:])
        else:
            idx = 3
            while idx < len(parts) and (parts[idx] == "-" or parts[idx].isdigit()):
                idx += 1
            command = " ".join(parts[idx:]) if idx < len(parts) else ""

        entries.append((gpu_idx, pid, ptype, command.strip()))
    return entries


def check_gpu3_zero_occupancy(
    nvidia_smi_output: Optional[str] = None,
    pmon_output: Optional[str] = None,
    gpu_map_output: Optional[str] = None,
    ignore_external_unowned: bool = True,
) -> Gpu3Status:
    """Verify GPU 3 has zero occupancy by workspace/local processes.

    On multi-tenant hosts, external container PIDs appear in `nvidia-smi` as
    `[Not Found]` and are absent from `/proc/<pid>`. A compute entry on GPU 3 is
    flagged as a violation iff `process_name != '[Not Found]'` OR
    `os.path.exists(f'/proc/{pid}')` OR `not ignore_external_unowned`.
    """
    raw_sections: List[str] = []

    if nvidia_smi_output is None:
        try:
            if gpu_map_output is None:
                map_proc = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,uuid,pci.bus_id",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                    check=False,
                )
                if map_proc.returncode != 0:
                    return Gpu3Status(
                        ok=False,
                        gpu3_pids=[],
                        gpu3_mem_mib=0,
                        raw_output=map_proc.stderr.strip() or map_proc.stdout.strip(),
                        notes=f"nvidia-smi --query-gpu failed (rc={map_proc.returncode})",
                    )
                gpu_map_output = map_proc.stdout
                raw_sections.append(f"[gpu_map]\n{gpu_map_output.strip()}")

            apps_proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=gpu_bus_id,gpu_uuid,pid,process_name,used_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False,
            )
            if apps_proc.returncode != 0:
                return Gpu3Status(
                    ok=False,
                    gpu3_pids=[],
                    gpu3_mem_mib=0,
                    raw_output=apps_proc.stderr.strip() or apps_proc.stdout.strip(),
                    notes=f"nvidia-smi --query-compute-apps failed (rc={apps_proc.returncode})",
                )
            nvidia_smi_output = apps_proc.stdout
            raw_sections.append(f"[compute_apps]\n{nvidia_smi_output.strip()}")

            if pmon_output is None:
                pmon_proc = subprocess.run(
                    ["nvidia-smi", "pmon", "-c", "1"],
                    capture_output=True,
                    text=True,
                    timeout=10.0,
                    check=False,
                )
                if pmon_proc.returncode == 0:
                    pmon_output = pmon_proc.stdout
                    raw_sections.append(f"[pmon]\n{pmon_output.strip()}")
                else:
                    pmon_output = ""
        except (OSError, subprocess.TimeoutExpired) as exc:
            return Gpu3Status(
                ok=False,
                gpu3_pids=[],
                gpu3_mem_mib=0,
                raw_output=str(exc),
                notes=f"Failed to execute nvidia-smi: {exc}",
            )
    else:
        raw_sections.append(f"[compute_apps]\n{nvidia_smi_output.strip()}")
        if gpu_map_output:
            raw_sections.append(f"[gpu_map]\n{gpu_map_output.strip()}")
        if pmon_output:
            raw_sections.append(f"[pmon]\n{pmon_output.strip()}")

    bus_suffixes, uuids = (
        _parse_gpu_map(gpu_map_output)
        if gpu_map_output
        else ({DEFAULT_GPU3_BUS_SUFFIX}, {DEFAULT_GPU3_UUID})
    )

    gpu3_pids: List[int] = []
    external_pids: List[int] = []
    gpu3_mem_mib = 0
    note_parts: List[str] = []

    for raw_line in nvidia_smi_output.splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("gpu_"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            gpu_id_fields = [parts[0]]
            pid_str, proc_name, mem_str = parts[1], parts[2], parts[3]
        elif len(parts) >= 5:
            gpu_id_fields = [parts[0], parts[1]]
            pid_str, proc_name, mem_str = parts[2], parts[3], parts[4]
        else:
            continue

        if not any(_matches_gpu3(f, bus_suffixes, uuids) for f in gpu_id_fields):
            continue

        try:
            pid = int(pid_str)
        except ValueError:
            continue

        mem_mib = _parse_mem_mib(mem_str)
        proc_exists = os.path.exists(f"/proc/{pid}")
        is_named = proc_name not in ("", "[Not Found]")
        is_violation = is_named or proc_exists or (not ignore_external_unowned)

        if is_violation:
            if pid not in gpu3_pids:
                gpu3_pids.append(pid)
            gpu3_mem_mib += mem_mib
            note_parts.append(
                f"GPU3 compute violation pid={pid} proc='{proc_name}' mem={mem_mib}MiB"
            )
        else:
            if pid not in external_pids:
                external_pids.append(pid)

    if pmon_output:
        for gpu_idx, pid, ptype, command in _parse_pmon_lines(pmon_output):
            if gpu_idx != 3:
                continue
            ptype_u = ptype.upper()
            if "G" in ptype_u:
                if SIM_RENDER_CMD_RE.search(command):
                    if pid not in gpu3_pids:
                        gpu3_pids.append(pid)
                    note_parts.append(
                        f"GPU3 graphics violation pid={pid} type={ptype} cmd='{command}'"
                    )
                else:
                    if pid not in external_pids and pid not in gpu3_pids:
                        external_pids.append(pid)
            elif "C" in ptype_u and pid not in gpu3_pids and pid not in external_pids:
                proc_exists = os.path.exists(f"/proc/{pid}")
                is_named = command not in ("", "-", "[Not Found]")
                if is_named or proc_exists or (not ignore_external_unowned):
                    gpu3_pids.append(pid)
                    note_parts.append(
                        f"GPU3 pmon compute violation pid={pid} cmd='{command}'"
                    )
                else:
                    external_pids.append(pid)

    if external_pids and not gpu3_pids:
        note_parts.append(
            f"Ignored {len(external_pids)} external unowned GPU3 PID(s): {external_pids}"
        )

    return Gpu3Status(
        ok=(len(gpu3_pids) == 0),
        gpu3_pids=gpu3_pids,
        gpu3_mem_mib=gpu3_mem_mib,
        raw_output="\n".join(raw_sections),
        external_pids=external_pids,
        notes="; ".join(note_parts),
    )


def check_cuda_env_excludes_gpu3(
    env_dict: Optional[Mapping[str, str]] = None,
) -> bool:
    """Return True iff CUDA_VISIBLE_DEVICES explicitly excludes GPU 3."""
    env = os.environ if env_dict is None else env_dict
    raw_val = env.get("CUDA_VISIBLE_DEVICES")
    if raw_val is None:
        return False

    stripped = raw_val.strip()
    if not stripped:
        return False

    low = stripped.lower()
    if low == "all":
        return False
    if low in {"-1", "none", "nodevfiles"}:
        return True

    tokens = [tok.strip() for tok in stripped.split(",")]
    if not tokens or any(tok == "" for tok in tokens):
        return False

    for tok in tokens:
        tok_low = tok.lower()
        if re.fullmatch(r"-?\d+", tok):
            idx = int(tok)
            if idx not in (0, 1, 2):
                return False
        elif tok_low in KNOWN_NON_GPU3_UUIDS:
            continue
        elif ":" in tok_low and _normalize_bus_suffix(tok_low) in KNOWN_NON_GPU3_BUS_SUFFIXES:
            continue
        else:
            return False

    return True


def check_ogre2_log_gpu3_excluded(log_path: Union[str, Path]) -> OgreLogAudit:
    """Audit an Ogre2 log file to verify GPU 3 was never enumerated or initialized."""
    path = Path(log_path)
    if not path.is_file():
        return OgreLogAudit(
            ok=False,
            detected_devices=[],
            violations=[f"Ogre2 log file does not exist: {path}"],
            num_egl_devices=None,
        )

    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return OgreLogAudit(
            ok=False,
            detected_devices=[],
            violations=[f"Failed to read Ogre2 log {path}: {exc}"],
            num_egl_devices=None,
        )

    violations: List[str] = []
    detected_devices: List[str] = []
    seen_devices: Set[str] = set()
    num_egl_devices: Optional[int] = None

    num_matches = re.findall(r"Found Num EGL Devices:\s*(\d+)", content)
    if not num_matches:
        violations.append("Missing 'Found Num EGL Devices' line in ogre2.log")
    else:
        num_egl_devices = int(num_matches[-1])
        if num_egl_devices == 0 or num_egl_devices >= 4:
            violations.append(
                f"Found Num EGL Devices={num_egl_devices} (expected 1..3; >=4 exposes GPU 3)"
            )

    dev_pattern = re.compile(
        r"(?:EGL Device:|Trying to init device:|Created GL .*? context for device|Destroying device:)\s*(.*?#(\d+))",
        re.IGNORECASE,
    )
    for line in content.splitlines():
        m = dev_pattern.search(line)
        if m:
            dev_desc = f"#{m.group(2)}"
            if dev_desc not in seen_devices:
                seen_devices.add(dev_desc)
                detected_devices.append(dev_desc)
            dev_idx = int(m.group(2))
            if dev_idx >= 3:
                violations.append(
                    f"Forbidden GPU/EGL device #{dev_idx} referenced: {line.strip()}"
                )

        low_line = line.lower()
        if "d9:00.0" in low_line or "renderd131" in low_line or "gpu-617ae266" in low_line:
            violations.append(f"Forbidden GPU 3 identifier in ogre2.log: {line.strip()}")

        if "gl_renderer" in low_line and any(
            sw in low_line for sw in ("llvmpipe", "softpipe", "swrast")
        ):
            violations.append(f"Software renderer detected in ogre2.log: {line.strip()}")

    if not detected_devices:
        violations.append("No EGL device entries detected in ogre2.log")

    return OgreLogAudit(
        ok=(len(violations) == 0),
        detected_devices=detected_devices,
        violations=violations,
        num_egl_devices=num_egl_devices,
    )


def check_git_anchors(
    repo_dir: Union[str, Path] = DEFAULT_REPO_DIR,
    expected: Optional[Dict[str, str]] = None,
) -> GitAnchorResult:
    """Verify protected rollback git branches match their expected commits."""
    repo = Path(repo_dir)
    anchors = dict(DEFAULT_EXPECTED_ANCHORS if expected is None else expected)
    actual: Dict[str, str] = {}
    errors: List[str] = []

    if not repo.exists():
        return GitAnchorResult(
            ok=False,
            actual=actual,
            errors=[f"Repository path does not exist: {repo}"],
        )

    probe = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        timeout=10.0,
        check=False,
    )
    if probe.returncode != 0 or probe.stdout.strip().lower() != "true":
        return GitAnchorResult(
            ok=False,
            actual=actual,
            errors=[
                f"Path is not a git work tree: {repo} ({probe.stderr.strip() or probe.stdout.strip()})"
            ],
        )

    for branch, exp_hash in anchors.items():
        full_proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--verify", branch],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
        if full_proc.returncode != 0:
            errors.append(
                f"Git branch '{branch}' missing or invalid: {full_proc.stderr.strip()}"
            )
            continue

        full_sha = full_proc.stdout.strip().lower()
        short_proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", branch],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
        short_sha = (
            short_proc.stdout.strip().lower()
            if short_proc.returncode == 0
            else full_sha[:7]
        )
        actual[branch] = short_sha

        exp_low = exp_hash.strip().lower()
        if not (full_sha.startswith(exp_low) or exp_low.startswith(short_sha)):
            errors.append(
                f"Git anchor mismatch for '{branch}': expected '{exp_hash}', actual '{short_sha}' ({full_sha})"
            )

    return GitAnchorResult(
        ok=(len(errors) == 0 and len(actual) == len(anchors)),
        actual=actual,
        errors=errors,
    )
