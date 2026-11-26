#!/usr/bin/env python3
"""Dual-mode E2E test runner CLI for ros2_ws/tests_e2e."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

TESTS_E2E_ROOT = Path(__file__).resolve().parent
ROS2_WS_ROOT = TESTS_E2E_ROOT.parent
PX4_ROS2_PYTHON = Path("/home/hkw/miniconda3/envs/px4_ros2/bin/python3")
PYTEST_INI_PATH = TESTS_E2E_ROOT / "pytest.ini"
FEATURES = [f"F{i}" for i in range(1, 12)]


def _ensure_safe_env(env: Dict[str, str]) -> Dict[str, str]:
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # Inline check before helpers import in case we re-exec early
    cuda_val = env.get("CUDA_VISIBLE_DEVICES", "").strip()
    valid_tokens = {"0", "1", "2"}
    ok_cuda = False
    if cuda_val:
        low = cuda_val.lower()
        if low in {"-1", "none", "nodevfiles"}:
            ok_cuda = True
        else:
            toks = [t.strip() for t in cuda_val.split(",")]
            if toks and all(t in valid_tokens for t in toks):
                ok_cuda = True
    if not ok_cuda:
        env["CUDA_VISIBLE_DEVICES"] = "2"
    return env


def _maybe_reexec_px4_python() -> None:
    if importlib.util.find_spec("pytest") is not None:
        return
    if os.environ.get("TESTS_E2E_REEXECED") == "1":
        return
    if PX4_ROS2_PYTHON.is_file() and os.access(PX4_ROS2_PYTHON, os.X_OK):
        env = _ensure_safe_env(os.environ.copy())
        env["TESTS_E2E_REEXECED"] = "1"
        os.execve(str(PX4_ROS2_PYTHON), [str(PX4_ROS2_PYTHON), *sys.argv], env)


_maybe_reexec_px4_python()
_ensure_safe_env(os.environ)

if str(TESTS_E2E_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_E2E_ROOT))

from helpers.hardware_git_guard import check_cuda_env_excludes_gpu3  # noqa: E402


def _parse_junit_xml(xml_path: Path) -> Dict[str, int]:
    if not xml_path.is_file():
        return {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return {"total": 0, "passed": 0, "failed": 0, "errors": 1, "skipped": 0}

    root = tree.getroot()
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    total = 0
    errors = 0
    failures = 0
    skipped = 0
    for suite in suites:
        total += int(suite.attrib.get("tests", 0))
        errors += int(suite.attrib.get("errors", 0))
        failures += int(suite.attrib.get("failures", 0))
        skipped += int(suite.attrib.get("skipped", 0))
    passed = max(0, total - errors - failures - skipped)
    return {
        "total": total,
        "passed": passed,
        "failed": failures,
        "errors": errors,
        "skipped": skipped,
    }


def _run_pytest_subprocess(
    target_paths: List[Path],
    mode: str,
    feature: Optional[str],
    marker_expr: Optional[str],
) -> Dict[str, Any]:
    import subprocess

    existing_targets = [str(p) for p in target_paths if p.exists()]
    if not existing_targets:
        return {
            "returncode": 5,
            "total": 0,
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "stdout": "",
            "stderr": "",
        }

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as tmp:
        junit_path = Path(tmp.name)

    try:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(PYTEST_INI_PATH),
            "--e2e-mode",
            mode,
            f"--junitxml={junit_path}",
            "-v",
        ]
        if feature:
            cmd.extend(["--feature", feature])
        if marker_expr:
            cmd.extend(["-m", marker_expr])
        cmd.extend(existing_targets)

        env = os.environ.copy()
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        if not check_cuda_env_excludes_gpu3(env):
            env["CUDA_VISIBLE_DEVICES"] = "2"

        proc = subprocess.run(
            cmd,
            cwd=str(TESTS_E2E_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        counts = _parse_junit_xml(junit_path)
        return {
            "returncode": proc.returncode,
            **counts,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    finally:
        if junit_path.exists():
            junit_path.unlink()


def _collect_inventory_via_pytest() -> Dict[str, Any]:
    """Use pytest --collect-only in a clean subprocess to audit tier1..tier4 coverage."""
    import subprocess

    tier_dirs = [TESTS_E2E_ROOT / f"tier{i}" for i in range(1, 5)]
    test_files: List[Path] = []
    for td in tier_dirs:
        if td.is_dir():
            test_files.extend(sorted(td.rglob("test_*.py")))

    tier1_counts = {f: 0 for f in FEATURES}
    tier2_counts = {f: 0 for f in FEATURES}
    tier3_count = 0
    tier4_count = 0
    tier_totals = {"tier1": 0, "tier2": 0, "tier3": 0, "tier4": 0}
    collection_errors: List[str] = []

    existing_tiers = [str(td) for td in tier_dirs if td.is_dir()]
    if test_files and existing_tiers:
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            inv_json = Path(tmp.name)
        plugin_code = f"""
import json, sys
import pytest

records = []

class InventoryPlugin:
    def pytest_collection_modifyitems(self, session, config, items):
        for item in items:
            tiers = [m.name for m in item.iter_markers() if m.name in ("tier1","tier2","tier3","tier4")]
            feats = []
            for m in item.iter_markers("feature"):
                if "id" in m.kwargs and m.kwargs["id"] is not None:
                    feats.append(str(m.kwargs["id"]).strip().upper())
                elif m.args:
                    feats.append(str(m.args[0]).strip().upper())
            records.append({{"nodeid": item.nodeid, "tiers": tiers, "features": feats}})

rc = pytest.main(
    ["-c", {repr(str(PYTEST_INI_PATH))}, "--collect-only", "-q", *sys.argv[1:]],
    plugins=[InventoryPlugin()],
)
with open({repr(str(inv_json))}, "w", encoding="utf-8") as fh:
    json.dump({{"rc": rc, "records": records}}, fh)
sys.exit(0 if rc in (0, 5) else rc)
"""
        try:
            env = os.environ.copy()
            env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
            if not check_cuda_env_excludes_gpu3(env):
                env["CUDA_VISIBLE_DEVICES"] = "2"

            proc = subprocess.run(
                [sys.executable, "-c", plugin_code, *existing_tiers],
                cwd=str(TESTS_E2E_ROOT),
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                collection_errors.append(
                    f"Inventory collection failed (rc={proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}"
                )
            elif inv_json.is_file():
                payload = json.loads(inv_json.read_text(encoding="utf-8"))
                for rec in payload.get("records", []):
                    nodeid = rec["nodeid"]
                    tiers = set(rec.get("tiers", []))
                    feats = set(rec.get("features", []))
                    if "tier1" in tiers or nodeid.startswith("tier1/"):
                        tier_totals["tier1"] += 1
                        for f in feats:
                            if f in tier1_counts:
                                tier1_counts[f] += 1
                    if "tier2" in tiers or nodeid.startswith("tier2/"):
                        tier_totals["tier2"] += 1
                        for f in feats:
                            if f in tier2_counts:
                                tier2_counts[f] += 1
                    if "tier3" in tiers or nodeid.startswith("tier3/"):
                        tier_totals["tier3"] += 1
                        tier3_count += 1
                    if "tier4" in tiers or nodeid.startswith("tier4/"):
                        tier_totals["tier4"] += 1
                        tier4_count += 1
        finally:
            if inv_json.exists():
                inv_json.unlink()

    return {
        "tier_test_file_count": len(test_files),
        "tier_totals": tier_totals,
        "tier1_by_feature": tier1_counts,
        "tier2_by_feature": tier2_counts,
        "tier3_total": tier3_count,
        "tier4_total": tier4_count,
        "collection_errors": collection_errors,
    }


def _evaluate_inventory(inv: Dict[str, Any], strict_inventory: bool) -> tuple[str, bool, List[str]]:
    errors: List[str] = list(inv.get("collection_errors", []))
    if inv["tier_test_file_count"] == 0:
        if strict_inventory:
            errors.append(
                "Strict inventory check failed: no tier1..tier4 test files exist yet (pre-E2E-M2..M4)."
            )
            return ("partial_pre_m2_m4", False, errors)
        return ("partial_pre_m2_m4", len(errors) == 0, errors)

    for f in FEATURES:
        c1 = inv["tier1_by_feature"].get(f, 0)
        if c1 < 5:
            errors.append(f"Tier 1 coverage for {f}: {c1} < 5 required")
        c2 = inv["tier2_by_feature"].get(f, 0)
        if c2 < 5:
            errors.append(f"Tier 2 coverage for {f}: {c2} < 5 required")

    if inv["tier3_total"] < 11:
        errors.append(f"Tier 3 pairwise test count: {inv['tier3_total']} < 11 required")
    if inv["tier4_total"] < 5:
        errors.append(f"Tier 4 real-world scenario count: {inv['tier4_total']} < 5 required")

    status = "complete" if not errors else "incomplete"
    return (status, len(errors) == 0, errors)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run opaque-box E2E verification tiers under GPU-3 isolation."
    )
    parser.add_argument(
        "--tier",
        required=True,
        choices=("1", "2", "3", "4", "all", "self_check"),
        help="Test tier to execute.",
    )
    parser.add_argument(
        "--mode",
        default="full_e2e",
        choices=("pre_integration", "full_e2e"),
        help="Execution mode (default: full_e2e).",
    )
    parser.add_argument(
        "--feature",
        default=None,
        help="Filter tests by feature ID (F1..F11).",
    )
    parser.add_argument(
        "--strict-inventory",
        action="store_true",
        help="Fail in pre_integration mode if tier1..tier4 inventory thresholds are not yet met.",
    )
    parser.add_argument(
        "--json-report",
        default=None,
        help="Optional output path for structured JSON test report.",
    )
    args = parser.parse_args(argv)

    feature: Optional[str] = None
    if args.feature is not None:
        feature = args.feature.strip().upper()
        if feature not in FEATURES:
            print(
                f"ERROR: Invalid --feature '{args.feature}'. Expected one of {FEATURES}.",
                file=sys.stderr,
            )
            return 2

    diagnostics: List[str] = []
    tier_map = {
        "1": [TESTS_E2E_ROOT / "tier1"],
        "2": [TESTS_E2E_ROOT / "tier2"],
        "3": [TESTS_E2E_ROOT / "tier3"],
        "4": [TESTS_E2E_ROOT / "tier4"],
        "self_check": [TESTS_E2E_ROOT / "self_check"],
    }

    if args.mode == "full_e2e":
        if args.tier == "all":
            targets = [
                TESTS_E2E_ROOT / "self_check",
                TESTS_E2E_ROOT / "tier1",
                TESTS_E2E_ROOT / "tier2",
                TESTS_E2E_ROOT / "tier3",
                TESTS_E2E_ROOT / "tier4",
            ]
        else:
            targets = tier_map[args.tier]
        marker_expr = None
    else:
        if args.tier == "all":
            targets = [
                TESTS_E2E_ROOT / "self_check",
                TESTS_E2E_ROOT / "tier1",
                TESTS_E2E_ROOT / "tier2",
                TESTS_E2E_ROOT / "tier3",
                TESTS_E2E_ROOT / "tier4",
            ]
        else:
            targets = tier_map[args.tier]
        marker_expr = "self_check or pre_integration_safe"

    res = _run_pytest_subprocess(targets, args.mode, feature, marker_expr)
    if res["stdout"]:
        print(res["stdout"], end="" if res["stdout"].endswith("\n") else "\n")
    if res["stderr"]:
        print(res["stderr"], file=sys.stderr, end="" if res["stderr"].endswith("\n") else "\n")

    inventory_data: Optional[Dict[str, Any]] = None
    inventory_status = "not_applicable"
    inventory_ok = True
    inventory_errors: List[str] = []

    if args.mode == "pre_integration" and args.tier in ("all", "1", "2", "3", "4"):
        inventory_data = _collect_inventory_via_pytest()
        inventory_status, inventory_ok, inventory_errors = _evaluate_inventory(
            inventory_data, args.strict_inventory
        )
        diagnostics.extend(inventory_errors)

    ok = True
    if args.mode == "full_e2e":
        if res["returncode"] != 0:
            ok = False
            diagnostics.append(f"pytest exited with code {res['returncode']}")
        if res["total"] <= 0:
            ok = False
            diagnostics.append("Zero tests collected in full_e2e mode (exit code 5 / empty suite forbidden)")
        if res["failed"] > 0 or res["errors"] > 0 or res["skipped"] > 0:
            ok = False
            diagnostics.append(
                "full_e2e requires 0 failed/errors/skipped; "
                f"got failed={res['failed']}, errors={res['errors']}, skipped={res['skipped']}"
            )
        if args.tier == "all" and feature is None:
            inv_check = _collect_inventory_via_pytest()
            inventory_data = inv_check
            for tname in ("tier1", "tier2", "tier3", "tier4"):
                if inv_check["tier_totals"].get(tname, 0) <= 0:
                    ok = False
                    diagnostics.append(
                        f"full_e2e --tier all requires non-empty {tname}/ tests (found 0)"
                    )
            inventory_status, _, inv_errs = _evaluate_inventory(inv_check, strict_inventory=True)
            if inv_errs:
                ok = False
                diagnostics.extend(inv_errs)
        elif args.tier in ("1", "2", "3", "4"):
            tdir = TESTS_E2E_ROOT / f"tier{args.tier}"
            if not tdir.is_dir():
                ok = False
                diagnostics.append(f"Requested tier directory does not exist: {tdir}")
    else:
        # pre_integration mode
        if args.tier == "self_check":
            if res["returncode"] != 0 or res["total"] <= 0:
                ok = False
                diagnostics.append(
                    f"self_check requires collected > 0 and rc == 0; got total={res['total']}, rc={res['returncode']}"
                )
        elif args.tier == "all":
            if feature is None and res["total"] <= 0:
                ok = False
                diagnostics.append("pre_integration --tier all expected >=1 self_check/baseline test")
            if res["returncode"] not in (0, 5):
                ok = False
                diagnostics.append(f"pytest failed with rc={res['returncode']}")
            if feature is not None and res["returncode"] == 5:
                # Acceptable if no pre_integration_safe tests exist for that feature yet
                pass
        else:
            # Individual tier 1..4 in pre_integration: exit code 5 (0 pre_integration_safe tests) is non-fatal
            if res["returncode"] not in (0, 5):
                ok = False
                diagnostics.append(f"pytest failed with rc={res['returncode']}")

        if res["failed"] > 0 or res["errors"] > 0 or res["skipped"] > 0:
            ok = False
            diagnostics.append(
                "pre_integration requires 0 failed/errors/skipped; "
                f"got failed={res['failed']}, errors={res['errors']}, skipped={res['skipped']}"
            )
        if not inventory_ok:
            ok = False

    exit_code = 0 if ok else 1

    report: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tier": args.tier,
        "mode": args.mode,
        "feature": feature,
        "strict_inventory": bool(args.strict_inventory),
        "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "pytest_disable_plugin_autoload": os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD", ""),
        "total": res["total"],
        "passed": res["passed"],
        "failed": res["failed"],
        "errors": res["errors"],
        "skipped": res["skipped"],
        "pytest_returncode": res["returncode"],
        "inventory_status": inventory_status,
        "inventory": inventory_data,
        "inventory_errors": inventory_errors,
        "diagnostics": diagnostics,
        "ok": ok,
        "exit_code": exit_code,
    }

    if args.json_report:
        out_path = Path(args.json_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary_line = (
        f"[tests_e2e runner] mode={args.mode} tier={args.tier} "
        f"total={res['total']} passed={res['passed']} failed={res['failed']} "
        f"errors={res['errors']} skipped={res['skipped']} "
        f"inventory={inventory_status} ok={ok}"
    )
    print(summary_line)
    for diag in diagnostics:
        print(f"[tests_e2e runner] DIAGNOSTIC: {diag}", file=sys.stderr)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
