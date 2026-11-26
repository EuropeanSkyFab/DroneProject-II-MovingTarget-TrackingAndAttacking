"""Pytest configuration, CLI options, feature filtering, and shared fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import pytest

VALID_FEATURES = {f"F{i}" for i in range(1, 12)}


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register custom E2E CLI options."""
    group = parser.getgroup("tests_e2e", "Opaque-box E2E test suite options")
    group.addoption(
        "--e2e-mode",
        action="store",
        default="full_e2e",
        choices=("pre_integration", "full_e2e"),
        help="Execution mode: pre_integration or full_e2e (default: full_e2e).",
    )
    group.addoption(
        "--feature",
        action="store",
        default=None,
        help="Optional feature filter (F1..F11).",
    )


def _extract_feature_ids(item: pytest.Item) -> List[str]:
    ids: List[str] = []
    for marker in item.iter_markers(name="feature"):
        if "id" in marker.kwargs and marker.kwargs["id"] is not None:
            ids.append(str(marker.kwargs["id"]).strip().upper())
        elif marker.args:
            ids.append(str(marker.args[0]).strip().upper())
    return ids


def pytest_collection_modifyitems(config: pytest.Config, items: List[pytest.Item]) -> None:
    """Filter collected tests by --feature when specified."""
    raw_feature = config.getoption("--feature")
    if raw_feature is None:
        return

    target_feature = str(raw_feature).strip().upper()
    if target_feature not in VALID_FEATURES:
        raise pytest.UsageError(
            f"Invalid --feature '{raw_feature}'. Expected one of {sorted(VALID_FEATURES)}."
        )

    selected: List[pytest.Item] = []
    deselected: List[pytest.Item] = []
    for item in items:
        if target_feature in _extract_feature_ids(item):
            selected.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


@pytest.fixture(scope="session")
def e2e_mode(request: pytest.FixtureRequest) -> str:
    """Return active E2E mode ('pre_integration' or 'full_e2e')."""
    return str(request.config.getoption("--e2e-mode"))


@pytest.fixture(scope="session")
def selected_feature(request: pytest.FixtureRequest) -> Optional[str]:
    """Return selected --feature filter (F1..F11) or None."""
    val = request.config.getoption("--feature")
    return str(val).strip().upper() if val is not None else None


@pytest.fixture(scope="session")
def tests_e2e_root() -> Path:
    """Return path to ros2_ws/tests_e2e."""
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def ros2_ws_root(tests_e2e_root: Path) -> Path:
    """Return path to ros2_ws."""
    return tests_e2e_root.parent


@pytest.fixture(scope="session")
def project_root(ros2_ws_root: Path) -> Path:
    """Return path to /home/hkw/flying."""
    return ros2_ws_root.parent


@pytest.fixture(scope="session")
def showcase_output_dir() -> Path:
    """Return showcase output artifact directory."""
    return Path("/home/hkw/.gemini/antigravity/brain/d204ee53-0b23-4a81-ac6e-d2bd544d5472")
