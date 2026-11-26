"""Tier 2 BVA & corner-case tests for F2: Complex 3D <collision> Arena & Registry."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from helpers.arena_parser import (
    check_line_of_sight,
    load_obstacles_yaml,
    parse_sdf_collisions,
    point_min_clearance,
)

pytestmark = [pytest.mark.tier2, pytest.mark.feature(id="F2"), pytest.mark.requires_m1]

ROS2_WS = Path("/home/hkw/flying/ros2_ws")
OBSTACLES_YAML = ROS2_WS / "src" / "flight_ctrl_pkg" / "config" / "obstacles.yaml"


def test_f02_bva_grazing_clearance_vs_safety_margin() -> None:
    """Verify exact 0.0m surface grazing vs safety_margin_m expansion on cylinder and box."""
    cfg = load_obstacles_yaml(OBSTACLES_YAML)
    obs_by_id = {o["id"]: o for o in cfg["obstacles"]}

    pylon = obs_by_id["obs_comm_pylon_center"]
    pylon_z = 0.5 * (pylon["z_min"] + pylon["z_max"])
    pylon_surface_pt = (pylon["cx"] + pylon["radius"], pylon["cy"], pylon_z)
    clr_raw = point_min_clearance(pylon_surface_pt, cfg, include_nfz=False, include_safety_margin=False)
    clr_margin = point_min_clearance(pylon_surface_pt, cfg, include_nfz=False, include_safety_margin=True)
    assert clr_raw == pytest.approx(0.0, abs=1e-6)
    assert clr_margin == pytest.approx(-pylon["safety_margin_m"], abs=1e-6)

    tower = obs_by_id["obs_canyon_tower_nw"]
    tower_z = 0.5 * (tower["z_min"] + tower["z_max"])
    box_surface_pt = (tower["cx"], tower["cy"] - 0.5 * tower["sy"], tower_z)
    box_raw = point_min_clearance(box_surface_pt, cfg, include_nfz=False, include_safety_margin=False)
    box_margin = point_min_clearance(box_surface_pt, cfg, include_nfz=False, include_safety_margin=True)
    assert box_raw == pytest.approx(0.0, abs=1e-6)
    assert box_margin == pytest.approx(-tower["safety_margin_m"], abs=1e-6)


def test_f02_bva_inside_obstacle_and_nfz_negative_clearance() -> None:
    """Verify negative signed clearance inside obstacles and inside NFZ-only airspace."""
    cfg = load_obstacles_yaml(OBSTACLES_YAML)
    obs_by_id = {o["id"]: o for o in cfg["obstacles"]}
    nfz_by_id = {n["id"]: n for n in cfg["no_fly_zones"]}

    pylon = obs_by_id["obs_comm_pylon_center"]
    inside_obs = (pylon["cx"], pylon["cy"], 0.5 * (pylon["z_min"] + pylon["z_max"]))
    expected_obs_clr = -min(pylon["radius"], 0.5 * (pylon["z_max"] - pylon["z_min"]))
    assert point_min_clearance(inside_obs, cfg, include_nfz=False) == pytest.approx(expected_obs_clr, abs=1e-6)

    bunker = obs_by_id["obs_bunker_mid_n"]
    nfz = nfz_by_id["nfz_command_core"]
    z_above = 0.5 * (bunker["z_max"] + nfz["z_max"])
    inside_nfz_above_bunker = (nfz["cx"], nfz["cy"], z_above)
    clr_no_nfz = point_min_clearance(inside_nfz_above_bunker, cfg, include_nfz=False)
    clr_with_nfz = point_min_clearance(inside_nfz_above_bunker, cfg, include_nfz=True)
    assert clr_no_nfz > 0.0, f"Expected positive clearance above bunker when NFZ ignored, got {clr_no_nfz}"
    assert clr_with_nfz == pytest.approx(-nfz["radius"], abs=1e-6)


def test_f02_bva_arena_bounds_corners_and_z_limits() -> None:
    """Verify arena_bounds exact boundary (0.0), interior (+eps), and out-of-bounds (-eps)."""
    cfg = load_obstacles_yaml(OBSTACLES_YAML)
    b = cfg["arena_bounds"]

    for pt in (
        (b["x_min"], 0.0, 10.0),
        (b["x_max"], 0.0, 10.0),
        (20.0, b["y_min"], 10.0),
        (20.0, b["y_max"], 10.0),
        (20.0, 0.0, b["z_max"]),
    ):
        assert point_min_clearance(pt, cfg, include_bounds=True) == pytest.approx(0.0, abs=1e-6)

    outside_pt = (b["x_max"] + 0.5, 0.0, 10.0)
    assert point_min_clearance(outside_pt, cfg, include_bounds=True) == pytest.approx(-0.5, abs=1e-6)

    high_inside = (20.0, 0.0, b["z_max"] - 0.25)
    assert point_min_clearance(high_inside, cfg, include_bounds=True) == pytest.approx(0.25, abs=1e-6)


def test_f02_bva_narrow_canyon_corridor_pinch_clearance_and_los() -> None:
    """Verify canyon gap between obs_canyon_tower_nw and obs_canyon_tower_sw."""
    cfg = load_obstacles_yaml(OBSTACLES_YAML)
    obs_by_id = {o["id"]: o for o in cfg["obstacles"]}
    nw = obs_by_id["obs_canyon_tower_nw"]
    sw = obs_by_id["obs_canyon_tower_sw"]

    nw_south_edge = nw["cy"] - 0.5 * nw["sy"]
    sw_north_edge = sw["cy"] + 0.5 * sw["sy"]
    mid_x = 0.5 * (nw["cx"] + sw["cx"])
    mid_y = 0.5 * (nw_south_edge + sw_north_edge)
    expected_half_gap = 0.5 * (nw_south_edge - sw_north_edge)
    z_canyon = 0.5 * min(nw["z_max"], sw["z_max"])
    x_lo = nw["cx"] - 0.5 * nw["sx"] - 5.0
    x_hi = nw["cx"] + 0.5 * nw["sx"] + 5.0

    mid = (mid_x, mid_y, z_canyon)
    clr_mid = point_min_clearance(mid, cfg, include_nfz=False, include_safety_margin=False)
    assert clr_mid == pytest.approx(expected_half_gap, abs=1e-6)

    assert check_line_of_sight((x_lo, mid_y, z_canyon), (x_hi, mid_y, z_canyon), cfg, margin_m=0.0) is True
    assert check_line_of_sight((x_lo, nw["cy"], z_canyon), (x_hi, nw["cy"], z_canyon), cfg, margin_m=0.0) is False
    assert check_line_of_sight((nw["cx"], nw["cy"], z_canyon), (nw["cx"], nw["cy"], z_canyon), cfg) is False


def test_f02_bva_reject_malformed_yaml_and_invalid_sdf_pose(tmp_path: Path) -> None:
    """Verify load_obstacles_yaml() and parse_sdf_collisions() reject malformed inputs."""
    missing_section_yaml = tmp_path / "missing_section.yaml"
    missing_section_yaml.write_text(
        "arena_bounds: {x_min: 0, x_max: 1, y_min: 0, y_max: 1, z_min: 0, z_max: 1}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Missing required top-level section"):
        load_obstacles_yaml(missing_section_yaml)

    bad_radius_yaml = tmp_path / "bad_radius.yaml"
    bad_radius_yaml.write_text(
        "arena_bounds: {x_min: -10, x_max: 10, y_min: -10, y_max: 10, z_min: 0, z_max: 20}\n"
        "helipads: []\n"
        "obstacles:\n"
        "  - {id: o1, type: cylinder, cx: 0, cy: 0, z_min: 0, z_max: 5, radius: 0.0}\n"
        "no_fly_zones: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="radius must be > 0"):
        load_obstacles_yaml(bad_radius_yaml)

    bad_pose_sdf = tmp_path / "bad_pose.sdf"
    bad_pose_sdf.write_text(
        "<sdf version='1.9'><world name='bad'>"
        "<model name='m1'><pose>1 2 3</pose><link name='l1'><collision name='c1'>"
        "<geometry><box><size>1 1 1</size></box></geometry>"
        "</collision></link></model></world></sdf>",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Expected 6 pose values"):
        parse_sdf_collisions(bad_pose_sdf)

    bad_xml_sdf = tmp_path / "bad_xml.sdf"
    bad_xml_sdf.write_text("<sdf><world><unclosed></sdf>", encoding="utf-8")
    with pytest.raises(ET.ParseError):
        parse_sdf_collisions(bad_xml_sdf)
