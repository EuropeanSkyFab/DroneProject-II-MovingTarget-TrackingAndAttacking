"""Tier 1 functional tests for F2: Complex 3D <collision> Arena & Registry."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from helpers.arena_parser import (
    check_line_of_sight,
    load_obstacles_yaml,
    parse_sdf_collisions,
    point_min_clearance,
)

pytestmark = [pytest.mark.tier1, pytest.mark.feature(id="F2"), pytest.mark.requires_m1]

ROS2_WS = Path("/home/hkw/flying/ros2_ws")
FC_PKG = ROS2_WS / "src" / "flight_ctrl_pkg"
OBSTACLES_YAML = FC_PKG / "config" / "obstacles.yaml"
URBAN_ARENA_SDF = FC_PKG / "sim" / "worlds" / "urban_counter_swarm_arena.sdf"
MOVING_ARENA_SDF = FC_PKG / "sim" / "worlds" / "moving_target_arena.sdf"
X500_DOWN_CAM_SDF = FC_PKG / "sim" / "models" / "x500_down_cam" / "model.sdf"


def test_f02_obstacles_yaml_schema_and_required_sections() -> None:
    """Verify obstacles.yaml validates cleanly with arena_bounds, helipads, obstacles, and NFZs."""
    cfg = load_obstacles_yaml(OBSTACLES_YAML)
    assert set(cfg.keys()) == {"arena_bounds", "helipads", "obstacles", "no_fly_zones"}

    bounds = cfg["arena_bounds"]
    assert bounds["x_min"] < bounds["x_max"]
    assert bounds["y_min"] < bounds["y_max"]
    assert bounds["z_min"] < bounds["z_max"]
    assert len(cfg["helipads"]) >= 6
    assert len(cfg["obstacles"]) >= 6
    assert len(cfg["no_fly_zones"]) >= 3


def test_f02_sdf_collisions_match_obstacles_yaml_one_to_one() -> None:
    """Verify urban_counter_swarm_arena.sdf collisions match obstacles.yaml 1-to-1 and moving_target_arena is valid."""
    cfg = load_obstacles_yaml(OBSTACLES_YAML)
    urban_audit = parse_sdf_collisions(URBAN_ARENA_SDF)
    non_ground = [c for c in urban_audit.collisions if not c.is_ground]

    assert len(non_ground) == len(cfg["obstacles"]), (
        f"Expected {len(cfg['obstacles'])} non-ground SDF collisions, got {len(non_ground)}"
    )

    by_model = {c.model: c for c in non_ground}
    assert len(by_model) == len(cfg["obstacles"]), "Duplicate obstacle model names in SDF"

    for obs in cfg["obstacles"]:
        oid = obs["id"]
        assert oid in by_model, f"Obstacle '{oid}' from obstacles.yaml missing in SDF"
        col = by_model[oid]
        assert col.geometry_type == obs["type"], f"Type mismatch for {oid}"

        exp_z = 0.5 * (obs["z_min"] + obs["z_max"])
        exp_len = obs["z_max"] - obs["z_min"]
        assert col.world_pose[0] == pytest.approx(obs["cx"], abs=1e-4)
        assert col.world_pose[1] == pytest.approx(obs["cy"], abs=1e-4)
        assert col.world_pose[2] == pytest.approx(exp_z, abs=1e-4)

        if obs["type"] == "cylinder":
            assert col.dimensions["radius"] == pytest.approx(obs["radius"], abs=1e-4)
            assert col.dimensions["length"] == pytest.approx(exp_len, abs=1e-4)
        else:
            assert col.dimensions["sx"] == pytest.approx(obs["sx"], abs=1e-4)
            assert col.dimensions["sy"] == pytest.approx(obs["sy"], abs=1e-4)
            assert col.dimensions["sz"] == pytest.approx(exp_len, abs=1e-4)

    moving_audit = parse_sdf_collisions(MOVING_ARENA_SDF)
    moving_non_plane = [c for c in moving_audit.collisions if c.geometry_type != "plane"]
    assert len(moving_non_plane) >= 5, "Legacy moving_target_arena.sdf missing building collisions"


def test_f02_nfz_and_obstacle_diversity_with_visual_only_markers() -> None:
    """Verify >=3 NFZs, >=6 diverse obstacles (cylinder + box), and collision-free NFZ markers."""
    cfg = load_obstacles_yaml(OBSTACLES_YAML)
    obstacles = cfg["obstacles"]
    nfzs = cfg["no_fly_zones"]

    assert len(obstacles) >= 6
    assert len(nfzs) >= 3
    types = {o["type"] for o in obstacles}
    assert types == {"cylinder", "box"}, f"Expected cylinder and box obstacles, got {types}"

    urban_audit = parse_sdf_collisions(URBAN_ARENA_SDF)
    nfz_model = [m for m in urban_audit.models if m["name"] == "nfz_ground_markers"]
    assert len(nfz_model) == 1, "Expected nfz_ground_markers model in urban arena SDF"
    assert nfz_model[0]["collision_count"] == 0, "NFZ ground markers must be visual-only"
    assert nfz_model[0]["visual_count"] >= len(nfzs)


def test_f02_golden_hour_lighting_and_cinematic_plugins() -> None:
    """Verify golden-hour directional shadows, LensFlare, ThermalSensor, ParticleEmitter, and Breadcrumbs."""
    urban_audit = parse_sdf_collisions(URBAN_ARENA_SDF)
    sun_lights = [lt for lt in urban_audit.lights if lt["name"] == "sunUTC"]
    assert len(sun_lights) == 1, "Missing sunUTC directional light"
    assert sun_lights[0]["type"] == "directional"
    assert sun_lights[0]["cast_shadows"] is True

    root = ET.parse(URBAN_ARENA_SDF).getroot()
    world = root.find("world")
    assert world is not None
    sky_time = world.findtext("./scene/sky/time")
    assert sky_time is not None and 16.0 <= float(sky_time) <= 19.0, (
        f"Expected golden-hour sky time, got {sky_time!r}"
    )

    plugin_names = {p["name"] for p in urban_audit.plugins}
    assert "gz::sim::systems::LensFlare" in plugin_names
    assert "gz::sim::systems::ThermalSensor" in plugin_names
    assert "gz::sim::systems::ParticleEmitter" in plugin_names
    assert "gz::sim::systems::Thermal" in plugin_names

    emitter_models = [m["name"] for m in urban_audit.models if m["has_particle_emitter"]]
    for expected in ("obs_cooling_silo_s", "strike_burst_1", "strike_burst_2", "strike_burst_3"):
        assert expected in emitter_models, f"Missing particle emitter on {expected}"

    drone_audit = parse_sdf_collisions(X500_DOWN_CAM_SDF)
    drone_plugins = {p["name"] for p in drone_audit.plugins}
    assert "gz::sim::systems::Breadcrumbs" in drone_plugins
    assert "gz::sim::systems::ThermalSensor" in drone_plugins


def test_f02_positive_corridor_clearance_and_line_of_sight() -> None:
    """Verify positive clearance and unobstructed LoS along central arena flight corridors."""
    cfg = load_obstacles_yaml(OBSTACLES_YAML)
    obs_by_id = {o["id"]: o for o in cfg["obstacles"]}
    pad_by_drone = {h["drone_id"]: h for h in cfg["helipads"]}
    h1 = pad_by_drone["drone_1"]
    nw = obs_by_id["obs_canyon_tower_nw"]
    sw = obs_by_id["obs_canyon_tower_sw"]
    pylon = obs_by_id["obs_comm_pylon_center"]
    ew = obs_by_id["obs_east_watchtower"]
    ewh = obs_by_id["obs_east_warehouse"]

    canyon_x = 0.5 * (nw["cx"] + sw["cx"])
    canyon_y = 0.5 * ((nw["cy"] - 0.5 * nw["sy"]) + (sw["cy"] + 0.5 * sw["sy"]))
    pylon_bypass_y = pylon["cy"] - (pylon["radius"] + pylon["safety_margin_m"] + 2.0)
    east_gap_x = 0.5 * (ew["cx"] + ewh["cx"])
    east_gap_y = 0.5 * (ew["cy"] + ewh["cy"])
    z_cruise = 6.0

    waypoints = [
        (h1["x"], h1["y"], z_cruise),
        (0.5 * (h1["x"] + canyon_x), 0.5 * (h1["y"] + canyon_y) - 2.0, z_cruise),
        (canyon_x, canyon_y, z_cruise),
        (pylon["cx"], pylon_bypass_y, z_cruise),
        (east_gap_x, east_gap_y, z_cruise),
    ]
    for pt in waypoints:
        clr = point_min_clearance(pt, cfg, include_nfz=True, include_bounds=True, include_safety_margin=True)
        assert math.isfinite(clr) and clr > 1.5, f"Corridor waypoint {pt} clearance too small: {clr}"

    for p1, p2 in zip(waypoints, waypoints[1:]):
        assert check_line_of_sight(p1, p2, cfg, margin_m=0.5, include_nfz=True) is True, (
            f"Expected clear corridor LoS between {p1} and {p2}"
        )
