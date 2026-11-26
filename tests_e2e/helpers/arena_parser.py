"""Arena YAML & Gazebo SDF 3D geometry parser, clearance SDF, and LoS oracle."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml

Pose6 = Tuple[float, float, float, float, float, float]


@dataclass
class SdfCollision:
    """Parsed SDF collision geometry with local and composed world poses."""

    model: str
    link: str
    name: str
    geometry_type: str
    dimensions: Dict[str, float]
    local_pose: Pose6
    world_pose: Pose6
    is_ground: bool


@dataclass
class SdfWorldAudit:
    """Structured audit of an SDF world or model file."""

    world_name: str
    collisions: List[SdfCollision]
    lights: List[Dict[str, Any]]
    plugins: List[Dict[str, str]]
    models: List[Dict[str, Any]]


def _is_finite_number(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def _require_finite_float(mapping: Dict[str, Any], key: str, ctx: str) -> float:
    if key not in mapping:
        raise ValueError(f"Missing required key '{key}' in {ctx}")
    val = mapping[key]
    if not _is_finite_number(val):
        raise ValueError(f"Key '{key}' in {ctx} must be a finite number, got {val!r}")
    return float(val)


def load_obstacles_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load and validate obstacles.yaml into a normalized dictionary."""
    yaml_path = Path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Obstacles YAML file not found: {yaml_path}")

    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected top-level YAML mapping in {yaml_path}")

    for section in ("arena_bounds", "helipads", "obstacles", "no_fly_zones"):
        if section not in raw:
            raise ValueError(f"Missing required top-level section '{section}' in {yaml_path}")

    bounds_raw = raw["arena_bounds"]
    if not isinstance(bounds_raw, dict):
        raise ValueError("arena_bounds must be a mapping")
    arena_bounds = {
        k: _require_finite_float(bounds_raw, k, "arena_bounds")
        for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
    }
    if not (
        arena_bounds["x_min"] < arena_bounds["x_max"]
        and arena_bounds["y_min"] < arena_bounds["y_max"]
        and arena_bounds["z_min"] < arena_bounds["z_max"]
    ):
        raise ValueError(f"Invalid arena_bounds ordering: {arena_bounds}")

    helipads_raw = raw["helipads"]
    if not isinstance(helipads_raw, list):
        raise ValueError("helipads must be a list")
    helipads: List[Dict[str, Any]] = []
    for i, hp in enumerate(helipads_raw):
        if not isinstance(hp, dict):
            raise ValueError(f"helipads[{i}] must be a mapping")
        hid = str(hp.get("id", "")).strip()
        drone_id = str(hp.get("drone_id", "")).strip()
        if not hid or not drone_id:
            raise ValueError(f"helipads[{i}] requires non-empty 'id' and 'drone_id'")
        helipads.append(
            {
                "id": hid,
                "drone_id": drone_id,
                "x": _require_finite_float(hp, "x", f"helipads[{i}]"),
                "y": _require_finite_float(hp, "y", f"helipads[{i}]"),
                "z": _require_finite_float(hp, "z", f"helipads[{i}]"),
                "yaw": float(hp.get("yaw", 0.0))
                if _is_finite_number(hp.get("yaw", 0.0))
                else _require_finite_float(hp, "yaw", f"helipads[{i}]"),
            }
        )

    obstacles_raw = raw["obstacles"]
    if not isinstance(obstacles_raw, list):
        raise ValueError("obstacles must be a list")
    obstacles: List[Dict[str, Any]] = []
    for i, obs in enumerate(obstacles_raw):
        if not isinstance(obs, dict):
            raise ValueError(f"obstacles[{i}] must be a mapping")
        oid = str(obs.get("id", "")).strip()
        otype = str(obs.get("type", "")).strip().lower()
        if not oid:
            raise ValueError(f"obstacles[{i}] requires non-empty 'id'")
        if otype not in ("cylinder", "box"):
            raise ValueError(f"obstacles[{i}] invalid type '{otype}' (expected cylinder|box)")
        cx = _require_finite_float(obs, "cx", f"obstacles[{i}]")
        cy = _require_finite_float(obs, "cy", f"obstacles[{i}]")
        z_min = _require_finite_float(obs, "z_min", f"obstacles[{i}]")
        z_max = _require_finite_float(obs, "z_max", f"obstacles[{i}]")
        if z_min >= z_max:
            raise ValueError(f"obstacles[{i}] z_min ({z_min}) must be < z_max ({z_max})")

        margin_raw = obs.get("safety_margin_m", 0.0)
        if not _is_finite_number(margin_raw) or float(margin_raw) < 0.0:
            raise ValueError(f"obstacles[{i}] safety_margin_m must be >= 0")

        yaw_raw = obs.get("yaw", 0.0)
        if not _is_finite_number(yaw_raw):
            raise ValueError(f"obstacles[{i}] yaw must be finite")

        entry: Dict[str, Any] = {
            "id": oid,
            "type": otype,
            "cx": cx,
            "cy": cy,
            "z_min": z_min,
            "z_max": z_max,
            "yaw": float(yaw_raw),
            "safety_margin_m": float(margin_raw),
            "visual_style": str(obs.get("visual_style", "")),
        }
        if otype == "cylinder":
            radius = _require_finite_float(obs, "radius", f"obstacles[{i}]")
            if radius <= 0.0:
                raise ValueError(f"obstacles[{i}] cylinder radius must be > 0")
            entry["radius"] = radius
        else:
            sx = _require_finite_float(obs, "sx", f"obstacles[{i}]")
            sy = _require_finite_float(obs, "sy", f"obstacles[{i}]")
            if sx <= 0.0 or sy <= 0.0:
                raise ValueError(f"obstacles[{i}] box sx/sy must be > 0")
            entry["sx"] = sx
            entry["sy"] = sy
        obstacles.append(entry)

    nfz_raw = raw["no_fly_zones"]
    if not isinstance(nfz_raw, list):
        raise ValueError("no_fly_zones must be a list")
    no_fly_zones: List[Dict[str, Any]] = []
    for i, nfz in enumerate(nfz_raw):
        if not isinstance(nfz, dict):
            raise ValueError(f"no_fly_zones[{i}] must be a mapping")
        nid = str(nfz.get("id", "")).strip()
        if not nid:
            raise ValueError(f"no_fly_zones[{i}] requires non-empty 'id'")
        cx = _require_finite_float(nfz, "cx", f"no_fly_zones[{i}]")
        cy = _require_finite_float(nfz, "cy", f"no_fly_zones[{i}]")
        z_min = _require_finite_float(nfz, "z_min", f"no_fly_zones[{i}]")
        z_max = _require_finite_float(nfz, "z_max", f"no_fly_zones[{i}]")
        if z_min >= z_max:
            raise ValueError(f"no_fly_zones[{i}] z_min ({z_min}) must be < z_max ({z_max})")
        radius = _require_finite_float(nfz, "radius", f"no_fly_zones[{i}]")
        if radius <= 0.0:
            raise ValueError(f"no_fly_zones[{i}] radius must be > 0")
        no_fly_zones.append(
            {
                "id": nid,
                "cx": cx,
                "cy": cy,
                "z_min": z_min,
                "z_max": z_max,
                "radius": radius,
            }
        )

    return {
        "arena_bounds": arena_bounds,
        "helipads": helipads,
        "obstacles": obstacles,
        "no_fly_zones": no_fly_zones,
    }


def _parse_pose_text(text: Optional[str]) -> Pose6:
    if not text or not text.strip():
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    vals = [float(x) for x in text.strip().split()]
    if len(vals) != 6:
        raise ValueError(f"Expected 6 pose values, got {len(vals)}: {text!r}")
    return (vals[0], vals[1], vals[2], vals[3], vals[4], vals[5])


def _pose_to_matrix(pose: Pose6) -> np.ndarray:
    x, y, z, roll, pitch, yaw = pose
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    rot = rz @ ry @ rx
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = [x, y, z]
    return mat


def _matrix_to_pose(mat: np.ndarray) -> Pose6:
    x, y, z = float(mat[0, 3]), float(mat[1, 3]), float(mat[2, 3])
    rot = mat[:3, :3]
    sy = -float(rot[2, 0])
    sy_clamped = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy_clamped)
    if abs(abs(sy_clamped) - 1.0) > 1e-9:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = math.atan2(-float(rot[1, 2]), float(rot[1, 1]))
        yaw = 0.0
    return (x, y, z, roll, pitch, yaw)


def _compose_poses(*poses: Pose6) -> Pose6:
    mat = np.eye(4, dtype=np.float64)
    for p in poses:
        mat = mat @ _pose_to_matrix(p)
    return _matrix_to_pose(mat)


def _parse_geometry(geom_elem: Optional[ET.Element]) -> Tuple[str, Dict[str, float]]:
    if geom_elem is None:
        return ("unknown", {})
    box = geom_elem.find("box")
    if box is not None:
        size_txt = box.findtext("size", default="0 0 0")
        sx, sy, sz = [float(v) for v in size_txt.strip().split()[:3]]
        return ("box", {"sx": sx, "sy": sy, "sz": sz})
    cyl = geom_elem.find("cylinder")
    if cyl is not None:
        radius = float(cyl.findtext("radius", default="0"))
        length = float(cyl.findtext("length", default="0"))
        return ("cylinder", {"radius": radius, "length": length})
    sph = geom_elem.find("sphere")
    if sph is not None:
        radius = float(sph.findtext("radius", default="0"))
        return ("sphere", {"radius": radius})
    plane = geom_elem.find("plane")
    if plane is not None:
        norm_txt = plane.findtext("normal", default="0 0 1")
        size_txt = plane.findtext("size", default="0 0")
        nx, ny, nz = [float(v) for v in norm_txt.strip().split()[:3]]
        sx, sy = [float(v) for v in size_txt.strip().split()[:2]]
        return ("plane", {"nx": nx, "ny": ny, "nz": nz, "sx": sx, "sy": sy})
    mesh = geom_elem.find("mesh")
    if mesh is not None:
        return ("mesh", {})
    return ("unknown", {})


def parse_sdf_collisions(sdf_path: Union[str, Path]) -> SdfWorldAudit:
    """Parse SDF world/model collisions, lights, plugins, and model summaries."""
    path = Path(sdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"SDF file not found: {path}")

    tree = ET.parse(path)
    root = tree.getroot()
    world = root.find("world")
    if world is None:
        world = root
    world_name = world.attrib.get("name", path.stem)

    collisions: List[SdfCollision] = []
    models: List[Dict[str, Any]] = []
    lights: List[Dict[str, Any]] = []
    plugins: List[Dict[str, str]] = []

    for plug in world.findall(".//plugin"):
        inner_tags = [child.tag for child in plug]
        plugins.append(
            {
                "name": plug.attrib.get("name", ""),
                "filename": plug.attrib.get("filename", ""),
                "inner_tags": ",".join(inner_tags),
            }
        )

    for light in world.findall("light"):
        lpose = _parse_pose_text(light.findtext("pose"))
        cast_txt = (light.findtext("cast_shadows", default="false") or "false").strip().lower()
        intensity_txt = light.findtext("intensity")
        lights.append(
            {
                "name": light.attrib.get("name", ""),
                "type": light.attrib.get("type", ""),
                "pose": lpose,
                "cast_shadows": cast_txt == "true",
                "intensity": float(intensity_txt) if intensity_txt else None,
                "direction": light.findtext("direction", default="").strip(),
            }
        )

    model_elems = world.findall("model")
    if not model_elems and world.tag == "model":
        model_elems = [world]

    for model in model_elems:
        m_name = model.attrib.get("name", "")
        m_pose = _parse_pose_text(model.findtext("pose"))
        m_static = (model.findtext("static", default="false") or "false").strip().lower() == "true"
        m_collisions = 0
        m_visuals = 0
        m_sensors: List[str] = []

        for link in model.findall("link"):
            l_name = link.attrib.get("name", "")
            l_pose = _parse_pose_text(link.findtext("pose"))
            m_visuals += len(link.findall("visual"))
            for sensor in link.findall("sensor"):
                m_sensors.append(sensor.attrib.get("name", sensor.attrib.get("type", "")))

            for col in link.findall("collision"):
                c_name = col.attrib.get("name", "")
                c_pose = _parse_pose_text(col.findtext("pose"))
                g_type, dims = _parse_geometry(col.find("geometry"))
                world_pose = _compose_poses(m_pose, l_pose, c_pose)
                is_ground = (g_type == "plane") or ("ground" in m_name.lower())
                collisions.append(
                    SdfCollision(
                        model=m_name,
                        link=l_name,
                        name=c_name,
                        geometry_type=g_type,
                        dimensions=dims,
                        local_pose=c_pose,
                        world_pose=world_pose,
                        is_ground=is_ground,
                    )
                )
                m_collisions += 1

        models.append(
            {
                "name": m_name,
                "static": m_static,
                "pose": m_pose,
                "collision_count": m_collisions,
                "visual_count": m_visuals,
                "sensors": m_sensors,
                "has_particle_emitter": model.find(".//particle_emitter") is not None,
            }
        )

    return SdfWorldAudit(
        world_name=world_name,
        collisions=collisions,
        lights=lights,
        plugins=plugins,
        models=models,
    )


def _validate_point3(point_xyz: Sequence[float], name: str = "point") -> Tuple[float, float, float]:
    if len(point_xyz) != 3:
        raise ValueError(f"{name} must have length 3, got {len(point_xyz)}")
    x, y, z = point_xyz
    if not (_is_finite_number(x) and _is_finite_number(y) and _is_finite_number(z)):
        raise ValueError(f"{name} must contain finite coordinates, got {point_xyz!r}")
    return (float(x), float(y), float(z))


def _cylinder_sdf(
    x: float,
    y: float,
    z: float,
    cx: float,
    cy: float,
    z_min: float,
    z_max: float,
    radius: float,
) -> float:
    d_xy = math.hypot(x - cx, y - cy) - radius
    d_z = max(z_min - z, z - z_max)
    if d_xy > 0.0 and d_z > 0.0:
        return math.hypot(d_xy, d_z)
    return max(d_xy, d_z)


def _oriented_box_sdf(
    x: float,
    y: float,
    z: float,
    cx: float,
    cy: float,
    z_min: float,
    z_max: float,
    sx: float,
    sy: float,
    yaw: float,
) -> float:
    dx = x - cx
    dy = y - cy
    cz = 0.5 * (z_min + z_max)
    c = math.cos(yaw)
    s = math.sin(yaw)
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    lz = z - cz

    hx = 0.5 * sx
    hy = 0.5 * sy
    hz = 0.5 * (z_max - z_min)

    qx = abs(lx) - hx
    qy = abs(ly) - hy
    qz = abs(lz) - hz

    out = math.sqrt(max(qx, 0.0) ** 2 + max(qy, 0.0) ** 2 + max(qz, 0.0) ** 2)
    inside = min(max(qx, qy, qz), 0.0)
    return out + inside


def point_min_clearance(
    point_xyz: Sequence[float],
    obstacles_cfg: Dict[str, Any],
    include_nfz: bool = True,
    include_bounds: bool = False,
    include_safety_margin: bool = False,
) -> float:
    """Compute exact minimum 3D signed distance (meters) from a point to arena primitives."""
    x, y, z = _validate_point3(point_xyz, "point_xyz")
    min_dist = float("inf")

    for obs in obstacles_cfg.get("obstacles", []):
        margin = float(obs.get("safety_margin_m", 0.0)) if include_safety_margin else 0.0
        z_min = float(obs["z_min"]) - (margin if include_safety_margin else 0.0)
        z_max = float(obs["z_max"]) + (margin if include_safety_margin else 0.0)
        if obs["type"] == "cylinder":
            d = _cylinder_sdf(
                x,
                y,
                z,
                float(obs["cx"]),
                float(obs["cy"]),
                z_min,
                z_max,
                float(obs["radius"]) + margin,
            )
        else:
            d = _oriented_box_sdf(
                x,
                y,
                z,
                float(obs["cx"]),
                float(obs["cy"]),
                z_min,
                z_max,
                float(obs["sx"]) + 2.0 * margin,
                float(obs["sy"]) + 2.0 * margin,
                float(obs.get("yaw", 0.0)),
            )
        if d < min_dist:
            min_dist = d

    if include_nfz:
        for nfz in obstacles_cfg.get("no_fly_zones", []):
            d = _cylinder_sdf(
                x,
                y,
                z,
                float(nfz["cx"]),
                float(nfz["cy"]),
                float(nfz["z_min"]),
                float(nfz["z_max"]),
                float(nfz["radius"]),
            )
            if d < min_dist:
                min_dist = d

    if include_bounds and "arena_bounds" in obstacles_cfg:
        b = obstacles_cfg["arena_bounds"]
        d_bounds = min(
            x - float(b["x_min"]),
            float(b["x_max"]) - x,
            y - float(b["y_min"]),
            float(b["y_max"]) - y,
            z - float(b["z_min"]),
            float(b["z_max"]) - z,
        )
        if d_bounds < min_dist:
            min_dist = d_bounds

    return min_dist


def _segment_intersects_box(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    cx: float,
    cy: float,
    z_min: float,
    z_max: float,
    sx: float,
    sy: float,
    yaw: float,
    margin_m: float,
) -> bool:
    """Kay-Kajiya 3D slab test in box-local coordinates."""
    cz = 0.5 * (z_min + z_max)
    c = math.cos(yaw)
    s = math.sin(yaw)

    def to_local(pt: Tuple[float, float, float]) -> Tuple[float, float, float]:
        dx, dy, dz = pt[0] - cx, pt[1] - cy, pt[2] - cz
        return (c * dx + s * dy, -s * dx + c * dy, dz)

    a = to_local(p1)
    b = to_local(p2)
    d = (b[0] - a[0], b[1] - a[1], b[2] - a[2])
    half = (
        0.5 * sx + margin_m,
        0.5 * sy + margin_m,
        0.5 * (z_max - z_min) + margin_m,
    )

    t0, t1 = 0.0, 1.0
    eps = 1e-12
    for i in range(3):
        if abs(d[i]) < eps:
            if a[i] < -half[i] or a[i] > half[i]:
                return False
        else:
            inv_d = 1.0 / d[i]
            t_near = (-half[i] - a[i]) * inv_d
            t_far = (half[i] - a[i]) * inv_d
            if t_near > t_far:
                t_near, t_far = t_far, t_near
            t0 = max(t0, t_near)
            t1 = min(t1, t_far)
            if t0 > t1:
                return False
    return True


def _segment_intersects_cylinder(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    cx: float,
    cy: float,
    z_min: float,
    z_max: float,
    radius: float,
    margin_m: float,
) -> bool:
    """Exact analytical segment vs finite vertical cylinder intersection."""
    ox = p1[0] - cx
    oy = p1[1] - cy
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    dz = p2[2] - p1[2]

    z_lo = z_min - margin_m
    z_hi = z_max + margin_m
    t0, t1 = 0.0, 1.0
    eps = 1e-12

    if abs(dz) < eps:
        if p1[2] < z_lo or p1[2] > z_hi:
            return False
    else:
        tz0 = (z_lo - p1[2]) / dz
        tz1 = (z_hi - p1[2]) / dz
        if tz0 > tz1:
            tz0, tz1 = tz1, tz0
        t0 = max(t0, tz0)
        t1 = min(t1, tz1)
        if t0 > t1:
            return False

    r_eff = radius + margin_m
    if r_eff <= 0.0:
        return False

    a = dx * dx + dy * dy
    c = ox * ox + oy * oy - r_eff * r_eff
    if a < eps:
        return c <= 0.0

    b = 2.0 * (ox * dx + oy * dy)
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False

    sqrt_disc = math.sqrt(disc)
    tr0 = (-b - sqrt_disc) / (2.0 * a)
    tr1 = (-b + sqrt_disc) / (2.0 * a)
    t_enter = max(t0, tr0)
    t_exit = min(t1, tr1)
    return t_enter <= t_exit


def check_line_of_sight(
    p1_xyz: Sequence[float],
    p2_xyz: Sequence[float],
    obstacles_cfg: Dict[str, Any],
    margin_m: float = 0.0,
    include_nfz: bool = False,
) -> bool:
    """Return True if segment [p1, p2] is unobstructed by obstacles (and optionally NFZs)."""
    p1 = _validate_point3(p1_xyz, "p1_xyz")
    p2 = _validate_point3(p2_xyz, "p2_xyz")
    if not _is_finite_number(margin_m) or float(margin_m) < 0.0:
        raise ValueError(f"margin_m must be finite and >= 0, got {margin_m!r}")
    margin = float(margin_m)

    seg_len_sq = (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2
    if seg_len_sq < 1e-24:
        return point_min_clearance(p1, obstacles_cfg, include_nfz=include_nfz) > margin

    for obs in obstacles_cfg.get("obstacles", []):
        if obs["type"] == "cylinder":
            hit = _segment_intersects_cylinder(
                p1,
                p2,
                float(obs["cx"]),
                float(obs["cy"]),
                float(obs["z_min"]),
                float(obs["z_max"]),
                float(obs["radius"]),
                margin,
            )
        else:
            hit = _segment_intersects_box(
                p1,
                p2,
                float(obs["cx"]),
                float(obs["cy"]),
                float(obs["z_min"]),
                float(obs["z_max"]),
                float(obs["sx"]),
                float(obs["sy"]),
                float(obs.get("yaw", 0.0)),
                margin,
            )
        if hit:
            return False

    if include_nfz:
        for nfz in obstacles_cfg.get("no_fly_zones", []):
            if _segment_intersects_cylinder(
                p1,
                p2,
                float(nfz["cx"]),
                float(nfz["cy"]),
                float(nfz["z_min"]),
                float(nfz["z_max"]),
                float(nfz["radius"]),
                margin,
            ):
                return False

    return True
