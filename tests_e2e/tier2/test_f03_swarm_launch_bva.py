"""Tier 2 BVA & corner-case tests for F3: Scalable N >= 6 Swarm Launch & Drone SDF."""

from __future__ import annotations

import json
import math
import os
import shlex
import signal
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from helpers.arena_parser import load_obstacles_yaml
from helpers.ros_probe import run_command_with_gpu_guard

pytestmark = [pytest.mark.tier2, pytest.mark.feature(id="F3"), pytest.mark.requires_m1]

ROS2_WS = Path("/home/hkw/flying/ros2_ws")
FC_PKG = ROS2_WS / "src" / "flight_ctrl_pkg"
OBSTACLES_YAML = FC_PKG / "config" / "obstacles.yaml"
START_SWARM_SH = FC_PKG / "scripts" / "start_px4_swarm.sh"
X500_DOWN_CAM_SDF = FC_PKG / "sim" / "models" / "x500_down_cam" / "model.sdf"


def _inspect_installed_swarm_launch(num_drones: int) -> List[Dict[str, Any]]:
    """Expand installed flight_ctrl_swarm.launch.py via ROS 2 public launch APIs."""
    py = (
        "import json, os, yaml\n"
        "from ament_index_python.packages import get_package_share_directory\n"
        "from launch import LaunchContext\n"
        "from launch.actions import IncludeLaunchDescription\n"
        "from launch.launch_description_sources import PythonLaunchDescriptionSource\n"
        "from launch.utilities import perform_substitutions\n"
        "from launch_ros.actions import Node\n"
        "share = get_package_share_directory('flight_ctrl_pkg')\n"
        "ctx = LaunchContext()\n"
        "inc = IncludeLaunchDescription(\n"
        "    PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'flight_ctrl_swarm.launch.py')),\n"
        f"    launch_arguments={{'num_drones': '{num_drones}', 'enable_camera': 'true'}}.items(),\n"
        ")\n"
        "def walk(entities):\n"
        "    out = []\n"
        "    for e in entities:\n"
        "        if isinstance(e, Node):\n"
        "            out.append(e)\n"
        "        elif hasattr(e, 'visit'):\n"
        "            sub = e.visit(ctx)\n"
        "            if sub:\n"
        "                out.extend(walk(sub))\n"
        "    return out\n"
        "def _unyaml(s):\n"
        "    if isinstance(s, str) and '\\n' in s:\n"
        "        loaded = yaml.safe_load(s)\n"
        "        return '' if loaded is None else loaded\n"
        "    return s\n"
        "def resolve_val(v):\n"
        "    if hasattr(v, 'perform'):\n"
        "        return _unyaml(v.perform(ctx))\n"
        "    if isinstance(v, (list, tuple)):\n"
        "        if len(v) == 1 and hasattr(v[0], 'perform'):\n"
        "            return _unyaml(v[0].perform(ctx))\n"
        "        if (v and any(hasattr(x, 'perform') for x in v)\n"
        "                and all(isinstance(x, str) or hasattr(x, 'perform') for x in v)):\n"
        "            return _unyaml(perform_substitutions(ctx, [x for x in v]))\n"
        "        return [resolve_val(x) for x in v]\n"
        "    if isinstance(v, dict):\n"
        "        return {str(resolve_val(dk)): resolve_val(dv) for dk, dv in v.items()}\n"
        "    return v\n"
        "nodes = []\n"
        "for n in walk([inc]):\n"
        "    # Humble launch_ros: Node.node_name raises RuntimeError before execute(),\n"
        "    # and Node has no public parameters getter; use public pkg/exe properties.\n"
        "    name = resolve_val(getattr(n, '_Node__node_name', None))\n"
        "    pkg = resolve_val(n.node_package)\n"
        "    exe = resolve_val(n.node_executable)\n"
        "    params = {}\n"
        "    for p in (getattr(n, '_Node__parameters', None) or []):\n"
        "        if isinstance(p, dict):\n"
        "            for k, v in p.items():\n"
        "                params[str(resolve_val(k))] = resolve_val(v)\n"
        "    nodes.append({'name': name, 'package': pkg, 'executable': exe, 'params': params})\n"
        "print(json.dumps(nodes, default=str))\n"
    )
    res = run_command_with_gpu_guard(
        ["python3", "-c", py],
        timeout_s=15.0,
        source_env=True,
    )
    assert res.returncode == 0, (
        f"Failed to expand installed flight_ctrl_swarm.launch.py (num_drones={num_drones}): "
        f"{res.stderr.strip() or res.stdout.strip()}"
    )
    return json.loads(res.stdout.strip().splitlines()[-1])


def test_f03_bva_minimal_n1_single_drone_launch_config() -> None:
    """Verify N=1 minimal config produces exactly 1 px4_bridge_node with px4_ns='' and H1 pose."""
    nodes = _inspect_installed_swarm_launch(1)
    bridges = [n for n in nodes if n["executable"] == "px4_bridge_node"]
    assert len(bridges) == 1, f"Expected 1 px4_bridge_node for N=1, got {len(bridges)}"
    p = bridges[0]["params"]
    assert p.get("drone_id") == "drone_1"
    assert p.get("px4_ns") == ""
    assert p.get("target_system_id") == 1
    assert "offset_x" in p and "offset_y" in p, f"Missing helipad offset params: {p}"


@pytest.mark.parametrize("num_drones", [8, 12])
def test_f03_bva_n8_and_n12_scaling_and_port_non_collision(num_drones: int) -> None:
    """Verify N=8 and N=12 scale cleanly with unique namespaces, sys_ids, offsets, and MAVLink ports."""
    nodes = _inspect_installed_swarm_launch(num_drones)
    bridges = [n for n in nodes if n["executable"] == "px4_bridge_node"]
    assert len(bridges) == num_drones, (
        f"Expected {num_drones} px4_bridge_node actions, got {len(bridges)}"
    )

    drone_ids = [b["params"]["drone_id"] for b in bridges]
    px4_nss = [b["params"]["px4_ns"] for b in bridges]
    sys_ids = [b["params"]["target_system_id"] for b in bridges]
    offsets = [(float(b["params"]["offset_x"]), float(b["params"]["offset_y"])) for b in bridges]

    assert len(set(drone_ids)) == num_drones
    assert len(set(px4_nss)) == num_drones
    assert px4_nss[0] == ""
    assert px4_nss[1:] == [f"px4_{i}" for i in range(1, num_drones)]
    assert sys_ids == list(range(1, num_drones + 1))

    # Verify non-overlapping spawn offsets and arena_bounds containment (including fallback helipads)
    cfg = load_obstacles_yaml(OBSTACLES_YAML)
    bounds = cfg["arena_bounds"]
    for ox, oy in offsets:
        assert bounds["x_min"] <= ox <= bounds["x_max"], f"Spawn offset x={ox} outside arena_bounds"
        assert bounds["y_min"] <= oy <= bounds["y_max"], f"Spawn offset y={oy} outside arena_bounds"
    for i in range(num_drones):
        for j in range(i + 1, num_drones):
            dist = math.hypot(offsets[i][0] - offsets[j][0], offsets[i][1] - offsets[j][1])
            assert dist >= 2.5, f"Spawn collision between drone_{i+1} and drone_{j+1}: {dist:.2f}m"

    # Verify camera_relay_node scaling for enable_camera=true
    relays = [
        n for n in nodes
        if n["executable"] == "camera_relay_node" and n["params"].get("out_namespace") == "camera"
    ]
    assert len(relays) == num_drones, f"Expected {num_drones} camera_relay_node actions, got {len(relays)}"
    assert [r["params"]["drone_id"] for r in relays] == drone_ids
    assert [r["params"]["gz_model"] for r in relays] == [f"x500_down_cam_{i}" for i in range(num_drones)]

    # Verify start_px4_swarm.sh per-instance isolation and hook wiring
    swarm_txt = START_SWARM_SH.read_text(encoding="utf-8")
    for token in (
        '-i "$i"',
        'INST_DIR="/tmp/px4_sitl_instances/${DRONE_NAME}"',
        'GZ_MODEL_INST="x500_down_cam_${i}"',
        'PX4_GZ_MODEL_NAME="${GZ_MODEL_INST}"',
        'UXRCE_NS="px4_${i}"',
        'PX4_UXRCE_DDS_NS="${UXRCE_NS}"',
        'PATH="${SCRIPT_DIR}:${PATH}"',
    ):
        assert token in swarm_txt, f"start_px4_swarm.sh missing per-instance token: {token}"

    # Execute actual scripts/px4-rc.mavlink -> PX4 etc/init.d-posix/px4-rc.mavlink in /bin/sh with stubs
    px4_rc_mavlink = FC_PKG / "scripts" / "px4-rc.mavlink"
    px4_rc_params = FC_PKG / "scripts" / "px4-rc.params"
    assert px4_rc_mavlink.is_file() and px4_rc_params.is_file()
    sh_script = (
        "set -e\n"
        'mavlink() { if [ "$1" = "start" ]; then echo "MAVLINK|$px4_instance|$*"; fi; }\n'
        'param() { echo "PARAM|$px4_instance|$*"; }\n'
        'export PX4_SIMULATOR=gz\n'
        'export R="$PX4_AUTOPILOT_DIR/build/px4_sitl_default/"\n'
        f'export PX4_RC_PARAMS_POST={shlex.quote(str(px4_rc_params))}\n'
        f'for px4_instance in $(seq 0 {num_drones - 1}); do\n'
        f'  . {shlex.quote(str(px4_rc_mavlink))}\n'
        'done\n'
    )
    mav_res = run_command_with_gpu_guard(
        ["/bin/sh", "-c", sh_script],
        timeout_s=15.0,
        source_env=True,
    )
    assert mav_res.returncode == 0, f"px4-rc.mavlink stub probe failed: {mav_res.stderr}"

    local_ports: List[int] = []
    remote_ports: List[int] = []
    mavlink_counts: Dict[int, int] = {}
    nav_dll_act: Dict[int, str] = {}
    for line in mav_res.stdout.splitlines():
        if line.startswith("MAVLINK|"):
            _, inst_s, rest = line.split("|", 2)
            inst = int(inst_s)
            mavlink_counts[inst] = mavlink_counts.get(inst, 0) + 1
            tokens = shlex.split(rest)
            if "-u" in tokens:
                local_ports.append(int(tokens[tokens.index("-u") + 1]))
            if "-o" in tokens:
                remote_ports.append(int(tokens[tokens.index("-o") + 1]))
        elif line.startswith("PARAM|"):
            _, inst_s, rest = line.split("|", 2)
            parts = shlex.split(rest)
            if len(parts) >= 3 and parts[0] == "set" and parts[1] == "NAV_DLL_ACT":
                nav_dll_act[int(inst_s)] = parts[2]

    for inst in range(num_drones):
        assert mavlink_counts.get(inst) == 4, (
            f"Expected 4 mavlink start lines for px4_instance={inst}, got {mavlink_counts.get(inst)}"
        )
        assert nav_dll_act.get(inst) == "0", (
            f"Expected post-load NAV_DLL_ACT=0 for px4_instance={inst}, got {nav_dll_act.get(inst)!r}"
        )
    assert len(local_ports) == 4 * num_drones
    assert len(set(local_ports)) == 4 * num_drones, f"Local UDP port collision: {local_ports}"
    assert set(remote_ports).isdisjoint(set(local_ports)), (
        f"Remote ports overlap local bind ports: {set(remote_ports) & set(local_ports)}"
    )


@pytest.mark.parametrize("bad_n", ["0", "-1", "-5"])
def test_f03_bva_reject_zero_or_negative_drone_count(tmp_path: Path, bad_n: str) -> None:
    """Verify start_px4_swarm.sh rejects NUM_DRONES <= 0 immediately without launching Gazebo/PX4."""
    log_file = tmp_path / f"swarm_{bad_n}.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    timed_out = False
    with log_file.open("w", encoding="utf-8") as fp:
        proc = subprocess.Popen(
            ["bash", str(START_SWARM_SH), bad_n, "1", "1"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=fp,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Send SIGTERM first so start_px4_swarm.sh trap cleanup stops MicroXRCEAgent/gz sim
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=1.0)
            rc = -1

    output = log_file.read_text(encoding="utf-8", errors="replace")
    assert not timed_out, (
        f"start_px4_swarm.sh hung/launched simulator instead of rejecting NUM_DRONES={bad_n}:\n{output}"
    )
    assert rc != 0, f"start_px4_swarm.sh must reject NUM_DRONES={bad_n}"


def test_f03_bva_helipad_inter_pad_minimum_separation() -> None:
    """Verify every pair of helipads in obstacles.yaml maintains >= 2.5m minimum separation."""
    cfg = load_obstacles_yaml(OBSTACLES_YAML)
    pads = cfg["helipads"]
    min_sep = float("inf")
    for i in range(len(pads)):
        for j in range(i + 1, len(pads)):
            d = math.hypot(pads[i]["x"] - pads[j]["x"], pads[i]["y"] - pads[j]["y"])
            min_sep = min(min_sep, d)
            assert d >= 2.5, (
                f"Helipads {pads[i]['id']} and {pads[j]['id']} separation {d:.2f}m < 2.5m"
            )
    assert min_sep >= 2.5


def _pose_to_se3(pose_text: str) -> np.ndarray:
    vals = [float(v) for v in (pose_text or "0 0 0 0 0 0").split()]
    x, y, z, r, p, yw = vals
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(yw), math.sin(yw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    tf = np.eye(4, dtype=float)
    tf[:3, :3] = rz @ ry @ rx
    tf[:3, 3] = [x, y, z]
    return tf


def _resolve_link_tf(root: ET.Element, link_name: str) -> np.ndarray:
    tf = np.eye(4, dtype=float)
    cur = link_name
    seen = set()
    while cur != "base_link":
        assert cur not in seen, f"Cyclic relative_to chain at {cur}"
        seen.add(cur)
        link_el = root.find(f".//link[@name='{cur}']")
        assert link_el is not None, f"Missing SDF link {cur}"
        pose_el = link_el.find("pose")
        assert pose_el is not None and pose_el.text, f"Missing pose on link {cur}"
        tf = _pose_to_se3(pose_el.text) @ tf
        cur = pose_el.attrib.get("relative_to", "base_link")
    return tf


def test_f03_bva_down_cam_fov_near_body_clearance() -> None:
    """Verify gimbal sphere/lens and landing skids stay strictly outside down_cam optical frustum."""
    root = ET.parse(X500_DOWN_CAM_SDF).getroot()
    down_link = root.find(".//link[@name='down_cam_link']")
    assert down_link is not None
    cam_pose = [float(v) for v in (down_link.findtext("pose") or "").split()]
    cam_pos = np.array(cam_pose[:3], dtype=float)
    pitch = cam_pose[4]
    cam = down_link.find("./sensor[@name='down_cam']/camera")
    assert cam is not None
    hfov = float(cam.findtext("horizontal_fov"))
    w = float(cam.findtext("./image/width"))
    h = float(cam.findtext("./image/height"))
    vfov = 2.0 * math.atan(math.tan(hfov / 2.0) * (h / w))

    cp, sp = math.cos(pitch), math.sin(pitch)
    r_flu_gz = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)

    # Derive gimbal sphere and lens tip dynamically in base_link from SDF link chain
    tf_pitch = _resolve_link_tf(root, "gimbal_pitch_link")
    pitch_link = root.find(".//link[@name='gimbal_pitch_link']")
    assert pitch_link is not None
    sphere_r_txt = pitch_link.findtext("./visual[@name='gimbal_pod_sphere']/geometry/sphere/radius")
    assert sphere_r_txt is not None
    sphere_r = float(sphere_r_txt)
    lens_pose = [float(v) for v in (pitch_link.findtext("./sensor[@name='gimbal_cam']/pose") or "0 0 0 0 0 0").split()]
    rot_pitch = tf_pitch[:3, :3]
    sphere_center = tf_pitch[:3, 3]
    lens_tip = sphere_center + rot_pitch @ np.array(lens_pose[:3], dtype=float)
    gimbal_points = [
        sphere_center,
        sphere_center + rot_pitch @ np.array([sphere_r, 0.0, 0.0]),
        lens_tip,
        sphere_center + rot_pitch @ np.array([0.0, sphere_r, 0.0]),
        sphere_center - rot_pitch @ np.array([0.0, sphere_r, 0.0]),
    ]
    for pt in gimbal_points:
        v_gz = r_flu_gz.T @ (pt - cam_pos)
        assert v_gz[0] < 0.0, f"Gimbal point {pt.tolist()} entered forward half-space: {v_gz}"

    # Derive x500_base landing skid boxes dynamically when PX4 x500_base/model.sdf is present
    px4_dir = Path(os.environ.get("PX4_AUTOPILOT_DIR", "/home/hkw/flying/third_party/PX4-Autopilot"))
    x500_base_sdf = px4_dir / "Tools" / "simulation" / "gz" / "models" / "x500_base" / "model.sdf"
    if x500_base_sdf.is_file():
        base_root = ET.parse(x500_base_sdf).getroot()
        skid_points = []
        for col_name in ("base_link_collision_3", "base_link_collision_4"):
            col_el = base_root.find(f".//collision[@name='{col_name}']")
            assert col_el is not None, f"Missing {col_name} in {x500_base_sdf}"
            c_pose = [float(v) for v in (col_el.findtext("pose") or "0 0 0 0 0 0").split()]
            b_size = [float(v) for v in (col_el.findtext("./geometry/box/size") or "").split()]
            assert len(b_size) == 3
            for dx in (-0.5 * b_size[0], 0.0, 0.5 * b_size[0]):
                skid_points.append(np.array([c_pose[0] + dx, c_pose[1], c_pose[2]], dtype=float))
    else:
        skid_points = [
            np.array([sx, sy, -0.2195])
            for sx in (-0.125, 0.0, 0.125)
            for sy in (-0.132, 0.132)
        ]
    for pt in skid_points:
        v_gz = r_flu_gz.T @ (pt - cam_pos)
        if v_gz[0] > 0.0:
            az = math.atan2(abs(v_gz[1]), v_gz[0])
            el = math.atan2(abs(v_gz[2]), v_gz[0])
            assert az > 0.5 * hfov or el > 0.5 * vfov, (
                f"Landing skid {pt.tolist()} intersects down_cam FoV: az={az:.3f}, el={el:.3f}"
            )
