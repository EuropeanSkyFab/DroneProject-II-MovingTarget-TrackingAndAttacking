# -*- coding: utf-8 -*-
import math
import sys
import os

from flight_ctrl_pkg.coord_utils import (
    clamp_step_3d,
    compute_gimbal_pitch_ned,
    compute_shadow_and_ff,
    compute_shadow_ff_accel,
    compute_yawspeed_ned,
    enu_to_ned_pos,
    ned_to_enu_pos,
    ned_to_enu_vel,
    quat_frd_ned_to_flu_enu,
    slew_yaw_ned,
    yaw_enu_to_ned,
    yaw_ned_to_enu,
)


def _quat_xyzw_to_mat(qx, qy, qz, qw):
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def test_pos_vel_roundtrip():
    for ox, oy in [(0.0, 0.0), (0.0, 2.0), (0.0, 4.0)]:
        x, y, z = 12.5, -7.3, 8.2
        n, e, d = enu_to_ned_pos(x, y, z, ox, oy)
        assert math.isclose(n, y - oy, abs_tol=1e-7)
        assert math.isclose(e, x - ox, abs_tol=1e-7)
        assert math.isclose(d, -z, abs_tol=1e-7)
        x2, y2, z2 = ned_to_enu_pos(n, e, d, ox, oy)
        assert math.isclose(x2, x, abs_tol=1e-7)
        assert math.isclose(y2, y, abs_tol=1e-7)
        assert math.isclose(z2, z, abs_tol=1e-7)

    vx, vy, vz = ned_to_enu_vel(3.0, -2.0, -1.5)
    assert (vx, vy, vz) == (-2.0, 3.0, 1.5)


def test_yaw_conversions():
    assert math.isclose(yaw_enu_to_ned(0.0), math.pi / 2.0, abs_tol=1e-7)
    assert math.isclose(yaw_enu_to_ned(math.pi / 2.0), 0.0, abs_tol=1e-7)
    assert math.isclose(yaw_ned_to_enu(0.0), math.pi / 2.0, abs_tol=1e-7)
    assert math.isclose(yaw_ned_to_enu(math.pi / 2.0), 0.0, abs_tol=1e-7)


def test_quaternion_frd_ned_to_flu_enu():
    # Case 1: PX4 leveled facing North -> q_wxyz = (1, 0, 0, 0)
    # In ENU: Body Forward (+x FLU) should map to +North (0, 1, 0)
    q_enu = quat_frd_ned_to_flu_enu((1.0, 0.0, 0.0, 0.0))
    R = _quat_xyzw_to_mat(*q_enu)
    assert math.isclose(R[0][0], 0.0, abs_tol=1e-6)
    assert math.isclose(R[1][0], 1.0, abs_tol=1e-6)
    assert math.isclose(R[2][0], 0.0, abs_tol=1e-6)
    # Body Up (+z FLU) should map to +Up (0, 0, 1)
    assert math.isclose(R[2][2], 1.0, abs_tol=1e-6)

    # Case 2: PX4 leveled facing East (yaw_ned = +90 deg) -> q_wxyz = (cos(pi/4), 0, 0, sin(pi/4))
    s = math.sqrt(0.5)
    q_east = quat_frd_ned_to_flu_enu((s, 0.0, 0.0, s))
    R_east = _quat_xyzw_to_mat(*q_east)
    # Forward (+x FLU) maps to +East (1, 0, 0)
    assert math.isclose(R_east[0][0], 1.0, abs_tol=1e-6)
    assert math.isclose(R_east[1][0], 0.0, abs_tol=1e-6)
    assert math.isclose(R_east[1][1], 1.0, abs_tol=1e-6)

    # Case 3: PX4 pitch nose up +30 deg while facing North
    half = math.radians(15.0)
    q_pitch_up = quat_frd_ned_to_flu_enu((math.cos(half), 0.0, math.sin(half), 0.0))
    R_up = _quat_xyzw_to_mat(*q_pitch_up)
    # Forward axis in ENU should have positive North (y) and positive Up (z)
    assert math.isclose(R_up[1][0], math.cos(math.radians(30.0)), abs_tol=1e-6)
    assert math.isclose(R_up[2][0], math.sin(math.radians(30.0)), abs_tol=1e-6)


def test_pixel_to_world_integration():
    # Directly verify compatibility with vision_pkg.geo_utils
    vision_pkg_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "vision_pkg")
    )
    if os.path.isdir(vision_pkg_root) and vision_pkg_root not in sys.path:
        sys.path.insert(0, vision_pkg_root)
    from vision_pkg.geo_utils import camera_mount_matrix, pixel_to_world

    # Drone at ENU (10, 20, 10), leveled facing East (PX4 yaw_ned = 90 deg)
    s = math.sqrt(0.5)
    q_flu_enu = quat_frd_ned_to_flu_enu((s, 0.0, 0.0, s))
    R_bc = camera_mount_matrix(0.0, 0.0, 0.0)
    k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
    # Center pixel (320, 240) straight down -> should hit ground at (10, 20, 0)
    pt = pixel_to_world(320.0, 240.0, k, (10.0, 20.0, 10.0), q_flu_enu, R_bc, ground_z=0.0)
    assert pt is not None
    assert math.isclose(pt[0], 10.0, abs_tol=1e-5)
    assert math.isclose(pt[1], 20.0, abs_tol=1e-5)
    assert math.isclose(pt[2], 0.0, abs_tol=1e-5)


def test_clamp_step_3d():
    n, e, d = clamp_step_3d(0.0, 0.0, -5.0, 10.0, 0.0, -5.0, 0.5)
    assert math.isclose(n, 0.5, abs_tol=1e-7)
    assert math.isclose(e, 0.0, abs_tol=1e-7)
    assert math.isclose(d, -5.0, abs_tol=1e-7)

    n2, e2, d2 = clamp_step_3d(0.0, 0.0, -5.0, 0.1, 0.1, -5.0, 0.5)
    assert (n2, e2, d2) == (0.1, 0.1, -5.0)


def test_forward_35deg_camera_pixel_to_world():
    vision_pkg_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "vision_pkg")
    )
    if os.path.isdir(vision_pkg_root) and vision_pkg_root not in sys.path:
        sys.path.insert(0, vision_pkg_root)
    from vision_pkg.geo_utils import camera_mount_matrix, pixel_to_world

    # Drone at ENU (0, 0, 9.5) leveled facing East (+X, PX4 yaw_ned = 90 deg)
    # SDF pitch=+35 deg below horizon corresponds to vision camera_mount_rpy=[pi/2 - 35deg, 0, -pi/2]
    s = math.sqrt(0.5)
    q_flu_enu = quat_frd_ned_to_flu_enu((s, 0.0, 0.0, s))
    pitch_down = math.radians(35.0)
    R_bc = camera_mount_matrix(math.pi / 2.0 - pitch_down, 0.0, -math.pi / 2.0)
    k = [430.0, 0.0, 320.0, 0.0, 430.0, 240.0, 0.0, 0.0, 1.0]
    # Optical axis pitched 35 deg forward-down intersects z=3.5 at x = (9.5 - 3.5) / tan(35 deg)
    pt = pixel_to_world(320.0, 240.0, k, (0.0, 0.0, 9.5), q_flu_enu, R_bc, ground_z=3.5)
    assert pt is not None
    assert math.isclose(pt[0], 6.0 / math.tan(pitch_down), abs_tol=1e-4)
    assert math.isclose(pt[1], 0.0, abs_tol=1e-4)
    assert math.isclose(pt[2], 3.5, abs_tol=1e-4)
    # Pixel to right of center (u > cx) projects to South (-Y in ENU when facing East)
    pt_r = pixel_to_world(420.0, 240.0, k, (0.0, 0.0, 9.5), q_flu_enu, R_bc, ground_z=3.5)
    assert pt_r is not None and pt_r[1] < -1.0


def test_slew_yaw_and_ff():
    # Wrap across +pi / -pi shortest arc
    y = slew_yaw_ned(math.radians(170.0), math.radians(-170.0), math.radians(10.0))
    assert math.isclose(y, math.radians(-180.0), abs_tol=1e-6) or math.isclose(
        abs(y), math.pi, abs_tol=1e-6
    )

    sh, vel = compute_shadow_and_ff(
        (0.0, 0.0, -5.0),
        (10.0, 0.0, -5.0),
        (20.0, 0.0, -5.0),
        (1.0, 0.0, 0.0),
        max_step=0.55,
        max_tether=3.0,
        max_speed=11.0,
        gain_p=3.5,
        deadband_m=0.08,
        max_vz_up=3.0,
        max_vz_dn=4.0,
    )
    # Shadow tethered within 3.0m before stepping +0.55m -> sh[0] = 10 - 3 + 0.55 = 7.55
    assert math.isclose(sh[0], 7.55, abs_tol=1e-6)
    assert math.isclose(vel[0], 11.0, abs_tol=1e-6)
    assert math.isclose(vel[2], 0.0, abs_tol=1e-6)


def test_accel_ff_yawspeed_and_gimbal_pitch():
    # Acceleration feedforward accelerates positively during velocity ramp and clamps safely
    sh, vel, acc = compute_shadow_ff_accel(
        (0.0, 0.0, -10.0),
        (0.0, 0.0, -10.0),
        (25.0, 0.0, -5.5),
        (1.5, 0.0, 0.0),
        prev_vel_cmd_ned=(0.0, 0.0, 0.0),
        prev_acc_cmd_ned=(0.0, 0.0, 0.0),
        dt=0.02,
        max_step=0.25,
        max_tether=3.0,
        max_speed=12.4,
        gain_p=3.2,
        deadband_m=0.06,
        max_vz_up=4.5,
        max_vz_dn=5.5,
        max_acc_hor=10.0,
        max_acc_vert=5.0,
    )
    assert math.hypot(sh[0], sh[2] + 10.0) <= 0.25 + 1e-6
    assert vel[0] > 5.0 and vel[2] > 0.0
    assert 0.0 < acc[0] <= 10.0 + 1e-6
    assert 0.0 <= acc[2] <= 5.0 + 1e-6

    # Shortest-arc yaw rate wrap across +/-pi
    yr = compute_yawspeed_ned(math.radians(170.0), math.radians(-170.0), tau=0.25, max_yaw_rate_rad=math.pi)
    assert yr > 0.5

    # Gimbal pitch from 10m drone looking at 5.5m aerial target at 5.2m horizontal range -> negative (nose down)
    pitch = compute_gimbal_pitch_ned(10.0, 5.5, 5.2)
    assert math.isclose(pitch, math.degrees(math.atan2(-4.5, 5.2)), abs_tol=1e-4)
