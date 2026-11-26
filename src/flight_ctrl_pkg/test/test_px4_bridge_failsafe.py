# -*- coding: utf-8 -*-
"""失控保护返航 (FAILSAFE_RTL) 与起飞原点锁定单元/闭环测试。"""

import math
from pathlib import Path
import re
import time

from diagnostic_msgs.msg import DiagnosticStatus
from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import (
    EstimatorStatusFlags,
    VehicleCommand,
    VehicleLocalPosition,
    VehicleStatus,
)
import rclpy
from rclpy.parameter import Parameter
from std_msgs.msg import Empty

from flight_ctrl_pkg.px4_bridge_node import Px4BridgeNode


def _make_lpos(n: float, e: float, d: float, vn: float = 0.0, ve: float = 0.0, vd: float = 0.0):
    msg = VehicleLocalPosition()
    msg.xy_valid = True
    msg.z_valid = True
    msg.x = float(n)
    msg.y = float(e)
    msg.z = float(d)
    msg.vx = float(vn)
    msg.vy = float(ve)
    msg.vz = float(vd)
    msg.heading = 0.0
    return msg


def _make_status(arming_state: int, nav_state: int, preflight_ok: bool = True):
    st = VehicleStatus()
    st.arming_state = int(arming_state)
    st.nav_state = int(nav_state)
    st.pre_flight_checks_pass = bool(preflight_ok)
    return st


def _make_est(ok: bool = True):
    ef = EstimatorStatusFlags()
    ef.cs_tilt_align = ok
    ef.cs_yaw_align = ok
    return ef


def _make_target_enu(x: float, y: float, z: float):
    m = PoseStamped()
    m.header.frame_id = 'world'
    m.pose.position.x = float(x)
    m.pose.position.y = float(y)
    m.pose.position.z = float(z)
    m.pose.orientation.w = 1.0
    return m


class _BridgeHarness:
    """Capture outbound PX4 commands, trajectory setpoints, and diagnostics cleanly."""

    def __init__(self, params=None) -> None:
        overrides = [Parameter(k, value=v) for k, v in (params or {}).items()]
        self.node = Px4BridgeNode(parameter_overrides=overrides)
        self.cmds = []
        self.trajs = []
        self.diags = []

        self.node.cmd_pub.publish = lambda m: self.cmds.append(m)
        self.node.traj_pub.publish = lambda m: self.trajs.append(m)
        self.node.offboard_mode_pub.publish = lambda _m: None
        self.node.fc_status_pub.publish = lambda m: self.diags.append(m)
        self.node.state_pub.publish = lambda _m: None

    def close(self) -> None:
        self.node.destroy_node()


def test_heartbeat_loss_returns_to_non_zero_takeoff_origin_before_landing():
    """Verify airborne heartbeat loss enters FAILSAFE_RTL, climbs, flies home, then lands."""
    rclpy.init()
    h = None
    try:
        h = _BridgeHarness(
            {
                'prestream_ticks': 2,
                'heartbeat_timeout': 1.0,
                'control_hz': 50.0,
                'failsafe_rtl_alt': 18.0,
                'failsafe_rtl_alt_step': 1.2,
                'failsafe_rtl_speed': 6.0,
                'failsafe_rtl_xy_tol': 0.45,
                'failsafe_rtl_land_speed': 0.5,
            }
        )
        n = h.node

        # 1. Spawn on non-zero takeoff pad (n=4.0, e=6.0) while DISARMED
        n._status_callback(_make_status(VehicleStatus.ARMING_STATE_DISARMED, VehicleStatus.NAVIGATION_STATE_AUTO_LOITER))
        n._estimator_flags_callback(_make_est(True))
        n._local_pos_callback(_make_lpos(4.0, 6.0, 0.0))
        n._heartbeat_callback(Empty())
        n._target_callback(_make_target_enu(6.0, 4.0, 8.0))
        n._control_loop()
        assert n.flight_phase == 'ARMING'
        assert (n.takeoff_n, n.takeoff_e, n.takeoff_d) == (4.0, 6.0, 0.0)

        # 2. Prestream -> ARMED + OFFBOARD -> AIRBORNE
        for _ in range(3):
            n._control_loop()
        n._status_callback(_make_status(VehicleStatus.ARMING_STATE_ARMED, VehicleStatus.NAVIGATION_STATE_OFFBOARD))
        n._control_loop()
        assert n.flight_phase == 'AIRBORNE'

        # 3. Vehicle flies away to (n=26.0, e=22.0, d=-8.0), then heartbeats drop (> 1.0s)
        pos_n, pos_e, pos_d = 26.0, 22.0, -8.0
        vel_n, vel_e, vel_d = 2.0, 1.5, 0.0
        n._local_pos_callback(_make_lpos(pos_n, pos_e, pos_d, vel_n, vel_e, vel_d))
        n.shadow_n, n.shadow_e, n.shadow_d = pos_n, pos_e, pos_d
        n.last_heartbeat_time = time.time() - 2.0
        h.cmds.clear()
        h.trajs.clear()

        n._control_loop()
        assert n.flight_phase == 'FAILSAFE_RTL'
        assert not any(
            c.command == VehicleCommand.VEHICLE_CMD_NAV_LAND for c in h.cmds
        ), 'Must NOT trigger direct NAV_LAND far from takeoff origin'

        # 4. Run closed-loop kinematics under FAILSAFE_RTL until arrival over (4.0, 6.0) triggers LANDING
        saw_climb_before_cruise = False
        land_triggered_at = None
        for _ in range(600):
            n._local_pos_callback(_make_lpos(pos_n, pos_e, pos_d, vel_n, vel_e, vel_d))
            n._control_loop()
            if n.flight_phase == 'LANDING':
                land_triggered_at = (pos_n, pos_e, math.hypot(vel_n, vel_e))
                break
            assert h.trajs, 'FAILSAFE_RTL must stream TrajectorySetpoint'
            sp = h.trajs[-1]
            # Verify initial climb holds XY near (26, 22) while ascending toward >= 18m (-18m D)
            if -pos_d < 16.5 and math.hypot(pos_n - 4.0, pos_e - 6.0) > 5.0:
                if sp.position[2] < pos_d - 0.1 and math.hypot(sp.position[0] - 26.0, sp.position[1] - 22.0) < 2.0:
                    saw_climb_before_cruise = True
            # Simple 1st-order plant tracking Bridge velocity+position command
            vn_c, ve_c, vd_c = sp.velocity
            kp = 1.6
            v_des_n = (0.0 if math.isnan(vn_c) else vn_c) + kp * (sp.position[0] - pos_n)
            v_des_e = (0.0 if math.isnan(ve_c) else ve_c) + kp * (sp.position[1] - pos_e)
            v_des_d = (0.0 if math.isnan(vd_c) else vd_c) + kp * (sp.position[2] - pos_d)
            alpha = min(1.0, n.dt / 0.18)
            vel_n += alpha * (v_des_n - vel_n)
            vel_e += alpha * (v_des_e - vel_e)
            vel_d += alpha * (v_des_d - vel_d)
            pos_n += vel_n * n.dt
            pos_e += vel_e * n.dt
            pos_d += vel_d * n.dt

        assert saw_climb_before_cruise, 'Expected climb above canyon obstacles before cruising home'
        assert land_triggered_at is not None, f'Did not arrive back at takeoff pad, ended at ({pos_n:.2f},{pos_e:.2f})'
        arr_err = math.hypot(land_triggered_at[0] - 4.0, land_triggered_at[1] - 6.0)
        assert arr_err <= 0.45, f'Switched to LANDING too far from takeoff origin: {arr_err:.3f}m'
        assert any(c.command == VehicleCommand.VEHICLE_CMD_NAV_LAND for c in h.cmds)

        # 5. Confirm fc_status reports flight_phase / DISARM returns to GROUND cleanly
        n._status_callback(_make_status(VehicleStatus.ARMING_STATE_DISARMED, VehicleStatus.NAVIGATION_STATE_AUTO_LAND))
        n._control_loop()
        assert n.flight_phase == 'GROUND'
    finally:
        if h is not None:
            h.close()
        if rclpy.ok():
            rclpy.shutdown()


def test_arming_prelaunch_heartbeat_loss_aborts_to_ground():
    """Verify heartbeat loss on ground during ARMING safely returns to GROUND without landing cmd."""
    rclpy.init()
    h = None
    try:
        h = _BridgeHarness({'prestream_ticks': 10, 'heartbeat_timeout': 1.0})
        n = h.node
        n._status_callback(_make_status(VehicleStatus.ARMING_STATE_DISARMED, VehicleStatus.NAVIGATION_STATE_AUTO_LOITER))
        n._estimator_flags_callback(_make_est(True))
        n._local_pos_callback(_make_lpos(0.0, 0.0, 0.0))
        n._heartbeat_callback(Empty())
        n._target_callback(_make_target_enu(0.0, 0.0, 8.0))
        n._control_loop()
        assert n.flight_phase == 'ARMING'

        n.last_heartbeat_time = time.time() - 2.0
        h.cmds.clear()
        n._control_loop()
        assert n.flight_phase == 'GROUND'
        assert not any(c.command == VehicleCommand.VEHICLE_CMD_NAV_LAND for c in h.cmds)
    finally:
        if h is not None:
            h.close()
        if rclpy.ok():
            rclpy.shutdown()


def test_multi_drone_failsafe_rtl_staggered_altitudes():
    """Verify target_system_id staggers return altitudes across multi-UAV swarm."""
    rclpy.init()
    h1 = h3 = None
    try:
        h1 = _BridgeHarness({'target_system_id': 1, 'failsafe_rtl_alt': 18.0, 'failsafe_rtl_alt_step': 1.2})
        h3 = _BridgeHarness({'drone_id': 'drone_3', 'target_system_id': 3, 'failsafe_rtl_alt': 18.0, 'failsafe_rtl_alt_step': 1.2})

        sp_alt = {}
        for key, harness in [(1, h1), (3, h3)]:
            nd = harness.node
            nd._status_callback(_make_status(VehicleStatus.ARMING_STATE_DISARMED, VehicleStatus.NAVIGATION_STATE_AUTO_LOITER))
            nd._local_pos_callback(_make_lpos(0.0, 0.0, 0.0))
            nd._status_callback(_make_status(VehicleStatus.ARMING_STATE_ARMED, VehicleStatus.NAVIGATION_STATE_OFFBOARD))
            nd.flight_phase = 'FAILSAFE_RTL'
            nd._local_pos_callback(_make_lpos(25.0, 10.0, -8.0))
            nd.shadow_n, nd.shadow_e, nd.shadow_d = 25.0, 10.0, -8.0
            nd.rtl_climb_n, nd.rtl_climb_e = 25.0, 10.0
            harness.trajs.clear()
            nd._control_loop()
            sp_alt[key] = float(harness.trajs[-1].velocity[2])
            # Also push shadow well up to observe settled cruise altitude target directly:
            nd.real_d = -18.5 if key == 1 else -20.9
            nd.shadow_d = nd.real_d
            nd._control_loop()
            sp_alt[f'{key}_vel'] = harness.trajs[-1].velocity[2]

        # At 18.5m: sys_id=1 reached target 18.0 (vel_d ~= 0), while sys_id=3 climbs toward 20.4 (vel_d < 0)
        assert sp_alt[1] < -0.5 and sp_alt[3] < -0.5
        assert abs(sp_alt['1_vel']) <= 0.1
        # Step sys_id=3 at d=-18.5 -> target is 20.4, so it still climbs (< 0)
        nd3 = h3.node
        nd3._local_pos_callback(_make_lpos(25.0, 10.0, -18.5))
        nd3.shadow_d = -18.5
        h3.trajs.clear()
        nd3._control_loop()
        assert h3.trajs[-1].velocity[2] < -1.0
    finally:
        if h1 is not None:
            h1.close()
        if h3 is not None:
            h3.close()
        if rclpy.ok():
            rclpy.shutdown()


def test_failsafe_rtl_offboard_lost_falls_back_to_px4_nav_rtl():
    """Verify if PX4 exits OFFBOARD during FAILSAFE_RTL, bridge commands NAV_RETURN_TO_LAUNCH."""
    rclpy.init()
    h = None
    try:
        h = _BridgeHarness({'failsafe_use_px4_rtl': False})
        n = h.node
        n._status_callback(_make_status(VehicleStatus.ARMING_STATE_DISARMED, VehicleStatus.NAVIGATION_STATE_AUTO_LOITER))
        n._local_pos_callback(_make_lpos(0.0, 0.0, 0.0))
        n._status_callback(_make_status(VehicleStatus.ARMING_STATE_ARMED, VehicleStatus.NAVIGATION_STATE_POSCTL))
        n._local_pos_callback(_make_lpos(18.0, 12.0, -8.0))
        n.flight_phase = 'FAILSAFE_RTL'
        h.cmds.clear()
        h.trajs.clear()

        n._control_loop()
        assert any(c.command == VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH for c in h.cmds)
        assert len(h.trajs) == 0, 'Must not fight PX4 Navigator with Offboard setpoints after losing OFFBOARD'
    finally:
        if h is not None:
            h.close()
        if rclpy.ok():
            rclpy.shutdown()


def test_failsafe_auto_resume_latch_vs_recovery():
    """Verify failsafe latches by default even if heartbeat returns, and recovers when enabled."""
    for allow_resume in (False, True):
        rclpy.init()
        h = None
        try:
            h = _BridgeHarness({'failsafe_auto_resume': allow_resume, 'heartbeat_timeout': 1.0})
            n = h.node
            n._status_callback(_make_status(VehicleStatus.ARMING_STATE_DISARMED, VehicleStatus.NAVIGATION_STATE_AUTO_LOITER))
            n._local_pos_callback(_make_lpos(0.0, 0.0, 0.0))
            n._status_callback(_make_status(VehicleStatus.ARMING_STATE_ARMED, VehicleStatus.NAVIGATION_STATE_OFFBOARD))
            n.flight_phase = 'FAILSAFE_RTL'
            n._local_pos_callback(_make_lpos(12.0, 5.0, -9.0))
            n.rtl_climb_n, n.rtl_climb_e = 12.0, 5.0

            n._heartbeat_callback(Empty())
            n._target_callback(_make_target_enu(5.0, 12.0, 9.0))
            n.last_diag_pub_time = 0.0
            h.diags.clear()
            n._control_loop()

            if allow_resume:
                assert n.flight_phase == 'AIRBORNE'
            else:
                assert n.flight_phase == 'FAILSAFE_RTL'
                assert h.diags and h.diags[-1].level == DiagnosticStatus.WARN
                kv = {item.key: item.value for item in h.diags[-1].values}
                assert kv.get('flight_phase') == 'FAILSAFE_RTL'
                assert kv.get('failsafe_rtl_active') == 'True'
        finally:
            if h is not None:
                h.close()
            if rclpy.ok():
                rclpy.shutdown()


def test_px4_rc_params_configures_return_mode_on_offboard_loss():
    """Verify px4-rc.params enforces COM_OBL_RC_ACT=3 (Return) and RTL_RETURN_ALT >= 18m."""
    params_path = Path(__file__).resolve().parent.parent / 'scripts' / 'px4-rc.params'
    text = params_path.read_text(encoding='utf-8')
    m_act = re.search(r'^param\s+set\s+COM_OBL_RC_ACT\s+(\d+)', text, re.MULTILINE)
    m_alt = re.search(r'^param\s+set\s+RTL_RETURN_ALT\s+([0-9.]+)', text, re.MULTILINE)
    assert m_act is not None and int(m_act.group(1)) == 3, 'COM_OBL_RC_ACT must be 3 (Return mode)'
    assert m_alt is not None and float(m_alt.group(1)) >= 18.0, 'RTL_RETURN_ALT must clear 16.5m obstacles'
