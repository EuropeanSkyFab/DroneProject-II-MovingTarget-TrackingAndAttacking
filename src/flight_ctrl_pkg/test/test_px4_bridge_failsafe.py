# -*- coding: utf-8 -*-
"""Unit tests for minimal PX4 failsafe RTL behavior in Px4BridgeNode."""

import re
import time
from pathlib import Path

from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import VehicleCommand, VehicleLocalPosition, VehicleStatus
import rclpy
from std_msgs.msg import Empty

from flight_ctrl_pkg.px4_bridge_node import Px4BridgeNode


class _CapturePub:
    def __init__(self) -> None:
        self.msgs = []

    def publish(self, msg) -> None:
        self.msgs.append(msg)


def _make_lpos(n: float, e: float, d: float, vx: float = 0.0, vy: float = 0.0, vz: float = 0.0) -> VehicleLocalPosition:
    msg = VehicleLocalPosition()
    msg.x = float(n)
    msg.y = float(e)
    msg.z = float(d)
    msg.vx = float(vx)
    msg.vy = float(vy)
    msg.vz = float(vz)
    msg.xy_valid = True
    msg.z_valid = True
    msg.heading = 0.0
    return msg


def _make_status(arming_state: int, nav_state: int) -> VehicleStatus:
    msg = VehicleStatus()
    msg.arming_state = int(arming_state)
    msg.nav_state = int(nav_state)
    msg.pre_flight_checks_pass = True
    return msg


def _make_target(x: float, y: float, z: float) -> PoseStamped:
    msg = PoseStamped()
    msg.pose.position.x = float(x)
    msg.pose.position.y = float(y)
    msg.pose.position.z = float(z)
    return msg


def _setup_node(heartbeat_timeout: float = 1.0):
    node = Px4BridgeNode()
    node.timer.cancel()
    node.heartbeat_timeout = float(heartbeat_timeout)
    cmds = _CapturePub()
    offboards = _CapturePub()
    trajs = _CapturePub()
    node.cmd_pub = cmds
    node.offboard_mode_pub = offboards
    node.traj_pub = trajs
    node._status_callback(
        _make_status(VehicleStatus.ARMING_STATE_DISARMED, VehicleStatus.NAVIGATION_STATE_AUTO_LOITER)
    )
    node._local_pos_callback(_make_lpos(0.0, 0.0, 0.0))
    return node, cmds, offboards, trajs


def test_heartbeat_loss_airborne_triggers_px4_rtl_without_offboard_fight():
    """Verify center heartbeat loss triggers NAV_RETURN_TO_LAUNCH and yields Offboard control."""
    rclpy.init()
    node = None
    try:
        node, cmds, offboards, trajs = _setup_node(heartbeat_timeout=0.5)
        node._status_callback(
            _make_status(VehicleStatus.ARMING_STATE_ARMED, VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        )
        node.flight_phase = 'AIRBORNE'
        node._local_pos_callback(_make_lpos(20.0, 15.0, -9.0, vx=2.0))
        node._target_callback(_make_target(15.0, 20.0, 9.0))
        node._heartbeat_callback(Empty())
        node.last_heartbeat_time = time.time() - 1.0

        cmds.msgs.clear()
        offboards.msgs.clear()
        trajs.msgs.clear()
        node._control_loop()

        assert node.flight_phase == 'FAILSAFE_RTL'
        assert node.latest_target is None
        assert len(offboards.msgs) == 0
        assert len(trajs.msgs) == 0
        assert [c.command for c in cmds.msgs if c.command in (
            VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH,
            VehicleCommand.VEHICLE_CMD_NAV_LAND,
        )] == [VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH]

        # While PX4 is executing AUTO_RTL, bridge must not overwrite it with NAV_LAND
        node._status_callback(
            _make_status(VehicleStatus.ARMING_STATE_ARMED, VehicleStatus.NAVIGATION_STATE_AUTO_RTL)
        )
        cmds.msgs.clear()
        node.last_cmd_retry_time = time.time() - 2.0
        node._control_loop()
        assert len(offboards.msgs) == 0
        assert len(trajs.msgs) == 0
        assert not any(c.command == VehicleCommand.VEHICLE_CMD_NAV_LAND for c in cmds.msgs)
        assert not any(c.command == VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH for c in cmds.msgs)

        # If PX4 is still in OFFBOARD (e.g. first RTL packet dropped), retry RTL cleanly
        node._status_callback(
            _make_status(VehicleStatus.ARMING_STATE_ARMED, VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        )
        node.last_cmd_retry_time = time.time() - 2.0
        node._control_loop()
        assert any(c.command == VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH for c in cmds.msgs)

        # Even if near ground shortly after entering FAILSAFE_RTL (e.g. climbing out), do not disarm early
        node._status_callback(
            _make_status(VehicleStatus.ARMING_STATE_ARMED, VehicleStatus.NAVIGATION_STATE_AUTO_RTL)
        )
        node._local_pos_callback(_make_lpos(2.0, 1.0, -0.1, vx=0.0, vy=0.0, vz=0.0))
        node._terminal_enter_time = time.time() - 1.5
        node.last_cmd_retry_time = time.time() - 2.0
        cmds.msgs.clear()
        node._control_loop()
        assert not any(c.command == VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM for c in cmds.msgs)

        # After sufficient flight/return time and touchdown on pad, assist disarm if still armed
        node._local_pos_callback(_make_lpos(0.0, 0.0, -0.05))
        node._terminal_enter_time = time.time() - 5.0
        node.last_cmd_retry_time = time.time() - 2.0
        cmds.msgs.clear()
        node._control_loop()
        assert any(c.command == VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM for c in cmds.msgs)

        # After RTL returns home and lands/disarms, bridge resets cleanly to GROUND
        node._status_callback(
            _make_status(VehicleStatus.ARMING_STATE_DISARMED, VehicleStatus.NAVIGATION_STATE_AUTO_RTL)
        )
        cmds.msgs.clear()
        node._control_loop()
        assert node.flight_phase == 'GROUND'
        assert node.last_heartbeat_time == 0.0
        assert any(c.command == VehicleCommand.VEHICLE_CMD_DO_SET_MODE for c in cmds.msgs)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_heartbeat_loss_on_ground_prelaunch_aborts_to_ground_not_rtl():
    """Verify heartbeat timeout while still on ground during ARMING aborts cleanly instead of RTL."""
    rclpy.init()
    node = None
    try:
        node, cmds, _offboards, _trajs = _setup_node(heartbeat_timeout=0.5)
        node.flight_phase = 'ARMING'
        node.last_heartbeat_time = time.time() - 1.0
        cmds.msgs.clear()

        node._control_loop()
        assert node.flight_phase == 'GROUND'
        assert not any(c.command == VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH for c in cmds.msgs)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_normal_vertical_land_target_keeps_landing_behavior():
    """Verify intentional landing setpoint (z < 0.3) still triggers NAV_LAND normally."""
    rclpy.init()
    node = None
    try:
        node, cmds, _offboards, _trajs = _setup_node(heartbeat_timeout=1.5)
        node._status_callback(
            _make_status(VehicleStatus.ARMING_STATE_ARMED, VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        )
        node.flight_phase = 'AIRBORNE'
        node._local_pos_callback(_make_lpos(0.0, 0.0, -3.0, vx=0.0, vy=0.0))
        node._target_callback(_make_target(0.0, 0.0, 0.0))
        node._heartbeat_callback(Empty())
        cmds.msgs.clear()

        node._control_loop()
        assert node.flight_phase == 'LANDING'
        assert any(c.command == VehicleCommand.VEHICLE_CMD_NAV_LAND for c in cmds.msgs)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_px4_rc_params_sets_return_mode_on_offboard_loss():
    """Verify px4-rc.params configures COM_OBL_RC_ACT=3 (Return mode)."""
    params_path = Path(__file__).resolve().parent.parent / 'scripts' / 'px4-rc.params'
    text = params_path.read_text(encoding='utf-8')
    m_act = re.search(r'^param\s+set\s+COM_OBL_RC_ACT\s+(\d+)', text, re.MULTILINE)
    assert m_act is not None and int(m_act.group(1)) == 3
