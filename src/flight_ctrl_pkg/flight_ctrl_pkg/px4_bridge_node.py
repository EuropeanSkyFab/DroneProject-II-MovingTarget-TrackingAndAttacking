# -*- coding: utf-8 -*-
"""
px4_bridge_node.py —— 课题二飞控调试与 PX4 uXRCE-DDS 高阶桥接主节点

严格遵守《课题二内部接口规范 V3.0》与《中枢节点接口协议》：
  【订阅中枢输出】
    - /{drone_id}/fmu/in/trajectory_setpoint  (geometry_msgs/PoseStamped, 全局 ENU)
    - /{drone_id}/action                      (std_msgs/String, 动作标签限幅)
    - /{drone_id}/heartbeat                   (std_msgs/Empty, 中枢心跳看门狗)
    - /{drone_id}/fmu/in/gimbal_setpoint      (geometry_msgs/Vector3, 云台协同)
  【订阅 PX4 uXRCE-DDS 状态】
    - {px4_ns}/fmu/out/vehicle_local_position (px4_msgs/VehicleLocalPosition)
    - {px4_ns}/fmu/out/vehicle_attitude       (px4_msgs/VehicleAttitude)
    - {px4_ns}/fmu/out/vehicle_odometry       (px4_msgs/VehicleOdometry)
    - {px4_ns}/fmu/out/vehicle_status         (px4_msgs/VehicleStatus)
    - {px4_ns}/fmu/out/battery_status         (px4_msgs/BatteryStatus)
    - {px4_ns}/fmu/out/vehicle_command_ack    (px4_msgs/VehicleCommandAck)
    - {px4_ns}/fmu/out/failsafe_flags         (px4_msgs/FailsafeFlags)
    - {px4_ns}/fmu/out/estimator_status_flags (px4_msgs/EstimatorStatusFlags)
  【发布】
    - /{drone_id}/state                       (nav_msgs/Odometry, 全局 ENU + FLU 四元数)
    - /{drone_id}/fmu/out/battery_raw         (sensor_msgs/BatteryState, 6S 真实电量)
    - /{drone_id}/fmu/out/fc_status           (diagnostic_msgs/DiagnosticStatus, PX4 健康诊断)
    - {px4_ns}/fmu/in/offboard_control_mode   (px4_msgs/OffboardControlMode, 位置+速度+加速度)
    - {px4_ns}/fmu/in/trajectory_setpoint     (px4_msgs/TrajectorySetpoint, 50Hz NED 前馈)
    - {px4_ns}/fmu/in/vehicle_command         (px4_msgs/VehicleCommand, 含云台指令)
"""

import copy
import math
import time
from typing import Optional

from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3
from nav_msgs.msg import Odometry
from px4_msgs.msg import (
    BatteryStatus,
    EstimatorStatusFlags,
    FailsafeFlags,
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleAttitude,
    VehicleCommand,
    VehicleCommandAck,
    VehicleLocalPosition,
    VehicleOdometry,
    VehicleStatus,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import BatteryState
from std_msgs.msg import Empty, String

from flight_ctrl_pkg.coord_utils import (
    compute_gimbal_pitch_ned,
    compute_shadow_ff_accel,
    compute_yawspeed_ned,
    enu_to_ned_pos,
    ned_to_enu_pos,
    ned_to_enu_vel,
    quat_frd_ned_to_flu_enu,
    slew_yaw_ned,
    versioned_px4_topic,
)


_REF_STEP_HZ = 20.0
_GIMBAL_FLAGS_WORLD_LOCK = 4.0 + 8.0 + 16.0  # ROLL_LOCK | PITCH_LOCK | YAW_LOCK
_GIMBAL_FLAGS_HORIZON_LOCK = 4.0 + 8.0       # ROLL_LOCK | PITCH_LOCK (body-relative yaw)


class Px4BridgeNode(Node):
    """基于 uXRCE-DDS 的 PX4 多机高机动飞控桥接节点。"""

    def __init__(self, **kwargs) -> None:
        super().__init__('px4_bridge_node', **kwargs)

        # -------------------- 参数声明 --------------------
        self.declare_parameter('drone_id', 'drone_1')
        self.declare_parameter('px4_ns', '')
        self.declare_parameter('target_system_id', 1)
        self.declare_parameter('offset_x', 0.0)
        self.declare_parameter('offset_y', 0.0)
        self.declare_parameter('control_hz', 50.0)
        # 保留 max_step_* 为 20Hz 等效参考步长，运行频率变化时自动换算 m/s 上限以保持完全兼容
        self.declare_parameter('max_step_default', 0.28)
        self.declare_parameter('max_step_intercept', 0.55)
        self.declare_parameter('max_step_attack', 0.82)
        self.declare_parameter('heartbeat_timeout', 1.5)
        self.declare_parameter('target_fresh_timeout', 2.0)
        self.declare_parameter('land_d_threshold', -0.25)
        self.declare_parameter('prestream_ticks', 25)
        self.declare_parameter('align_yaw', True)
        self.declare_parameter('enable_gimbal_cmd', True)
        self.declare_parameter('min_track_alt', 8.0)
        self.declare_parameter('target_ref_alt', 5.5)
        self.declare_parameter('use_vel_feedforward', True)
        self.declare_parameter('use_acc_feedforward', True)
        self.declare_parameter('max_yaw_rate_deg', 75.0)
        self.declare_parameter('failsafe_rtl_alt', 18.0)
        self.declare_parameter('failsafe_rtl_alt_step', 1.2)
        self.declare_parameter('failsafe_rtl_speed', 5.5)
        self.declare_parameter('failsafe_rtl_xy_tol', 0.45)
        self.declare_parameter('failsafe_rtl_land_speed', 0.45)
        self.declare_parameter('failsafe_auto_resume', False)
        self.declare_parameter('failsafe_use_px4_rtl', False)

        p = self.get_parameter
        self.drone_id: str = p('drone_id').value
        raw_ns: str = str(p('px4_ns').value).strip().strip('/')
        self.px4_prefix: str = f'/{raw_ns}' if raw_ns else ''
        self.target_system_id: int = int(p('target_system_id').value)
        self.offset_x: float = float(p('offset_x').value)
        self.offset_y: float = float(p('offset_y').value)
        self.control_hz: float = max(float(p('control_hz').value), 10.0)
        self.dt: float = 1.0 / self.control_hz
        self.max_step_default: float = float(p('max_step_default').value)
        self.max_step_intercept: float = float(p('max_step_intercept').value)
        self.max_step_attack: float = float(p('max_step_attack').value)
        self.heartbeat_timeout: float = float(p('heartbeat_timeout').value)
        self.target_fresh_timeout: float = float(p('target_fresh_timeout').value)
        self.land_d_threshold: float = float(p('land_d_threshold').value)
        self.prestream_ticks: int = int(p('prestream_ticks').value)
        self.align_yaw: bool = bool(p('align_yaw').value)
        self.enable_gimbal_cmd: bool = bool(p('enable_gimbal_cmd').value)
        self.min_track_alt: float = float(p('min_track_alt').value)
        self.target_ref_alt: float = float(p('target_ref_alt').value)
        self.use_vel_feedforward: bool = bool(p('use_vel_feedforward').value)
        self.use_acc_feedforward: bool = bool(p('use_acc_feedforward').value)
        self.max_yaw_rate_rad: float = math.radians(max(float(p('max_yaw_rate_deg').value), 10.0))
        self.failsafe_rtl_alt: float = max(float(p('failsafe_rtl_alt').value), 2.0)
        self.failsafe_rtl_alt_step: float = max(float(p('failsafe_rtl_alt_step').value), 0.0)
        self.failsafe_rtl_speed: float = max(float(p('failsafe_rtl_speed').value), 1.0)
        self.failsafe_rtl_xy_tol: float = max(float(p('failsafe_rtl_xy_tol').value), 0.15)
        self.failsafe_rtl_land_speed: float = max(float(p('failsafe_rtl_land_speed').value), 0.1)
        self.failsafe_auto_resume: bool = bool(p('failsafe_auto_resume').value)
        self.failsafe_use_px4_rtl: bool = bool(p('failsafe_use_px4_rtl').value)

        # -------------------- PX4 DDS QoS --------------------
        px4_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # -------------------- 订阅中枢接口 --------------------
        self.create_subscription(
            PoseStamped,
            f'/{self.drone_id}/fmu/in/trajectory_setpoint',
            self._target_callback,
            10,
        )
        self.create_subscription(
            TwistStamped,
            f'/{self.drone_id}/fmu/in/trajectory_ff',
            self._ff_callback,
            10,
        )
        self.create_subscription(
            String,
            f'/{self.drone_id}/action',
            self._action_callback,
            10,
        )
        self.create_subscription(
            Empty,
            f'/{self.drone_id}/heartbeat',
            self._heartbeat_callback,
            10,
        )
        self.create_subscription(
            Vector3,
            f'/{self.drone_id}/fmu/in/gimbal_setpoint',
            self._gimbal_callback,
            10,
        )

        # -------------------- 订阅 PX4 状态与诊断话题 --------------------
        self.create_subscription(
            VehicleLocalPosition,
            versioned_px4_topic(self.px4_prefix, '/fmu/out/vehicle_local_position', VehicleLocalPosition),
            self._local_pos_callback,
            px4_qos,
        )
        self.create_subscription(
            VehicleAttitude,
            versioned_px4_topic(self.px4_prefix, '/fmu/out/vehicle_attitude', VehicleAttitude),
            self._attitude_callback,
            px4_qos,
        )
        self.create_subscription(
            VehicleOdometry,
            versioned_px4_topic(self.px4_prefix, '/fmu/out/vehicle_odometry', VehicleOdometry),
            self._odometry_callback,
            px4_qos,
        )
        self.create_subscription(
            VehicleStatus,
            versioned_px4_topic(self.px4_prefix, '/fmu/out/vehicle_status', VehicleStatus),
            self._status_callback,
            px4_qos,
        )
        self.create_subscription(
            BatteryStatus,
            versioned_px4_topic(self.px4_prefix, '/fmu/out/battery_status', BatteryStatus),
            self._battery_callback,
            px4_qos,
        )
        self.create_subscription(
            VehicleCommandAck,
            versioned_px4_topic(self.px4_prefix, '/fmu/out/vehicle_command_ack', VehicleCommandAck),
            self._command_ack_callback,
            px4_qos,
        )
        self.create_subscription(
            FailsafeFlags,
            versioned_px4_topic(self.px4_prefix, '/fmu/out/failsafe_flags', FailsafeFlags),
            self._failsafe_flags_callback,
            px4_qos,
        )
        self.create_subscription(
            EstimatorStatusFlags,
            versioned_px4_topic(self.px4_prefix, '/fmu/out/estimator_status_flags', EstimatorStatusFlags),
            self._estimator_flags_callback,
            px4_qos,
        )

        # -------------------- 发布器 --------------------
        self.state_pub = self.create_publisher(Odometry, f'/{self.drone_id}/state', 10)
        self.battery_pub = self.create_publisher(
            BatteryState, f'/{self.drone_id}/fmu/out/battery_raw', 10
        )
        self.fc_status_pub = self.create_publisher(
            DiagnosticStatus, f'/{self.drone_id}/fmu/out/fc_status', 10
        )

        self.offboard_mode_pub = self.create_publisher(
            OffboardControlMode,
            versioned_px4_topic(self.px4_prefix, '/fmu/in/offboard_control_mode', OffboardControlMode),
            px4_qos,
        )
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint,
            versioned_px4_topic(self.px4_prefix, '/fmu/in/trajectory_setpoint', TrajectorySetpoint),
            px4_qos,
        )
        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            versioned_px4_topic(self.px4_prefix, '/fmu/in/vehicle_command', VehicleCommand),
            px4_qos,
        )

        # -------------------- 运行时状态 --------------------
        self.local_pos: Optional[VehicleLocalPosition] = None
        self.attitude: Optional[VehicleAttitude] = None
        self.px4_odom: Optional[VehicleOdometry] = None
        self.vehicle_status: Optional[VehicleStatus] = None
        self.failsafe_flags: Optional[FailsafeFlags] = None
        self.estimator_flags: Optional[EstimatorStatusFlags] = None

        self.latest_target: Optional[PoseStamped] = None
        self.last_target_time: float = 0.0
        self.prev_tgt_ned: Optional[tuple] = None
        self.prev_tgt_time: float = 0.0
        self.tgt_vel_ned: tuple = (0.0, 0.0, 0.0)
        self.latest_ff_ned: Optional[tuple] = None
        self.last_ff_time: float = 0.0
        self.prev_vel_cmd_ned: tuple = (0.0, 0.0, 0.0)
        self.prev_acc_cmd_ned: tuple = (0.0, 0.0, 0.0)

        self.current_action: str = 'HOVER'
        self.last_heartbeat_time: float = 0.0
        self.last_gimbal_recv_time: float = 0.0
        self.last_gimbal_sent_time: float = 0.0
        self.last_gimbal_cfg_try: float = 0.0
        self.gimbal_configured: bool = False
        self.gimbal_target_yaw_ned: Optional[float] = None
        self.gimbal_target_pitch_rad: float = -0.35
        self.land_brake_start: Optional[float] = None
        self.last_diag_pub_time: float = 0.0

        # 状态机: GROUND -> ARMING -> AIRBORNE -> (FAILSAFE_RTL) -> LANDING -> GROUND
        self.flight_phase: str = 'GROUND'
        self.arm_tick_count: int = 0
        self.last_cmd_retry_time: float = 0.0

        # 局部 NED 真实位置、本次起飞原点与影子平滑位置
        self.real_n: float = 0.0
        self.real_e: float = 0.0
        self.real_d: float = 0.0
        self.takeoff_n: float = 0.0
        self.takeoff_e: float = 0.0
        self.takeoff_d: float = 0.0
        self.takeoff_recorded: bool = False
        self.rtl_climb_n: float = 0.0
        self.rtl_climb_e: float = 0.0
        self.rtl_cmd_sent: bool = False
        self.home_d_offset: Optional[float] = None
        self.shadow_n: float = 0.0
        self.shadow_e: float = 0.0
        self.shadow_d: float = 0.0
        self.desired_yaw_ned: float = 0.0

        self.timer = self.create_timer(self.dt, self._control_loop)
        self.get_logger().info(
            f'[{self.drone_id}] uXRCE-DDS 高阶桥接已启动({self.control_hz:.0f}Hz): '
            f'px4_ns="{self.px4_prefix or "/"}", sys_id={self.target_system_id}, '
            f'gimbal={self.enable_gimbal_cmd}, acc_ff={self.use_acc_feedforward}'
        )

    # -------------------- 内部时间戳工具 --------------------
    def _px4_timestamp_us(self) -> int:
        return int(self.get_clock().now().nanoseconds / 1000)

    # -------------------- 中枢输入回调 --------------------
    def _target_callback(self, msg: PoseStamped) -> None:
        now = time.time()
        self.latest_target = copy.deepcopy(msg)
        self.last_target_time = now
        tn, te, td = enu_to_ned_pos(
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
            self.offset_x,
            self.offset_y,
        )
        if self.prev_tgt_ned is not None:
            dt = now - self.prev_tgt_time
            jump = math.sqrt(
                (tn - self.prev_tgt_ned[0]) ** 2
                + (te - self.prev_tgt_ned[1]) ** 2
                + (td - self.prev_tgt_ned[2]) ** 2
            )
            jump_lim = 2.8 if self.current_action in ('INTERCEPT', 'ATTACK') else 1.8
            if 0.02 <= dt <= 0.5 and jump < jump_lim:
                vx = (tn - self.prev_tgt_ned[0]) / dt
                vy = (te - self.prev_tgt_ned[1]) / dt
                vz = (td - self.prev_tgt_ned[2]) / dt
                a = 0.30
                nvx = (1.0 - a) * self.tgt_vel_ned[0] + a * vx
                nvy = (1.0 - a) * self.tgt_vel_ned[1] + a * vy
                nvz = (1.0 - a) * self.tgt_vel_ned[2] + a * vz
                h = math.hypot(nvx, nvy)
                v_ff_cap = 13.5 if self.current_action == 'ATTACK' else (10.5 if self.current_action == 'INTERCEPT' else 5.5)
                if h > v_ff_cap:
                    nvx *= v_ff_cap / h
                    nvy *= v_ff_cap / h
                nvz = max(-3.5, min(3.5, nvz))
                self.tgt_vel_ned = (nvx, nvy, nvz)
            elif dt > 0.5 or jump >= jump_lim:
                self.tgt_vel_ned = (0.0, 0.0, 0.0)
        self.prev_tgt_ned = (tn, te, td)
        self.prev_tgt_time = now

    def _action_callback(self, msg: String) -> None:
        if msg.data != self.current_action:
            self.current_action = msg.data
            self.tgt_vel_ned = (0.0, 0.0, 0.0)
            self.latest_ff_ned = None
            self.last_ff_time = 0.0
            self.prev_acc_cmd_ned = (0.0, 0.0, 0.0)
            self.get_logger().info(f'[{self.drone_id}] 动作切换 -> {self.current_action}')

    def _ff_callback(self, msg: TwistStamped) -> None:
        vx_enu = float(msg.twist.linear.x)
        vy_enu = float(msg.twist.linear.y)
        vz_enu = float(msg.twist.linear.z)
        if not (math.isfinite(vx_enu) and math.isfinite(vy_enu) and math.isfinite(vz_enu)):
            self.get_logger().warn(
                f'[{self.drone_id}] 拒绝含 NaN/Inf 的速度前馈 trajectory_ff',
                throttle_duration_sec=2.0,
            )
            return
        # ENU (vx_east, vy_north, vz_up) -> NED (vn=vy_north, ve=vx_east, vd=-vz_up)
        vn, ve, vd = vy_enu, vx_enu, -vz_enu
        h = math.hypot(vn, ve)
        v_cap = 16.0
        if h > v_cap and h > 1e-6:
            vn *= v_cap / h
            ve *= v_cap / h
        vd = max(-6.0, min(6.0, vd))
        self.latest_ff_ned = (vn, ve, vd)
        self.last_ff_time = time.time()

    def _heartbeat_callback(self, _msg: Empty) -> None:
        self.last_heartbeat_time = time.time()

    def _gimbal_callback(self, msg: Vector3) -> None:
        self.gimbal_target_pitch_rad = float(msg.x)
        self.gimbal_target_yaw_ned = float(msg.y)
        self.last_gimbal_recv_time = time.time()

    # -------------------- PX4 状态与诊断回调 --------------------
    def _local_pos_callback(self, msg: VehicleLocalPosition) -> None:
        self.local_pos = msg
        if self.home_d_offset is None and self.vehicle_status is not None:
            if (
                msg.z_valid
                and self.flight_phase == 'GROUND'
                and self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED
            ):
                self.home_d_offset = float(msg.z)
                self.get_logger().info(
                    f'[{self.drone_id}] 地面高度基准校准: home_d_offset={self.home_d_offset:.3f}m'
                )
            elif self.vehicle_status.arming_state != VehicleStatus.ARMING_STATE_DISARMED:
                self.home_d_offset = 0.0

        if self.home_d_offset is None:
            return

        offset_d = self.home_d_offset
        self.real_n = float(msg.x)
        self.real_e = float(msg.y)
        self.real_d = float(msg.z) - offset_d
        if self.flight_phase == 'GROUND':
            self.shadow_n = self.real_n
            self.shadow_e = self.real_e
            self.shadow_d = self.real_d
            if (
                msg.xy_valid
                and msg.z_valid
                and self.vehicle_status is not None
                and self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED
            ):
                self.takeoff_n = self.real_n
                self.takeoff_e = self.real_e
                self.takeoff_d = self.real_d
                self.takeoff_recorded = True
            if math.isfinite(msg.heading):
                self.desired_yaw_ned = float(msg.heading)
        self._publish_enu_odometry()

    def _attitude_callback(self, msg: VehicleAttitude) -> None:
        self.attitude = msg

    def _odometry_callback(self, msg: VehicleOdometry) -> None:
        self.px4_odom = msg

    def _status_callback(self, msg: VehicleStatus) -> None:
        self.vehicle_status = msg

    def _failsafe_flags_callback(self, msg: FailsafeFlags) -> None:
        self.failsafe_flags = msg

    def _estimator_flags_callback(self, msg: EstimatorStatusFlags) -> None:
        self.estimator_flags = msg

    def _command_ack_callback(self, msg: VehicleCommandAck) -> None:
        if msg.command == VehicleCommand.VEHICLE_CMD_DO_GIMBAL_MANAGER_CONFIGURE:
            # 忽略启动前 Navigator 留下的 TRANSIENT_LOCAL 历史 ACK，仅响应本节点发出的配置请求
            if (
                self.last_gimbal_cfg_try > 0.0
                and msg.target_system == 1
                and msg.target_component == 1
                and msg.result == VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED
            ):
                if not self.gimbal_configured:
                    self.gimbal_configured = True
                    self.last_diag_pub_time = 0.0
                    self.get_logger().info(f'[{self.drone_id}] PX4 原生云台控制器已接管并确认 (ACK=ACCEPTED)')
        elif msg.command == VehicleCommand.VEHICLE_CMD_DO_GIMBAL_MANAGER_PITCHYAW:
            if msg.result == VehicleCommandAck.VEHICLE_CMD_RESULT_DENIED:
                # 若被 Navigator（如降落/RTL阶段）释放所有权，自动清零标志并在下个周期重新申领
                self.gimbal_configured = False
        elif msg.result not in (
            VehicleCommandAck.VEHICLE_CMD_RESULT_ACCEPTED,
            VehicleCommandAck.VEHICLE_CMD_RESULT_IN_PROGRESS,
        ):
            self.get_logger().warn(
                f'[{self.drone_id}] PX4 Command {msg.command} ACK result={msg.result}',
                throttle_duration_sec=2.0,
            )

    def _battery_callback(self, msg: BatteryStatus) -> None:
        out = BatteryState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = f'{self.drone_id}_base_link'
        v_filt = float(getattr(msg, 'voltage_filtered_v', 0.0))
        out.voltage = v_filt if v_filt > 0.0 else float(msg.voltage_v)
        c_filt = getattr(msg, 'current_filtered_a', msg.current_a)
        out.current = float(c_filt if msg.current_a >= 0.0 else 0.0)
        out.percentage = float(msg.remaining) if msg.remaining >= 0.0 else 1.0
        out.temperature = float(msg.temperature) if math.isfinite(msg.temperature) else 25.0
        out.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
        self.battery_pub.publish(out)

    # -------------------- 发布 ENU Odometry 与 PX4 诊断 --------------------
    def _publish_enu_odometry(self) -> None:
        if self.local_pos is None:
            return
        lp = self.local_pos
        if not (lp.xy_valid and lp.z_valid):
            return

        offset_d = self.home_d_offset if self.home_d_offset is not None else 0.0
        wx, wy, wz = ned_to_enu_pos(
            lp.x, lp.y, float(lp.z) - offset_d, self.offset_x, self.offset_y
        )
        vx, vy, vz = ned_to_enu_vel(lp.vx, lp.vy, lp.vz)

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = 'world'
        odom.child_frame_id = f'{self.drone_id}_base_link'
        odom.pose.pose.position.x = wx
        odom.pose.pose.position.y = wy
        odom.pose.pose.position.z = wz

        if self.attitude is not None:
            qx, qy, qz, qw = quat_frd_ned_to_flu_enu(self.attitude.q)
        elif self.px4_odom is not None:
            qx, qy, qz, qw = quat_frd_ned_to_flu_enu(self.px4_odom.q)
        else:
            qx, qy, qz, qw = (0.0, 0.0, 0.0, 1.0)

        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw

        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.linear.z = vz

        if self.px4_odom is not None:
            av = self.px4_odom.angular_velocity
            if all(math.isfinite(float(v)) for v in av):
                odom.twist.twist.angular.x = float(av[0])
                odom.twist.twist.angular.y = -float(av[1])
                odom.twist.twist.angular.z = -float(av[2])

        self.state_pub.publish(odom)

    def _publish_fc_diagnostics(self, now: float) -> None:
        if now - self.last_diag_pub_time < 0.5:
            return
        self.last_diag_pub_time = now
        st = DiagnosticStatus()
        st.hardware_id = self.drone_id
        st.name = f'PX4 Bridge ({self.flight_phase}/{self.current_action})'
        ekf_ok = True
        if self.estimator_flags is not None:
            ekf_ok = bool(self.estimator_flags.cs_tilt_align and self.estimator_flags.cs_yaw_align)
        fs_breach = False
        fs_obl = False
        bat_warn = 0
        if self.failsafe_flags is not None:
            fs_breach = bool(self.failsafe_flags.geofence_breached)
            fs_obl = bool(self.failsafe_flags.offboard_control_signal_lost)
            bat_warn = int(self.failsafe_flags.battery_warning)

        if fs_breach or bat_warn >= 2:
            st.level = DiagnosticStatus.ERROR
            st.message = 'PX4 Failsafe Active'
        elif self.flight_phase == 'FAILSAFE_RTL':
            st.level = DiagnosticStatus.WARN
            st.message = 'Failsafe RTL Active'
        elif not ekf_ok or fs_obl:
            st.level = DiagnosticStatus.WARN
            st.message = 'EKF / Offboard Warning'
        else:
            st.level = DiagnosticStatus.OK
            st.message = 'PX4 Nominal'

        nav_state = getattr(self.vehicle_status, 'nav_state', -1)
        arm_state = getattr(self.vehicle_status, 'arming_state', -1)
        st.values = [
            KeyValue(key='nav_state', value=str(nav_state)),
            KeyValue(key='arming_state', value=str(arm_state)),
            KeyValue(key='ekf_aligned', value=str(ekf_ok)),
            KeyValue(key='geofence_breached', value=str(fs_breach)),
            KeyValue(key='gimbal_configured', value=str(self.gimbal_configured)),
            KeyValue(key='acc_ff', value=str(self.use_acc_feedforward)),
            KeyValue(key='flight_phase', value=self.flight_phase),
            KeyValue(key='failsafe_rtl_active', value=str(self.flight_phase == 'FAILSAFE_RTL')),
        ]
        self.fc_status_pub.publish(st)

    # -------------------- PX4 指令与云台协同下发 --------------------
    def _publish_offboard_control_mode(self) -> None:
        msg = OffboardControlMode()
        msg.timestamp = self._px4_timestamp_us()
        msg.position = True
        msg.velocity = bool(self.use_vel_feedforward)
        msg.acceleration = bool(self.use_acc_feedforward)
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        self.offboard_mode_pub.publish(msg)

    def _publish_trajectory_setpoint(
        self,
        n: float,
        e: float,
        d: float,
        yaw_ned: float,
        vel_ned: Optional[tuple] = None,
        acc_ned: Optional[tuple] = None,
        yawspeed_ned: float = float('nan'),
    ) -> None:
        msg = TrajectorySetpoint()
        msg.timestamp = self._px4_timestamp_us()
        msg.position = [float(n), float(e), float(d)]
        if self.use_vel_feedforward and vel_ned is not None:
            msg.velocity = [float(vel_ned[0]), float(vel_ned[1]), float(vel_ned[2])]
        else:
            msg.velocity = [float('nan'), float('nan'), float('nan')]
        if self.use_acc_feedforward and acc_ned is not None:
            msg.acceleration = [float(acc_ned[0]), float(acc_ned[1]), float(acc_ned[2])]
        else:
            msg.acceleration = [float('nan'), float('nan'), float('nan')]
        msg.jerk = [float('nan'), float('nan'), float('nan')]
        msg.yaw = float(yaw_ned)
        msg.yawspeed = float(yawspeed_ned)
        self.traj_pub.publish(msg)

    def _publish_vehicle_command(
        self,
        command: int,
        target_system: Optional[int] = None,
        target_component: int = 1,
        **params: float,
    ) -> None:
        msg = VehicleCommand()
        msg.timestamp = self._px4_timestamp_us()
        msg.command = int(command)
        msg.param1 = float(params.get('param1', 0.0))
        msg.param2 = float(params.get('param2', 0.0))
        msg.param3 = float(params.get('param3', 0.0))
        msg.param4 = float(params.get('param4', 0.0))
        msg.param5 = float(params.get('param5', 0.0))
        msg.param6 = float(params.get('param6', 0.0))
        msg.param7 = float(params.get('param7', 0.0))
        msg.target_system = self.target_system_id if target_system is None else int(target_system)
        msg.target_component = int(target_component)
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)

    def _update_px4_gimbal(self, now: float) -> None:
        if not self.enable_gimbal_cmd:
            return
        if not self.gimbal_configured:
            if now - self.last_gimbal_cfg_try >= 1.0:
                self.last_gimbal_cfg_try = now
                self._publish_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_DO_GIMBAL_MANAGER_CONFIGURE,
                    target_system=0,
                    target_component=0,
                    param1=1.0,
                    param2=1.0,
                    param3=-1.0,
                    param4=-1.0,
                )
            return

        if self.flight_phase == 'AIRBORNE':
            if now - self.last_gimbal_sent_time < 0.05:
                return
            self.last_gimbal_sent_time = now
            drone_alt = -self.real_d
            gimbal_fresh = (
                self.gimbal_target_yaw_ned is not None
                and (now - self.last_gimbal_recv_time) < 1.0
            )
            if gimbal_fresh:
                yaw_deg = math.degrees(self.gimbal_target_yaw_ned)
                if abs(self.gimbal_target_pitch_rad - (-0.2)) > 1e-4:
                    pitch_deg = max(-55.0, min(12.0, math.degrees(self.gimbal_target_pitch_rad)))
                else:
                    pitch_deg = compute_gimbal_pitch_ned(
                        drone_alt, self.target_ref_alt, 5.2, min_pitch_deg=-52.0, max_pitch_deg=10.0
                    )
            elif self.latest_target is not None and self.current_action in ('TRACK', 'INTERCEPT', 'ATTACK'):
                tgt = self.latest_target.pose.position
                tn, te, _ = enu_to_ned_pos(tgt.x, tgt.y, tgt.z, self.offset_x, self.offset_y)
                dn = tn - self.real_n
                de = te - self.real_e
                h_dist = math.hypot(dn, de)
                yaw_deg = math.degrees(math.atan2(de, dn)) if h_dist > 0.5 else math.degrees(self.desired_yaw_ned)
                pitch_deg = compute_gimbal_pitch_ned(
                    drone_alt, tgt.z, h_dist, min_pitch_deg=-52.0, max_pitch_deg=10.0
                )
            else:
                yaw_deg = math.degrees(self.desired_yaw_ned)
                pitch_deg = -18.0

            self._publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_GIMBAL_MANAGER_PITCHYAW,
                target_system=0,
                target_component=0,
                param1=pitch_deg,
                param2=yaw_deg,
                param3=float('nan'),
                param4=float('nan'),
                param5=_GIMBAL_FLAGS_WORLD_LOCK,
            )
        else:
            if now - self.last_gimbal_sent_time < 0.5:
                return
            self.last_gimbal_sent_time = now
            self._publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_GIMBAL_MANAGER_PITCHYAW,
                target_system=0,
                target_component=0,
                param1=0.0,
                param2=0.0,
                param3=float('nan'),
                param4=float('nan'),
                param5=_GIMBAL_FLAGS_HORIZON_LOCK,
            )

    def _engage_offboard_and_arm(self) -> None:
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

    def _land(self) -> None:
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)

    def _rtl_px4(self) -> None:
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_RETURN_TO_LAUNCH)

    def _disarm(self) -> None:
        self._publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)

    def _reset_to_ground(self, reset_mode: bool = True) -> None:
        self.flight_phase = 'GROUND'
        self.latest_target = None
        self.latest_ff_ned = None
        self.prev_tgt_ned = None
        self.tgt_vel_ned = (0.0, 0.0, 0.0)
        self.prev_vel_cmd_ned = (0.0, 0.0, 0.0)
        self.prev_acc_cmd_ned = (0.0, 0.0, 0.0)
        self.land_brake_start = None
        self.rtl_cmd_sent = False
        self.arm_tick_count = 0
        self.last_heartbeat_time = 0.0
        if reset_mode:
            self._publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=4.0, param3=3.0
            )

    def _handle_failsafe_rtl(self, now: float) -> None:
        """Guide vehicle smoothly back to recorded takeoff origin before landing."""
        if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
            self._reset_to_ground(reset_mode=True)
            self.get_logger().info(f'[{self.drone_id}] 失控返航已触地并 disarm，回到 GROUND 待命')
            return

        if (
            self.failsafe_auto_resume
            and self.last_heartbeat_time > 0.0
            and (now - self.last_heartbeat_time) <= 0.5 * self.heartbeat_timeout
            and self.latest_target is not None
            and (now - self.last_target_time) <= 0.5
            and self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
        ):
            self.flight_phase = 'AIRBORNE'
            self.rtl_cmd_sent = False
            self.land_brake_start = None
            self.get_logger().info(f'[{self.drone_id}] 中枢心跳与设定点已恢复，退出失控返航恢复正常飞控')
            return

        nav_state = self.vehicle_status.nav_state
        if self.failsafe_use_px4_rtl or nav_state != VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            if nav_state not in (
                VehicleStatus.NAVIGATION_STATE_AUTO_RTL,
                VehicleStatus.NAVIGATION_STATE_AUTO_LAND,
            ):
                if not self.rtl_cmd_sent or (now - self.last_cmd_retry_time > 1.0):
                    self.last_cmd_retry_time = now
                    self.rtl_cmd_sent = True
                    self._rtl_px4()
            return

        self._publish_offboard_control_mode()
        dn = self.takeoff_n - self.real_n
        de = self.takeoff_e - self.real_e
        d_xy = math.hypot(dn, de)
        h_spd = math.hypot(self.local_pos.vx, self.local_pos.vy)

        if d_xy <= self.failsafe_rtl_xy_tol:
            if self.land_brake_start is None:
                self.land_brake_start = now
            if h_spd <= self.failsafe_rtl_land_speed or (now - self.land_brake_start) >= 2.0:
                self.land_brake_start = None
                self.flight_phase = 'LANDING'
                self.last_cmd_retry_time = now
                self._land()
                self.get_logger().info(
                    f'[{self.drone_id}] 失控返航已抵达起飞点 (误差={d_xy:.2f}m)，切换 PX4 自动降落'
                )
                return
        else:
            self.land_brake_start = None

        stagger_alt = self.failsafe_rtl_alt + self.failsafe_rtl_alt_step * max(
            self.target_system_id - 1, 0
        )
        cur_alt = -(self.real_d - self.takeoff_d)
        cruise_alt = min(38.0, max(cur_alt, stagger_alt))
        cruise_d = self.takeoff_d - cruise_alt

        if d_xy > 1.5 and cur_alt < (stagger_alt - 1.0):
            tgt_n, tgt_e, tgt_d = self.rtl_climb_n, self.rtl_climb_e, cruise_d
        elif d_xy > 1.5:
            tgt_n, tgt_e, tgt_d = self.takeoff_n, self.takeoff_e, cruise_d
        elif d_xy > self.failsafe_rtl_xy_tol:
            final_alt = min(cruise_alt, max(4.0, cur_alt - 1.5 if h_spd <= 0.8 else cur_alt))
            tgt_n, tgt_e, tgt_d = self.takeoff_n, self.takeoff_e, self.takeoff_d - final_alt
        else:
            tgt_n, tgt_e, tgt_d = self.takeoff_n, self.takeoff_e, self.real_d

        yawspeed_cmd = 0.0
        if self.align_yaw and d_xy > 0.6:
            target_yaw = math.atan2(de, dn)
            yawspeed_cmd = compute_yawspeed_ned(
                self.desired_yaw_ned,
                target_yaw,
                tau=0.42,
                max_yaw_rate_rad=self.max_yaw_rate_rad,
            )
            self.desired_yaw_ned = slew_yaw_ned(
                self.desired_yaw_ned,
                target_yaw,
                self.max_yaw_rate_rad * self.dt,
            )

        (self.shadow_n, self.shadow_e, self.shadow_d), vel_cmd, acc_cmd = compute_shadow_ff_accel(
            (self.shadow_n, self.shadow_e, self.shadow_d),
            (self.real_n, self.real_e, self.real_d),
            (tgt_n, tgt_e, tgt_d),
            (0.0, 0.0, 0.0),
            self.prev_vel_cmd_ned,
            self.prev_acc_cmd_ned,
            dt=self.dt,
            max_step=self.failsafe_rtl_speed * self.dt,
            max_tether=2.0,
            max_speed=self.failsafe_rtl_speed,
            gain_p=1.2,
            deadband_m=0.08,
            max_vz_up=3.2,
            max_vz_dn=2.5,
            max_acc_hor=6.0,
            max_acc_vert=4.0,
        )
        self.prev_vel_cmd_ned = vel_cmd
        self.prev_acc_cmd_ned = acc_cmd

        offset_d = self.home_d_offset if self.home_d_offset is not None else 0.0
        self._publish_trajectory_setpoint(
            self.shadow_n,
            self.shadow_e,
            self.shadow_d + offset_d,
            self.desired_yaw_ned,
            vel_ned=vel_cmd,
            acc_ned=acc_cmd,
            yawspeed_ned=yawspeed_cmd,
        )

    def _select_max_speed(self) -> float:
        if self.current_action == 'ATTACK':
            return self.max_step_attack * _REF_STEP_HZ
        if self.current_action == 'INTERCEPT':
            return self.max_step_intercept * _REF_STEP_HZ
        return self.max_step_default * _REF_STEP_HZ

    # -------------------- 50Hz 主控制循环 --------------------
    def _control_loop(self) -> None:
        now = time.time()

        if self.local_pos is None or self.vehicle_status is None:
            self.get_logger().info(
                f'[{self.drone_id}] 等待 PX4 uXRCE-DDS 状态话题 ({self.px4_prefix}/fmu/out/*)...',
                throttle_duration_sec=3.0,
            )
            return

        self._publish_fc_diagnostics(now)
        self._update_px4_gimbal(now)

        # 1. 心跳看门狗：中枢失联触发失控返航（飞回起飞点），不再原地直接降落
        if (
            self.last_heartbeat_time > 0.0
            and (now - self.last_heartbeat_time > self.heartbeat_timeout)
            and self.flight_phase in ('ARMING', 'AIRBORNE')
        ):
            on_ground_prelaunch = (
                self.flight_phase == 'ARMING'
                and (
                    self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED
                    or self.real_d > self.takeoff_d - 0.35
                )
            )
            self.latest_target = None
            self.latest_ff_ned = None
            self.tgt_vel_ned = (0.0, 0.0, 0.0)
            self.land_brake_start = None
            if on_ground_prelaunch:
                self.get_logger().warn(
                    f'[{self.drone_id}] 地面预流期心跳失联 ({self.heartbeat_timeout:.1f}s)，中止起飞退回 GROUND',
                    throttle_duration_sec=1.0,
                )
                if self.vehicle_status.arming_state != VehicleStatus.ARMING_STATE_DISARMED:
                    self._disarm()
                self._reset_to_ground(reset_mode=False)
                return

            if not self.takeoff_recorded:
                self.takeoff_n, self.takeoff_e, self.takeoff_d = (0.0, 0.0, 0.0)
                self.takeoff_recorded = True

            self.flight_phase = 'FAILSAFE_RTL'
            self.rtl_climb_n = self.real_n
            self.rtl_climb_e = self.real_e
            self.rtl_cmd_sent = False
            self.current_action = 'RETURN'
            self.prev_tgt_ned = None
            self.last_cmd_retry_time = 0.0
            self.get_logger().error(
                f'[{self.drone_id}] 中枢心跳失联超过 {self.heartbeat_timeout:.1f}s，触发失控返航 -> '
                f'起飞点 ({self.takeoff_n:.2f}, {self.takeoff_e:.2f})！',
                throttle_duration_sec=1.0,
            )
            if self.failsafe_use_px4_rtl:
                self._rtl_px4()
                self.rtl_cmd_sent = True
            return

        # 2. 失控返航态（飞回记录的起飞原点后再降落）
        if self.flight_phase == 'FAILSAFE_RTL':
            self._handle_failsafe_rtl(now)
            return

        # 3. 目标保鲜期检查：超 2 秒无新坐标则清空旧目标原地悬停
        if self.latest_target is not None and (now - self.last_target_time > self.target_fresh_timeout):
            self.latest_target = None
            self.get_logger().warn(
                f'[{self.drone_id}] 中枢长时间未发新坐标，维持原地悬停',
                throttle_duration_sec=2.0,
            )

        # 4. 地面待命态（EKF 对齐门禁 + Offboard 预流）
        if self.flight_phase == 'GROUND':
            if self.latest_target is not None and self.latest_target.pose.position.z >= 0.3:
                ekf_ready = True
                if self.estimator_flags is not None:
                    ekf_ready = bool(self.estimator_flags.cs_tilt_align and self.estimator_flags.cs_yaw_align)
                if not (self.local_pos.xy_valid and self.local_pos.z_valid and ekf_ready):
                    self.get_logger().warn(
                        f'[{self.drone_id}] EKF2 位置/航向尚未收敛，等待起飞...',
                        throttle_duration_sec=2.0,
                    )
                    return
                self.shadow_n = self.real_n
                self.shadow_e = self.real_e
                self.shadow_d = self.real_d
                self.takeoff_n = self.real_n
                self.takeoff_e = self.real_e
                self.takeoff_d = self.real_d
                self.takeoff_recorded = True
                self.rtl_cmd_sent = False
                self.prev_vel_cmd_ned = (0.0, 0.0, 0.0)
                self.prev_acc_cmd_ned = (0.0, 0.0, 0.0)
                self.arm_tick_count = 0
                self.flight_phase = 'ARMING'
                self.get_logger().info(
                    f'[{self.drone_id}] 收到首个起飞目标 (原点={self.takeoff_n:.2f},{self.takeoff_e:.2f})，开始预流并切入 Offboard...'
                )
            return

        # 4. 预流与解锁态
        if self.flight_phase == 'ARMING':
            offset_d = self.home_d_offset if self.home_d_offset is not None else 0.0
            self._publish_offboard_control_mode()
            self._publish_trajectory_setpoint(
                self.real_n,
                self.real_e,
                self.real_d + offset_d,
                self.desired_yaw_ned,
                vel_ned=(0.0, 0.0, 0.0),
                acc_ned=(0.0, 0.0, 0.0),
                yawspeed_ned=0.0,
            )
            self.arm_tick_count += 1

            if self.arm_tick_count >= self.prestream_ticks:
                armed = self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_ARMED
                in_offboard = self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD
                if armed and in_offboard:
                    self.flight_phase = 'AIRBORNE'
                    self.get_logger().info(f'[{self.drone_id}] 已解锁并进入 Offboard 飞行态')
                elif not self.vehicle_status.pre_flight_checks_pass:
                    self.get_logger().warn(
                        f'[{self.drone_id}] 等待 PX4 起飞前自检通过...',
                        throttle_duration_sec=2.0,
                    )
                elif now - self.last_cmd_retry_time > 0.5:
                    self.last_cmd_retry_time = now
                    self._engage_offboard_and_arm()
            return

        # 5. 空中 50Hz 3-DOF Offboard 执行态
        if self.flight_phase == 'AIRBORNE':
            self._publish_offboard_control_mode()
            vel_cmd = (0.0, 0.0, 0.0)
            acc_cmd = (0.0, 0.0, 0.0)
            yawspeed_cmd = 0.0

            if self.latest_target is not None:
                tgt_pos = self.latest_target.pose.position
                tgt_n, tgt_e, tgt_d = enu_to_ned_pos(
                    tgt_pos.x, tgt_pos.y, tgt_pos.z, self.offset_x, self.offset_y
                )
                # 垂直降落前水平刹车
                if tgt_pos.z < 0.3:
                    h_spd = math.hypot(self.local_pos.vx, self.local_pos.vy)
                    if self.land_brake_start is None:
                        self.land_brake_start = now
                    if h_spd > 0.8 and (now - self.land_brake_start) < 1.2:
                        offset_d = self.home_d_offset if self.home_d_offset is not None else 0.0
                        self.shadow_n, self.shadow_e = tgt_n, tgt_e
                        self._publish_trajectory_setpoint(
                            tgt_n,
                            tgt_e,
                            self.real_d + offset_d,
                            self.desired_yaw_ned,
                            vel_ned=(0.0, 0.0, 0.0),
                            acc_ned=(0.0, 0.0, 0.0),
                            yawspeed_ned=0.0,
                        )
                        return
                    self.land_brake_start = None
                    self.flight_phase = 'LANDING'
                    self.latest_target = None
                    self._land()
                    self.get_logger().info(f'[{self.drone_id}] 收到垂直降落目标，切换 PX4 自动降落')
                    return
                self.land_brake_start = None

                if tgt_d > 0.0:
                    tgt_d = 0.0

                if self.current_action in ('TRACK', 'HOVER') and self.min_track_alt > 0.0:
                    if tgt_d > -self.min_track_alt:
                        tgt_d = -self.min_track_alt

                if self.align_yaw:
                    dn = tgt_n - self.real_n
                    de = tgt_e - self.real_e
                    gimbal_fresh = (
                        self.gimbal_target_yaw_ned is not None
                        and (now - self.last_gimbal_recv_time) < 1.5
                    )
                    target_yaw = self.desired_yaw_ned
                    if self.current_action in ('TRACK', 'INTERCEPT', 'ATTACK') and gimbal_fresh:
                        target_yaw = self.gimbal_target_yaw_ned
                    elif math.hypot(dn, de) > 0.6:
                        target_yaw = math.atan2(de, dn)
                    elif gimbal_fresh:
                        target_yaw = self.gimbal_target_yaw_ned
                    yawspeed_cmd = compute_yawspeed_ned(
                        self.desired_yaw_ned,
                        target_yaw,
                        tau=0.42,
                        max_yaw_rate_rad=self.max_yaw_rate_rad,
                    )
                    self.desired_yaw_ned = slew_yaw_ned(
                        self.desired_yaw_ned,
                        target_yaw,
                        self.max_yaw_rate_rad * self.dt,
                    )

                max_speed = self._select_max_speed()
                max_step_tick = max_speed * self.dt
                if self.current_action == 'ATTACK':
                    gain_p, deadband_m, max_tether, vz_up, vz_dn = 4.2, 0.02, 3.0, 4.8, 5.0
                    max_ah, max_av = 11.8, 5.5
                elif self.current_action == 'INTERCEPT':
                    gain_p, deadband_m, max_tether, vz_up, vz_dn = 1.8, 0.12, 2.4, 3.8, 4.0
                    max_ah, max_av = 8.0, 4.5
                else:
                    gain_p, deadband_m, max_tether, vz_up, vz_dn = 0.40, 0.22, 0.9, 3.0, 2.5
                    max_ah, max_av = 5.0, 3.5

                if self.latest_ff_ned is not None and (now - self.last_ff_time) <= 0.22:
                    vel_ff_base = self.latest_ff_ned
                elif (now - self.last_target_time) < 0.35:
                    vel_ff_base = self.tgt_vel_ned
                else:
                    vel_ff_base = (0.0, 0.0, 0.0)
                (self.shadow_n, self.shadow_e, self.shadow_d), vel_cmd, acc_cmd = compute_shadow_ff_accel(
                    (self.shadow_n, self.shadow_e, self.shadow_d),
                    (self.real_n, self.real_e, self.real_d),
                    (tgt_n, tgt_e, tgt_d),
                    vel_ff_base,
                    self.prev_vel_cmd_ned,
                    self.prev_acc_cmd_ned,
                    dt=self.dt,
                    max_step=max_step_tick,
                    max_tether=max_tether,
                    max_speed=max_speed,
                    gain_p=gain_p,
                    deadband_m=deadband_m,
                    max_vz_up=vz_up,
                    max_vz_dn=vz_dn,
                    max_acc_hor=max_ah,
                    max_acc_vert=max_av,
                )
                self.prev_vel_cmd_ned = vel_cmd
                self.prev_acc_cmd_ned = acc_cmd

            offset_d = self.home_d_offset if self.home_d_offset is not None else 0.0
            self._publish_trajectory_setpoint(
                self.shadow_n,
                self.shadow_e,
                self.shadow_d + offset_d,
                self.desired_yaw_ned,
                vel_ned=vel_cmd,
                acc_ned=acc_cmd,
                yawspeed_ned=yawspeed_cmd,
            )
            return

        # 7. 降落与自动上锁态
        if self.flight_phase == 'LANDING':
            if self.vehicle_status.arming_state == VehicleStatus.ARMING_STATE_DISARMED:
                self._reset_to_ground(reset_mode=True)
                self.get_logger().info(f'[{self.drone_id}] 已安全触地并 disarm，回到 GROUND 待命')
                return

            if now - self.last_cmd_retry_time > 1.0:
                self.last_cmd_retry_time = now
                speed = math.sqrt(
                    self.local_pos.vx ** 2 + self.local_pos.vy ** 2 + self.local_pos.vz ** 2
                )
                if self.real_d > self.land_d_threshold and speed < 0.25:
                    self._disarm()
                elif self.vehicle_status.nav_state != VehicleStatus.NAVIGATION_STATE_AUTO_LAND:
                    self._land()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Px4BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
