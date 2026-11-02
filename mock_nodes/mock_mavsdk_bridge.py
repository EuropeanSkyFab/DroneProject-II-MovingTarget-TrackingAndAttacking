import rclpy
import time
import asyncio
import math
import threading
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty, String
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw


class MavsdkBridge(Node):
    def __init__(self):
        super().__init__('mavsdk_bridge')

        # 订阅中枢指令
        self.create_subscription(PoseStamped, '/drone_1/fmu/in/trajectory_setpoint', self.pose_callback, 10)
        self.create_subscription(String, '/drone_1/action', self.action_callback, 10)
        self.create_subscription(Empty, '/drone_1/heartbeat', self.heartbeat_callback, 10)

        # 发布己方状态给中枢
        self.state_pub = self.create_publisher(Odometry, '/drone_1/state', 10)

        self.latest_target = None
        self.last_target_time = 0.0
        self.current_action = "HOVER"
        self.last_heartbeat_time = 0.0

        # ★ 飞行阶段标志
        self.is_airborne = False      # False = 地面 disarm 待命；True = 已在 Offboard 飞行
        self.is_landing = False

        # ★ 影子位置（限幅累加用）
        self.current_n = 0.0
        self.current_e = 0.0
        self.current_d = 0.0

        # ★ 真实位置（遥测始终维护，供落地检测用）
        self.real_n = 0.0
        self.real_e = 0.0
        self.real_d = 0.0

        # ★ 阈值
        self.HEARTBEAT_TIMEOUT = 1.0
        self.TARGET_FRESH_TIMEOUT = 2.0
        self.LAND_D_THRESHOLD = -0.2      # NED down 接近 0 视为落地

        self.drone = System(port=50052)
        self.get_logger().info("MAVSDK 桥接节点已启动...")

        self.mavsdk_thread = threading.Thread(target=self.run_mavsdk_loop, daemon=True)
        self.mavsdk_thread.start()

    def pose_callback(self, msg):
        self.latest_target = msg
        self.last_target_time = time.time()

    def action_callback(self, msg):
        self.current_action = msg.data
        self.get_logger().warn(f"⚡ 收到动作标签: {self.current_action}")

    def heartbeat_callback(self, msg):
        self.last_heartbeat_time = time.time()

    def run_mavsdk_loop(self):
        asyncio.run(self.mavsdk_connect_and_fly())

    # ★ 遥测独立协程：始终维护 real_*，无目标时对齐 current_*
    async def telemetry_loop(self):
        async for pos in self.drone.telemetry.position_velocity_ned():
            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = "world"
            odom.child_frame_id = "drone_1_base_link"
            odom.pose.pose.position.x = pos.position.east_m
            odom.pose.pose.position.y = pos.position.north_m
            odom.pose.pose.position.z = -pos.position.down_m
            odom.twist.twist.linear.x = pos.velocity.east_m_s
            odom.twist.twist.linear.y = pos.velocity.north_m_s
            odom.twist.twist.linear.z = -pos.velocity.down_m_s
            self.state_pub.publish(odom)

            # ★ 真实位置始终更新
            self.real_n = pos.position.north_m
            self.real_e = pos.position.east_m
            self.real_d = pos.position.down_m

            # ★ 无目标时用真实位置对齐影子位置
            if self.latest_target is None:
                self.current_n = self.real_n
                self.current_e = self.real_e
                self.current_d = self.real_d

    async def mavsdk_connect_and_fly(self):
        await self.drone.connect(system_address="udp://:14540")

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                break

        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                break

        # ★ 地面 disarm 待命，不 arm、不 offboard
        self.get_logger().info("已连接 PX4，地面 disarm 待命，等待中枢 TAKEOFF...")
        self.telemetry_task = asyncio.create_task(self.telemetry_loop())

        while rclpy.ok():
            # 1. 心跳看门狗（★ 仅空中才触发降落）
            if self.last_heartbeat_time > 0.0 and (time.time() - self.last_heartbeat_time > self.HEARTBEAT_TIMEOUT):
                if self.is_airborne and not self.is_landing:
                    self.is_landing = True
                    self.get_logger().error("💀 中枢彻底失联！触发强制降落！")
                    try:
                        await self.drone.offboard.stop()
                    except OffboardError:
                        pass
                    await self.drone.action.land()
                    await self._wait_and_disarm()
                    self.is_airborne = False
                    self.is_landing = False
                await asyncio.sleep(0.05)
                continue

            # 2. 数据保鲜期
            if self.latest_target is not None and (time.time() - self.last_target_time > self.TARGET_FRESH_TIMEOUT):
                self.latest_target = None
                self.get_logger().warn("⚠️ 中枢长时间未发新坐标，原地悬停。", throttle_duration_sec=1.0)

            # ★ 3. 地面待命 → 收到首个目标点，执行起飞流程
            if not self.is_airborne and not self.is_landing and self.latest_target is not None:
                self.get_logger().info("收到首个目标点，执行 arm + offboard 起飞流程...")
                try:
                    await self.drone.action.arm()
                    await self.drone.offboard.set_position_ned(
                        PositionNedYaw(self.current_n, self.current_e, self.current_d, 0.0)
                    )
                    await self.drone.offboard.start()
                    self.is_airborne = True
                    self.get_logger().info("Offboard 已启动，进入飞行模式。")
                except Exception as e:
                    self.get_logger().error(f"起飞流程失败：{e}")
                    await asyncio.sleep(0.05)
                    continue

            # 4. 飞行中：限幅逻辑
            if self.is_airborne and self.latest_target is not None:
                if self.current_action == "ATTACK":
                    max_step = 0.5
                elif self.current_action == "INTERCEPT":
                    max_step = 0.3
                else:
                    max_step = 0.15

                target_n = self.latest_target.pose.position.y
                target_e = self.latest_target.pose.position.x
                target_d = -self.latest_target.pose.position.z

                if target_d > 0.0:
                    target_d = 0.0

                diff_n = target_n - self.current_n
                diff_e = target_e - self.current_e
                diff_d = target_d - self.current_d
                distance = math.sqrt(diff_n**2 + diff_e**2 + diff_d**2)

                if distance > max_step:
                    self.current_n += (diff_n / distance) * max_step
                    self.current_e += (diff_e / distance) * max_step
                    self.current_d += (diff_d / distance) * max_step
                else:
                    self.current_n, self.current_e, self.current_d = target_n, target_e, target_d

                await self.drone.offboard.set_position_ned(
                    PositionNedYaw(self.current_n, self.current_e, self.current_d, 0.0)
                )

            # ★ 5. 落地检测：目标是地面点且已贴地 → land + disarm
            if self.is_airborne and not self.is_landing and self.latest_target is not None:
                target_z_enu = self.latest_target.pose.position.z
                if target_z_enu < 0.3 and self.real_d > self.LAND_D_THRESHOLD:
                    self.get_logger().info("已到达地面点，执行 land + disarm...")
                    self.is_landing = True
                    try:
                        await self.drone.offboard.stop()
                    except OffboardError:
                        pass
                    await self.drone.action.land()
                    await self._wait_and_disarm()
                    self.is_airborne = False
                    self.is_landing = False
                    self.get_logger().info("已回到地面 disarm 待命。")

            await asyncio.sleep(0.05)  # 20Hz

        if getattr(self, 'telemetry_task', None) is not None:
            self.telemetry_task.cancel()

    # ★ 用 real_d 判断落地并 disarm，不开新遥测流
    async def _wait_and_disarm(self):
        for _ in range(200):  # 最多 10s
            if self.real_d > self.LAND_D_THRESHOLD:
                try:
                    await self.drone.action.disarm()
                    self.get_logger().info("已落地并 disarm。")
                except Exception as e:
                    self.get_logger().warn(f"disarm 失败：{e}")
                return
            await asyncio.sleep(0.05)
        self.get_logger().warn("等待落地超时，未能确认 disarm。")


def main(args=None):
    rclpy.init(args=args)
    node = MavsdkBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()