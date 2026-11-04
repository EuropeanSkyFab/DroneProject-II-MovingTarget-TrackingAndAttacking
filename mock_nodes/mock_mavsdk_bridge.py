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
        # ★ 节点名字设为通用，避免冲突
        super().__init__('mavsdk_bridge')

        # ================= 动态参数配置 =================
        self.declare_parameter('drone_id', 'drone_1')
        self.declare_parameter('port', 50052)
        
        self.drone_id = self.get_parameter('drone_id').value
        self.port = self.get_parameter('port').value
        # ===============================================

        # 订阅中枢指令（★ 动态拼接话题）
        self.create_subscription(PoseStamped, f'/{self.drone_id}/fmu/in/trajectory_setpoint', self.pose_callback, 10)
        self.create_subscription(String, f'/{self.drone_id}/action', self.action_callback, 10)
        self.create_subscription(Empty, f'/{self.drone_id}/heartbeat', self.heartbeat_callback, 10)

        # 发布己方状态给中枢
        self.state_pub = self.create_publisher(Odometry, f'/{self.drone_id}/state', 10)

        self.latest_target = None
        self.last_target_time = 0.0
        self.current_action = "HOVER"
        self.last_heartbeat_time = 0.0

        self.is_airborne = False
        self.is_landing = False

        self.current_n = 0.0
        self.current_e = 0.0
        self.current_d = 0.0

        self.real_n = 0.0
        self.real_e = 0.0
        self.real_d = 0.0

        self.HEARTBEAT_TIMEOUT = 3.0
        self.TARGET_FRESH_TIMEOUT = 2.0
        self.LAND_D_THRESHOLD = -0.2

        # ★ 动态连接端口
        self.drone = System(port=self.port)
        self.get_logger().info(f"[{self.drone_id}] MAVSDK 桥接已启动，监听端口 {self.port}...")

        self.mavsdk_thread = threading.Thread(target=self.run_mavsdk_loop, daemon=True)
        self.mavsdk_thread.start()

    def pose_callback(self, msg):
        self.latest_target = msg
        self.last_target_time = time.time()

    def action_callback(self, msg):
        self.current_action = msg.data
        self.get_logger().warn(f"[{self.drone_id}] 收到动作标签: {self.current_action}")

    def heartbeat_callback(self, msg):
        self.last_heartbeat_time = time.time()

    def run_mavsdk_loop(self):
        asyncio.run(self.mavsdk_connect_and_fly())

    async def telemetry_loop(self):
        async for pos in self.drone.telemetry.position_velocity_ned():
            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = "world"
            odom.child_frame_id = f"{self.drone_id}_base_link" # ★ 动态 frame_id
            odom.pose.pose.position.x = pos.position.east_m
            odom.pose.pose.position.y = pos.position.north_m
            odom.pose.pose.position.z = -pos.position.down_m
            odom.twist.twist.linear.x = pos.velocity.east_m_s
            odom.twist.twist.linear.y = pos.velocity.north_m_s
            odom.twist.twist.linear.z = -pos.velocity.down_m_s
            self.state_pub.publish(odom)

            self.real_n = pos.position.north_m
            self.real_e = pos.position.east_m
            self.real_d = pos.position.down_m

            if self.latest_target is None and not self.is_landing:
                self.current_n = self.real_n
                self.current_e = self.real_e
                self.current_d = self.real_d

    async def mavsdk_connect_and_fly(self):
        # ★ 注意：这里的连接地址可能需要根据实际端口微调，通常 MAVSDK 是通过特定端口监听
        await self.drone.connect(system_address=f"udp://:{14540 + (self.port - 50052)}") 
        # 这里的 14540 + offset 只是个示例，具体取决于 PX4 SITL 的 UDP 端口分配

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                break

        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                break

        self.get_logger().info(f"[{self.drone_id}] 已连接 PX4，等待起飞...")
        self.telemetry_task = asyncio.create_task(self.telemetry_loop())

        while rclpy.ok():
            # 心跳看门狗
            if self.last_heartbeat_time > 0.0 and (time.time() - self.last_heartbeat_time > self.HEARTBEAT_TIMEOUT):
                if self.is_airborne and not self.is_landing:
                    self.is_landing = True
                    self.get_logger().error(f"[{self.drone_id}] 中枢失联！强制降落！")
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

            # 保鲜期
            if self.latest_target is not None and (time.time() - self.last_target_time > self.TARGET_FRESH_TIMEOUT):
                self.latest_target = None
                self.get_logger().warn(f"[{self.drone_id}] 中枢长时间未发新坐标，原地悬停。", throttle_duration_sec=1.0)

            # 起飞
            if not self.is_airborne and not self.is_landing and self.latest_target is not None:
                self.get_logger().info(f"[{self.drone_id}] 收到目标点，执行起飞流程...")
                try:
                    await self.drone.action.arm()
                    await self.drone.offboard.set_position_ned(
                        PositionNedYaw(self.current_n, self.current_e, self.current_d, 0.0)
                    )
                    await self.drone.offboard.start()
                    self.is_airborne = True
                except Exception as e:
                    self.get_logger().error(f"[{self.drone_id}] 起飞失败：{e}")
                    await asyncio.sleep(0.05)
                    continue

            # 飞行中
            if self.is_airborne and not self.is_landing and self.latest_target is not None:
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

            # 降落
            if self.is_airborne and not self.is_landing and self.latest_target is not None:
                target_z_enu = self.latest_target.pose.position.z
                if target_z_enu < 0.3 and self.real_d > self.LAND_D_THRESHOLD:
                    self.is_landing = True 
                    self.get_logger().info(f"[{self.drone_id}] 到达地面点，执行降落...")
                    try:
                        await self.drone.offboard.stop()
                    except OffboardError:
                        pass
                    await self.drone.action.land()
                    await self._wait_and_disarm()
                    
                    self.is_airborne = False
                    self.is_landing = False 
                    self.latest_target = None 
                    self.get_logger().info(f"[{self.drone_id}] 已回到地面待命。")

            await asyncio.sleep(0.05)

        if getattr(self, 'telemetry_task', None) is not None:
            self.telemetry_task.cancel()

    async def _wait_and_disarm(self):
        for _ in range(200):
            if self.real_d > self.LAND_D_THRESHOLD:
                try:
                    await self.drone.action.disarm()
                    self.get_logger().info(f"[{self.drone_id}] 已落地并 disarm。")
                    return
                except Exception as e:
                    await asyncio.sleep(0.5)
            await asyncio.sleep(0.05)

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