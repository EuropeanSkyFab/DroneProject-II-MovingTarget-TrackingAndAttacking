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

        # ================= 动态参数配置 =================
        self.declare_parameter('drone_id', 'drone_1')
        self.declare_parameter('port', 50052)       # MAVSDK 本地监听端口
        self.declare_parameter('mav_port', 14540)   # PX4 UDP 端口
        
        self.drone_id = self.get_parameter('drone_id').value
        self.port = self.get_parameter('port').value
        self.mav_port = self.get_parameter('mav_port').value
        # ===============================================

        # ================= 世界坐标系偏移配置 =================
        # 这里的偏移量必须与你在 Gazebo 里启动时的 PX4_GZ_MODEL_POSE 保持一致
        self.offset_x = 0.0
        self.offset_y = 0.0
        if self.drone_id == "drone_1":
            self.offset_x, self.offset_y = 0.0, 0.0
        elif self.drone_id == "drone_2":
            self.offset_x, self.offset_y = 0.0, 2.0   # 对应 PX4_GZ_MODEL_POSE="0,2"
        elif self.drone_id == "drone_3":
            self.offset_x, self.offset_y = 0.0, 4.0   # 对应 PX4_GZ_MODEL_POSE="0,4"
        # =======================================================

        # 订阅中枢指令（动态拼接话题）
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

        # 影子位置（NED）
        self.current_n = 0.0
        self.current_e = 0.0
        self.current_d = 0.0

        # 真实位置（NED）
        self.real_n = 0.0
        self.real_e = 0.0
        self.real_d = 0.0

        self.HEARTBEAT_TIMEOUT = 3.0
        self.TARGET_FRESH_TIMEOUT = 2.0
        self.LAND_D_THRESHOLD = -0.2

        # 动态连接端口
        self.drone = System(port=self.port)
        self.get_logger().info(f"[{self.drone_id}] MAVSDK 桥接已启动，监听端口 {self.port}，连接 PX4 端口 {self.mav_port}...")

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
            odom.child_frame_id = f"{self.drone_id}_base_link"
            
            # 1. PX4 原本反馈的局部坐标（以起飞点为原点）
            local_x = pos.position.east_m
            local_y = pos.position.north_m
            local_z = -pos.position.down_m
            
            # 2. 转换为全局世界坐标（加上物理偏移）
            world_x = local_x + self.offset_x
            world_y = local_y + self.offset_y
            
            odom.pose.pose.position.x = world_x
            odom.pose.pose.position.y = world_y
            odom.pose.pose.position.z = local_z
            
            odom.twist.twist.linear.x = pos.velocity.east_m_s
            odom.twist.twist.linear.y = pos.velocity.north_m_s
            odom.twist.twist.linear.z = -pos.velocity.down_m_s
            self.state_pub.publish(odom)

            # 3. 更新内部影子变量
            self.real_n = local_y + self.offset_y
            self.real_e = local_x + self.offset_x
            self.real_d = pos.position.down_m

            if self.latest_target is None and not self.is_landing:
                self.current_n = self.real_n
                self.current_e = self.real_e
                self.current_d = self.real_d

    async def mavsdk_connect_and_fly(self):
        # 直接使用动态传入的 mav_port
        await self.drone.connect(system_address=f"udp://:{self.mav_port}")

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
                    # 起飞前先发一个初始位置（使用局部坐标）
                    init_n = self.real_n - self.offset_y
                    init_e = self.real_e - self.offset_x
                    await self.drone.offboard.set_position_ned(
                        PositionNedYaw(init_n, init_e, self.real_d, 0.0)
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

                # 中枢发来的是世界坐标，我们要减掉偏移，转回 PX4 的局部坐标
                target_e = self.latest_target.pose.position.x - self.offset_x
                target_n = self.latest_target.pose.position.y - self.offset_y
                target_d = -self.latest_target.pose.position.z

                if target_d > 0.0:
                    target_d = 0.0

                # 影子位置计算需要把世界坐标转回局部
                diff_n = target_n - (self.current_n - self.offset_y)
                diff_e = target_e - (self.current_e - self.offset_x)
                diff_d = target_d - self.current_d
                
                distance = math.sqrt(diff_n**2 + diff_e**2 + diff_d**2)

                if distance > max_step:
                    step_n = (diff_n / distance) * max_step
                    step_e = (diff_e / distance) * max_step
                    step_d = (diff_d / distance) * max_step
                else:
                    step_n = diff_n
                    step_e = diff_e
                    step_d = diff_d

                # 更新影子位置（局部坐标系下）
                local_current_n = self.current_n - self.offset_y + step_n
                local_current_e = self.current_e - self.offset_x + step_e
                self.current_d += step_d

                # 存回世界坐标系
                self.current_n = local_current_n + self.offset_y
                self.current_e = local_current_e + self.offset_x

                await self.drone.offboard.set_position_ned(
                    PositionNedYaw(local_current_n, local_current_e, self.current_d, 0.0)
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