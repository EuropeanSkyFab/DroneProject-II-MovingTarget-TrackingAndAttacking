import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
import asyncio
import threading

class MavsdkBridge(Node):
    def __init__(self):
        super().__init__('mavsdk_bridge')
        
        # 订阅中枢节点发来的标准 ROS 2 坐标
        self.subscription = self.create_subscription(
            PoseStamped,
            '/drone_1/fmu/in/trajectory_setpoint',
            self.pose_callback,
            10
        )
        
        self.latest_target = None
        self.drone = System(port=50052) 
        self.get_logger().info("MAVSDK 桥接节点已启动，准备连接 PX4...")

        # 启动后台异步线程处理 MAVSDK
        self.mavsdk_thread = threading.Thread(target=self.run_mavsdk_loop, daemon=True)
        self.mavsdk_thread.start()

    def pose_callback(self, msg):
        # 把中枢发来的 ENU 坐标存下来
        self.latest_target = msg

    def run_mavsdk_loop(self):
        asyncio.run(self.mavsdk_connect_and_fly())

    async def mavsdk_connect_and_fly(self):
        await self.drone.connect(system_address="udp://:14540")
        self.get_logger().info("等待 PX4 连接...")

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                self.get_logger().info("PX4 已连接！")
                break

        self.get_logger().info("等待 GPS 锁定...")
        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok and health.is_home_position_ok:
                self.get_logger().info("GPS 已就绪！")
                break

        # 解锁
        await self.drone.action.arm()
        
        # 记录当前平滑后的坐标，初始设为起点(0,0,-3) 悬停
        # 注意：NED 坐标系下，D为负代表高度向上
        current_n = 0.0
        current_e = 0.0
        current_d = -3.0

        # 发送初始起飞指令
        await self.drone.offboard.set_position_ned(PositionNedYaw(current_n, current_e, current_d, 0.0))
        
        try:
            await self.drone.offboard.start()
            self.get_logger().info("Offboard 模式已启动！开始接收中枢指令...")
        except OffboardError as error:
            self.get_logger().error(f"启动失败: {error._result.result}")
            await self.drone.action.disarm()
            return

        # 核心控制循环（20Hz）
        while rclpy.ok():
            if self.latest_target is not None:
                # 1. 提取中枢发来的 ENU 目标点
                target_enu_x = self.latest_target.pose.position.x
                target_enu_y = self.latest_target.pose.position.y
                target_enu_z = self.latest_target.pose.position.z

                # 2. 将 ENU 转换为 NED (注意：北=ENU_y, 东=ENU_x, 地=-ENU_z)
                target_n = target_enu_y
                target_e = target_enu_x
                target_d = -target_enu_z

                # ================= 核心安全保护 =================
                # 保护 1：高度限幅（防止钻地）
                # 假设 NED 中 target_d > -0.5 就说明高度不到 0.5 米了，强制截断
                if target_d > -0.5:
                    target_d = -0.5
                
                # 保护 2：平滑逼近（防瞬移抽搐）
                max_step = 0.05  # 每次循环（0.05秒）最多移动 5 厘米
                
                # 计算当前位置与目标位置的差值
                diff_n = target_n - current_n
                diff_e = target_e - current_e
                diff_d = target_d - current_d

                # 把差值限制在 max_step 范围内，累加到 current 上
                current_n += max(-max_step, min(diff_n, max_step))
                current_e += max(-max_step, min(diff_e, max_step))
                current_d += max(-max_step, min(diff_d, max_step))
                # ================================================

                # 3. 发送平滑处理后的坐标给 PX4
                await self.drone.offboard.set_position_ned(
                    PositionNedYaw(current_n, current_e, current_d, 0.0)
                )
                
            await asyncio.sleep(0.05) # 20Hz

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