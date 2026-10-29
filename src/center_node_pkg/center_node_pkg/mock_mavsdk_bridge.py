import rclpy
import time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry  
from mavsdk import System
from mavsdk.offboard import OffboardError, PositionNedYaw
from std_msgs.msg import Empty
import asyncio
import threading
import math

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
        self.last_target_time = 0.0 # 记录最后一次收到中枢坐标的时间
        self.is_landing = False  # 防止重复发送降落指令的锁
        
        self.drone = System(port=50052) 
        self.get_logger().info("MAVSDK 桥接节点已启动，准备连接 PX4...")
        
        # 向中枢发送当前己方无人机的坐标
        # ================= 多机桥接架构预留 =================
        # 字典键是飞机 ID，值是对应的 MAVSDK 端口（以后加飞机就加一行）
        self.swarm_ports = {
            'drone_1': 50052,
            # 'drone_2': 50053,
        }
        
        # 存放多机发布者的字典
        self.state_pubs = {}
        for drone_id in self.swarm_ports.keys():
            topic_name = f'/{drone_id}/state'
            self.state_pubs[drone_id] = self.create_publisher(Odometry, topic_name, 10)
        # ====================================================

        # 启动后台异步线程处理 MAVSDK
        self.mavsdk_thread = threading.Thread(target=self.run_mavsdk_loop, daemon=True)
        self.mavsdk_thread.start()
        
        # 记录心跳
        self.last_heartbeat_time = 0.0
        # 订阅中枢的心跳
        self.create_subscription(Empty, '/drone_1/heartbeat', self.heartbeat_callback, 10)

    # 处理心跳的回调函数
    def heartbeat_callback(self, msg):
        self.last_heartbeat_time = time.time() # 收到心跳就刷新时间

    def pose_callback(self, msg):
        # 把中枢发来的 ENU 坐标存下来
        self.latest_target = msg
        self.last_target_time = time.time() # 只要收到新坐标，就刷新时间

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
            
            # 【第一部分】：检测中枢心跳
            if self.last_heartbeat_time > 0.0 and (time.time() - self.last_heartbeat_time > 1.0):
                
                # 如果还没触发过降落，就主动下发降落指令
                if not self.is_landing:
                    self.get_logger().error("💀 中枢彻底失联！桥接主动触发 PX4 降落！", throttle_duration_sec=1.0)
                    self.latest_target = None 
                    self.is_landing = True # 上锁，防止重复调用
                    
                    # 【核心修改】：主动让 MAVSDK 接管并执行降落
                    await self.drone.action.land()
                
                # 降落过程中，循环直接跳过
                await asyncio.sleep(0.05)
                continue 


            # 【第二部分】：向中枢回传己方飞机的真实位置
            try:
                async for pos in self.drone.telemetry.position_velocity_ned():
                    for drone_id, pub in self.state_pubs.items():
                        if drone_id == 'drone_1': 
                            odom = Odometry()
                            odom.header.stamp = self.get_clock().now().to_msg()
                            odom.header.frame_id = "world"
                            odom.child_frame_id = drone_id
                            
                            # NED 转 ENU
                            odom.pose.pose.position.x = pos.position.east_m
                            odom.pose.pose.position.y = pos.position.north_m
                            odom.pose.pose.position.z = -pos.position.down_m
                            
                            pub.publish(odom)
                    break
            except Exception:
                pass # 如果读取失败，安静地跳过，别让程序崩溃


            # 【第三部分】：判断是否悬停
            # 如果超过 2 秒没收到中枢发来的新坐标，强行清空目标
            if self.latest_target is not None and (time.time() - self.last_target_time > 2.0):
                self.get_logger().warn("⚠️ 中枢长时间未发新坐标，清除旧目标！", throttle_duration_sec=1.0)
                self.latest_target = None
                
            if self.latest_target is None:
                # 中枢活着，但没发新目标。让飞机停在当前位置
                await self.drone.offboard.set_position_ned(
                    PositionNedYaw(current_n, current_e, current_d, 0.0)
                )
                await asyncio.sleep(0.05)
                continue # 悬停指令发完了，直接跳过本次循环


            # 【第四部分】：正常飞向中枢发来的目标
            target_enu_x = self.latest_target.pose.position.x
            target_enu_y = self.latest_target.pose.position.y
            target_enu_z = self.latest_target.pose.position.z

            # ENU 转 NED
            target_n = target_enu_y
            target_e = target_enu_x
            target_d = -target_enu_z

            # 安全保护 1：高度限幅
            if target_d > -0.5:
                target_d = -0.5
            
            # 安全保护 2：平滑逼近（修改了之前走折线不走直线的bug）
            max_step = 0.15 
            diff_n = target_n - current_n
            diff_e = target_e - current_e
            diff_d = target_d - current_d

            distance = math.sqrt(diff_n**2 + diff_e**2 + diff_d**2)

            # 如果距离大于最小步长，则按比例分配步长，保证走直线
            if distance > max_step:
                step_n = (diff_n / distance) * max_step
                step_e = (diff_e / distance) * max_step
                step_d = (diff_d / distance) * max_step
            else:
                # 如果距离极近，就直接到达目标（防止在终点附近无限震荡）
                step_n = diff_n
                step_e = diff_e
                step_d = diff_d

            # 更新当前目标点
            current_n += step_n
            current_e += step_e
            current_d += step_d

            # 最终下发
            await self.drone.offboard.set_position_ned(
                PositionNedYaw(current_n, current_e, current_d, 0.0)
            )
            
            # 循环休眠，维持 20Hz
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