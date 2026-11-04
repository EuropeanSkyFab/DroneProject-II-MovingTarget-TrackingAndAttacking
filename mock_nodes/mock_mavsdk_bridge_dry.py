import rclpy
import math
import time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String

class MockBridgeDry(Node):
    def __init__(self):
        super().__init__('mock_bridge_dry')
        
        # ================= 动态参数配置 =================
        self.declare_parameter('drone_id', 'drone_1')
        self.drone_id = self.get_parameter('drone_id').value
        # ===============================================

        # 1. 发布假状态给中枢
        self.state_pub = self.create_publisher(Odometry, f'/{self.drone_id}/state', 10)
        
        # 2. 订阅中枢发来的坐标和动作
        self.create_subscription(PoseStamped, f'/{self.drone_id}/fmu/in/trajectory_setpoint', self.target_callback, 10)
        self.create_subscription(String, f'/{self.drone_id}/action', self.action_callback, 10)
        
        # 3. 模拟内部状态
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_z = 0.0  # 初始在地面
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.max_step = 0.15  # 默认速度 3m/s
        self.has_target = False
        
        # 4. 定时器，20Hz 广播假位置并模拟运动
        self.timer = self.create_timer(0.05, self.timer_callback)
        
        self.get_logger().info(f"✅ [{self.drone_id}] 轻量化假桥接已启动！无需 PX4 和 Gazebo。")

    def action_callback(self, msg):
        action = msg.data
        if action == "ATTACK":
            self.max_step = 0.5
        elif action == "INTERCEPT":
            self.max_step = 0.3
        else:
            self.max_step = 0.15

    def target_callback(self, msg):
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
        self.target_z = msg.pose.position.z
        self.has_target = True

    def timer_callback(self):
        # 1. 模拟运动学：向着目标点移动
        if self.has_target:
            dx = self.target_x - self.current_x
            dy = self.target_y - self.current_y
            dz = self.target_z - self.current_z
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            if dist > self.max_step:
                self.current_x += (dx / dist) * self.max_step
                self.current_y += (dy / dist) * self.max_step
                self.current_z += (dz / dist) * self.max_step
            else:
                self.current_x = self.target_x
                self.current_y = self.target_y
                self.current_z = self.target_z

        # 2. 组装并发布 Odometry
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.child_frame_id = f"{self.drone_id}_base_link"
        
        msg.pose.pose.position.x = self.current_x
        msg.pose.pose.position.y = self.current_y
        msg.pose.pose.position.z = self.current_z
        msg.pose.pose.orientation.w = 1.0
        
        # 计算一下当前速度，供中枢落地检测使用
        msg.twist.twist.linear.x = (self.target_x - self.current_x) * 0.1
        msg.twist.twist.linear.y = (self.target_y - self.current_y) * 0.1
        msg.twist.twist.linear.z = (self.target_z - self.current_z) * 0.1
        
        self.state_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = MockBridgeDry()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()