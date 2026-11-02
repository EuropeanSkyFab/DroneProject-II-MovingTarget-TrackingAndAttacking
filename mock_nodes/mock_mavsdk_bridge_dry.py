import rclpy
import math
import time
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

class MockBridgeDry(Node):
    def __init__(self):
        super().__init__('mock_bridge_dry')
        
        # 1. 假装自己是底层，发假状态给中枢/规划
        self.state_pub = self.create_publisher(Odometry, '/drone_1/state', 10)
        
        # 2. 假装自己是中枢，接收规划/视觉发出的坐标
        self.create_subscription(PoseStamped, '/drone_1/fmu/in/trajectory_setpoint', self.target_callback, 10)
        
        # 3. 定时器，20Hz 广播假位置
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.t = 0.0
        
        self.get_logger().info("✅ 轻量化假桥接已启动！无需 PX4 和 Gazebo。")
        self.get_logger().info("✅ 正在模拟发送 /drone_1/state，并监听收到的坐标...")

    def timer_callback(self):
        self.t += 0.05
        # 假装无人机在原地（0,0,0）轻微晃动
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.child_frame_id = "drone_1_base_link"
        
        msg.pose.pose.position.x = 0.0
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = 0.1 * math.sin(self.t) # 高度轻微波动
        msg.pose.pose.orientation.w = 1.0
        
        msg.twist.twist.linear.x = 0.0 # 速度
        self.state_pub.publish(msg)

    def target_callback(self, msg):
        # 只要收到中枢（或真实模块）发来的坐标，就打印出来
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.get_logger().info(f"📥 收到目标点: x={x:.2f}, y={y:.2f}, z={z:.2f}", throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = MockBridgeDry()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()