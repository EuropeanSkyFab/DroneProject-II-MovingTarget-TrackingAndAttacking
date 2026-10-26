import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class MockVisionNode(Node):
    def __init__(self):
        super().__init__('mock_vision_node')
        
        # 创建发布者，发布到视觉话题
        self.publisher_ = self.create_publisher(PoseStamped, '/vision/target_pose', 10)
        
        # 10Hz 频率
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # 假视觉目标坐标 (这里用你之前测试的坐标)
        self.target_x = 3.5
        self.target_y = 2.5
        self.target_z = 5.0

        self.get_logger().info('假视觉节点已启动！准备发送数据...')

    def timer_callback(self):
        msg = PoseStamped()
        
        # 【极其重要】：必须打上时间戳，否则会被中枢的时间安检门拦截
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        
        # 填入坐标
        msg.pose.position.x = self.target_x
        msg.pose.position.y = self.target_y
        msg.pose.position.z = self.target_z
        
        # 姿态朝向（四元数）
        msg.pose.orientation.w = 1.0
        
        self.publisher_.publish(msg)
        
        # 限制打印频率，防止终端刷屏卡顿
        self.get_logger().info(f'发送视觉目标 -> x:{msg.pose.position.x}, y:{msg.pose.position.y}, z:{msg.pose.position.z}', throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = MockVisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()