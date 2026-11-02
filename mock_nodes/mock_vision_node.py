import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class MockVisionNode(Node):
    def __init__(self):
        super().__init__('mock_vision_node')
        
        # 接收来自 GUI 的内部注入数据
        self.sub = self.create_subscription(
            PoseStamped, '~/vision_inject', self.inject_callback, 10)
            
        # 发布到正式的对外话题
        self.pub = self.create_publisher(PoseStamped, '/vision/target_pose', 10)
        self.get_logger().info('假视觉已就绪，等待 GUI 注入数据...')

    def inject_callback(self, msg):
        # 强制刷新时间戳，防止被中枢当作过期数据丢弃
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)
        self.get_logger().info(f'👁️ 转发视觉坐标: ({msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z})')

def main(args=None):
    rclpy.init(args=args)
    node = MockVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()