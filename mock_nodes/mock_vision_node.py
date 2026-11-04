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
        
        # 记忆变量，存储 GUI 最后一次注入的坐标
        self.latest_msg = None
        
        # 10Hz 定时器，持续往外发
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('假视觉已就绪，等待 GUI 注入数据...')

    def inject_callback(self, msg):
        # 收到 GUI 发来的数据时，存起来（这里也可以直接把时间戳刷新）
        msg.header.stamp = self.get_clock().now().to_msg()
        self.latest_msg = msg
        self.get_logger().info(f'👁️ 收到新视觉注入: ({msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z})')

    def timer_callback(self):
        # 如果收到过数据，就以 10Hz 持续往外发
        if self.latest_msg is not None:
            # ★ 必须每次发之前刷新时间戳，否则会被中枢的安检门拦截
            self.latest_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub.publish(self.latest_msg)

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