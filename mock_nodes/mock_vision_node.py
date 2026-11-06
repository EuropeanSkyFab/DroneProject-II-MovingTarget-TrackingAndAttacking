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
        
        # ★ 核心修改：给一个默认的视觉目标（即使你没点注入，它也一直发）
        self.latest_msg = PoseStamped()
        self.latest_msg.header.frame_id = "world"
        self.latest_msg.pose.position.x = 15.0
        self.latest_msg.pose.position.y = 15.0
        self.latest_msg.pose.position.z = 10.0
        
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('假视觉已就绪，正在持续发送全局视觉目标 (默认 15, 15, 10)')

    def inject_callback(self, msg):
        # 收到 GUI 发来的新坐标，直接覆盖
        msg.header.frame_id = "world"
        self.latest_msg = msg
        self.get_logger().info(f'👁️ 收到新视觉注入: ({msg.pose.position.x}, {msg.pose.position.y}, {msg.pose.position.z})')

    def timer_callback(self):
        # 10Hz 持续广播，每次都刷新时间戳
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