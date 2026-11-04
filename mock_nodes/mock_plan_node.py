import rclpy
import json
from rclpy.node import Node
from std_msgs.msg import String

class MockPlanNode(Node):
    def __init__(self):
        super().__init__('mock_plan_node')
        
        # 接收来自 GUI 的内部注入数据
        self.sub = self.create_subscription(
            String, '~/planning_inject', self.inject_callback, 10)
            
        # 发布到正式的对外话题
        self.pub = self.create_publisher(String, '/planning/target_pose', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('假规划已就绪，等待 GUI 注入数据...')

    def inject_callback(self, msg):
        # 收到 GUI 发来的指令时，保存下来，并打上最新时间戳
        try:
            self.data = json.loads(msg.data)
            self.get_logger().info(f'收到新注入规划点...')
        except Exception as e:
            self.get_logger().error(f'解析错误: {e}')
            
    def timer_callback(self):
        if hasattr(self, 'data') and self.data is not None:
            self.data["stamp"] = self.get_clock().now().nanoseconds / 1e9
            out_msg = String()
            out_msg.data = json.dumps(self.data)
            self.pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = MockPlanNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()