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
        self.get_logger().info('假规划已就绪，等待 GUI 注入数据...')

    def inject_callback(self, msg):
        # 尝试解析 JSON，并强制刷新 stamp
        try:
            data = json.loads(msg.data)
            data["stamp"] = self.get_clock().now().nanoseconds / 1e9 # 重新打时间戳
            
            out_msg = String()
            out_msg.data = json.dumps(data)
            self.pub.publish(out_msg)
            self.get_logger().info(f'🧠 转发规划坐标: ({data.get("x")}, {data.get("y")}, {data.get("z")})')
        except Exception as e:
            self.get_logger().error(f'解析错误: {e}')

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