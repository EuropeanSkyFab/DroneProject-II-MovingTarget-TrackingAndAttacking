import rclpy
import json
import math
import time
from rclpy.node import Node
from std_msgs.msg import String

class MockPlanNode(Node):
    def __init__(self):
        super().__init__('mock_plan_node')
        # 发布 String 类型的 JSON 数据
        self.publisher_ = self.create_publisher(String, '/planning/target_pose', 10)
        self.timer = self.create_timer(0.1, self.timer_callback) # 10Hz
        self.t = 0.0
        self.get_logger().info('假规划节点已启动，准备发送 JSON 坐标...')

    def timer_callback(self):
        self.t += 0.1
        # 模拟目标在 15 米高度，半径 10 米的圆上移动
        x = 10.0 * math.cos(self.t * 0.5) 
        y = 10.0 * math.sin(self.t * 0.5)
        z = 15.0 
        
        data = {
            "stamp": time.time(), # 极其重要：带时间戳防止被中枢拦截
            "x": x,
            "y": y,
            "z": z
        }
        
        msg = String()
        msg.data = json.dumps(data)
        self.publisher_.publish(msg)
        
        # 降低打印频率，防刷屏
        self.get_logger().info(f'发送假规划: x={x:.2f}, y={y:.2f}, z={z}', throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = MockPlanNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()