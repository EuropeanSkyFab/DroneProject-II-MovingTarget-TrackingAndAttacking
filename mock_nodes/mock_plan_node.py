import rclpy
import json
from rclpy.node import Node
from std_msgs.msg import String

class MockPlanNode(Node):
    def __init__(self):
        super().__init__('mock_plan_node')
        
        self.sub = self.create_subscription(
            String, '~/planning_inject', self.inject_callback, 10)
            
        self.pub = self.create_publisher(String, '/planning/target_pose', 10)
        
        # ★ 核心修改：使用字典记住每架飞机的最新目标
        self.drone_targets = {} 
        
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('假规划已就绪，等待 GUI 注入数据...')

    def inject_callback(self, msg):
        try:
            data = json.loads(msg.data)
            drone_id = data.get("drone_id", "drone_1")
            
            # 只更新被注入的那架飞机的数据，不影响别人
            self.drone_targets[drone_id] = data
            self.get_logger().info(f'收到新注入规划点 -> 目标: {drone_id}')
        except Exception as e:
            self.get_logger().error(f'解析错误: {e}')
            
    def timer_callback(self):
        # 以 10Hz 频率，把字典里所有飞机的目标都广播出去
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        for drone_id, data in self.drone_targets.items():
            # 强制刷新时间戳，防止被中枢保鲜期清空
            data["stamp"] = current_time
            
            out_msg = String()
            out_msg.data = json.dumps(data)
            self.pub.publish(out_msg)
            
        if len(self.drone_targets) > 0:
            # 日志降频，防止刷屏
            self.get_logger().info(f'正在广播 {list(self.drone_targets.keys())} 的规划坐标', throttle_duration_sec=2.0)

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