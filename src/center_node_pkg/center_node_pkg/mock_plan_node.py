import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class FakePlanningPublisher(Node):
    def __init__(self):
        super().__init__('fake_planning_pub')
        # 创建发布者，话题名必须对应中枢订阅的规划话题
        self.publisher_ = self.create_publisher(PoseStamped, '/planning/target_pose', 10)
        
        # 设置 10Hz 的频率发送
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # 你可以在这里修改你想要的假目标坐标
        self.target_x = 10.0
        self.target_y = 20.0
        self.target_z = 5.0

        self.get_logger().info('假规划数据发布器已启动！')

    def timer_callback(self):
        msg = PoseStamped()
        
        # 【关键】：获取当前仿真时间并填入 header
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        
        # 填入坐标
        msg.pose.position.x = self.target_x
        msg.pose.position.y = self.target_y
        msg.pose.position.z = self.target_z
        
        # 姿态（四元数默认单位四元数即可）
        msg.pose.orientation.w = 1.0
        
        # 发布
        self.publisher_.publish(msg)
        
        # 打印日志，方便你看它有没有在发
        self.get_logger().info(f'发送规划目标 -> x:{msg.pose.position.x}, y:{msg.pose.position.y}, z:{msg.pose.position.z}', throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = FakePlanningPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()