import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import Detection2D # 暂时注释掉，防止没装这个包报错
# from px4_msgs.msg import TrajectorySetpoint # 暂时注释掉

class CenterNode(Node):
    def __init__(self):
        super().__init__('center_node')
        
        # 1. 订阅规划目标（ENU）
        self.plan_sub = self.create_subscription(
            PoseStamped, '/planning/target_pose', self.plan_callback, 10)
            
        # 2. 订阅视觉数据
        self.vision_sub = self.create_subscription(
            Detection2D, '/vision/target_pixel', self.vision_callback, 10)
        
        # 3. 发布给飞控
        # 先暂时用 PoseStamped 模拟发布，等飞控同学配好 px4_msgs 再换
        self.fmu_pub = self.create_publisher(PoseStamped, '/fmu/in/trajectory_setpoint', 10)
        
        # 状态机变量
        self.current_mode = "GUIDANCE" # 默认听规划的
        self.latest_plan_target = None
        
        # 20Hz 定时器
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("中枢节点已启动，等待数据...")

    def plan_callback(self, msg):
        self.latest_plan_target = msg
        self.get_logger().info(f"收到规划目标: x={msg.pose.position.x}, y={msg.pose.position.y}, z={msg.pose.position.z}")

    def timer_callback(self):
        if self.current_mode == "GUIDANCE" and self.latest_plan_target is not None:
            # 把规划的目标直接透传给飞控
            out_msg = self.latest_plan_target
            # 强制把 frame_id 改成 world
            out_msg.header.frame_id = "world"
            self.fmu_pub.publish(out_msg)
            
    def vision_callback(self, msg):
        # 先简单打印一下，证明收到了
        x = msg.bbox.center.position.x
        y = msg.bbox.center.position.y
        self.get_logger().info(f"收到视觉目标像素: x={x}, y={y}")

def main(args=None):
    rclpy.init(args=args)
    node = CenterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()