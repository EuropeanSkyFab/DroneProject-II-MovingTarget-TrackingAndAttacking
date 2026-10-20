import rclpy
import time
import copy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
# from px4_msgs.msg import TrajectorySetpoint # 暂时注释掉

class CenterNode(Node):
    def __init__(self):
        super().__init__('center_node')
        
        # 1. 订阅规划目标（ENU）
        self.plan_sub = self.create_subscription(
            PoseStamped, 
            '/planning/target_pose', 
            self.plan_callback, 
            10
            )
            
        # 2. 订阅视觉数据
        self.vision_sub = self.create_subscription(
            PoseStamped, 
            '/vision/target_pose', 
            self.vision_callback, 
            10
            )
        
        # 3. 发布给飞控
        # 先暂时用 PoseStamped 模拟发布，等飞控同学配好 px4_msgs 再换
        self.fmu_pub = self.create_publisher(
            PoseStamped, 
            '/fmu/in/trajectory_setpoint', 
            10
            )
        
        
        
        self.current_mode = "GUIDANCE" # 状态机变量，默认听规划的
        self.last_vision_time = 0.0 #计时变量
        self.latest_plan_target = None
        self.latest_vision_target = None
        
        
        # 20Hz 定时器
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("中枢节点已启动，等待数据...")

    # 路径规划的回调函数
    def plan_callback(self, msg):
        self.latest_plan_target = copy.deepcopy(msg)
        self.get_logger().info(f"收到规划目标: x={msg.pose.position.x}, y={msg.pose.position.y}, z={msg.pose.position.z}")
        
    # 机载视觉的回调函数
    def vision_callback(self, msg):
        # 因为现在 msg 直接就是 PoseStamped，里面包含 x,y,z
        self.latest_vision_target = copy.deepcopy(msg)
        self.last_vision_time = time.time() 
        
        if self.current_mode != "VISION":
            self.current_mode = "VISION"
            self.get_logger().info("🚨 发现目标！切换到视觉跟踪模式！")
            
            
    # 状态机代码
    def timer_callback(self):
        
        # 安全检查：如果在 VISION 模式下，超过 2 秒没收到视觉数据
        if self.current_mode == "VISION":
            if time.time() - self.last_vision_time > 2.0:
                self.current_mode = "GUIDANCE"
                self.get_logger().warn("⚠️ 目标丢失超时！降级切回引导模式！")
                self.latest_vision_target = None

        
        if self.current_mode == "GUIDANCE" and self.latest_plan_target is not None:
            # 把规划的目标透传给飞控
            out_msg = copy.deepcopy(self.latest_plan_target)
            out_msg.header.frame_id = "world"
            self.fmu_pub.publish(out_msg)
            
        elif self.current_mode == "VISION" and self.latest_vision_target is not None:
            # 直接透传视觉发来的三维坐标
            out_msg = copy.deepcopy(self.latest_vision_target)
            out_msg.header.frame_id = "world"
            self.fmu_pub.publish(out_msg)
            


def main(args=None):
    rclpy.init(args=args)
    node = CenterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()