import rclpy
import time
import copy
import math
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
        
        # 变量
        self.current_mode = "GUIDANCE" # 状态机变量，默认听规划的
        self.last_vision_time = 0.0 # 计时变量
        self.latest_plan_target = None # 判断是否收到规划发出的
        self.latest_vision_target = None # 判断是否收到视觉发出的
        # 用于防止状态机高频切换状态
        self.vision_hit_count = 0  # 连续看到目标的次数
        self.vision_miss_count = 0 # 连续丢失目标的次数
        #  物理安全边界配置（按需修改）
        self.MAX_XY_DIST = 1000.0   # 水平最大允许坐标（米），超过则丢弃
        self.MAX_ALT = 100.0        # 最大相对高度（米），比如上限 100 米
        self.MIN_ALT = -50.0        # 最低相对高度（米，因为向下为负，比如限制最低到 -50 米）

        
        # 20Hz 定时器
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info("中枢节点已启动，等待数据...")

    # 路径规划的回调函数
    def plan_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        
        # 规划数据也可能发疯，同样拦截
        if math.isnan(x) or math.isnan(y) or math.isnan(z) or math.isinf(x) or math.isinf(y) or math.isinf(z):
            self.get_logger().warn("⚠️ 规划数据非法 (NaN 或 Inf)！已丢弃。",throttle_duration_sec=1.0 )
            return
        if abs(x) > self.MAX_XY_DIST or abs(y) > self.MAX_XY_DIST or z > self.MAX_ALT or z < self.MIN_ALT:
            self.get_logger().warn(f"⚠️ 规划目标超出安全围栏！x={x}, y={y}, z={z}。已丢弃。",throttle_duration_sec=1.0 )
            return

        self.latest_plan_target = copy.deepcopy(msg)
        self.get_logger().info(f"收到规划目标: x={x}, y={y}, z={z}",throttle_duration_sec=1.0 )
        
    # 机载视觉的回调函数
    def vision_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        
        # 拦截 NaN 和 Inf
        if math.isnan(x) or math.isnan(y) or math.isnan(z) or math.isinf(x) or math.isinf(y) or math.isinf(z):
            self.get_logger().warn("⚠️ 拦截到非法数据 (NaN 或 Inf)！已丢弃。",throttle_duration_sec=1.0 )
            return
            
        # 拦截超出物理边界的数据（引用在 __init__ 里定义的变量）
        if abs(x) > self.MAX_XY_DIST or abs(y) > self.MAX_XY_DIST or z > self.MAX_ALT or z < self.MIN_ALT:
            self.get_logger().warn(f"⚠️ 目标超出安全围栏！x={x}, y={y}, z={z}。已丢弃。",throttle_duration_sec=1.0 )
            return

        
        self.latest_vision_target = copy.deepcopy(msg)
        self.last_vision_time = time.time() 
        
        self.vision_hit_count += 1
        self.vision_miss_count = 0 # 只要收到数据，丢失计数立刻清零
        
        # 【核心防抖】：必须连续收到 5 帧（约0.5秒），才真正切到 VISION
        if self.current_mode != "VISION" and self.vision_hit_count >= 10:
            self.current_mode = "VISION"
            self.get_logger().info("🚨 连续锁定目标！切换到视觉跟踪模式！")
            
            
    # 状态机代码
    def timer_callback(self):
        
        # 安全检查：如果在 VISION 模式，检查心跳
        if self.current_mode == "VISION":
            if time.time() - self.last_vision_time > 2.0: # 超过2秒没收到新数据
                self.vision_miss_count += 1
                
                # 【核心防抖】：必须连续判定丢失 20 次（定时器是20Hz，即1秒），才真正切回 GUIDANCE
                if self.vision_miss_count >= 10:
                    self.current_mode = "GUIDANCE"
                    self.get_logger().warn("⚠️ 目标彻底丢失，降级切回引导模式！")
                    self.latest_vision_target = None
                    self.vision_miss_count = 0
                    self.vision_hit_count = 0

        
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