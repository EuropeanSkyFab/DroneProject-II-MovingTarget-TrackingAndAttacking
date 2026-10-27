import rclpy
import time
import copy
import math
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Vector3   # 云台
from nav_msgs.msg import Odometry # 订阅己方无人机位置
from std_msgs.msg import Empty # 心跳发布
# from px4_msgs.msg import TrajectorySetpoint  （暂时注释掉）

class CenterNode(Node):
    def __init__(self):
        super().__init__('center_node')
        
        # 存放多机集群的字典
        self.swarm_config = {
            'drone_1': {'offset_x': 0.0, 'offset_y': 0.0, 'offset_z': 0.0}, # 'offset'指集群阵型中，该无人机相对目标中心点的空间偏移量
            # 以后要加飞机，直接在这里加 'drone_2': {...}
        }
        
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
        
        # 3. 订阅己方无人机状态
        for drone_id in self.swarm_config.keys():
            topic_name = f'/{drone_id}/state'
            self.create_subscription(
                Odometry,
                topic_name,
                lambda msg, d_id=drone_id: self.drone_state_callback(msg, d_id),
                10
            )
            
            
        # 4. 发布给飞控（升级为集群命名空间）
        self.fmu_pubs = {}
        for drone_id in self.swarm_config.keys():
            topic_name = f'/{drone_id}/fmu/in/trajectory_setpoint'
            self.fmu_pubs[drone_id] = self.create_publisher(
                PoseStamped, 
                topic_name,
                10
                )
            
        # 5. 发布给云台（已配置集群命名空间）
        self.gimbal_pubs = {}
        for drone_id in self.swarm_config.keys():
            gimbal_topic = f'/{drone_id}/fmu/in/gimbal_setpoint'
            self.gimbal_pubs[drone_id] = self.create_publisher(
                Vector3, 
                gimbal_topic,
                10
            )
        # --------------------------- 变量 --------------------------------
        
        # 基础变量
        self.current_mode = "GUIDANCE" # 状态机变量，默认听规划的
        self.last_vision_time = 0.0 # 计时变量
        self.latest_plan_target = None # 判断是否收到规划发出的
        self.latest_vision_target = None # 判断是否收到视觉发出的
        
        # 初始化多机状态字典
        self.drone_states = {} 
        for drone_id in self.swarm_config.keys():
            self.drone_states[drone_id] = None 
        
        # 用于防止状态机高频切换状态
        self.vision_hit_count = 0  # 连续看到目标的次数
        self.vision_miss_count = 0 # 连续丢失目标的次数
        
        #  物理安全边界配置（按需修改）
        self.MAX_XY_DIST = 1000.0   # 水平最大允许坐标（米），超过则丢弃
        self.MAX_ALT = 100.0        # 最大相对高度（米），比如上限 100 米
        self.MIN_ALT = -50.0        # 最低相对高度（米，因为向下为负，比如限制最低到 -50 米）
        
        # 延迟阈值（秒），用于检测数据的时效与可信度（按需修改）
        self.max_vision_delay = 0.2 # 视觉模块数据的时效
        self.max_plan_delay = 0.5 # 规划模块数据的时效
        # 检测内存的老数据，以便后续清出
        self.last_plan_time = 0.0 # 记录最后一次收到规划数据的时间
        # ----------------------------------------------------------------


        # 20Hz 定时器
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.heartbeat_pub = self.create_publisher(Empty, '/drone_1/heartbeat', 10) # 心跳话题发送
        self.get_logger().info("中枢节点已启动，等待数据...")
        

    # -----------------------------------辅助函数------------------------------------
    # 根据基础目标点，分发带有阵型偏移的指令给所有无人机（用于下面关于视觉与规划的回调函数）
    def publish_swarm_target(self, base_msg):
        if base_msg is None:
            return

        base_x = base_msg.pose.position.x
        base_y = base_msg.pose.position.y
        base_z = base_msg.pose.position.z

        for drone_id, config in self.swarm_config.items():
            out_msg = copy.deepcopy(base_msg)
            out_msg.header.frame_id = "world"
            
            # 加上物理偏移量
            out_msg.pose.position.x = base_x + config['offset_x']
            out_msg.pose.position.y = base_y + config['offset_y']
            out_msg.pose.position.z = base_z + config['offset_z']
            
            # 从字典取出对应飞机的发布者并发送
            self.fmu_pubs[drone_id].publish(out_msg)
            self.get_logger().info("🔥🔥🔥 数据已发出！", throttle_duration_sec=1.0)

    # 计算云台角度以控制云台转动
    def publish_gimbal_target(self, target_msg):
        if target_msg is None:
            return
            
        target_x = target_msg.pose.position.x
        target_y = target_msg.pose.position.y
        
        
        # 遍历集群中的每一架飞机
        for drone_id in self.swarm_config.keys():
            # 获取该飞机的真实位置
            my_state = self.drone_states.get(drone_id)
            if my_state is None:
                continue # 如果还没收到这架飞机的位置，先跳过它
                
            drone_x = my_state.pose.pose.position.x
            drone_y = my_state.pose.pose.position.y
            
            # 算这架飞机到目标的相对偏移
            delta_x = target_x - drone_x
            delta_y = target_y - drone_y
            
            # 算云台偏航角
            target_yaw_rad = math.atan2(delta_x, delta_y) 
            
            # 组装指令
            gimbal_cmd = Vector3()
            gimbal_cmd.x = -0.2 # Pitch (俯仰)
            gimbal_cmd.y = target_yaw_rad # Yaw (偏航)
            gimbal_cmd.z = 0.0
            
            # 精准投递给对应的飞机
            self.gimbal_pubs[drone_id].publish(gimbal_cmd)
            
    # 
    def drone_state_callback(self, msg, drone_id):
        self.drone_states[drone_id] = copy.deepcopy(msg)
    # -------------------------------- 对应模块的回调函数 -----------------------------------

    # 路径规划的回调函数
    def plan_callback(self, msg):
        
         # 时间延迟过滤检测 
        current_time = self.get_clock().now()
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        delay = (current_time - msg_time).nanoseconds / 1e9
        if delay > self.max_plan_delay:
            self.get_logger().warn(f"⚠️ [规划] 数据延迟过高 ({delay:.3f}s)，已丢弃！", throttle_duration_sec=1.0)
            return
        
        # 物理边界检查
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
        self.last_plan_time = time.time() # 记录最新时间
        self.get_logger().info(f"收到规划目标(!): x={x}, y={y}, z={z}",throttle_duration_sec=1.0 )
        
        
        
    # 机载视觉的回调函数
    def vision_callback(self, msg):
        
        # 时间延迟过滤检测
        current_time = self.get_clock().now()
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)
        delay = (current_time - msg_time).nanoseconds / 1e9
        if delay > self.max_vision_delay:
            self.get_logger().warn(f"⚠️ [视觉] 数据延迟过高 ({delay:.3f}s)，已丢弃！", throttle_duration_sec=1.0)
            return
        
        # 物理边界检查
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

    # ------------------------------------------------------------------------------
            
            
    # ------------------------------ 状态机代码 -----------------------------------
    def timer_callback(self):
        
        # 只要中枢还活着，每次循环（20Hz）都发一次心跳
        self.heartbeat_pub.publish(Empty())
        
        # =================【全局数据保鲜期检查】=================
        # 无论当前是什么模式，只要数据断供超时，必须强制清空（防幽灵变量）
        
        # 1. 视觉数据过期检查
        if time.time() - self.last_vision_time > 2.0:
            self.latest_vision_target = None
            
        # 2. 规划数据过期检查 (解决飞向幽灵坐标的核心补丁！)
        if time.time() - self.last_plan_time > 2.0:
            self.latest_plan_target = None
        # ==========================================================
        
        # 【探针 1】打印当前模式，看看它到底在想什么
        #self.get_logger().info(f"当前模式: {self.current_mode}, 规划数据是否为空: {self.latest_plan_target is None}", throttle_duration_sec=2.0)

        # 安全检查：如果在 VISION 模式，检查心跳
        if self.current_mode == "VISION":
            if self.latest_vision_target is None: 
                self.vision_miss_count += 1
                if self.vision_miss_count >= 10:
                    self.current_mode = "GUIDANCE"
                    self.get_logger().warn("⚠️ 目标彻底丢失，降级切回引导模式！")
                    self.vision_miss_count = 0
                    self.vision_hit_count = 0

        
        if self.current_mode == "GUIDANCE" and self.latest_plan_target is not None:
            # 【探针 2】看看有没有进入发送判断
            #self.get_logger().info("✅ 满足 GUIDANCE 发送条件，准备调用发送函数", throttle_duration_sec=2.0)
            self.publish_swarm_target(self.latest_plan_target)
            
        elif self.current_mode == "VISION" and self.latest_vision_target is not None:
            self.publish_swarm_target(self.latest_vision_target)
            self.publish_gimbal_target(self.latest_vision_target) # 发送云台指令
        # 【探针 3】如果没进上面的条件，看看是卡在哪了    
        #else:
            #self.get_logger().warn(f"❌ 未进入发送条件！模式:{self.current_mode}, plan为None? {self.latest_plan_target is None}", throttle_duration_sec=2.0)
            
    # ------------------------------------------------------------------------------------


def main(args=None):
    rclpy.init(args=args)
    node = CenterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()