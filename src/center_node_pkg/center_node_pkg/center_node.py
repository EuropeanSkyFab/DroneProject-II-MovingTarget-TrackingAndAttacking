import rclpy
import time
import copy
import math
import json
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Vector3   # 云台
from nav_msgs.msg import Odometry # 订阅己方无人机位置
from std_msgs.msg import Empty # 心跳发布
from std_msgs.msg import String
from sensor_msgs.msg import BatteryState
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue



class CenterNode(Node):
    def __init__(self):
        super().__init__('center_node')
        
        # 存放多机集群的列表
        self.swarm_config = ["drone_1"]
        
        # --------------------------- 发布器 --------------------------------
        # 内部发布
        self.heartbeat_pubs = {}    # 给每架无人机发送心跳 
        self.fmu_pubs = {}   # 给每架无人机发送轨迹指令
        self.gimbal_pubs = {}   # 给每架无人机发送云台指令
        self.action_pubs = {}   # 给每架无人机发送动作标签（解除限幅）
        
        # 外部发布
        self.fleet_state_pubs = {}  # 给中控发送每架无人机发送态势信息
        self.battery_pubs = {}  # 给中控发送每架无人机发送电池信息
        self.link_status_pubs = {} # 给中控发送每架无人机发送链路状态信息
        
        # --------------------------- 订阅信息 --------------------------------
        # 1. 订阅规划目标（ENU）
        self.plan_sub = self.create_subscription(
            String, 
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
        
        # 3. 订阅课题三的战术指令
        self.strategy_sub = self.create_subscription(
            String, 
            '/strategy/cmd',
            self.strategy_callback,
            10
            )
        
        # 4. 订阅己方无人机状态
        for drone_id in self.swarm_config:
            topic_name = f'/{drone_id}/state'
            self.create_subscription(
                Odometry,
                topic_name,
                lambda msg, d_id=drone_id: self.drone_state_callback(msg, d_id),
                10
                )
            
        # 5. 发布给飞控（升级为集群命名空间）
        for drone_id in self.swarm_config:
            topic_name = f'/{drone_id}/fmu/in/trajectory_setpoint'
            self.fmu_pubs[drone_id] = self.create_publisher(
                PoseStamped, 
                topic_name,
                10
                )
            
        # 6. 发布给云台（已配置集群命名空间）
        for drone_id in self.swarm_config:
            gimbal_topic = f'/{drone_id}/fmu/in/gimbal_setpoint'
            self.gimbal_pubs[drone_id] = self.create_publisher(
                Vector3, 
                gimbal_topic,
                10
                )
            
        # 7.发布动作标签给桥接（用于解除限幅）
        for drone_id in self.swarm_config:
            self.action_pubs[drone_id] = self.create_publisher(
                String, 
                f'/{drone_id}/action',
                10
                )

        # 8. 发布给课题三的态势信息 (中枢作为安全代理，转发或聚合)
        for drone_id in self.swarm_config:
            # 发布此时无人机的位姿状态
            self.fleet_state_pubs[drone_id] = self.create_publisher(
                Odometry, 
                f'/{drone_id}/state', 
                10
                )
            # 发布此时无人机的电池状态
            self.battery_pubs[drone_id] = self.create_publisher(
                BatteryState, 
                f'/{drone_id}/battery', 
                10
                ) 
            # 发布此时无人机的信号状态
            self.link_status_pubs[drone_id] = self.create_publisher(
                DiagnosticStatus, 
                f'/{drone_id}/link_status', 
                10
                )
            
        # --------------------------- 变量 --------------------------------
        
        # ======================= 状态机变量 ========================
        self.current_mode = "GROUND"    # 初始模式为 GROUND，表示无人机在地面上待命
        # ==========================================================
        
        # 基础变量
        self.active_drone_id = "drone_1" # 默认执行者
        self.last_vision_time = 0.0 # 计时变量
        self.latest_plan_target = None # 判断是否收到规划发出的
        self.latest_vision_target = None # 判断是否收到视觉发出的
        self.last_battery_time = 0.0    # 计时变量，降频发送电池状态
        
        # 初始化多机状态字典
        self.drone_states = {} 
        for drone_id in self.swarm_config:
            self.drone_states[drone_id] = None 
        
        # 用于防止状态机高频切换状态
        self.vision_hit_count = 0  # 连续看到目标的次数
        self.vision_miss_count = 0 # 连续丢失目标的次数
        
        #  物理安全边界配置（按需修改）
        self.MAX_XY_DIST = 1000.0   # 水平最大允许坐标（米），超过则丢弃
        self.MAX_ALT = 100.0        # 最大相对高度（米），比如上限 100 米
        self.MIN_ALT = -50.0        # 最低相对高度（米，因为向下为负，比如限制最低到 -50 米）
        
        # 落地判定阈值（用于 RETURN 后自动回 GROUND）
        self.LAND_ALT_THRESHOLD = 0.3      # 高度低于此值视为落地（米）
        self.LAND_SPEED_THRESHOLD = 0.2    # 速度低于此值视为静止（m/s）
        
        # 追踪模式安全距离：己方与目标水平距离小于此值，强制原地悬停（防止撞上去）
        self.TRACK_SAFE_DIST = 5.0   # 单位：米，按实测调
        
        # 延迟阈值（秒），用于检测数据的时效与可信度（按需修改）
        self.max_vision_delay = 0.2     # 视觉模块数据的时效
        self.max_plan_delay = 0.5       # 规划模块数据的时效
        
        # 检测内存的老数据，以便后续清出
        self.last_plan_time = 0.0 # 记录最后一次收到规划数据的时间
        # ----------------------------------------------------------------


        # 创建心跳发布器，给每架无人机发送心跳
        for drone_id in self.swarm_config:
            self.heartbeat_pubs[drone_id] = self.create_publisher(
                Empty, 
                f'/{drone_id}/heartbeat', 
                10
                )
        # 启动定时器（20Hz）    
        self.timer = self.create_timer(0.05, self.timer_callback)    
        self.get_logger().info("中枢节点已启动，等待数据...")
        

    # -----------------------------------辅助函数------------------------------------
    # 根据当前中控指定的无人机，发送单机目标点
    def send_single_target(self, target_msg):
        if target_msg is None:
            return
            
        # 直接拿中控当前指定的飞机 ID
        drone_id = self.active_drone_id
        
        # 安全检查：确保这个 ID 在我们的列表里
        if drone_id not in self.fmu_pubs:
            return
            
        out_msg = copy.deepcopy(target_msg)
        out_msg.header.frame_id = "world"
        
        # 去掉所有偏移量计算，指哪打哪
        self.fmu_pubs[drone_id].publish(out_msg)
        self.get_logger().info(f"🔥🔥🔥 {drone_id} 数据已发出！", throttle_duration_sec=1.0)


    # 计算云台角度以控制云台转动
    def publish_gimbal_target(self, target_msg):
        if target_msg is None:
            return
            
        drone_id = self.active_drone_id
        if drone_id not in self.swarm_config:
            return
            
        my_state = self.drone_states.get(drone_id)
        if my_state is None:
            return # 如果还没收到这架飞机的位置，直接跳过
            
        # 获取目标和自己的坐标
        target_x = target_msg.pose.position.x
        target_y = target_msg.pose.position.y
        drone_x = my_state.pose.pose.position.x
        drone_y = my_state.pose.pose.position.y
        
        # 算相对偏移和云台偏航角
        delta_x = target_x - drone_x
        delta_y = target_y - drone_y
        target_yaw_rad = math.atan2(delta_x, delta_y) 
        
        # 组装指令并精准投递
        gimbal_cmd = Vector3()
        gimbal_cmd.x = -0.2 # Pitch (俯仰)
        gimbal_cmd.y = target_yaw_rad # Yaw (偏航)
        gimbal_cmd.z = 0.0
        
        self.gimbal_pubs[drone_id].publish(gimbal_cmd)
            
            
    # 返回当前的无人机状态
    def drone_state_callback(self, msg, drone_id):
        self.drone_states[drone_id] = copy.deepcopy(msg)
        # 【新增】：记录收到这架飞机状态的时间，用于算延迟
        setattr(self, f'last_state_time_{drone_id}', time.time()) 

    # 路径规划的回调函数
    def plan_callback(self, msg):
        try:
            data = json.loads(msg.data)
            msg_time = data.get("stamp", 0.0)
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            # 时间延迟过滤
            if current_time - msg_time > self.max_plan_delay:
                self.get_logger().warn("⚠️ [规划] 数据延迟过高，已丢弃！", throttle_duration_sec=1.0)
                return
                
            x = float(data.get("x", 0.0))
            y = float(data.get("y", 0.0))
            z = float(data.get("z", 0.0))
            
            # 物理边界过滤
            if math.isnan(x) or math.isnan(y) or math.isnan(z) or math.isinf(x) or math.isinf(y) or math.isinf(z):
                self.get_logger().warn("⚠️ 规划数据非法 (NaN 或 Inf)！已丢弃。", throttle_duration_sec=1.0)
                return
            if abs(x) > self.MAX_XY_DIST or abs(y) > self.MAX_XY_DIST or z > self.MAX_ALT or z < self.MIN_ALT:
                self.get_logger().warn(f"⚠️ 规划目标超出安全围栏！已丢弃。", throttle_duration_sec=1.0)
                return

            # 重新打包成 PoseStamped 存起来
            target = PoseStamped()
            target.header.stamp = self.get_clock().now().to_msg()
            target.header.frame_id = "world"
            target.pose.position.x = x
            target.pose.position.y = y
            target.pose.position.z = z
            
            self.latest_plan_target = target
            self.last_plan_time = time.time()
            self.get_logger().info(f"收到规划 JSON: x={x}, y={y}, z={z}", throttle_duration_sec=1.0)
            
        except Exception as e:
            self.get_logger().error(f"解析规划 JSON 失败: {e}")
        
        
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


    #  中控（课题三）回调函数 
    def strategy_callback(self, msg):
        try:
            data = json.loads(msg.data)
            new_drone_id = data.get("drone_id", "drone_1")
            action = data.get("action", "HOVER")
            
            # 【安全拦截】：在地上时，只允许起飞指令！
            if self.current_mode == "GROUND":
                if action != "TAKEOFF":
                    self.get_logger().warn(f"⚠️ 拒绝执行 {action}：无人机在地上，请先发送 TAKEOFF！")
                    return
            
            # 如果通过了安检，更新状态
            self.active_drone_id = new_drone_id
            self.current_mode = action 
            
            # 给底层桥接发动作标签
            if self.active_drone_id in self.action_pubs:
                act_msg = String()
                act_msg.data = action
                self.action_pubs[self.active_drone_id].publish(act_msg)
                
            self.get_logger().warn(f"🚨 收到指令：{self.active_drone_id} 执行 {action}！")

        except Exception as e:
            self.get_logger().error(f"解析中控指令失败: {e}")
    # ------------------------------------------------------------------------------
            
            
    # ------------------------------ 状态机代码 -----------------------------------
    def timer_callback(self):
        
        # 遍历列表，给所有在线飞机发心跳
        for drone_id in self.heartbeat_pubs:
            self.heartbeat_pubs[drone_id].publish(Empty())
        
        # =================【全局数据保鲜期检查】=================
        # 无论当前是什么模式，只要数据断供超时，必须强制清空（防幽灵变量）
        # 视觉数据过期检查
        if time.time() - self.last_vision_time > 2.0:
            self.latest_vision_target = None
            
        # 规划数据过期检查 (解决飞向幽灵坐标的核心补丁！)
        if time.time() - self.last_plan_time > 2.0:
            self.latest_plan_target = None
        # ==========================================================
        
        # ========================= 单层扁平化状态机分发 ===========================
        # 地面待命：什么都不发
        if self.current_mode == "GROUND":
            pass
        
        # 起飞：直接发一个起飞点
        elif self.current_mode == "TAKEOFF":
            # 直接发一个起飞点，起飞点为 (0, 0, 10)
            target = PoseStamped()
            target.header.stamp = self.get_clock().now().to_msg()
            target.header.frame_id = "world"
            target.pose.position.x = 0.0
            target.pose.position.y = 0.0
            target.pose.position.z = 10.0 # 飞到 10 米高
            self.send_single_target(target)

        # 悬停：读取当前位置，原地发给自己
        elif self.current_mode == "HOVER":
            my_state = self.drone_states.get(self.active_drone_id)
            if my_state is not None:
                target = PoseStamped()
                target.header.stamp = self.get_clock().now().to_msg()
                target.header.frame_id = "world"
                # 直接拿当前位置
                target.pose.position.x = my_state.pose.pose.position.x
                target.pose.position.y = my_state.pose.pose.position.y
                target.pose.position.z = my_state.pose.pose.position.z
                self.send_single_target(target)

        # 返航：给个原点坐标
        elif self.current_mode == "RETURN":
            target = PoseStamped()
            target.header.stamp = self.get_clock().now().to_msg()
            target.header.frame_id = "world"
            target.pose.position.x = 0.0
            target.pose.position.y = 0.0
            target.pose.position.z = 0.0
            self.send_single_target(target)
            
            # 落地检测：高度和速度都接近 0，判定已落地，回 GROUND 待命
            my_state = self.drone_states.get(self.active_drone_id)
            if my_state is not None:
                alt = my_state.pose.pose.position.z
                vx = my_state.twist.twist.linear.x
                vy = my_state.twist.twist.linear.y
                vz = my_state.twist.twist.linear.z
                speed = math.sqrt(vx*vx + vy*vy + vz*vz)
                if alt < self.LAND_ALT_THRESHOLD and speed < self.LAND_SPEED_THRESHOLD:
                    self.current_mode = "GROUND"
                    self.get_logger().info(f"🛬 {self.active_drone_id} 已落地，状态机回到 GROUND")

        # 追踪模式：视觉优先（锁定），规划兜底
        elif self.current_mode == "TRACK":
            target = self.latest_vision_target if self.latest_vision_target is not None else self.latest_plan_target
            if target is not None:
                # 算己方与目标的水平距离
                my_state = self.drone_states.get(self.active_drone_id)
                near = False
                if my_state is not None:
                    dx = target.pose.position.x - my_state.pose.pose.position.x
                    dy = target.pose.position.y - my_state.pose.pose.position.y
                    horiz_dist = math.sqrt(dx * dx + dy * dy)
                    if horiz_dist < self.TRACK_SAFE_DIST:
                        near = True

                if near and my_state is not None:
                    # 太近了：发当前坐标，原地悬停，不切状态机
                    hold = PoseStamped()
                    hold.header.stamp = self.get_clock().now().to_msg()
                    hold.header.frame_id = "world"
                    hold.pose.position.x = my_state.pose.pose.position.x
                    hold.pose.position.y = my_state.pose.pose.position.y
                    hold.pose.position.z = my_state.pose.pose.position.z
                    self.send_single_target(hold)
                    self.get_logger().warn(
                        f"⚠️ 距目标 {horiz_dist:.1f}m < {self.TRACK_SAFE_DIST}m，强制悬停防撞机",
                        throttle_duration_sec=1.0
                    )
                else:
                    # 正常追踪
                    self.send_single_target(target)
                    if self.latest_vision_target is not None:
                        self.publish_gimbal_target(target)

        # 拦截/撞击模式：抛弃视觉，只听规划的预测拦截点
        elif self.current_mode in ["INTERCEPT", "ATTACK"]:
            if self.latest_plan_target is not None:
                self.send_single_target(self.latest_plan_target)
        
        # =============================================================        
                
        # =================【转发给课题三的态势数据】=================
        for drone_id in self.swarm_config:
            # 1. 转发 Odometry 状态（如果中枢收到了底层桥接发来的状态）
            my_state = self.drone_states.get(drone_id)
            if my_state is not None:
                # 顺手转发给课题三
                self.fleet_state_pubs[drone_id].publish(my_state)
                
                # 2. 顺便生成链路健康度 DiagnosticStatus
                # （因为你是中枢，你收到 state 的速度就代表了链路健康）
                status = DiagnosticStatus()
                status.hardware_id = drone_id
                
                # 假设 0.1 秒以内收到算良好，0.5秒算微弱
                delay = time.time() - getattr(self, f'last_state_time_{drone_id}', 0.0)
                
                if delay < 0.2:
                    status.level = DiagnosticStatus.OK
                    status.name = "Link Status: GOOD"
                elif delay < 0.8:
                    status.level = DiagnosticStatus.WARN
                    status.name = "Link Status: WEAK"
                else:
                    status.level = DiagnosticStatus.ERROR
                    status.name = "Link Status: LOST"
                    
                kv_delay = KeyValue(key="delay_ms", value=str(round(delay * 1000, 1)))
                status.values = [kv_delay]
                self.link_status_pubs[drone_id].publish(status)
                
        # 降频转发：电量数据（每 2 秒发一次）
        if time.time() - getattr(self, 'last_battery_time', 0.0) > 2.0:
            self.last_battery_time = time.time()
            
            batt_msg = BatteryState()
            batt_msg.header.stamp = self.get_clock().now().to_msg()
            batt_msg.percentage = 1.0  # 仿真先写死 100%
            batt_msg.voltage = 24.0
            batt_msg.power_supply_status = BatteryState.POWER_SUPPLY_STATUS_DISCHARGING
            
            for drone_id in self.swarm_config:
                self.battery_pubs[drone_id].publish(batt_msg)
                
    # ------------------------------------------------------------------------------------


def main(args=None):
    rclpy.init(args=args)
    node = CenterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()