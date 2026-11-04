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
        self.swarm_config = ["drone_1","drone_2","drone_3"]  # 可根据实际情况修改无人机 ID 列表
        
        # --------------------------- 发布器 --------------------------------
        # 内部发布
        self.heartbeat_pubs = {}    # 给每架无人机发送心跳 
        self.fmu_pubs = {}   # 给每架无人机发送轨迹指令
        self.gimbal_pubs = {}   # 给每架无人机发送云台指令
        self.action_pubs = {}   # 给每架无人机发送动作标签（解除限幅）
        self.drone_modes = {}   # 记录每架飞机的状态
        for drone_id in self.swarm_config:
            self.drone_modes[drone_id] = "GROUND"
        
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
    def send_single_target(self, drone_id, target_msg):
        if target_msg is None or drone_id not in self.fmu_pubs:
            return
        out_msg = copy.deepcopy(target_msg)
        out_msg.header.frame_id = "world"
        self.fmu_pubs[drone_id].publish(out_msg)


    # 计算云台角度以控制云台转动
    def publish_gimbal_target(self, drone_id, target_msg, my_state):
        if target_msg is None or drone_id not in self.gimbal_pubs:
            return
            
        target_x = target_msg.pose.position.x
        target_y = target_msg.pose.position.y
        drone_x = my_state.pose.pose.position.x
        drone_y = my_state.pose.pose.position.y
        
        delta_x = target_x - drone_x
        delta_y = target_y - drone_y
        target_yaw_rad = math.atan2(delta_x, delta_y) 
        
        gimbal_cmd = Vector3()
        gimbal_cmd.x = -0.2
        gimbal_cmd.y = target_yaw_rad
        gimbal_cmd.z = 0.0
        
        self.gimbal_pubs[drone_id].publish(gimbal_cmd)
            
            
    # 返回当前的无人机状态
    def drone_state_callback(self, msg, drone_id):
        self.drone_states[drone_id] = copy.deepcopy(msg)
        # 【新增】：记录收到这架飞机状态的时间，用于算延迟
        setattr(self, f'last_state_time_{drone_id}', time.time()) 
        
        
# 基于优先级的动态势场法（切向）
    def apply_collision_avoidance(self, drone_id, my_state, target, my_mode):
        my_index = self.swarm_config.index(drone_id)
        
        for other_id in self.swarm_config:
            if other_id == drone_id: 
                continue
            
            other_state = self.drone_states.get(other_id)
            other_mode = self.drone_modes.get(other_id, "GROUND")
            
            if other_state is not None and other_mode != "GROUND":
                odx = my_state.pose.pose.position.x - other_state.pose.pose.position.x
                ody = my_state.pose.pose.position.y - other_state.pose.pose.position.y
                odz = my_state.pose.pose.position.z - other_state.pose.pose.position.z
                
                # 必须是三维总距离，否则上下重叠时会炸机
                dist = math.sqrt(odx**2 + ody**2 + odz**2)
                
                # 安全距离定为 4 米
                if dist < 4.0 and dist > 0.01:
                    
                    if dist < 2.0:
                        push_force = 3.0  # 紧急硬核推力，强制弹开
                    else:
                        push_force = (4.0 - dist) * 1.5 # 预警圈内的渐进推力
                        
                    # 1. 计算两机连线在水平面的单位向量
                    horiz_dist = math.sqrt(odx**2 + ody**2)
                    if horiz_dist < 0.01:
                        # 极端情况：如果两架飞机在水平面上完全垂直重叠了，
                        # 给一个默认的 X 方向，强行把它们推开
                        ux = 1.0
                        uy = 0.0
                    else:
                        ux = odx / horiz_dist
                        uy = ody / horiz_dist
                    
                    # 2. 计算垂直向量（切向）
                    tangent_x = -uy
                    tangent_y = ux
                    
                    # 3. 决定是往左滑还是往右滑（优先级判定）
                    is_higher_priority = False
                    if my_mode == "ATTACK" and other_mode != "ATTACK":
                        is_higher_priority = True
                    elif my_mode != "ATTACK" and other_mode == "ATTACK":
                        is_higher_priority = False
                    elif my_mode == "ATTACK" and other_mode == "ATTACK":
                        is_higher_priority = my_index < self.swarm_config.index(other_id)
                    else:
                        is_higher_priority = my_index < self.swarm_config.index(other_id)

                    # 如果我是低权限，切向矢量翻转（往另一边躲）
                    if not is_higher_priority:
                        tangent_x = -tangent_x
                        tangent_y = -tangent_y

                    # 4. 叠加到目标点（侧向滑步）
                    target.pose.position.x += tangent_x * push_force
                    target.pose.position.y += tangent_y * push_force
                    
                    self.get_logger().warn(f"{drone_id} 切向侧滑避让 {other_id}", throttle_duration_sec=1.0)


    # 路径规划的回调函数
    def plan_callback(self, msg):
        try:
            data = json.loads(msg.data)
            msg_time = data.get("stamp", 0.0)
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            # 时间延迟过滤
            if current_time - msg_time > self.max_plan_delay:
                self.get_logger().warn("[规划] 数据延迟过高，已丢弃！", throttle_duration_sec=1.0)
                return
                
            x = float(data.get("x", 0.0))
            y = float(data.get("y", 0.0))
            z = float(data.get("z", 0.0))
            
            # 物理边界过滤
            if math.isnan(x) or math.isnan(y) or math.isnan(z) or math.isinf(x) or math.isinf(y) or math.isinf(z):
                self.get_logger().warn("规划数据非法 (NaN 或 Inf)！已丢弃。", throttle_duration_sec=1.0)
                return
            if abs(x) > self.MAX_XY_DIST or abs(y) > self.MAX_XY_DIST or z > self.MAX_ALT or z < self.MIN_ALT:
                self.get_logger().warn(f"规划目标超出安全围栏！已丢弃。", throttle_duration_sec=1.0)
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
            self.get_logger().warn(f"[视觉] 数据延迟过高 ({delay:.3f}s)，已丢弃！", throttle_duration_sec=1.0)
            return
        
        # 物理边界检查
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        
        # 拦截 NaN 和 Inf
        if math.isnan(x) or math.isnan(y) or math.isnan(z) or math.isinf(x) or math.isinf(y) or math.isinf(z):
            self.get_logger().warn("拦截到非法数据 (NaN 或 Inf)！已丢弃。",throttle_duration_sec=1.0 )
            return
            
        # 拦截超出物理边界的数据（引用在 __init__ 里定义的变量）
        if abs(x) > self.MAX_XY_DIST or abs(y) > self.MAX_XY_DIST or z > self.MAX_ALT or z < self.MIN_ALT:
            self.get_logger().warn(f"目标超出安全围栏！x={x}, y={y}, z={z}。已丢弃。",throttle_duration_sec=1.0 )
            return

        
        self.latest_vision_target = copy.deepcopy(msg)
        self.last_vision_time = time.time() 

 
    #  中控（课题三）回调函数 
    def strategy_callback(self, msg):
        try:
            data = json.loads(msg.data)
            new_drone_id = data.get("drone_id", "drone_1")
            action = data.get("action", "HOVER")
            
            # 1. 安全拦截：校验无人机 ID 是否存在
            if new_drone_id not in self.swarm_config:
                self.get_logger().warn(f"拒绝执行：系统中不存在 {new_drone_id}！")
                return
                
            # 2. 获取这架无人机当前的状态
            current_drone_mode = self.drone_modes.get(new_drone_id, "GROUND")

            # 3. 【状态安检门】基于该无人机的当前状态进行拦截
            if current_drone_mode == "GROUND":
                if action != "TAKEOFF":
                    self.get_logger().warn(f"拒绝执行 {action}：{new_drone_id} 在地上，请先发送 TAKEOFF！")
                    return
            else:
                if action == "TAKEOFF":
                    self.get_logger().warn(f"拒绝执行 TAKEOFF：{new_drone_id} 已在空中，请勿重复起飞！")
                    return
            
            # 4. 通过安检，更新该无人机的专属状态
            self.drone_modes[new_drone_id] = action
            
            # （可选）如果你代码里还有遗留的全局 active_drone_id，为了方便兼容单机逻辑，可以同步一下，但多机核心看 drone_modes
            # self.active_drone_id = new_drone_id 
            
            # 5. 给底层桥接发动作标签
            if new_drone_id in self.action_pubs:
                act_msg = String()
                act_msg.data = action
                self.action_pubs[new_drone_id].publish(act_msg)
                
            self.get_logger().warn(f"收到指令：{new_drone_id} 执行 {action}！")

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
        
# ========================= 单层扁平化状态机分发（多机并发版） ===========================
        for drone_id in self.swarm_config:
            # 1. 获取该飞机的专属状态（如果字典里没有，默认在地面）
            mode = self.drone_modes.get(drone_id, "GROUND")
            
            # 2. 获取该飞机的实时状态（可能还没收到，是 None）
            my_state = self.drone_states.get(drone_id)

            # 地面待命
            if mode == "GROUND":
                continue 
                
            # 起飞
            elif mode == "TAKEOFF":
                if my_state is not None:
                    current_x = my_state.pose.pose.position.x
                    current_y = my_state.pose.pose.position.y
                    
                    target = PoseStamped()
                    target.header.stamp = self.get_clock().now().to_msg()
                    target.header.frame_id = "world"
                    # 原地拔高到 10 米
                    target.pose.position.x = current_x
                    target.pose.position.y = current_y
                    target.pose.position.z = 10.0 
                    
                    # 调用避障函数（起飞时防止队友在头顶重叠）
                    self.apply_collision_avoidance(drone_id, my_state, target, mode)
                    
                    self.send_single_target(drone_id, target)

            # 悬停
            elif mode == "HOVER":
                if my_state is not None:
                    target = PoseStamped()
                    target.header.stamp = self.get_clock().now().to_msg()
                    target.header.frame_id = "world"
                    target.pose.position.x = my_state.pose.pose.position.x
                    target.pose.position.y = my_state.pose.pose.position.y
                    target.pose.position.z = my_state.pose.pose.position.z
                    # HOVER 状态不施加排斥力，做一颗安静的“石头”
                    self.send_single_target(drone_id, target)

            # 返航
            elif mode == "RETURN":
                if my_state is not None:
                    current_x = my_state.pose.pose.position.x
                    current_y = my_state.pose.pose.position.y
                    current_z = my_state.pose.pose.position.z
                    
                    target = PoseStamped()
                    target.header.stamp = self.get_clock().now().to_msg()
                    target.header.frame_id = "world"
                    
                    # 原地垂直降落
                    target.pose.position.x = current_x
                    target.pose.position.y = current_y
                    target.pose.position.z = 0.0 
                    
                    # 调用避障（防止降落时和队友垂直相撞）
                    self.apply_collision_avoidance(drone_id, my_state, target, mode)
                    
                    self.send_single_target(drone_id, target)

                    # 落地检测
                    alt = current_z
                    vx = my_state.twist.twist.linear.x
                    vy = my_state.twist.twist.linear.y
                    vz = my_state.twist.twist.linear.z
                    speed = math.sqrt(vx*vx + vy*vy + vz*vz)
                    
                    if alt < self.LAND_ALT_THRESHOLD and speed < self.LAND_SPEED_THRESHOLD:
                        # 状态改回 GROUND
                        self.drone_modes[drone_id] = "GROUND"
                        
                        # 通知桥接恢复默认限幅
                        if drone_id in self.action_pubs:
                            act_msg = String()
                            act_msg.data = "TRACK" 
                            self.action_pubs[drone_id].publish(act_msg)

                        self.get_logger().info(f"{drone_id} 已落地，状态机回到 GROUND")

            # 追踪（极限环环绕 + 切向防撞）
            elif mode == "TRACK":
                target = self.latest_vision_target if self.latest_vision_target is not None else self.latest_plan_target
                if target is not None and my_state is not None:
                    
                    target = copy.deepcopy(target)
                    
                    # 计算与目标的相对位置
                    dx = target.pose.position.x - my_state.pose.pose.position.x
                    dy = target.pose.position.y - my_state.pose.pose.position.y
                    horiz_dist = math.sqrt(dx * dx + dy * dy)

                    # 如果距离小于安全距离，触发切向环绕（极限环）
                    if horiz_dist < self.TRACK_SAFE_DIST and horiz_dist > 0.01:
                        # 计算连线单位向量
                        ux = dx / horiz_dist
                        uy = dy / horiz_dist
                        
                        # ★ 垂直于连线的切向向量（产生逆时针环绕力）
                        tangent_x = -uy
                        tangent_y = ux
                        
                        # 距离越近，环绕的向心力越强
                        push_force = (self.TRACK_SAFE_DIST - horiz_dist) * 1.5
                        
                        # 将原本要去的目标点，沿着切向偏移
                        target.pose.position.x += tangent_x * push_force
                        target.pose.position.y += tangent_y * push_force
                        
                        self.get_logger().info(f"{drone_id} 正在环绕目标", throttle_duration_sec=2.0)

                    # 然后再应用友军防撞（防止环绕的时候撞到队友）
                    self.apply_collision_avoidance(drone_id, my_state, target, mode)
                    
                    # 发送最终偏移后的目标点
                    self.send_single_target(drone_id, target)
                    
                    # 云台锁定原始目标
                    if self.latest_vision_target is not None:
                        self.publish_gimbal_target(drone_id, target, my_state)

            # 拦截/攻击
            elif mode in ["INTERCEPT", "ATTACK"]:
                if self.latest_plan_target is not None and my_state is not None:
                    # ★ 同样必须深拷贝
                    target = copy.deepcopy(self.latest_plan_target)
                    
                    # 攻击时也要加防撞（长机全速冲，僚机切向避开）
                    self.apply_collision_avoidance(drone_id, my_state, target, mode)
                    
                    self.send_single_target(drone_id, target)

        # =============================================================
                
        # =================【转发给课题三的态势数据】=================
        for drone_id in self.swarm_config:
            my_state = self.drone_states.get(drone_id)
            if my_state is not None:
                self.fleet_state_pubs[drone_id].publish(my_state)
                
                status = DiagnosticStatus()
                status.hardware_id = drone_id
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
                
        # 降频转发：电量数据
        if time.time() - getattr(self, 'last_battery_time', 0.0) > 2.0:
            self.last_battery_time = time.time()
            
            batt_msg = BatteryState()
            batt_msg.header.stamp = self.get_clock().now().to_msg()
            batt_msg.percentage = 1.0
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