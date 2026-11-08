# -*- coding: utf-8 -*-
"""
vision_node.py —— 机载视觉模块主节点（课题二 · 接口A 提供方 · 多机集群版）

======================= 接口清单（严格遵守《课题二内部接口规范》最新版）=======================

【订阅】（对 drone_ids 参数里的每一架飞机，各订阅三路）
  接口J  /drone_<id>/camera/image_raw    (sensor_msgs/msg/Image)      相机原始图像
  接口K  /drone_<id>/camera/camera_info  (sensor_msgs/msg/CameraInfo) 相机内参
  补充   /drone_<id>/state               (nav_msgs/msg/Odometry)      无人机自身 ENU 位姿

【发布】
  接口A  /vision/target_pose  (geometry_msgs/msg/PoseStamped)
         - 目标在 ENU 世界系下的绝对三维坐标，frame_id = "world"
         - header.stamp 必须是【图像采集的原始时间】（分工细节文档新规），不是推理结束时间
         - 目标丢失时【停止发送】
         - 视觉只做"眼睛"：不计算拦截点、速度预测、偏移量

【多机融合约定（接口J 补充条款）】
  同时订阅多路视频流，但运行【单实例】检测器，采用"主从切换策略"：
  只要任意一架无人机的画面捕获到目标，即采用【该飞机】的内参与位姿计算全局 ENU 坐标，
  发布至 /vision/target_pose 供中枢与规划共享。

【算力策略（参数 process_mode）】
  - "round_robin"（默认）：定时器在三路图像间轮转处理，每路约 per_stream_hz(默认10)Hz。
                          CPU/核显可跑，三路共用一个检测器实例。
  - "parallel"            ：每路图像回调里逐帧处理（可配合 process_every_n_frames 降载）。
                          适合有 RTX 级显卡的机器。
============================================================================================
"""

import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

from vision_pkg.geo_utils import camera_mount_matrix, pixel_to_world
from vision_pkg.detector import ColorBlobDetector, YoloDetector


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')

        # ------------------------------ 参数声明 ------------------------------
        # 多机集群：飞机 ID 列表（默认三机协同）
        self.declare_parameter('drone_ids', ['drone_1', 'drone_2', 'drone_3'])
        self.declare_parameter('process_mode', 'round_robin')   # 'round_robin' 或 'parallel'
        self.declare_parameter('per_stream_hz', 10.0)           # round_robin 模式下每路处理频率
        self.declare_parameter('detector_type', 'color')        # 'color' 或 'yolo'
        self.declare_parameter('ground_z', 0.0)                 # 目标所在平面高度（地面为 0）
        self.declare_parameter('process_every_n_frames', 1)     # parallel 模式下降载用
        # 颜色检测阈值（默认红色靶机，HSV 双区间）
        self.declare_parameter('hsv_lower1', [0, 120, 70])
        self.declare_parameter('hsv_upper1', [10, 255, 255])
        self.declare_parameter('hsv_lower2', [170, 120, 70])
        self.declare_parameter('hsv_upper2', [180, 255, 255])
        self.declare_parameter('min_area', 200)
        # YOLO 配置
        self.declare_parameter('yolo_model_path', 'yolov8n.pt')
        self.declare_parameter('yolo_conf', 0.4)
        self.declare_parameter('yolo_target_class', -1)
        # 相机安装外参（弧度，默认 0,0,0 = 垂直朝下安装）
        self.declare_parameter('camera_mount_rpy', [0.0, 0.0, 0.0])

        p = self.get_parameter
        self.drone_ids = list(p('drone_ids').value)
        self.process_mode = p('process_mode').value
        self.per_stream_hz = p('per_stream_hz').value
        self.ground_z = p('ground_z').value
        self.frame_skip = p('process_every_n_frames').value
        rpy = p('camera_mount_rpy').value
        self.R_bc = camera_mount_matrix(rpy[0], rpy[1], rpy[2])

        # ------------------------------ 检测器（全集群单实例） ------------------------------
        detector_type = p('detector_type').value
        if detector_type == 'yolo':
            self.detector = YoloDetector(
                model_path=p('yolo_model_path').value,
                conf_thres=p('yolo_conf').value,
                target_class=p('yolo_target_class').value)
            if not self.detector.available:
                self.get_logger().error(
                    'YOLO 初始化失败（未安装 ultralytics 或模型路径错误），'
                    '节点将收图但不产出。可改用 -p detector_type:=color')
        else:
            self.detector = ColorBlobDetector(
                hsv_lower1=tuple(p('hsv_lower1').value),
                hsv_upper1=tuple(p('hsv_upper1').value),
                hsv_lower2=tuple(p('hsv_lower2').value),
                hsv_upper2=tuple(p('hsv_upper2').value),
                min_area=p('min_area').value)
        self.get_logger().info(f'检测后端: {detector_type} | 处理模式: {self.process_mode}')

        # ------------------------------ 每架飞机的状态缓存 ------------------------------
        # drones[drone_id] = {'info': CameraInfo, 'state': Odometry,
        #                     'image': Image, 'last_stamp': (sec, nanosec), 'frames': int}
        self.drones = {d: {'info': None, 'state': None, 'image': None,
                           'last_stamp': None, 'frames': 0} for d in self.drone_ids}

        self.bridge = CvBridge()

        # ------------------------------ 订阅（每架飞机三路） ------------------------------
        for drone_id in self.drone_ids:
            self.create_subscription(
                Image, f'/{drone_id}/camera/image_raw',
                self._make_image_cb(drone_id), qos_profile_sensor_data)
            self.create_subscription(
                CameraInfo, f'/{drone_id}/camera/camera_info',
                self._make_info_cb(drone_id), qos_profile_sensor_data)
            self.create_subscription(
                Odometry, f'/{drone_id}/state',
                self._make_state_cb(drone_id), 10)

        # ------------------------------ 发布（接口A，唯一出口） ------------------------------
        self.target_pub = self.create_publisher(PoseStamped, '/vision/target_pose', 10)

        # ------------------------------ round_robin 轮转定时器 ------------------------------
        self._rr_index = 0
        if self.process_mode == 'round_robin':
            n = max(len(self.drone_ids), 1)
            period = 1.0 / max(self.per_stream_hz * n, 1.0)
            self.create_timer(period, self.round_robin_tick)
            self.get_logger().info(
                f'轮转处理: {n} 路 x {self.per_stream_hz}Hz (定时器 {1.0/period:.1f}Hz)')

        self.get_logger().info(
            f'vision_node 已启动：集群 {self.drone_ids}，发布 /vision/target_pose (ENU, world)')

    # ------------------------------ 回调工厂（闭包绑定 drone_id） ------------------------------
    def _make_image_cb(self, drone_id):
        def cb(msg):
            entry = self.drones[drone_id]
            entry['frames'] += 1
            if self.process_mode == 'parallel':
                # parallel：逐帧处理（可按 frame_skip 降载）
                if entry['frames'] % max(self.frame_skip, 1) != 0:
                    return
                self.process_frame(drone_id, msg)
            else:
                # round_robin：只缓存最新帧（深拷贝），由定时器统一调度
                entry['image'] = copy.deepcopy(msg)
        return cb

    def _make_info_cb(self, drone_id):
        def cb(msg):
            self.drones[drone_id]['info'] = copy.deepcopy(msg)
        return cb

    def _make_state_cb(self, drone_id):
        def cb(msg):
            self.drones[drone_id]['state'] = copy.deepcopy(msg)
        return cb

    # ------------------------------ round_robin 调度 ------------------------------
    def round_robin_tick(self):
        drone_id = self.drone_ids[self._rr_index % len(self.drone_ids)]
        self._rr_index += 1
        entry = self.drones[drone_id]
        msg = entry['image']
        if msg is None:
            return
        # 跳过已经处理过的同一帧（相机比轮转慢时会重复）
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == entry['last_stamp']:
            return
        entry['last_stamp'] = stamp
        self.process_frame(drone_id, msg)

    # ------------------------------ 核心流水线：一路图像 -> 可能发布一个目标 ------------------------------
    def process_frame(self, drone_id, image_msg):
        entry = self.drones[drone_id]

        # 前置条件：该飞机的内参和位姿缺一不可
        if entry['info'] is None:
            self.get_logger().warn(
                f'[{drone_id}] 等待 camera_info（接口K）...', throttle_duration_sec=2.0)
            return
        if entry['state'] is None:
            self.get_logger().warn(
                f'[{drone_id}] 等待无人机位姿 /state ...', throttle_duration_sec=2.0)
            return

        # 1) ROS Image -> OpenCV BGR
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'[{drone_id}] 图像转换失败: {e}', throttle_duration_sec=2.0)
            return

        # 2) 2D 检测（主从切换：这路没目标就直接返回，等下一路）
        result = self.detector.detect(cv_image)
        if result is None:
            self.get_logger().info(f'[{drone_id}] 画面无目标', throttle_duration_sec=5.0)
            return
        u, v = result

        # 3) 用【这架飞机】的内参 + 位姿做 2D -> 3D 反投影
        pose = entry['state'].pose.pose
        q = pose.orientation
        pos = pose.position
        world_xyz = pixel_to_world(
            u, v,
            entry['info'].k,
            (pos.x, pos.y, pos.z),
            (q.x, q.y, q.z, q.w),
            self.R_bc,
            ground_z=self.ground_z)
        if world_xyz is None:
            self.get_logger().warn(
                f'[{drone_id}] 反投影失败（射线朝天/平行地面或内参无效），本帧丢弃',
                throttle_duration_sec=2.0)
            return

        # 4) 组装并发布 PoseStamped
        out = PoseStamped()
        # 新规铁律：时间戳必须是【图像采集的原始时间】，不是推理结束时间
        if image_msg.header.stamp.sec == 0 and image_msg.header.stamp.nanosec == 0:
            out.header.stamp = self.get_clock().now().to_msg()  # 上游没打戳才兜底
        else:
            out.header.stamp = image_msg.header.stamp
        out.header.frame_id = 'world'
        out.pose.position.x = world_xyz[0]
        out.pose.position.y = world_xyz[1]
        out.pose.position.z = world_xyz[2]
        out.pose.orientation.w = 1.0
        self.target_pub.publish(out)

        self.get_logger().info(
            f'[{drone_id}] 捕获目标 -> ENU: x={world_xyz[0]:.2f}, '
            f'y={world_xyz[1]:.2f}, z={world_xyz[2]:.2f}',
            throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
