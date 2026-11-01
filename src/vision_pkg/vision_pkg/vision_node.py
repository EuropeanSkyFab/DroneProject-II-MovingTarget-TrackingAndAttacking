# -*- coding: utf-8 -*-
"""
vision_node.py —— 机载视觉模块主节点（课题二 · 接口A 提供方）

======================= 接口清单（严格遵守《课题二内部接口规范 V3.0》）=======================

【订阅】
  接口J  /drone_<id>/camera/image_raw    (sensor_msgs/msg/Image)      相机原始图像
  接口K  /drone_<id>/camera/camera_info  (sensor_msgs/msg/CameraInfo) 相机内参
  补充   /drone_<id>/state               (nav_msgs/msg/Odometry)      无人机自身 ENU 位姿
         —— 2D->3D 反投影物理上必须要自身位姿；分工表未列出，建议与组长确认。

【发布】
  接口A  /vision/target_pose  (geometry_msgs/msg/PoseStamped)
         - 敌方/靶机在 ENU 世界系下的绝对三维坐标，frame_id = "world"
         - header.stamp 必须是 publish 前的真实当前时间（中枢有 0.2s 延迟过滤！）
         - 目标丢失时【停止发送】（严禁发 (0,0,0) 或保持发旧数据）
         - 视觉只做"眼睛"：绝不计算拦截点、速度预测、偏移量，那是规划模块的活。

【坐标系】ENU（东-北-天，Z 向上），严禁在本模块内做 NED 转换。
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
        self.declare_parameter('drone_id', 'drone_1')
        self.declare_parameter('detector_type', 'color')        # 'color' 或 'yolo'
        self.declare_parameter('ground_z', 0.0)                 # 目标所在平面高度（地面为 0）
        self.declare_parameter('process_every_n_frames', 1)     # 每 N 帧处理一次，降载用
        # 颜色检测阈值（默认红色靶机，HSV 双区间）
        self.declare_parameter('hsv_lower1', [0, 120, 70])
        self.declare_parameter('hsv_upper1', [10, 255, 255])
        self.declare_parameter('hsv_lower2', [170, 120, 70])
        self.declare_parameter('hsv_upper2', [180, 255, 255])
        self.declare_parameter('min_area', 200)                 # 最小色块面积（像素²）
        # YOLO 配置
        self.declare_parameter('yolo_model_path', 'yolov8n.pt')
        self.declare_parameter('yolo_conf', 0.4)
        self.declare_parameter('yolo_target_class', -1)         # -1 不限类别
        # 相机安装外参（弧度，默认 0,0,0 = 垂直朝下安装）
        self.declare_parameter('camera_mount_rpy', [0.0, 0.0, 0.0])

        p = self.get_parameter
        self.drone_id = p('drone_id').value
        self.ground_z = p('ground_z').value
        self.frame_skip = p('process_every_n_frames').value
        rpy = p('camera_mount_rpy').value
        self.R_bc = camera_mount_matrix(rpy[0], rpy[1], rpy[2])

        # ------------------------------ 检测器 ------------------------------
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
        self.get_logger().info(f'检测后端: {detector_type}')

        # ------------------------------ 订阅（接口J / 接口K / 自身位姿） ------------------------------
        self.bridge = CvBridge()
        self.camera_info_msg = None   # 缓存相机内参（深拷贝）
        self.drone_state_msg = None   # 缓存无人机位姿（深拷贝）
        self.frame_count = 0

        self.create_subscription(
            Image,
            f'/{self.drone_id}/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data)   # 图像流必须用 sensor_data QoS，否则订阅不上 Gazebo 相机

        self.create_subscription(
            CameraInfo,
            f'/{self.drone_id}/camera/camera_info',
            self.camera_info_callback,
            qos_profile_sensor_data)

        self.create_subscription(
            Odometry,
            f'/{self.drone_id}/state',
            self.drone_state_callback,
            10)

        # ------------------------------ 发布（接口A，唯一出口） ------------------------------
        self.target_pub = self.create_publisher(PoseStamped, '/vision/target_pose', 10)

        self.get_logger().info(
            f'vision_node 已启动：订阅 /{self.drone_id}/camera/* 与 /{self.drone_id}/state，'
            f'发布 /vision/target_pose (ENU, frame_id=world)')

    # ------------------------------ 回调 ------------------------------
    def camera_info_callback(self, msg):
        # 铁律：存消息必须深拷贝，防止底层内存回收导致数据乱码
        self.camera_info_msg = copy.deepcopy(msg)

    def drone_state_callback(self, msg):
        self.drone_state_msg = copy.deepcopy(msg)

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % max(self.frame_skip, 1) != 0:
            return

        # 前置条件：内参和位姿缺一不可，缺了就不发（等数据到齐）
        if self.camera_info_msg is None:
            self.get_logger().warn('等待 camera_info（接口K）...', throttle_duration_sec=2.0)
            return
        if self.drone_state_msg is None:
            self.get_logger().warn('等待无人机位姿 /state ...', throttle_duration_sec=2.0)
            return

        # 1) ROS Image -> OpenCV BGR
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'图像转换失败: {e}', throttle_duration_sec=2.0)
            return

        # 2) 2D 检测
        result = self.detector.detect(cv_image)
        if result is None:
            # 目标丢失：按规范【停止发送】，直接返回
            self.get_logger().info('目标丢失', throttle_duration_sec=2.0)
            return
        u, v = result

        # 3) 2D 像素 -> 3D 世界 ENU 坐标
        pose = self.drone_state_msg.pose.pose
        q = pose.orientation
        pos = pose.position
        world_xyz = pixel_to_world(
            u, v,
            self.camera_info_msg.k,
            (pos.x, pos.y, pos.z),
            (q.x, q.y, q.z, q.w),
            self.R_bc,
            ground_z=self.ground_z)
        if world_xyz is None:
            self.get_logger().warn(
                '反投影失败（射线朝天/平行地面或内参无效），本帧丢弃',
                throttle_duration_sec=2.0)
            return

        # 4) 组装并发布 PoseStamped（时间戳铁律：publish 前取当前真实时间）
        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = 'world'
        out.pose.position.x = world_xyz[0]
        out.pose.position.y = world_xyz[1]
        out.pose.position.z = world_xyz[2]
        out.pose.orientation.w = 1.0   # 位置无姿态含义，置单位四元数
        self.target_pub.publish(out)

        self.get_logger().info(
            f'目标 ENU: x={world_xyz[0]:.2f}, y={world_xyz[1]:.2f}, z={world_xyz[2]:.2f}',
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
