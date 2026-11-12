# -*- coding: utf-8 -*-
"""
camera_relay_node.py —— 机载相机数据桥接与系统时间戳同步节点

职责（接口 J / K）：
  1. 支持直接通过 Gazebo Harmonic (gz.transport13) 订阅仿真相机原始图像与内参
     （避免 ROS 2 Humble 二进制 ros_gz_bridge 绑定旧版 ign-transport11 的兼容断层），
     同时也保留 source_mode='ros2' 兼容模式。
  2. 在全系统禁用 use_sim_time 的约束下，将仿真零点时间戳对齐到系统墙钟基准，
     同时保留帧间真实采集间隔，防止被 center_node.py 的 0.5s 延迟检查误杀。
  3. 发布到《课题二内部接口规范 V3.0》规定的标准话题：
       - /{drone_id}/camera/image_raw
       - /{drone_id}/camera/camera_info
"""

import array
import copy
from typing import Optional

import rclpy
from builtin_interfaces.msg import Time as RosTime
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image


_GZ_PIXEL_FORMAT_TO_ROS = {
    1: 'mono8',   # L_INT8
    2: 'mono16',  # L_INT16
    3: 'rgb8',    # RGB_INT8
    4: 'rgba8',   # RGBA_INT8
    5: 'bgra8',   # BGRA_INT8
    8: 'bgr8',    # BGR_INT8
}


class CameraRelayNode(Node):
    """单机相机图像/内参标准化与时间戳对齐转发节点。"""

    def __init__(self) -> None:
        super().__init__('camera_relay_node')

        self.declare_parameter('drone_id', 'drone_1')
        self.declare_parameter('source_mode', 'gz_transport')  # 'gz_transport' | 'ros2'
        self.declare_parameter('gz_world', 'moving_target_arena')
        self.declare_parameter('gz_model', 'x500_down_cam_0')
        self.declare_parameter('gz_link', 'down_cam_link')
        self.declare_parameter('gz_sensor', 'down_cam')
        self.declare_parameter('out_namespace', 'camera')
        self.declare_parameter('in_image_topic', '/gz/drone_1/camera/image_raw')
        self.declare_parameter('in_info_topic', '/gz/drone_1/camera/camera_info')
        self.declare_parameter('restamp_to_wall_clock', True)

        p = self.get_parameter
        self.drone_id: str = str(p('drone_id').value)
        self.source_mode: str = str(p('source_mode').value)
        self.gz_world: str = str(p('gz_world').value)
        self.gz_model: str = str(p('gz_model').value)
        self.gz_link: str = str(p('gz_link').value)
        self.gz_sensor: str = str(p('gz_sensor').value)
        self.out_ns: str = str(p('out_namespace').value).strip().strip('/') or 'camera'
        in_img: str = str(p('in_image_topic').value)
        in_info: str = str(p('in_info_topic').value)
        self.restamp: bool = bool(p('restamp_to_wall_clock').value)

        # wall_ns - sim_ns 偏移量（首次收到仿真戳时锁定，保留相对帧间时间间隔）
        self._wall_sim_offset_ns: Optional[int] = None

        self.pub_img = self.create_publisher(
            Image, f'/{self.drone_id}/{self.out_ns}/image_raw', qos_profile_sensor_data
        )
        self.pub_info = self.create_publisher(
            CameraInfo, f'/{self.drone_id}/{self.out_ns}/camera_info', qos_profile_sensor_data
        )

        self._gz_node = None
        if self.source_mode == 'gz_transport':
            self._init_gz_transport()
        else:
            self.create_subscription(Image, in_img, self._image_cb, qos_profile_sensor_data)
            self.create_subscription(CameraInfo, in_info, self._info_cb, qos_profile_sensor_data)
            self.get_logger().info(
                f'[{self.drone_id}] 相机转发节点(ROS2模式)已启动: {in_img} -> /{self.drone_id}/camera/image_raw'
            )

    def _init_gz_transport(self) -> None:
        try:
            import gz.transport13 as gzt
            from gz.msgs10 import camera_info_pb2, image_pb2
        except ImportError as exc:
            self.get_logger().error(f'[{self.drone_id}] 无法导入 gz.transport13 / gz.msgs10: {exc}')
            return

        base = (
            f'/world/{self.gz_world}/model/{self.gz_model}'
            f'/link/{self.gz_link}/sensor/{self.gz_sensor}'
        )
        gz_img_topic = f'{base}/image'
        gz_info_topic = f'{base}/camera_info'

        self._gz_node = gzt.Node()
        self._gz_node.subscribe(image_pb2.Image, gz_img_topic, self._gz_image_cb)
        self._gz_node.subscribe(camera_info_pb2.CameraInfo, gz_info_topic, self._gz_info_cb)
        self.get_logger().info(
            f'[{self.drone_id}] 相机直连桥接(gz.transport13)已启动: {gz_img_topic} -> /{self.drone_id}/{self.out_ns}/image_raw'
        )

    def _align_stamp_sec_nsec(self, sec: int, nanosec: int) -> RosTime:
        out_stamp = RosTime()
        if not self.restamp:
            out_stamp.sec = int(sec)
            out_stamp.nanosec = int(nanosec)
            return out_stamp

        now_ns = self.get_clock().now().nanoseconds
        sim_ns = int(sec) * 1_000_000_000 + int(nanosec)
        if sim_ns <= 0:
            return self.get_clock().now().to_msg()

        # 若消息时间与当前墙钟偏差超过 60 秒，判定为仿真零点时钟，执行平滑偏移对齐
        # 同时将滞后控制在 [-10ms, +80ms] 内，避免 Gazebo 实时率波动导致中枢 >0.5s 丢帧
        if abs(now_ns - sim_ns) > 60_000_000_000:
            if self._wall_sim_offset_ns is None:
                self._wall_sim_offset_ns = now_ns - sim_ns
            else:
                drift_ns = now_ns - (sim_ns + self._wall_sim_offset_ns)
                if drift_ns > 80_000_000 or drift_ns < -10_000_000:
                    self._wall_sim_offset_ns = now_ns - sim_ns
            aligned_ns = max(0, sim_ns + self._wall_sim_offset_ns)
            out_stamp.sec = int(aligned_ns // 1_000_000_000)
            out_stamp.nanosec = int(aligned_ns % 1_000_000_000)
            return out_stamp

        out_stamp.sec = int(sec)
        out_stamp.nanosec = int(nanosec)
        return out_stamp

    def _align_stamp(self, msg_stamp) -> RosTime:
        return self._align_stamp_sec_nsec(int(msg_stamp.sec), int(msg_stamp.nanosec))

    # -------------------- gz.transport13 回调 --------------------
    def _gz_image_cb(self, gz_msg) -> None:
        if not rclpy.ok():
            return
        ros_img = Image()
        ros_img.header.stamp = self._align_stamp_sec_nsec(
            int(gz_msg.header.stamp.sec), int(gz_msg.header.stamp.nsec)
        )
        ros_img.header.frame_id = f'{self.drone_id}_{self.out_ns}_optical_link'
        ros_img.height = int(gz_msg.height)
        ros_img.width = int(gz_msg.width)
        ros_img.encoding = _GZ_PIXEL_FORMAT_TO_ROS.get(int(gz_msg.pixel_format_type), 'rgb8')
        ros_img.is_bigendian = 0
        ros_img.step = int(gz_msg.step) if gz_msg.step > 0 else int(gz_msg.width * 3)
        img_arr = array.array('B')
        img_arr.frombytes(gz_msg.data)
        ros_img.data = img_arr
        self.pub_img.publish(ros_img)

    def _gz_info_cb(self, gz_msg) -> None:
        if not rclpy.ok():
            return
        ros_info = CameraInfo()
        ros_info.header.stamp = self._align_stamp_sec_nsec(
            int(gz_msg.header.stamp.sec), int(gz_msg.header.stamp.nsec)
        )
        ros_info.header.frame_id = f'{self.drone_id}_{self.out_ns}_optical_link'
        ros_info.height = int(gz_msg.height)
        ros_info.width = int(gz_msg.width)
        ros_info.distortion_model = 'plumb_bob'
        if len(gz_msg.distortion.k) > 0:
            ros_info.d = [float(v) for v in gz_msg.distortion.k]
        else:
            ros_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        if len(gz_msg.intrinsics.k) == 9:
            ros_info.k = [float(v) for v in gz_msg.intrinsics.k]
        if len(gz_msg.rectification_matrix) == 9:
            ros_info.r = [float(v) for v in gz_msg.rectification_matrix]
        else:
            ros_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        if len(gz_msg.projection.p) == 12:
            ros_info.p = [float(v) for v in gz_msg.projection.p]
        self.pub_info.publish(ros_info)

    # -------------------- ROS 2 兼容回调 --------------------
    def _image_cb(self, msg: Image) -> None:
        out = copy.deepcopy(msg)
        out.header.stamp = self._align_stamp(msg.header.stamp)
        out.header.frame_id = f'{self.drone_id}_{self.out_ns}_optical_link'
        self.pub_img.publish(out)

    def _info_cb(self, msg: CameraInfo) -> None:
        out = copy.deepcopy(msg)
        out.header.stamp = self._align_stamp(msg.header.stamp)
        out.header.frame_id = f'{self.drone_id}_{self.out_ns}_optical_link'
        self.pub_info.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
