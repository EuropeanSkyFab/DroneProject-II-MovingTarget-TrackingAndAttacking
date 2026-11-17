# -*- coding: utf-8 -*-
"""
vision_node.py —— 机载视觉模块主节点（课题二 · 接口 A 提供方 · 多机立体三角测量增强版）

严格遵守《课题二内部接口规范 V3.0》：
  订阅（每架无人机三路）：
    - 接口 J: /drone_<id>/camera/image_raw    (sensor_msgs/Image)
    - 接口 K: /drone_<id>/camera/camera_info  (sensor_msgs/CameraInfo)
    - 位姿  : /drone_<id>/state               (nav_msgs/Odometry, 全局 ENU)
  发布：
    - 接口 A: /vision/target_pose             (geometry_msgs/PoseStamped, frame_id="world", 原始采集时间戳)
    - 诊断  : /vision/diagnostics             (std_msgs/String JSON, 立体视差残差与工作模式可观测性)
"""

from __future__ import annotations

import collections
import copy
import json
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from vision_pkg.detector import ColorBlobDetector, YoloDetector
from vision_pkg.geo_utils import (
    camera_mount_matrix,
    pixel_to_ray_world,
    pixel_to_world,
    triangulate_rays,
)


class VisionNode(Node):
    """多机单实例共享检测 + 异构视角三维射线三角测量视觉节点。"""

    def __init__(self) -> None:
        super().__init__('vision_node')

        # ------------------------------ 参数声明 ------------------------------
        self.declare_parameter('drone_ids', ['drone_1', 'drone_2', 'drone_3'])
        self.declare_parameter('process_mode', 'round_robin')
        self.declare_parameter('per_stream_hz', 15.0)
        self.declare_parameter('detector_type', 'color')
        self.declare_parameter('ground_z', 5.5)
        self.declare_parameter('process_every_n_frames', 1)
        # 颜色检测阈值（默认抗强光红度增强 + 远距小目标支持）
        self.declare_parameter('hsv_lower1', [0, 110, 55])
        self.declare_parameter('hsv_upper1', [12, 255, 255])
        self.declare_parameter('hsv_lower2', [168, 110, 55])
        self.declare_parameter('hsv_upper2', [180, 255, 255])
        self.declare_parameter('min_area', 24)
        self.declare_parameter('enable_glare_suppression', True)
        # YOLO 配置
        self.declare_parameter('yolo_model_path', 'yolov8n.pt')
        self.declare_parameter('yolo_conf', 0.4)
        self.declare_parameter('yolo_target_class', -1)
        # 相机安装外参（默认对齐 x500_down_cam 前斜 35° 下视导航相机）
        self.declare_parameter('camera_mount_rpy', [0.95993109, 0.0, -1.57079633])
        self.declare_parameter('camera_offset_body', [0.12, 0.0, -0.04])
        # 多机三维立体三角测量配置
        self.declare_parameter('enable_triangulation', True)
        self.declare_parameter('triangulation_sync_sec', 0.16)
        self.declare_parameter('min_parallax_deg', 11.5)
        self.declare_parameter('max_skew_rms_m', 0.55)
        self.declare_parameter('min_baseline_m', 3.4)

        p = self.get_parameter
        self.drone_ids: List[str] = [str(d) for d in p('drone_ids').value]
        self.process_mode: str = str(p('process_mode').value)
        self.per_stream_hz: float = float(p('per_stream_hz').value)
        self.ground_z: float = float(p('ground_z').value)
        self._adaptive_target_z: float = self.ground_z
        self.frame_skip: int = int(p('process_every_n_frames').value)

        rpy = list(p('camera_mount_rpy').value)
        self.R_bc: np.ndarray = camera_mount_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        off = list(p('camera_offset_body').value)
        self.cam_offset_body: Tuple[float, float, float] = (float(off[0]), float(off[1]), float(off[2]))

        self.enable_triangulation: bool = bool(p('enable_triangulation').value)
        self.tri_sync_sec: float = float(p('triangulation_sync_sec').value)
        self.min_parallax_deg: float = float(p('min_parallax_deg').value)
        self.max_skew_rms_m: float = float(p('max_skew_rms_m').value)
        self.min_baseline_m: float = float(p('min_baseline_m').value)

        # ------------------------------ 检测器（全集群单实例） ------------------------------
        detector_type = str(p('detector_type').value)
        if detector_type == 'yolo':
            self.detector = YoloDetector(
                model_path=str(p('yolo_model_path').value),
                conf_thres=float(p('yolo_conf').value),
                target_class=int(p('yolo_target_class').value),
            )
            if not self.detector.available:
                self.get_logger().error(
                    'YOLO 初始化失败（未安装 ultralytics 或模型不存在），自动回退至抗强光 ColorBlobDetector'
                )
                self.detector = ColorBlobDetector(
                    hsv_lower1=tuple(p('hsv_lower1').value),
                    hsv_upper1=tuple(p('hsv_upper1').value),
                    hsv_lower2=tuple(p('hsv_lower2').value),
                    hsv_upper2=tuple(p('hsv_upper2').value),
                    min_area=float(p('min_area').value),
                    enable_glare_suppression=bool(p('enable_glare_suppression').value),
                )
        else:
            self.detector = ColorBlobDetector(
                hsv_lower1=tuple(p('hsv_lower1').value),
                hsv_upper1=tuple(p('hsv_upper1').value),
                hsv_lower2=tuple(p('hsv_lower2').value),
                hsv_upper2=tuple(p('hsv_upper2').value),
                min_area=float(p('min_area').value),
                enable_glare_suppression=bool(p('enable_glare_suppression').value),
            )

        self.drones: Dict[str, Dict[str, object]] = {
            d: {
                'info': None,
                'state': None,
                'state_buf': collections.deque(maxlen=35),
                'image': None,
                'last_stamp': None,
                'frames': 0,
                'last_ray': None,  # (origin_np, unit_dir_np, wall_sec)
            }
            for d in self.drone_ids
        }

        self.bridge = CvBridge()
        self._last_diag_sec: float = 0.0
        self._last_stereo_sec: float = -1e9
        self._last_loc_mode: str = 'idle'
        self._last_skew_rms: float = 0.0
        self._stereo_hits: int = 0
        self._mono_hits: int = 0

        # ------------------------------ 订阅与发布 ------------------------------
        for drone_id in self.drone_ids:
            self.create_subscription(
                Image, f'/{drone_id}/camera/image_raw', self._make_image_cb(drone_id), qos_profile_sensor_data
            )
            self.create_subscription(
                CameraInfo, f'/{drone_id}/camera/camera_info', self._make_info_cb(drone_id), qos_profile_sensor_data
            )
            self.create_subscription(Odometry, f'/{drone_id}/state', self._make_state_cb(drone_id), 10)

        self.target_pub = self.create_publisher(PoseStamped, '/vision/target_pose', 10)
        self.diag_pub = self.create_publisher(String, '/vision/diagnostics', 10)

        self._rr_index = 0
        if self.process_mode == 'round_robin':
            n = max(len(self.drone_ids), 1)
            period = 1.0 / max(self.per_stream_hz * n, 1.0)
            self.create_timer(period, self.round_robin_tick)
            self.get_logger().info(
                f'轮转处理: {n} 路 x {self.per_stream_hz}Hz (总调度 {1.0 / period:.1f}Hz) | '
                f'多机立体三角化={self.enable_triangulation}'
            )

        self.get_logger().info(
            f'vision_node 已启动：集群={self.drone_ids}，参考高度={self.ground_z}m'
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _make_image_cb(self, drone_id: str):
        def cb(msg: Image) -> None:
            entry = self.drones[drone_id]
            entry['frames'] = int(entry['frames']) + 1
            if self.process_mode == 'parallel':
                if int(entry['frames']) % max(self.frame_skip, 1) != 0:
                    return
                self.process_frame(drone_id, msg)
            else:
                entry['image'] = copy.deepcopy(msg)

        return cb

    def _make_info_cb(self, drone_id: str):
        def cb(msg: CameraInfo) -> None:
            self.drones[drone_id]['info'] = copy.deepcopy(msg)

        return cb

    def _make_state_cb(self, drone_id: str):
        def cb(msg: Odometry) -> None:
            m = copy.deepcopy(msg)
            self.drones[drone_id]['state'] = m
            ts = int(m.header.stamp.sec) + int(m.header.stamp.nanosec) * 1e-9
            if ts <= 0.0:
                ts = self._now_sec()
            buf: Deque[Tuple[float, Odometry]] = self.drones[drone_id]['state_buf']  # type: ignore[assignment]
            buf.append((ts, m))

        return cb

    def _match_state_for_stamp(
        self, drone_id: str, img_stamp, render_lag_sec: float = 0.065
    ) -> Optional[Odometry]:
        entry = self.drones[drone_id]
        latest: Optional[Odometry] = entry['state']  # type: ignore[assignment]
        buf: Deque[Tuple[float, Odometry]] = entry['state_buf']  # type: ignore[assignment]
        if not buf or (int(img_stamp.sec) == 0 and int(img_stamp.nanosec) == 0):
            return latest
        target_ts = int(img_stamp.sec) + int(img_stamp.nanosec) * 1e-9 - render_lag_sec
        best_odom = latest
        best_dt = float('inf')
        for ts, odom in buf:
            dt = abs(ts - target_ts)
            if dt < best_dt:
                best_dt = dt
                best_odom = odom
        return best_odom

    def round_robin_tick(self) -> None:
        drone_id = self.drone_ids[self._rr_index % len(self.drone_ids)]
        self._rr_index += 1
        entry = self.drones[drone_id]
        msg: Optional[Image] = entry['image']  # type: ignore[assignment]
        if msg is None:
            return
        stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
        if stamp == entry['last_stamp']:
            return
        entry['last_stamp'] = stamp
        self.process_frame(drone_id, msg)

    def process_frame(self, drone_id: str, image_msg: Image) -> None:
        entry = self.drones[drone_id]
        info_msg: Optional[CameraInfo] = entry['info']  # type: ignore[assignment]
        lag = 0.0 if image_msg.header.frame_id == 'camera_link' else 0.065
        state_msg: Optional[Odometry] = self._match_state_for_stamp(
            drone_id, image_msg.header.stamp, render_lag_sec=lag
        )

        if info_msg is None:
            self.get_logger().warn(f'[{drone_id}] 等待 camera_info（接口K）...', throttle_duration_sec=3.0)
            return
        if state_msg is None:
            self.get_logger().warn(f'[{drone_id}] 等待无人机位姿 /state ...', throttle_duration_sec=3.0)
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'[{drone_id}] 图像转换失败: {exc}', throttle_duration_sec=2.0)
            return

        result = self.detector.detect(cv_image)
        now_s = self._now_sec()
        if result is None:
            entry['last_ray'] = None
            return

        u, v = result
        pose = state_msg.pose.pose
        pos_tuple = (float(pose.position.x), float(pose.position.y), float(pose.position.z))
        quat_tuple = (
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )

        # 若为假相机注入 (frame_id == 'camera_link')，不叠加真实机体杆臂偏移
        eff_offset = (0.0, 0.0, 0.0) if image_msg.header.frame_id == 'camera_link' else self.cam_offset_body

        ray = pixel_to_ray_world(
            u, v, info_msg.k, pos_tuple, quat_tuple, self.R_bc, cam_offset_body=eff_offset
        )
        if ray is None:
            return
        entry['last_ray'] = (ray[0], ray[1], now_s)

        # 1) 优先尝试多机同步视窗内的三维立体射线三角测量（零高度先验真 3D 解算）
        world_xyz: Optional[Tuple[float, float, float]] = None
        active_cams: List[str] = [drone_id]
        if self.enable_triangulation and len(self.drone_ids) >= 2:
            sync_rays: List[Tuple[np.ndarray, np.ndarray]] = [ray]
            for other_id in self.drone_ids:
                if other_id == drone_id:
                    continue
                other_ray = self.drones[other_id].get('last_ray')
                if other_ray is not None:
                    o_orig, o_dir, o_ts = other_ray  # type: ignore[misc]
                    if (now_s - float(o_ts)) <= self.tri_sync_sec:
                        sync_rays.append((o_orig, o_dir))
                        active_cams.append(other_id)
            if len(sync_rays) >= 2:
                eff_baseline = 0.0 if image_msg.header.frame_id == 'camera_link' else self.min_baseline_m
                tri_res = triangulate_rays(
                    sync_rays,
                    min_parallax_deg=self.min_parallax_deg,
                    max_skew_rms_m=self.max_skew_rms_m,
                    min_baseline_m=eff_baseline,
                )
                if tri_res is not None:
                    tx, ty, tz, skew_rms = tri_res
                    z_jump_ok = (self._stereo_hits < 2) or (abs(tz - self._adaptive_target_z) <= 2.4)
                    if 2.2 <= tz <= 30.0 and z_jump_ok:
                        world_xyz = (tx, ty, tz)
                        self._adaptive_target_z = 0.75 * self._adaptive_target_z + 0.25 * tz
                        self._last_stereo_sec = now_s
                        prev_mode = self._last_loc_mode
                        self._last_loc_mode = f'stereo_{len(sync_rays)}cam'
                        self._last_skew_rms = skew_rms
                        self._stereo_hits += 1
                    else:
                        prev_mode = self._last_loc_mode
                else:
                    prev_mode = self._last_loc_mode
            else:
                prev_mode = self._last_loc_mode
        else:
            prev_mode = self._last_loc_mode

        # 2) 立体主导锁：若过去 0.85s 内已有双目/三目真三维定位，抑制夹杂的单目平面帧以消除卡尔曼微分跳变
        if world_xyz is None:
            is_mock_cam = image_msg.header.frame_id == 'camera_link'
            if not is_mock_cam:
                if (now_s - self._last_stereo_sec) < 0.85:
                    return
                if (-float(ray[1][2])) < 0.36:
                    return
            world_xyz = pixel_to_world(
                u,
                v,
                info_msg.k,
                pos_tuple,
                quat_tuple,
                self.R_bc,
                ground_z=self._adaptive_target_z,
                cam_offset_body=eff_offset,
            )
            if world_xyz is None:
                return
            self._last_loc_mode = 'monocular_plane'
            self._mono_hits += 1

        if (self._last_loc_mode != prev_mode) or (now_s - self._last_diag_sec >= 0.4):
            self._last_diag_sec = now_s
            d_msg = String()
            d_msg.data = json.dumps({
                'stamp': round(now_s, 3),
                'mode': self._last_loc_mode,
                'active_cams': active_cams,
                'skew_rms_m': round(self._last_skew_rms, 4),
                'adaptive_z_m': round(self._adaptive_target_z, 3),
                'stereo_hits': self._stereo_hits,
                'mono_hits': self._mono_hits,
            })
            self.diag_pub.publish(d_msg)

        # 3) 组装并发布接口 A (/vision/target_pose)，严格保留图像原始采集时间戳
        out = PoseStamped()
        if image_msg.header.stamp.sec == 0 and image_msg.header.stamp.nanosec == 0:
            out.header.stamp = self.get_clock().now().to_msg()
        else:
            out.header.stamp = image_msg.header.stamp
        out.header.frame_id = 'world'
        out.pose.position.x = float(world_xyz[0])
        out.pose.position.y = float(world_xyz[1])
        out.pose.position.z = float(world_xyz[2])
        out.pose.orientation.w = 1.0
        self.target_pub.publish(out)

        self.get_logger().info(
            f'[{",".join(active_cams)}|{self._last_loc_mode}] 捕获目标 -> ENU: '
            f'({world_xyz[0]:.2f}, {world_xyz[1]:.2f}, {world_xyz[2]:.2f})',
            throttle_duration_sec=1.0,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionNode()
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
