# -*- coding: utf-8 -*-
"""
vision_node.py —— 多目标协同机载视觉节点（F4 多特征谱系检测 + F5 全配对极线三角测量与 3D 障碍物 LoS 检验）

输入订阅（每架无人机支持主话题与兼容别名双订阅）：
  - 主接口: /{drone_id}/camera/image_raw    (sensor_msgs/Image)
            /{drone_id}/camera/camera_info  (sensor_msgs/CameraInfo)
            /{drone_id}/state               (nav_msgs/Odometry, 全局 ENU)
  - 兼容别名 (subscribe_compat_aliases=True):
            /{drone_id}/down_cam/image_raw    (sensor_msgs/Image)
            /{drone_id}/down_cam/camera_info  (sensor_msgs/CameraInfo)
            /{drone_id}/pose                  (geometry_msgs/PoseStamped, 全局 ENU)

输出发布（严格对齐 PROJECT.md § Interface Contracts #3）：
  - /vision/targets      (std_msgs/String JSON, frame_id="world")
  - /vision/target_pose  (geometry_msgs/PoseStamped, frame_id="world", 保留向后兼容主目标)
  - /vision/diagnostics  (std_msgs/String JSON, 保留 mode/stereo_hits/mono_hits 及多目标/LoS/单路帧率指标)
"""

from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import math
import multiprocessing as mp
from multiprocessing import shared_memory
import os
import threading
import time
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple
import weakref

import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from vision_pkg.detector import (
    ColorBlobDetector,
    Detection,
    MultiSignatureDetector,
    YoloDetector,
    profiles_from_json,
)
from vision_pkg.geo_utils import (
    Obstacle3D,
    RayObservation,
    TriangulatedTarget,
    camera_mount_matrix,
    has_line_of_sight,
    load_obstacles_from_json,
    load_obstacles_from_yaml,
    pixel_to_ray_world,
    triangulate_multi_target,
)


_CLASS_PRIORITY = {
    'crimson_uav': 0,
    'amber_uav': 1,
    'slate_uav': 2,
}


@dataclass
class _DroneSlot:
    drone_id: str
    slot_idx: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    info_k: Optional[Tuple[float, ...]] = None
    latest_state: Optional[Tuple[float, Tuple[float, float, float], Tuple[float, float, float, float]]] = None
    state_buf: Deque[Tuple[float, Tuple[float, float, float], Tuple[float, float, float, float]]] = field(
        default_factory=lambda: collections.deque(maxlen=60)
    )
    latest_image: Optional[Image] = None
    latest_image_rx_wall: float = 0.0
    rx_seq: int = 0
    proc_seq: int = 0
    in_flight: bool = False
    worker_buf: int = 0
    inflight_ctx: Optional[Tuple[Any, ...]] = None
    pending_meta: Optional[Tuple[Any, Any]] = None
    last_rr_stamp: Optional[Tuple[int, int]] = None
    prev_bgr: Optional[np.ndarray] = None
    prev_gray: Optional[np.ndarray] = None
    # (wall_sec, img_stamp_sec, img_stamp_msg, is_mock_cam, ray_obs_list)
    latest_rays: Optional[Tuple[float, float, object, bool, List[RayObservation]]] = None
    rx_frames: int = 0
    proc_frames: int = 0
    coalesced_drops: int = 0
    last_detections: int = 0
    last_latency_ms: float = 0.0
    rx_times: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=40))
    proc_times: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=40))


def _cleanup_shm_obj(shm_obj: Any) -> None:
    if shm_obj is None:
        return
    try:
        shm_obj.close()
    except Exception:
        pass
    try:
        shm_obj.unlink()
    except FileNotFoundError:
        pass
    except Exception:
        pass


def _decode_shm_to_bgr(
    shm_buf: Any,
    buf_offset: int,
    h: int,
    w: int,
    step: int,
    enc: str,
) -> np.ndarray:
    ch = 1 if enc in ('mono8', '8uc1') else (4 if enc in ('bgra8', 'rgba8') else 3)
    raw_view = np.ndarray((h, step), dtype=np.uint8, buffer=shm_buf, offset=buf_offset)
    mat = raw_view[:, :w * ch]
    if ch == 1:
        return cv2.cvtColor(mat.reshape((h, w)), cv2.COLOR_GRAY2BGR)
    arr = mat.reshape((h, w, ch))
    if enc in ('bgr8', '8uc3'):
        return np.ascontiguousarray(arr)
    if enc == 'rgb8':
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if enc == 'bgra8':
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    if enc == 'rgba8':
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return np.ascontiguousarray(arr)


def _build_worker_detector(det_cfg: Dict[str, Any]):
    detector_type = str(det_cfg.get('detector_type', 'color')).lower()
    enable_multi_sig = bool(det_cfg.get('enable_multi_signature', True))
    profiles_json = str(det_cfg.get('profiles_json', ''))
    custom_profiles = profiles_from_json(profiles_json)
    min_area = float(det_cfg.get('min_area', 24.0))
    enable_glare = bool(det_cfg.get('enable_glare_suppression', True))
    if custom_profiles is not None:
        return MultiSignatureDetector(
            profiles=custom_profiles,
            min_area=min_area,
            enable_glare_suppression=enable_glare,
        )
    if enable_multi_sig or detector_type == 'multi_color':
        return MultiSignatureDetector(
            min_area=min_area,
            enable_glare_suppression=enable_glare,
        )
    return ColorBlobDetector(
        hsv_lower1=tuple(det_cfg.get('hsv_lower1', (0, 110, 55))),
        hsv_upper1=tuple(det_cfg.get('hsv_upper1', (12, 255, 255))),
        hsv_lower2=tuple(det_cfg.get('hsv_lower2', (168, 110, 55))),
        hsv_upper2=tuple(det_cfg.get('hsv_upper2', (180, 255, 255))),
        min_area=min_area,
        enable_glare_suppression=enable_glare,
    )


def _shm_worker_main(
    conn: Any,
    shm_name: str,
    slot_cap: int,
    drone_slot_index_map: Dict[str, int],
    det_cfg: Dict[str, Any],
    cv2_threads: int,
) -> None:
    try:
        from multiprocessing import resource_tracker
        resource_tracker.unregister(f'/{shm_name}', 'shared_memory')
    except Exception:
        pass

    if int(cv2_threads) >= 0:
        try:
            cv2.setNumThreads(int(cv2_threads))
        except Exception:
            pass

    shm = None
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        detector = _build_worker_detector(det_cfg)
        prev_gray_by_drone: Dict[str, np.ndarray] = {}

        while True:
            try:
                item = conn.recv()
            except (EOFError, BrokenPipeError, OSError):
                break
            if item is None:
                break

            drone_id, seq, write_buf, h, w, step, enc, raw_fallback = item
            if raw_fallback is not None:
                bgr = raw_fallback
            else:
                s_idx = drone_slot_index_map[drone_id]
                buf_offset = (s_idx * 2 + int(write_buf)) * slot_cap
                bgr = _decode_shm_to_bgr(shm.buf, buf_offset, int(h), int(w), int(step), str(enc))

            prev_g = prev_gray_by_drone.get(drone_id)
            dets = detector.detect_targets(bgr, prev_gray=prev_g)
            prev_gray_by_drone[drone_id] = (
                cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
            )
            compact_dets = [
                (float(d.u), float(d.v), str(d.class_id), float(d.conf), float(d.area_px))
                for d in dets
            ]
            t_end = time.monotonic()
            try:
                conn.send(('RES', drone_id, seq, compact_dets, t_end))
            except (BrokenPipeError, OSError):
                break
    finally:
        try:
            conn.close()
        except Exception:
            pass
        if shm is not None:
            try:
                shm.close()
            except Exception:
                pass


def _resolve_drone_ids(
    drone_ids: Sequence[str],
    drone_names: Sequence[str],
    num_drones: int,
) -> List[str]:
    cleaned_names = [str(d).strip() for d in drone_names if str(d).strip()]
    if cleaned_names:
        return cleaned_names
    if int(num_drones) > 0:
        return [f'drone_{i}' for i in range(1, int(num_drones) + 1)]
    cleaned_ids = [str(d).strip() for d in drone_ids if str(d).strip()]
    return cleaned_ids if cleaned_ids else ['drone_1', 'drone_2', 'drone_3']


def _default_pinhole_k(width: int, height: int, hfov_rad: float) -> Tuple[float, ...]:
    w = max(float(width), 1.0)
    h = max(float(height), 1.0)
    hfov = min(max(float(hfov_rad), 0.2), math.pi - 0.1)
    fx = (0.5 * w) / math.tan(0.5 * hfov)
    fy = fx
    cx = 0.5 * w
    cy = 0.5 * h
    return (fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)


def _discover_default_obstacles_yaml() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        cand = os.path.join(get_package_share_directory('flight_ctrl_pkg'), 'config', 'obstacles.yaml')
        if os.path.isfile(cand):
            return cand
    except Exception:
        pass
    local_cand = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', '..', 'flight_ctrl_pkg', 'config', 'obstacles.yaml')
    )
    if os.path.isfile(local_cand):
        return local_cand
    return ''


def _imgmsg_to_bgr(msg: Image, bridge: CvBridge) -> np.ndarray:
    """线程安全、零锁的高速 sensor_msgs/Image -> BGR uint8 转换，必要时回退至 CvBridge。"""
    enc = str(msg.encoding).lower()
    h = int(msg.height)
    w = int(msg.width)
    step = int(msg.step)
    if h > 0 and w > 0 and enc in ('bgr8', 'rgb8', 'bgra8', 'rgba8', 'mono8', '8uc3', '8uc1'):
        ch = 1 if enc in ('mono8', '8uc1') else (4 if enc in ('bgra8', 'rgba8') else 3)
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        row_bytes = w * ch
        if step >= row_bytes and raw.size >= h * step:
            mat = raw[:h * step].reshape((h, step))[:, :row_bytes]
            if ch == 1:
                gray = mat.reshape((h, w))
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            arr = mat.reshape((h, w, ch))
            if enc in ('bgr8', '8uc3'):
                return np.ascontiguousarray(arr)
            if enc == 'rgb8':
                return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            if enc == 'bgra8':
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            if enc == 'rgba8':
                return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')


def _compute_rate_hz(times: Deque[float], now_s: float) -> float:
    if len(times) < 2:
        return 0.0
    dt = times[-1] - times[0]
    if dt <= 1e-6 or (now_s - times[-1]) > 1.5:
        return 0.0
    return float((len(times) - 1) / dt)


class VisionNode(Node):
    """多机并行多目标检测 + 全配对极线三角测量与 3D 障碍物 LoS 检验视觉节点。"""

    def __init__(self) -> None:
        super().__init__('vision_node')

        # ------------------------------ 参数声明 ------------------------------
        self.declare_parameter('drone_ids', ['drone_1', 'drone_2', 'drone_3'])
        self.declare_parameter('drone_names', [''])
        self.declare_parameter('num_drones', 0)

        self.declare_parameter('process_mode', 'threaded_batch')
        self.declare_parameter('max_workers', 0)
        self.declare_parameter('cv2_num_threads', 1)
        self.declare_parameter('per_stream_hz', 15.0)
        self.declare_parameter('target_publish_hz', 20.0)
        self.declare_parameter('process_every_n_frames', 1)

        self.declare_parameter('subscribe_compat_aliases', True)
        self.declare_parameter('allow_default_camera_info', True)
        self.declare_parameter('default_hfov_rad', 1.35)

        self.declare_parameter('detector_type', 'color')
        self.declare_parameter('enable_multi_signature', True)
        self.declare_parameter('detector_profiles_json', '')
        self.declare_parameter('ground_z', 5.5)

        # 颜色检测参数（兼容单红配置）
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

        # 多目标立体三角化与 3D 障碍物 LoS 配置
        self.declare_parameter('enable_multi_target', True)
        self.declare_parameter('enable_triangulation', True)
        self.declare_parameter('triangulation_sync_sec', 0.18)
        self.declare_parameter('min_parallax_deg', 8.0)
        self.declare_parameter('max_skew_rms_m', 0.55)
        self.declare_parameter('min_baseline_m', 3.4)
        self.declare_parameter('min_depth_m', 0.4)
        self.declare_parameter('max_depth_m', 180.0)
        self.declare_parameter('target_z_min', 1.5)
        self.declare_parameter('target_z_max', 35.0)
        self.declare_parameter('cluster_radius_m', 1.5)
        self.declare_parameter('min_cluster_views', 2)
        self.declare_parameter('allow_monocular_fallback', True)
        self.declare_parameter('stereo_lock_sec', 0.85)
        self.declare_parameter('monocular_min_down_cos', 0.36)
        self.declare_parameter('obstacles_yaml_path', '')
        self.declare_parameter('obstacles_json', '')
        self.declare_parameter('enable_los_check', True)
        self.declare_parameter('los_clearance_margin_m', 0.0)

        p = self.get_parameter
        self.drone_ids: List[str] = _resolve_drone_ids(
            p('drone_ids').value,
            p('drone_names').value,
            int(p('num_drones').value),
        )
        self.process_mode: str = str(p('process_mode').value)
        req_workers = int(p('max_workers').value)
        self.cv2_num_threads: int = int(p('cv2_num_threads').value)
        if self.cv2_num_threads >= 0:
            try:
                cv2.setNumThreads(self.cv2_num_threads)
            except Exception:
                pass

        self.per_stream_hz: float = float(p('per_stream_hz').value)
        self.target_publish_hz: float = max(15.0, float(p('target_publish_hz').value))
        self.frame_skip: int = max(1, int(p('process_every_n_frames').value))

        self.subscribe_compat_aliases: bool = bool(p('subscribe_compat_aliases').value)
        self.allow_default_camera_info: bool = bool(p('allow_default_camera_info').value)
        self.default_hfov_rad: float = float(p('default_hfov_rad').value)

        self.ground_z: float = float(p('ground_z').value)
        self._adaptive_target_z: float = self.ground_z
        self._adaptive_z_by_class: Dict[str, float] = {}

        rpy = list(p('camera_mount_rpy').value)
        self.R_bc: np.ndarray = camera_mount_matrix(float(rpy[0]), float(rpy[1]), float(rpy[2]))
        off = list(p('camera_offset_body').value)
        self.cam_offset_body: Tuple[float, float, float] = (float(off[0]), float(off[1]), float(off[2]))

        self.enable_multi_target: bool = bool(p('enable_multi_target').value)
        self.enable_triangulation: bool = bool(p('enable_triangulation').value)
        self.tri_sync_sec: float = float(p('triangulation_sync_sec').value)
        self.min_parallax_deg: float = float(p('min_parallax_deg').value)
        self.max_skew_rms_m: float = float(p('max_skew_rms_m').value)
        self.min_baseline_m: float = float(p('min_baseline_m').value)
        self.min_depth_m: float = float(p('min_depth_m').value)
        self.max_depth_m: float = float(p('max_depth_m').value)
        self.target_z_min: float = float(p('target_z_min').value)
        self.target_z_max: float = float(p('target_z_max').value)
        self.cluster_radius_m: float = float(p('cluster_radius_m').value)
        self.min_cluster_views: int = max(2, int(p('min_cluster_views').value))
        self.allow_monocular_fallback: bool = bool(p('allow_monocular_fallback').value)
        self.stereo_lock_sec: float = float(p('stereo_lock_sec').value)
        self.monocular_min_down_cos: float = float(p('monocular_min_down_cos').value)

        self.enable_los_check: bool = bool(p('enable_los_check').value)
        self.los_clearance_margin_m: float = float(p('los_clearance_margin_m').value)
        self.obstacles: List[Obstacle3D] = []
        self.arena_bounds: Optional[Dict[str, float]] = None

        obs_json = str(p('obstacles_json').value).strip()
        obs_yaml = str(p('obstacles_yaml_path').value).strip()
        if obs_json:
            self.obstacles, self.arena_bounds = load_obstacles_from_json(obs_json)
        else:
            if not obs_yaml:
                obs_yaml = _discover_default_obstacles_yaml()
            if obs_yaml:
                self.obstacles, self.arena_bounds = load_obstacles_from_yaml(obs_yaml)

        # ------------------------------ 检测器初始化 ------------------------------
        detector_type = str(p('detector_type').value).lower()
        enable_multi_sig = bool(p('enable_multi_signature').value)
        profiles_json = str(p('detector_profiles_json').value)
        custom_profiles = profiles_from_json(profiles_json)

        det_cfg: Dict[str, Any] = {
            'detector_type': detector_type,
            'enable_multi_signature': enable_multi_sig,
            'profiles_json': profiles_json,
            'min_area': float(p('min_area').value),
            'enable_glare_suppression': bool(p('enable_glare_suppression').value),
            'hsv_lower1': tuple(int(v) for v in p('hsv_lower1').value),
            'hsv_upper1': tuple(int(v) for v in p('hsv_upper1').value),
            'hsv_lower2': tuple(int(v) for v in p('hsv_lower2').value),
            'hsv_upper2': tuple(int(v) for v in p('hsv_upper2').value),
        }

        def _make_color_detector():
            if custom_profiles is not None:
                return MultiSignatureDetector(
                    profiles=custom_profiles,
                    min_area=float(p('min_area').value),
                    enable_glare_suppression=bool(p('enable_glare_suppression').value),
                )
            if enable_multi_sig or detector_type == 'multi_color':
                return MultiSignatureDetector(
                    min_area=float(p('min_area').value),
                    enable_glare_suppression=bool(p('enable_glare_suppression').value),
                )
            return ColorBlobDetector(
                hsv_lower1=tuple(int(v) for v in p('hsv_lower1').value),
                hsv_upper1=tuple(int(v) for v in p('hsv_upper1').value),
                hsv_lower2=tuple(int(v) for v in p('hsv_lower2').value),
                hsv_upper2=tuple(int(v) for v in p('hsv_upper2').value),
                min_area=float(p('min_area').value),
                enable_glare_suppression=bool(p('enable_glare_suppression').value),
            )

        yolo_active = False
        if detector_type == 'yolo':
            self.detector = YoloDetector(
                model_path=str(p('yolo_model_path').value),
                conf_thres=float(p('yolo_conf').value),
                target_class=int(p('yolo_target_class').value),
            )
            if not self.detector.available:
                self.get_logger().error(
                    'YOLO 初始化失败（未安装 ultralytics 或模型不存在），自动回退至 ColorBlob/MultiSignature 检测器'
                )
                self.detector = _make_color_detector()
            else:
                yolo_active = True
        else:
            self.detector = _make_color_detector()

        self._has_detect_targets = hasattr(self.detector, 'detect_targets')
        self._use_sim_time = bool(self.get_clock().ros_time_is_active)

        # ------------------------------ 状态槽与并行池 ------------------------------
        self.drones: Dict[str, _DroneSlot] = {
            d: _DroneSlot(drone_id=d, slot_idx=idx) for idx, d in enumerate(self.drone_ids)
        }
        self.bridge = CvBridge()
        self._cb_group = ReentrantCallbackGroup()
        self._fusion_lock = threading.Lock()

        self._last_diag_sec: float = 0.0
        self._last_stereo_sec: float = -1e9
        self._last_loc_mode: str = 'idle'
        self._last_skew_rms: float = 0.0
        self._stereo_hits: int = 0
        self._mono_hits: int = 0
        self._los_rejected_count: int = 0

        self._shutdown_flag = False
        self._executor_pool: Optional[ThreadPoolExecutor] = None
        self._slot_cap: int = 640 * 480 * 4
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._shm_finalizer = None
        self._mp_workers: List[mp.Process] = []
        self._mp_conns: List[Any] = []
        self._mp_pipe_locks: List[threading.Lock] = []
        self._collector_threads: List[threading.Thread] = []
        self._drone_to_worker: Dict[str, int] = {}
        self._mp_enabled: bool = False

        # ------------------------------ 订阅与发布（先完成 DDS 注册，再启动工作池） ------------------------------
        for drone_id in self.drone_ids:
            self.create_subscription(
                Image,
                f'/{drone_id}/camera/image_raw',
                self._make_image_cb(drone_id),
                qos_profile_sensor_data,
                callback_group=self._cb_group,
            )
            self.create_subscription(
                CameraInfo,
                f'/{drone_id}/camera/camera_info',
                self._make_info_cb(drone_id),
                qos_profile_sensor_data,
                callback_group=self._cb_group,
            )
            self.create_subscription(
                Odometry,
                f'/{drone_id}/state',
                self._make_odom_cb(drone_id),
                10,
                callback_group=self._cb_group,
            )
            if self.subscribe_compat_aliases:
                self.create_subscription(
                    Image,
                    f'/{drone_id}/down_cam/image_raw',
                    self._make_image_cb(drone_id),
                    qos_profile_sensor_data,
                    callback_group=self._cb_group,
                )
                self.create_subscription(
                    CameraInfo,
                    f'/{drone_id}/down_cam/camera_info',
                    self._make_info_cb(drone_id),
                    qos_profile_sensor_data,
                    callback_group=self._cb_group,
                )
                self.create_subscription(
                    PoseStamped,
                    f'/{drone_id}/pose',
                    self._make_pose_cb(drone_id),
                    10,
                    callback_group=self._cb_group,
                )

        self.targets_pub = self.create_publisher(String, '/vision/targets', 10)
        self.target_pub = self.create_publisher(PoseStamped, '/vision/target_pose', 10)
        self.diag_pub = self.create_publisher(String, '/vision/diagnostics', 10)

        is_parallel_mode = self.process_mode in ('threaded_batch', 'parallel', 'mp_batch', 'process_pool')
        if is_parallel_mode and yolo_active:
            self.max_workers = req_workers if req_workers > 0 else min(12, max(4, len(self.drone_ids)))
            self._executor_pool = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix='vision_worker',
            )
        elif is_parallel_mode:
            n_d = max(1, len(self.drone_ids))
            if 1 <= req_workers <= 5:
                w_cnt = min(req_workers, n_d)
            else:
                w_cnt = min(5, max(2, (n_d + 1) // 2))
            self.max_workers = w_cnt
            try:
                self._shm = shared_memory.SharedMemory(create=True, size=n_d * 2 * self._slot_cap)
                self._shm_finalizer = weakref.finalize(self, _cleanup_shm_obj, self._shm)
                drone_slot_index_map = {d: idx for idx, d in enumerate(self.drone_ids)}
                for idx, d in enumerate(self.drone_ids):
                    self._drone_to_worker[d] = idx % w_cnt
                ctx = mp.get_context('spawn')
                for w_idx in range(w_cnt):
                    parent_conn, child_conn = ctx.Pipe(duplex=True)
                    proc = ctx.Process(
                        target=_shm_worker_main,
                        args=(
                            child_conn,
                            self._shm.name,
                            self._slot_cap,
                            drone_slot_index_map,
                            det_cfg,
                            self.cv2_num_threads,
                        ),
                        daemon=True,
                    )
                    proc.start()
                    child_conn.close()
                    self._mp_workers.append(proc)
                    self._mp_conns.append(parent_conn)
                    self._mp_pipe_locks.append(threading.Lock())
                for w_idx, parent_conn in enumerate(self._mp_conns):
                    t_col = threading.Thread(
                        target=self._collector_loop,
                        args=(w_idx, parent_conn),
                        name=f'vision_col_{w_idx}',
                        daemon=True,
                    )
                    t_col.start()
                    self._collector_threads.append(t_col)
                self._mp_enabled = True
            except Exception:
                self._shutdown_mp_pool()
                raise
        else:
            self.max_workers = req_workers if req_workers > 0 else min(12, max(4, len(self.drone_ids)))

        self._rr_index = 0
        self._rr_timer = None
        self._fusion_timer = None
        if self.process_mode == 'round_robin':
            n = max(len(self.drone_ids), 1)
            period = 1.0 / max(self.per_stream_hz * n, 1.0)
            self._rr_timer = self.create_timer(period, self.round_robin_tick, callback_group=self._cb_group)
        else:
            fusion_period = 1.0 / self.target_publish_hz
            self._fusion_timer = self.create_timer(
                fusion_period, self._fusion_timer_cb, callback_group=self._cb_group
            )

        self.get_logger().info(
            f'vision_node 已启动: mode={self.process_mode}, drones={self.drone_ids}, '
            f'workers={self.max_workers}, cv2_threads={self.cv2_num_threads}, '
            f'publish_hz={self.target_publish_hz:.1f}, obstacles={len(self.obstacles)}'
        )

    def _now_sec(self) -> float:
        if self._use_sim_time:
            return self.get_clock().now().nanoseconds / 1e9
        return time.time()

    def _compute_aligned_stamp(self, image_msg: Image, now_s: float) -> Tuple[object, float]:
        img_s = int(image_msg.header.stamp.sec)
        img_ns = int(image_msg.header.stamp.nanosec)
        if img_s == 0 and img_ns == 0:
            return self.get_clock().now().to_msg(), now_s
        raw_img_sec = img_s + img_ns * 1e-9
        if abs(now_s - raw_img_sec) > 60.0:
            return self.get_clock().now().to_msg(), now_s
        return image_msg.header.stamp, raw_img_sec

    def _send_worker_task(self, worker_idx: int, task: Any) -> bool:
        if self._shutdown_flag or worker_idx >= len(self._mp_conns):
            return False
        conn = self._mp_conns[worker_idx]
        p_lock = self._mp_pipe_locks[worker_idx]
        try:
            with p_lock:
                conn.send(task)
            return True
        except (BrokenPipeError, OSError):
            return False

    def _collector_loop(self, worker_idx: int, conn: Any) -> None:
        while not self._shutdown_flag:
            try:
                if not conn.poll(0.05):
                    continue
                msg = conn.recv()
            except (EOFError, BrokenPipeError, OSError):
                break
            if not isinstance(msg, tuple) or len(msg) != 5 or msg[0] != 'RES':
                continue
            _, drone_id, seq, compact_dets, t_end = msg
            slot = self.drones.get(drone_id)
            if slot is None:
                continue

            dispatch_next = None
            with slot.lock:
                ctx = slot.inflight_ctx
                slot.inflight_ctx = None
                if slot.pending_meta is not None and not self._shutdown_flag:
                    next_task, next_ctx = slot.pending_meta
                    slot.pending_meta = None
                    if next_task[2] >= 0:
                        slot.worker_buf = next_task[2]
                    slot.inflight_ctx = next_ctx
                    dispatch_next = next_task
                else:
                    slot.in_flight = False

            if ctx is None:
                if dispatch_next is not None and not self._shutdown_flag:
                    self._send_worker_task(worker_idx, dispatch_next)
                continue

            info_k, state_snap, aligned_stamp_sec, aligned_stamp_msg, _, is_mock_cam, rx_wall = ctx
            _, pos_tuple, quat_tuple = state_snap
            eff_offset = (0.0, 0.0, 0.0) if is_mock_cam else self.cam_offset_body

            ray_list: List[RayObservation] = []
            for idx, (du, dv, d_cls, d_conf, d_area) in enumerate(compact_dets):
                if not self.enable_multi_target and idx > 0:
                    break
                ray = pixel_to_ray_world(
                    du,
                    dv,
                    info_k,
                    pos_tuple,
                    quat_tuple,
                    self.R_bc,
                    cam_offset_body=eff_offset,
                )
                if ray is None:
                    continue
                ray_list.append(
                    RayObservation(
                        drone_id=drone_id,
                        origin=ray[0],
                        direction=ray[1],
                        class_id=d_cls,
                        conf=float(d_conf),
                        timestamp=aligned_stamp_sec,
                        det_id=idx,
                        pixel_uv=(du, dv),
                        area_px=float(d_area),
                    )
                )

            r_now_s = self._now_sec()
            latency_ms = (t_end - (rx_wall if rx_wall > 0.0 else t_end)) * 1000.0
            ray_bundle = (r_now_s, aligned_stamp_sec, aligned_stamp_msg, is_mock_cam, ray_list)
            with slot.lock:
                slot.latest_rays = ray_bundle
                slot.proc_frames += 1
                slot.proc_seq = seq
                slot.proc_times.append(t_end)
                slot.last_detections = len(ray_list)
                slot.last_latency_ms = latency_ms

            if dispatch_next is not None and not self._shutdown_flag:
                self._send_worker_task(worker_idx, dispatch_next)

    def _shutdown_mp_pool(self) -> None:
        self._shutdown_flag = True
        for w_idx, conn in enumerate(self._mp_conns):
            p_lock = self._mp_pipe_locks[w_idx] if w_idx < len(self._mp_pipe_locks) else None
            try:
                if p_lock is not None:
                    with p_lock:
                        conn.send(None)
                else:
                    conn.send(None)
            except Exception:
                pass
        for t_col in self._collector_threads:
            try:
                t_col.join(timeout=0.4)
            except Exception:
                pass
        self._collector_threads.clear()
        for proc in self._mp_workers:
            try:
                proc.join(timeout=0.4)
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=0.3)
            except Exception:
                pass
        self._mp_workers.clear()
        for conn in self._mp_conns:
            try:
                conn.close()
            except Exception:
                pass
        self._mp_conns.clear()
        self._mp_pipe_locks.clear()
        if self._shm_finalizer is not None:
            try:
                self._shm_finalizer()
            except Exception:
                pass
            self._shm_finalizer = None
        if self._shm is not None:
            _cleanup_shm_obj(self._shm)
            self._shm = None

    def _make_image_cb(self, drone_id: str):
        def cb(msg: Image) -> None:
            if self._shutdown_flag:
                return
            slot = self.drones[drone_id]
            now_wall = time.monotonic()

            if self._mp_enabled and self._shm is not None:
                now_s = self._now_sec()
                is_mock_cam = (msg.header.frame_id == 'camera_link')
                lag = 0.0 if is_mock_cam else 0.065
                with slot.lock:
                    slot.rx_frames += 1
                    slot.rx_times.append(now_wall)
                    if slot.rx_frames % self.frame_skip != 0:
                        return
                    info_k = slot.info_k
                    if info_k is None and self.allow_default_camera_info and msg.width > 0 and msg.height > 0:
                        info_k = _default_pinhole_k(msg.width, msg.height, self.default_hfov_rad)
                    state_snap = self._match_state_from_snapshot(
                        slot.latest_state,
                        slot.state_buf,
                        msg.header.stamp,
                        now_s,
                        render_lag_sec=lag,
                    )

                if info_k is None:
                    self.get_logger().warn(f'[{drone_id}] 等待 camera_info ...', throttle_duration_sec=3.0)
                    return
                if state_snap is None:
                    self.get_logger().warn(
                        f'[{drone_id}] 等待无人机位姿 /state 或 /pose ...', throttle_duration_sec=3.0
                    )
                    return

                aligned_stamp_msg, aligned_stamp_sec = self._compute_aligned_stamp(msg, now_s)
                enc = str(msg.encoding).lower()
                h = int(msg.height)
                w = int(msg.width)
                step = int(msg.step)
                ch = 1 if enc in ('mono8', '8uc1') else (4 if enc in ('bgra8', 'rgba8') else 3)
                need = h * step
                can_shm = (
                    h > 0
                    and w > 0
                    and enc in ('bgr8', 'rgb8', 'bgra8', 'rgba8', 'mono8', '8uc3', '8uc1')
                    and step >= w * ch
                    and 0 < need <= self._slot_cap
                    and len(msg.data) >= need
                )
                raw_fallback = None
                if not can_shm:
                    try:
                        raw_fallback = _imgmsg_to_bgr(msg, self.bridge)
                    except Exception as exc:
                        self.get_logger().error(f'[{drone_id}] 图像转换失败: {exc}', throttle_duration_sec=2.0)
                        return

                dispatch_task = None
                with slot.lock:
                    if self._shutdown_flag:
                        return
                    slot.rx_seq += 1
                    seq = slot.rx_seq
                    if can_shm:
                        write_buf = (1 - slot.worker_buf) if slot.in_flight else 0
                        buf_off = (slot.slot_idx * 2 + write_buf) * self._slot_cap
                        dst = np.ndarray((need,), dtype=np.uint8, buffer=self._shm.buf, offset=buf_off)
                        src = np.frombuffer(msg.data, dtype=np.uint8, count=need)
                        np.copyto(dst, src)
                    else:
                        write_buf = -1

                    if slot.pending_meta is not None:
                        slot.coalesced_drops += 1

                    ctx = (
                        info_k,
                        state_snap,
                        aligned_stamp_sec,
                        aligned_stamp_msg,
                        now_s,
                        is_mock_cam,
                        now_wall,
                    )
                    task = (drone_id, seq, write_buf, h, w, step, enc, raw_fallback)
                    if not slot.in_flight:
                        slot.in_flight = True
                        slot.worker_buf = max(0, write_buf)
                        slot.inflight_ctx = ctx
                        dispatch_task = task
                    else:
                        slot.pending_meta = (task, ctx)

                if dispatch_task is not None and not self._shutdown_flag:
                    w_idx = self._drone_to_worker[drone_id]
                    if not self._send_worker_task(w_idx, dispatch_task):
                        with slot.lock:
                            slot.in_flight = False
                            slot.inflight_ctx = None
                return

            should_submit = False
            with slot.lock:
                slot.rx_frames += 1
                slot.rx_times.append(now_wall)
                if slot.rx_frames % self.frame_skip != 0:
                    return
                if slot.latest_image is not None:
                    slot.coalesced_drops += 1
                slot.latest_image = msg
                slot.latest_image_rx_wall = now_wall
                slot.rx_seq += 1
                if (
                    self.process_mode in ('threaded_batch', 'parallel')
                    and self._executor_pool is not None
                    and not slot.in_flight
                ):
                    slot.in_flight = True
                    should_submit = True

            if should_submit and self._executor_pool is not None and not self._shutdown_flag:
                try:
                    self._executor_pool.submit(self._worker_run, drone_id)
                except RuntimeError:
                    with slot.lock:
                        slot.in_flight = False

        return cb

    def _make_info_cb(self, drone_id: str):
        def cb(msg: CameraInfo) -> None:
            k_tuple = tuple(float(v) for v in msg.k)
            if len(k_tuple) == 9 and k_tuple[0] > 1e-6 and k_tuple[4] > 1e-6:
                slot = self.drones[drone_id]
                with slot.lock:
                    slot.info_k = k_tuple

        return cb

    def _store_state(
        self,
        drone_id: str,
        stamp,
        px: float,
        py: float,
        pz: float,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> None:
        ts = int(stamp.sec) + int(stamp.nanosec) * 1e-9
        if ts <= 0.0:
            ts = self._now_sec()
        item = (ts, (float(px), float(py), float(pz)), (float(qx), float(qy), float(qz), float(qw)))
        slot = self.drones[drone_id]
        with slot.lock:
            slot.latest_state = item
            slot.state_buf.append(item)

    def _make_odom_cb(self, drone_id: str):
        def cb(msg: Odometry) -> None:
            pose = msg.pose.pose
            self._store_state(
                drone_id,
                msg.header.stamp,
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )

        return cb

    def _make_pose_cb(self, drone_id: str):
        def cb(msg: PoseStamped) -> None:
            pose = msg.pose
            self._store_state(
                drone_id,
                msg.header.stamp,
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            )

        return cb

    @staticmethod
    def _match_state_from_snapshot(
        latest_state,
        state_buf: Sequence[Tuple[float, Tuple[float, float, float], Tuple[float, float, float, float]]],
        img_stamp,
        now_s: float,
        render_lag_sec: float = 0.065,
    ):
        if latest_state is None:
            return None
        s = int(img_stamp.sec)
        ns = int(img_stamp.nanosec)
        if not state_buf or (s == 0 and ns == 0):
            return latest_state
        img_ts = s + ns * 1e-9
        if abs(now_s - img_ts) > 60.0:
            return latest_state
        target_ts = img_ts - render_lag_sec
        best = latest_state
        best_dt = float('inf')
        for item in reversed(state_buf):
            dt = abs(item[0] - target_ts)
            if dt < best_dt:
                best_dt = dt
                best = item
            elif dt > best_dt + 0.02:
                break
        return best

    def _detect_and_project(
        self,
        drone_id: str,
        image_msg: Image,
        rx_wall: float,
    ) -> Optional[Tuple[float, float, object, bool, List[RayObservation]]]:
        """在单路工作线程或同步调用中完成解码、多谱系检测与射线反投影。"""
        slot = self.drones[drone_id]
        t_start = time.monotonic()
        now_s = self._now_sec()
        is_mock_cam = (image_msg.header.frame_id == 'camera_link')
        lag = 0.0 if is_mock_cam else 0.065

        with slot.lock:
            info_k = slot.info_k
            if info_k is None and self.allow_default_camera_info and image_msg.width > 0 and image_msg.height > 0:
                info_k = _default_pinhole_k(image_msg.width, image_msg.height, self.default_hfov_rad)
            state_snap = self._match_state_from_snapshot(
                slot.latest_state,
                slot.state_buf,
                image_msg.header.stamp,
                now_s,
                render_lag_sec=lag,
            )
            prev_gray = slot.prev_gray

        if info_k is None:
            self.get_logger().warn(f'[{drone_id}] 等待 camera_info ...', throttle_duration_sec=3.0)
            return None
        if state_snap is None:
            self.get_logger().warn(f'[{drone_id}] 等待无人机位姿 /state 或 /pose ...', throttle_duration_sec=3.0)
            return None

        try:
            cv_image = _imgmsg_to_bgr(image_msg, self.bridge)
        except Exception as exc:
            self.get_logger().error(f'[{drone_id}] 图像转换失败: {exc}', throttle_duration_sec=2.0)
            return None

        if self._has_detect_targets:
            dets: List[Detection] = self.detector.detect_targets(cv_image, prev_gray=prev_gray)
        else:
            uv_list = self.detector.detect_all(cv_image)
            dets = [
                Detection(
                    u=float(u),
                    v=float(v),
                    area_px=36.0,
                    bbox_span_px=6.0,
                    bbox=(int(u), int(v), 6, 6),
                    class_id='crimson_uav',
                    conf=0.85,
                    aspect_ratio=1.0,
                    extent=1.0,
                    ring_delta_e=0.0,
                )
                for u, v in uv_list
            ]

        if not self.enable_multi_target and len(dets) > 1:
            dets = [dets[0]]

        _, pos_tuple, quat_tuple = state_snap
        eff_offset = (0.0, 0.0, 0.0) if is_mock_cam else self.cam_offset_body

        # 计算对齐的图像采集时间戳（用于同步与下游延迟审计）
        aligned_stamp_msg, aligned_stamp_sec = self._compute_aligned_stamp(image_msg, now_s)

        ray_list: List[RayObservation] = []
        for idx, det in enumerate(dets):
            ray = pixel_to_ray_world(
                det.u,
                det.v,
                info_k,
                pos_tuple,
                quat_tuple,
                self.R_bc,
                cam_offset_body=eff_offset,
            )
            if ray is None:
                continue
            ray_list.append(
                RayObservation(
                    drone_id=drone_id,
                    origin=ray[0],
                    direction=ray[1],
                    class_id=det.class_id,
                    conf=float(det.conf),
                    timestamp=aligned_stamp_sec,
                    det_id=idx,
                    pixel_uv=(det.u, det.v),
                    area_px=float(det.area_px),
                )
            )

        t_end = time.monotonic()
        latency_ms = (t_end - (rx_wall if rx_wall > 0.0 else t_start)) * 1000.0
        ray_bundle = (now_s, aligned_stamp_sec, aligned_stamp_msg, is_mock_cam, ray_list)

        cur_gray_next = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) if cv_image.ndim == 3 else cv_image
        with slot.lock:
            slot.prev_gray = cur_gray_next
            slot.latest_rays = ray_bundle
            slot.proc_frames += 1
            slot.proc_times.append(t_end)
            slot.last_detections = len(ray_list)
            slot.last_latency_ms = latency_ms

        return ray_bundle

    def _worker_run(self, drone_id: str) -> None:
        slot = self.drones[drone_id]
        try:
            for _ in range(2):
                if self._shutdown_flag:
                    break
                with slot.lock:
                    if slot.latest_image is None or slot.proc_seq >= slot.rx_seq:
                        break
                    msg = slot.latest_image
                    slot.latest_image = None
                    rx_wall = slot.latest_image_rx_wall
                    target_seq = slot.rx_seq
                self._detect_and_project(drone_id, msg, rx_wall)
                with slot.lock:
                    slot.proc_seq = target_seq
                    if slot.proc_seq >= slot.rx_seq:
                        break
        finally:
            resubmit = False
            with slot.lock:
                if (
                    not self._shutdown_flag
                    and slot.latest_image is not None
                    and slot.proc_seq < slot.rx_seq
                    and self._executor_pool is not None
                ):
                    resubmit = True
                else:
                    slot.in_flight = False
            if resubmit and self._executor_pool is not None:
                try:
                    self._executor_pool.submit(self._worker_run, drone_id)
                except RuntimeError:
                    with slot.lock:
                        slot.in_flight = False

    def round_robin_tick(self) -> None:
        drone_id = self.drone_ids[self._rr_index % len(self.drone_ids)]
        self._rr_index += 1
        slot = self.drones[drone_id]
        with slot.lock:
            msg = slot.latest_image
            if msg is None:
                return
            stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)
            if stamp == slot.last_rr_stamp and slot.proc_seq >= slot.rx_seq:
                return
            slot.latest_image = None
            slot.last_rr_stamp = stamp
            slot.proc_seq = slot.rx_seq
        self.process_frame(drone_id, msg)

    def _fusion_timer_cb(self) -> None:
        if self._shutdown_flag:
            return
        self._run_fusion_and_publish(trigger_msg=None)

    def process_frame(self, drone_id: str, image_msg: Image) -> None:
        """同步处理单路图像并立即执行多机融合发布（保持原有测试与单线程调用完全兼容）。"""
        if drone_id not in self.drones:
            return
        self._detect_and_project(drone_id, image_msg, time.monotonic())
        self._run_fusion_and_publish(trigger_msg=image_msg)

    def _run_fusion_and_publish(self, trigger_msg: Optional[Image] = None) -> None:
        with self._fusion_lock:
            now_s = self._now_sec()
            now_wall = time.monotonic()

            fresh_rays: List[RayObservation] = []
            active_cams_with_rays: List[str] = []
            any_active_cam = False
            any_mock_cam = False
            freshest_stamp_sec = -1.0
            freshest_stamp_msg = None

            for d_id in self.drone_ids:
                slot = self.drones[d_id]
                with slot.lock:
                    bundle = slot.latest_rays
                if bundle is None:
                    continue
                any_active_cam = True
                r_wall_ts, img_stamp_sec, img_stamp_msg, is_mock, r_list = bundle
                if (now_s - r_wall_ts) <= self.tri_sync_sec:
                    if is_mock:
                        any_mock_cam = True
                    if r_wall_ts >= freshest_stamp_sec:
                        freshest_stamp_sec = r_wall_ts
                        freshest_stamp_msg = img_stamp_msg
                    if r_list:
                        fresh_rays.extend(r_list)
                        active_cams_with_rays.append(d_id)

            if not any_active_cam and trigger_msg is None:
                return

            if trigger_msg is not None:
                t_s = int(trigger_msg.header.stamp.sec)
                t_ns = int(trigger_msg.header.stamp.nanosec)
                if t_s == 0 and t_ns == 0:
                    out_stamp_msg = self.get_clock().now().to_msg()
                    out_stamp_sec = now_s
                else:
                    raw_t = t_s + t_ns * 1e-9
                    if abs(now_s - raw_t) > 60.0:
                        out_stamp_msg = self.get_clock().now().to_msg()
                        out_stamp_sec = now_s
                    else:
                        out_stamp_msg = trigger_msg.header.stamp
                        out_stamp_sec = raw_t
            elif freshest_stamp_msg is not None:
                out_stamp_msg = freshest_stamp_msg
                out_stamp_sec = int(freshest_stamp_msg.sec) + int(freshest_stamp_msg.nanosec) * 1e-9
            else:
                out_stamp_msg = self.get_clock().now().to_msg()
                out_stamp_sec = now_s

            obstacles_active = self.obstacles if self.enable_los_check else []
            targets: List[TriangulatedTarget] = []
            prev_mode = self._last_loc_mode
            eff_baseline = 0.0 if any_mock_cam else self.min_baseline_m

            # 1) 多目标异面射线立体三角测量 + 3D LoS 门控
            distinct_drones = {r.drone_id for r in fresh_rays}
            if self.enable_triangulation and len(distinct_drones) >= self.min_cluster_views:
                targets = triangulate_multi_target(
                    fresh_rays,
                    obstacles=obstacles_active,
                    arena_bounds=self.arena_bounds,
                    triangulation_sync_sec=self.tri_sync_sec,
                    min_parallax_deg=self.min_parallax_deg,
                    max_skew_rms_m=self.max_skew_rms_m,
                    min_depth_m=self.min_depth_m,
                    max_depth_m=self.max_depth_m,
                    min_baseline_m=eff_baseline,
                    min_alt_m=self.target_z_min,
                    max_alt_m=self.target_z_max,
                    cluster_radius_m=self.cluster_radius_m,
                    min_cluster_views=self.min_cluster_views,
                    los_margin_m=self.los_clearance_margin_m,
                )

                if not self.enable_multi_target and len(targets) > 1:
                    targets = [targets[0]]

                if targets:
                    max_views = max(t.rays_used for t in targets)
                    mean_skew = sum(t.skew_rms_m for t in targets) / float(len(targets))
                    for t in targets:
                        prev_z = self._adaptive_z_by_class.get(t.class_id, self._adaptive_target_z)
                        self._adaptive_z_by_class[t.class_id] = 0.75 * prev_z + 0.25 * t.z
                    self._adaptive_target_z = 0.75 * self._adaptive_target_z + 0.25 * targets[0].z
                    self._last_stereo_sec = now_s
                    self._last_loc_mode = f'stereo_{max_views}cam'
                    self._last_skew_rms = mean_skew
                    self._stereo_hits += 1

            # 2) 单目参考平面回退（遵守立体主导锁及障碍物 LoS 防穿墙门控）
            los_rejected_now = 0
            if not targets and self.allow_monocular_fallback and fresh_rays:
                stereo_locked = (not any_mock_cam) and ((now_s - self._last_stereo_sec) < self.stereo_lock_sec)
                if not stereo_locked:
                    mono_candidates: List[TriangulatedTarget] = []
                    for r in fresh_rays:
                        if not any_mock_cam and (-float(r.direction[2])) < self.monocular_min_down_cos:
                            continue
                        ref_z = self._adaptive_z_by_class.get(r.class_id, self._adaptive_target_z)
                        dz = float(r.direction[2])
                        if abs(dz) < 1e-6:
                            continue
                        t_plane = (ref_z - float(r.origin[2])) / dz
                        if t_plane <= self.min_depth_m or t_plane > self.max_depth_m:
                            continue
                        pt = r.origin + t_plane * r.direction
                        if not np.all(np.isfinite(pt)):
                            continue
                        if obstacles_active and not has_line_of_sight(
                            r.origin,
                            pt,
                            obstacles_active,
                            margin_m=self.los_clearance_margin_m,
                        ):
                            los_rejected_now += 1
                            continue
                        # 空间去重（同类别不同机单目投影合并）
                        dup = False
                        for ex in mono_candidates:
                            if (
                                ex.class_id == r.class_id
                                and math.hypot(float(pt[0]) - ex.x, float(pt[1]) - ex.y) <= self.cluster_radius_m
                            ):
                                dup = True
                                break
                        if dup:
                            continue
                        mono_candidates.append(
                            TriangulatedTarget(
                                det_id=len(mono_candidates),
                                class_id=r.class_id,
                                x=float(pt[0]),
                                y=float(pt[1]),
                                z=float(pt[2]),
                                conf=float(min(0.65, max(0.20, 0.7 * r.conf))),
                                stereo=False,
                                rays_used=1,
                                parallax_deg=0.0,
                                skew_rms_m=0.0,
                                drone_ids=(r.drone_id,),
                            )
                        )
                        if not self.enable_multi_target:
                            break
                    if mono_candidates:
                        targets = mono_candidates
                        self._last_loc_mode = 'monocular_plane'
                        self._mono_hits += 1

            self._los_rejected_count += los_rejected_now

            # 3) 发布 /vision/targets (PROJECT.md § Interface Contracts #3)
            if self._shutdown_flag or not rclpy.ok():
                return
            targets_payload = {
                'stamp': round(float(out_stamp_sec), 6),
                'frame_id': 'world',
                'targets': [t.to_dict() for t in targets],
            }
            targets_msg = String()
            targets_msg.data = json.dumps(targets_payload)
            try:
                self.targets_pub.publish(targets_msg)
            except Exception:
                return

            # 4) 若存在有效目标，组装并发布向后兼容的 /vision/target_pose
            if targets:
                best = min(
                    targets,
                    key=lambda t: (
                        0 if t.stereo else 1,
                        _CLASS_PRIORITY.get(t.class_id, 5),
                        -t.conf,
                        -t.rays_used,
                    ),
                )
                pose_out = PoseStamped()
                pose_out.header.stamp = out_stamp_msg
                pose_out.header.frame_id = 'world'
                pose_out.pose.position.x = float(best.x)
                pose_out.pose.position.y = float(best.y)
                pose_out.pose.position.z = float(best.z)
                pose_out.pose.orientation.w = 1.0
                self.target_pub.publish(pose_out)

                self.get_logger().info(
                    f'[{",".join(active_cams_with_rays)}|{self._last_loc_mode}] '
                    f'检出 {len(targets)} 目标 -> 主目标 {best.class_id}: '
                    f'({best.x:.2f}, {best.y:.2f}, {best.z:.2f})',
                    throttle_duration_sec=1.0,
                )

            # 5) 发布 /vision/diagnostics（保留 mode, stereo_hits, mono_hits 等所有既有键）
            if (self._last_loc_mode != prev_mode) or (now_s - self._last_diag_sec >= 0.4):
                self._last_diag_sec = now_s
                per_cam: Dict[str, Dict[str, object]] = {}
                for d_id in self.drone_ids:
                    slot = self.drones[d_id]
                    with slot.lock:
                        per_cam[d_id] = {
                            'rx_hz': round(_compute_rate_hz(slot.rx_times, now_wall), 2),
                            'proc_hz': round(_compute_rate_hz(slot.proc_times, now_wall), 2),
                            'latency_ms': round(slot.last_latency_ms, 2),
                            'detections': int(slot.last_detections),
                            'coalesced_drops': int(slot.coalesced_drops),
                        }
                stereo_count = sum(1 for t in targets if t.stereo)
                mono_count = len(targets) - stereo_count
                d_msg = String()
                d_msg.data = json.dumps({
                    'stamp': round(now_s, 3),
                    'mode': self._last_loc_mode,
                    'active_cams': active_cams_with_rays,
                    'skew_rms_m': round(self._last_skew_rms, 4),
                    'adaptive_z_m': round(self._adaptive_target_z, 3),
                    'stereo_hits': self._stereo_hits,
                    'mono_hits': self._mono_hits,
                    'num_drones': len(self.drone_ids),
                    'target_count': len(targets),
                    'stereo_targets': stereo_count,
                    'mono_targets': mono_count,
                    'los_rejected_count': self._los_rejected_count,
                    'obstacles_loaded': len(self.obstacles),
                    'per_camera': per_cam,
                })
                self.diag_pub.publish(d_msg)

    def destroy_node(self) -> bool:
        self._shutdown_mp_pool()
        if self._executor_pool is not None:
            try:
                self._executor_pool.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass
            self._executor_pool = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionNode()
    # 使用单线程 rclpy 执行器负责零锁轻量入队与融合定时发布，避免多执行器线程与工作线程池竞争 Python GIL；
    # 全部图像解码、多谱系检测与射线反投影均由 node._executor_pool 并发执行。
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
