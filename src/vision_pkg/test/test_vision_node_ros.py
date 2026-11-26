# -*- coding: utf-8 -*-
"""F5 VisionNode ROS 2 接口契约、N 机参数化、双话题别名、LoS 门控与向后兼容集成测试。"""

import json
import math
import os
import threading
import time

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from vision_pkg.geo_utils import camera_mount_matrix, quat_to_matrix
from vision_pkg.vision_node import VisionNode


def _yaw_quat(yaw: float):
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def _project_world_to_pixel(target_enu, drone_pos, drone_quat, R_bc, K, cam_offset=(0.12, 0.0, -0.04)):
    R_wb = quat_to_matrix(*drone_quat)
    origin = np.asarray(drone_pos, dtype=float) + R_wb @ np.asarray(cam_offset, dtype=float)
    rel_world = np.asarray(target_enu, dtype=float) - origin
    d_cam = R_bc.T @ (R_wb.T @ rel_world)
    if d_cam[2] <= 0.2:
        return None
    u = K[0] * (d_cam[0] / d_cam[2]) + K[2]
    v = K[4] * (d_cam[1] / d_cam[2]) + K[5]
    return float(u), float(v)


def _make_bgr_image_msg(bgr: np.ndarray, stamp_sec: float, frame_id: str = 'down_cam') -> Image:
    msg = Image()
    sec = int(stamp_sec)
    nsec = int((stamp_sec - sec) * 1e9)
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = nsec
    msg.header.frame_id = frame_id
    msg.height, msg.width = bgr.shape[:2]
    msg.encoding = 'bgr8'
    msg.is_bigendian = 0
    msg.step = int(bgr.shape[1] * 3)
    msg.data = bgr.tobytes()
    return msg


def _make_camera_info_msg(width: int, height: int, K, stamp_sec: float) -> CameraInfo:
    msg = CameraInfo()
    sec = int(stamp_sec)
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = int((stamp_sec - sec) * 1e9)
    msg.width = width
    msg.height = height
    msg.k = [float(x) for x in K]
    return msg


def _make_odom_msg(pos, quat, stamp_sec: float) -> Odometry:
    msg = Odometry()
    sec = int(stamp_sec)
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = int((stamp_sec - sec) * 1e9)
    msg.header.frame_id = 'world'
    msg.pose.pose.position.x = float(pos[0])
    msg.pose.pose.position.y = float(pos[1])
    msg.pose.pose.position.z = float(pos[2])
    msg.pose.pose.orientation.x = float(quat[0])
    msg.pose.pose.orientation.y = float(quat[1])
    msg.pose.pose.orientation.z = float(quat[2])
    msg.pose.pose.orientation.w = float(quat[3])
    return msg


def _make_pose_msg(pos, quat, stamp_sec: float) -> PoseStamped:
    msg = PoseStamped()
    sec = int(stamp_sec)
    msg.header.stamp.sec = sec
    msg.header.stamp.nanosec = int((stamp_sec - sec) * 1e9)
    msg.header.frame_id = 'world'
    msg.pose.position.x = float(pos[0])
    msg.pose.position.y = float(pos[1])
    msg.pose.position.z = float(pos[2])
    msg.pose.orientation.x = float(quat[0])
    msg.pose.orientation.y = float(quat[1])
    msg.pose.orientation.z = float(quat[2])
    msg.pose.orientation.w = float(quat[3])
    return msg


def _get_isolated_domain_id() -> int:
    env_val = os.environ.get('ROS_DOMAIN_ID', '').strip()
    if env_val.isdigit() and int(env_val) > 0:
        return int(env_val)
    return 71 + (os.getpid() % 14)


def test_vision_node_drone_parameterization():
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    rclpy.init(domain_id=_get_isolated_domain_id())
    try:
        n_default = VisionNode()
        assert n_default.drone_ids == ['drone_1', 'drone_2', 'drone_3']
        n_default.destroy_node()

        n_6 = VisionNode.__new__(VisionNode)
        # 使用 ParameterOverrides 通过 rclpy 参数覆盖测试 num_drones=8 与 drone_names
        node8 = _create_node_with_params({'num_drones': 8})
        assert node8.drone_ids == [f'drone_{i}' for i in range(1, 9)]
        node8.destroy_node()

        node_named = _create_node_with_params({'drone_names': ['alpha', 'bravo', 'charlie', 'delta']})
        assert node_named.drone_ids == ['alpha', 'bravo', 'charlie', 'delta']
        node_named.destroy_node()
        del n_6
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def _create_node_with_params(overrides: dict) -> VisionNode:
    class _ParamVisionNode(VisionNode):
        def __init__(self):
            # 绕过 Node.__init__ 传参以便注入 parameter_overrides
            Node.__init__(
                self,
                'vision_node',
                parameter_overrides=[
                    rclpy.parameter.Parameter(k, value=v) for k, v in overrides.items()
                ],
            )
            # 仅初始化属性后复用 VisionNode 参数解析与资源建立：改用 rclpy ParameterOverrides 无法在 super().__init__ 后传，
            # 因此直接调用工厂函数更稳妥；此处用参数覆写创建完整 VisionNode。
    del _ParamVisionNode
    # 正确方式：通过 rclpy NodeOptions / monkeypatch super().__init__ 参数覆写
    orig_init = Node.__init__

    def patched_init(self_node, node_name, **kwargs):
        po = list(kwargs.get('parameter_overrides') or [])
        for k, v in overrides.items():
            po.append(rclpy.parameter.Parameter(k, value=v))
        kwargs['parameter_overrides'] = po
        return orig_init(self_node, node_name, **kwargs)

    Node.__init__ = patched_init
    try:
        return VisionNode()
    finally:
        Node.__init__ = orig_init


def test_six_drones_three_targets_ros_contracts_and_compat_aliases():
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    rclpy.init(domain_id=_get_isolated_domain_id())
    try:
        node = _create_node_with_params({
            'num_drones': 6,
            'process_mode': 'threaded_batch',
            'max_workers': 4,
            'target_publish_hz': 20.0,
            'enable_los_check': False,
        })
        client = rclpy.create_node('test_vision_client_6v3')
        targets_msgs = []
        pose_msgs = []
        diag_msgs = []
        lock = threading.Lock()

        def _on_targets(m):
            with lock:
                targets_msgs.append(json.loads(m.data))

        def _on_pose(m):
            with lock:
                pose_msgs.append(m)

        def _on_diag(m):
            with lock:
                diag_msgs.append(json.loads(m.data))

        client.create_subscription(String, '/vision/targets', _on_targets, 10)
        client.create_subscription(PoseStamped, '/vision/target_pose', _on_pose, 10)
        client.create_subscription(String, '/vision/diagnostics', _on_diag, 10)

        # 前 3 架使用主话题 (/camera + /state)，后 3 架使用兼容别名 (/down_cam + /pose，且不发 camera_info)
        pubs = {}
        for idx, d in enumerate(node.drone_ids):
            if idx < 3:
                pubs[d] = {
                    'img': client.create_publisher(Image, f'/{d}/camera/image_raw', 10),
                    'info': client.create_publisher(CameraInfo, f'/{d}/camera/camera_info', 10),
                    'state': client.create_publisher(Odometry, f'/{d}/state', 10),
                    'alias': False,
                }
            else:
                pubs[d] = {
                    'img': client.create_publisher(Image, f'/{d}/down_cam/image_raw', 10),
                    'info': None,
                    'state': client.create_publisher(PoseStamped, f'/{d}/pose', 10),
                    'alias': True,
                }

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.add_node(client)

        w, h = 640, 480
        hfov = 1.35
        fx = (0.5 * w) / math.tan(0.5 * hfov)
        K = [fx, 0.0, 0.5 * w, 0.0, fx, 0.5 * h, 0.0, 0.0, 1.0]
        R_bc = camera_mount_matrix(0.95993109, 0.0, -1.57079633)

        gt_specs = [
            ('crimson_uav', np.array([15.0, -3.0, 7.2]), (22, 22, 232)),
            ('amber_uav', np.array([21.0, 2.0, 8.4]), (20, 120, 245)),
            ('slate_uav', np.array([16.0, 5.0, 6.5]), (255, 46, 199)),
        ]
        drone_poses = [
            ('drone_1', (6.0, -8.0, 14.0)),
            ('drone_2', (26.0, -9.0, 14.5)),
            ('drone_3', (28.0, 3.0, 15.0)),
            ('drone_4', (24.0, 13.0, 14.2)),
            ('drone_5', (6.0, 11.0, 14.8)),
            ('drone_6', (5.0, 1.0, 15.2)),
        ]

        # 等待 DDS 匹配
        for _ in range(10):
            executor.spin_once(timeout_sec=0.02)

        for tick in range(8):
            now_s = time.time()
            for d_id, pos in drone_poses:
                yaw = math.atan2(1.0 - pos[1], 17.0 - pos[0])
                quat = _yaw_quat(yaw)
                bgr = np.full((h, w, 3), (55, 120, 70), dtype=np.uint8)
                for _, tgt_xyz, color in gt_specs:
                    uv = _project_world_to_pixel(tgt_xyz, pos, quat, R_bc, K)
                    if uv is not None and 8 <= uv[0] < w - 8 and 8 <= uv[1] < h - 8:
                        cx, cy = int(round(uv[0])), int(round(uv[1]))
                        bgr[cy - 6:cy + 7, cx - 6:cx + 7] = color
                p_entry = pubs[d_id]
                if p_entry['alias']:
                    p_entry['state'].publish(_make_pose_msg(pos, quat, now_s))
                else:
                    p_entry['info'].publish(_make_camera_info_msg(w, h, K, now_s))
                    p_entry['state'].publish(_make_odom_msg(pos, quat, now_s))
                p_entry['img'].publish(_make_bgr_image_msg(bgr, now_s, frame_id='down_cam'))
            for _ in range(10):
                executor.spin_once(timeout_sec=0.02)
            time.sleep(0.03)

        # 额外 spin 确保异步工作线程与融合定时器交付完成（待 >=3 视角协同完成）
        t_wait = time.time()
        while time.time() - t_wait < 1.5:
            executor.spin_once(timeout_sec=0.02)
            with lock:
                has_3 = any(
                    len(m.get('targets', [])) == 3
                    and all(t.get('rays_used', 0) >= 3 for t in m['targets'])
                    for m in targets_msgs
                )
                has_stereo_diag = any(str(d.get('mode', '')).startswith('stereo') for d in diag_msgs)
                if has_3 and pose_msgs and has_stereo_diag:
                    break

        with lock:
            valid_multi = [
                m for m in targets_msgs
                if len(m.get('targets', [])) == 3 and all(t.get('rays_used', 0) >= 3 for t in m['targets'])
            ]
            if not valid_multi:
                valid_multi = [m for m in targets_msgs if len(m.get('targets', [])) == 3]
            assert len(valid_multi) >= 1, f'Never saw 3 targets in /vision/targets: {targets_msgs[-3:]}'
            latest = valid_multi[-1]
            assert latest['frame_id'] == 'world'
            assert isinstance(latest['stamp'], float)

            required_keys = {
                'det_id', 'class_id', 'x', 'y', 'z', 'conf',
                'stereo', 'rays_used', 'parallax_deg', 'skew_rms_m',
            }
            seen_classes = set()
            for t in latest['targets']:
                assert required_keys.issubset(t.keys())
                assert isinstance(t['det_id'], int)
                assert isinstance(t['class_id'], str)
                assert isinstance(t['stereo'], bool) and t['stereo'] is True
                assert isinstance(t['rays_used'], int) and t['rays_used'] >= 3
                seen_classes.add(t['class_id'])
                gt_xyz = dict((c, xyz) for c, xyz, _ in gt_specs)[t['class_id']]
                err = math.sqrt((t['x'] - gt_xyz[0]) ** 2 + (t['y'] - gt_xyz[1]) ** 2 + (t['z'] - gt_xyz[2]) ** 2)
                assert err < 0.30, f"{t['class_id']} 3D error too large: {err:.3f}m"
            assert seen_classes == {'crimson_uav', 'amber_uav', 'slate_uav'}

            assert len(pose_msgs) >= 1
            assert pose_msgs[-1].header.frame_id == 'world'
            stereo_diags = [d for d in diag_msgs if str(d.get('mode', '')).startswith('stereo')]
            assert len(stereo_diags) >= 1, f'No stereo diag observed: {diag_msgs}'
            diag = stereo_diags[-1]
            assert diag['stereo_hits'] >= 1
            assert 'mono_hits' in diag
            assert diag['num_drones'] == 6

        executor.remove_node(node)
        executor.remove_node(client)
        executor.shutdown()
        client.destroy_node()
        node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


def test_sync_process_frame_and_obstacle_los_gating():
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    rclpy.init(domain_id=_get_isolated_domain_id())
    try:
        # 1) 测试同步 process_frame (frame_id='camera_link') 与障碍物 LoS 拦截单目穿墙伪影
        obs_json = json.dumps({
            'obstacles': [
                {
                    'id': 'blocker', 'type': 'box',
                    'cx': 12.0, 'cy': 0.0, 'z_min': 0.0, 'z_max': 20.0, 'sx': 2.0, 'sy': 20.0,
                }
            ]
        })
        node = _create_node_with_params({
            'num_drones': 2,
            'process_mode': 'round_robin',
            'obstacles_json': obs_json,
            'enable_los_check': True,
        })
        received_targets = []
        client = rclpy.create_node('test_sync_vision_client')
        client.create_subscription(String, '/vision/targets', lambda m: received_targets.append(json.loads(m.data)), 10)
        ex = SingleThreadedExecutor()
        ex.add_node(client)

        w, h = 640, 480
        K = [400.1, 0.0, 320.0, 0.0, 400.1, 240.0, 0.0, 0.0, 1.0]
        R_bc = camera_mount_matrix(0.95993109, 0.0, -1.57079633)
        now_s = time.time()

        # drone_1 在 x=4 看向 x=18 (被 x=12 的墙挡住！)
        pos1 = (4.0, 0.0, 10.0)
        quat1 = _yaw_quat(0.0)
        node._make_info_cb('drone_1')(_make_camera_info_msg(w, h, K, now_s))
        node._make_odom_cb('drone_1')(_make_odom_msg(pos1, quat1, now_s))

        uv1 = _project_world_to_pixel([18.0, 0.0, 5.5], pos1, quat1, R_bc, K, cam_offset=(0.0, 0.0, 0.0))
        assert uv1 is not None
        bgr = np.full((h, w, 3), (55, 120, 70), dtype=np.uint8)
        cx, cy = int(round(uv1[0])), int(round(uv1[1]))
        bgr[cy - 6:cy + 7, cx - 6:cx + 7] = (22, 22, 232)

        node.process_frame('drone_1', _make_bgr_image_msg(bgr, now_s, frame_id='camera_link'))
        for _ in range(5):
            ex.spin_once(timeout_sec=0.01)

        assert len(received_targets) >= 1
        assert received_targets[-1]['targets'] == []
        assert node._los_rejected_count >= 1

        # 2) 无障碍节点验证同步双目 process_frame 成功三角化
        node_clear = _create_node_with_params({
            'num_drones': 2,
            'process_mode': 'round_robin',
            'enable_los_check': False,
        })
        clear_targets = []
        client2 = rclpy.create_node('test_sync_vision_client_clear')
        client2.create_subscription(String, '/vision/targets', lambda m: clear_targets.append(json.loads(m.data)), 10)
        ex2 = SingleThreadedExecutor()
        ex2.add_node(client2)

        gt = np.array([18.0, 0.0, 7.5])
        for d_id, pos, yaw in (('drone_1', (10.0, -5.0, 12.0), 0.55), ('drone_2', (10.0, 5.0, 12.0), -0.55)):
            quat = _yaw_quat(yaw)
            node_clear._make_info_cb(d_id)(_make_camera_info_msg(w, h, K, now_s))
            node_clear._make_odom_cb(d_id)(_make_odom_msg(pos, quat, now_s))
            uv = _project_world_to_pixel(gt, pos, quat, R_bc, K, cam_offset=(0.0, 0.0, 0.0))
            assert uv is not None
            img = np.full((h, w, 3), (55, 120, 70), dtype=np.uint8)
            iu, iv = int(round(uv[0])), int(round(uv[1]))
            img[iv - 6:iv + 7, iu - 6:iu + 7] = (22, 22, 232)
            node_clear.process_frame(d_id, _make_bgr_image_msg(img, now_s, frame_id='camera_link'))

        for _ in range(5):
            ex2.spin_once(timeout_sec=0.01)
        assert len(clear_targets) >= 1
        assert len(clear_targets[-1]['targets']) == 1
        t0 = clear_targets[-1]['targets'][0]
        assert t0['stereo'] is True
        assert math.sqrt((t0['x'] - 18.0) ** 2 + (t0['y'] - 0.0) ** 2 + (t0['z'] - 7.5) ** 2) < 0.25

        ex.shutdown()
        ex2.shutdown()
        client.destroy_node()
        client2.destroy_node()
        node.destroy_node()
        node_clear.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
