# -*- coding: utf-8 -*-
"""F5 并行视觉节点吞吐量与端到端延迟 SLA 测试 (N=6..10 路相机 @ >=15 Hz)。"""

import json
import math
import multiprocessing as mp
import os
import sys
import threading
import time

from nav_msgs.msg import Odometry
import numpy as np
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from vision_pkg.geo_utils import camera_mount_matrix, quat_to_matrix
from vision_pkg.vision_node import VisionNode


def _create_vision_node_with_params(overrides: dict) -> VisionNode:
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


def _yaw_quat(yaw: float):
    return (0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw))


def _project(target_enu, drone_pos, drone_quat, R_bc, K, cam_offset=(0.12, 0.0, -0.04)):
    R_wb = quat_to_matrix(*drone_quat)
    origin = np.asarray(drone_pos, dtype=float) + R_wb @ np.asarray(cam_offset, dtype=float)
    rel = np.asarray(target_enu, dtype=float) - origin
    d_cam = R_bc.T @ (R_wb.T @ rel)
    if d_cam[2] <= 0.2:
        return None
    return float(K[0] * (d_cam[0] / d_cam[2]) + K[2]), float(K[4] * (d_cam[1] / d_cam[2]) + K[5])


def _get_isolated_domain_id(num_drones: int) -> int:
    env_val = os.environ.get('ROS_DOMAIN_ID', '').strip()
    if env_val.isdigit() and int(env_val) > 0:
        return int(env_val)
    return 86 + ((os.getpid() + int(num_drones)) % 15)


def _camera_driver_process(
    num_drones: int,
    width: int,
    height: int,
    stream_hz: float,
    node_ready,
    start_ev,
    done_ev,
    vision_pkg_root: str,
    domain_id: int,
) -> None:
    """独立进程模拟 N 路相机与飞控状态发布（隔离发布端序列化开销，真实还原 SITL 多进程架构）。"""
    if vision_pkg_root not in sys.path:
        sys.path.insert(0, vision_pkg_root)
    os.environ['ROS_DOMAIN_ID'] = str(domain_id)
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    rclpy.init(domain_id=domain_id)
    pub_node = rclpy.create_node(f'test_throughput_driver_{num_drones}')
    try:
        w, h = int(width), int(height)
        fx = (0.5 * w) / math.tan(0.5 * 1.35)
        K = [fx, 0.0, 0.5 * w, 0.0, fx, 0.5 * h, 0.0, 0.0, 1.0]
        R_bc = camera_mount_matrix(0.95993109, 0.0, -1.57079633)
        gt_targets = [
            (np.array([14.0, -3.0, 7.0]), (22, 22, 232)),
            (np.array([20.0, 1.0, 8.0]), (20, 120, 245)),
            (np.array([16.0, 5.0, 6.5]), (255, 46, 199)),
        ]
        drone_assets = []
        for idx in range(num_drones):
            d_id = f'drone_{idx + 1}'
            ang = (2.0 * math.pi * idx) / float(num_drones)
            pos = (16.5 + 11.0 * math.cos(ang), 1.0 + 11.0 * math.sin(ang), 14.5)
            yaw = math.atan2(1.0 - pos[1], 16.5 - pos[0])
            quat = _yaw_quat(yaw)
            bgr = np.full((h, w, 3), (55, 120, 70), dtype=np.uint8)
            for tgt_xyz, color in gt_targets:
                uv = _project(tgt_xyz, pos, quat, R_bc, K)
                if uv is not None and 8 <= uv[0] < w - 8 and 8 <= uv[1] < h - 8:
                    cx, cy = int(round(uv[0])), int(round(uv[1]))
                    bgr[cy - 6:cy + 7, cx - 6:cx + 7] = color

            info_msg = CameraInfo()
            info_msg.width = w
            info_msg.height = h
            info_msg.k = K

            odom_msg = Odometry()
            odom_msg.header.frame_id = 'world'
            odom_msg.pose.pose.position.x = pos[0]
            odom_msg.pose.pose.position.y = pos[1]
            odom_msg.pose.pose.position.z = pos[2]
            odom_msg.pose.pose.orientation.x = quat[0]
            odom_msg.pose.pose.orientation.y = quat[1]
            odom_msg.pose.pose.orientation.z = quat[2]
            odom_msg.pose.pose.orientation.w = quat[3]

            img_msg = Image()
            img_msg.header.frame_id = 'down_cam'
            img_msg.height = h
            img_msg.width = w
            img_msg.encoding = 'bgr8'
            img_msg.step = w * 3
            img_msg.data = bgr.tobytes()

            img_pub = pub_node.create_publisher(Image, f'/{d_id}/camera/image_raw', qos_profile_sensor_data)
            info_pub = pub_node.create_publisher(CameraInfo, f'/{d_id}/camera/camera_info', qos_profile_sensor_data)
            state_pub = pub_node.create_publisher(Odometry, f'/{d_id}/state', 10)
            drone_assets.append((img_pub, info_pub, state_pub, info_msg, odom_msg, img_msg))

        node_ready.wait(timeout=10.0)
        time.sleep(0.30)
        t_init = time.time()
        for _, info_pub, state_pub, info_msg, odom_msg, _ in drone_assets:
            sec = int(t_init)
            nsec = int((t_init - sec) * 1e9)
            info_msg.header.stamp.sec = sec
            info_msg.header.stamp.nanosec = nsec
            odom_msg.header.stamp.sec = sec
            odom_msg.header.stamp.nanosec = nsec
            info_pub.publish(info_msg)
            state_pub.publish(odom_msg)
        time.sleep(0.10)

        start_ev.set()
        duration_s = 3.0
        period = 1.0 / float(stream_hz)
        sub_step = period / float(max(num_drones, 1))
        t_start = time.monotonic()
        tick = 0
        while True:
            now_mono = time.monotonic()
            if now_mono - t_start >= duration_s:
                break
            # 错峰发送各机相机帧（模拟真实多机异步相机采样，避免瞬时 10x921KB 突发拥塞系统套接字缓冲区）
            for d_idx, (img_pub, _, state_pub, _, odom_msg, img_msg) in enumerate(drone_assets):
                if time.monotonic() - t_start >= duration_s:
                    break
                target_t = t_start + tick * period + d_idx * sub_step
                wait_dt = target_t - time.monotonic()
                if wait_dt > 0.0005:
                    time.sleep(wait_dt)
                now_wall = time.time()
                sec = int(now_wall)
                nsec = int((now_wall - sec) * 1e9)
                odom_msg.header.stamp.sec = sec
                odom_msg.header.stamp.nanosec = nsec
                state_pub.publish(odom_msg)

                img_msg.header.stamp.sec = sec
                img_msg.header.stamp.nanosec = nsec
                img_pub.publish(img_msg)

            tick += 1
        done_ev.set()
    finally:
        pub_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


@pytest.mark.parametrize('num_drones,width,height,stream_hz', [
    (6, 640, 480, 25.0),
    (8, 640, 480, 25.0),
    (10, 480, 360, 25.0),
    (10, 640, 480, 25.0),
])
def test_parallel_vision_node_throughput_sla(num_drones: int, width: int, height: int, stream_hz: float):
    pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dom_id = _get_isolated_domain_id(num_drones)
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    ctx = mp.get_context('spawn')
    node_ready = ctx.Event()
    start_ev = ctx.Event()
    done_ev = ctx.Event()
    drv_proc = ctx.Process(
        target=_camera_driver_process,
        args=(num_drones, width, height, stream_hz, node_ready, start_ev, done_ev, pkg_root, dom_id),
    )
    drv_proc.start()

    rclpy.init(domain_id=dom_id)
    try:
        node = _create_vision_node_with_params({
            'num_drones': num_drones,
            'process_mode': 'threaded_batch',
            'max_workers': min(12, num_drones),
            'cv2_num_threads': 1,
            'target_publish_hz': 17.0,
        })
        sub_node = rclpy.create_node(f'test_throughput_sub_{num_drones}')

        targets_rx_count = 0
        latencies = []
        max_targets_seen = 0
        rx_lock = threading.Lock()

        def _on_targets(msg: String):
            nonlocal targets_rx_count, max_targets_seen
            payload = json.loads(msg.data)
            now_s = time.time()
            with rx_lock:
                targets_rx_count += 1
                tgts = payload.get('targets') or []
                if len(tgts) > max_targets_seen:
                    max_targets_seen = len(tgts)
                if tgts:
                    latencies.append(max(0.0, now_s - float(payload['stamp'])))

        sub_node.create_subscription(String, '/vision/targets', _on_targets, 10)

        executor = SingleThreadedExecutor()
        executor.add_node(node)
        sub_executor = SingleThreadedExecutor()
        sub_executor.add_node(sub_node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        sub_spin_thread = threading.Thread(target=sub_executor.spin, daemon=True)
        spin_thread.start()
        sub_spin_thread.start()

        try:
            node_ready.set()
            assert start_ev.wait(timeout=10.0), 'Driver process did not start streaming'
            t_start = time.monotonic()
            assert done_ev.wait(timeout=10.0), 'Driver process did not finish streaming'
            stream_elapsed = time.monotonic() - t_start
            time.sleep(0.05)

            for d_id in node.drone_ids:
                slot = node.drones[d_id]
                rx_hz = slot.rx_frames / stream_elapsed
                proc_hz = slot.proc_frames / stream_elapsed
                assert rx_hz >= 15.0, (
                    f'N={num_drones} ({width}x{height}) {d_id} received {slot.rx_frames} frames in '
                    f'{stream_elapsed:.2f}s ({rx_hz:.1f} Hz < 15.0 Hz)'
                )
                assert proc_hz >= 15.0, (
                    f'N={num_drones} ({width}x{height}) {d_id} processed {slot.proc_frames} frames in '
                    f'{stream_elapsed:.2f}s ({proc_hz:.1f} Hz < 15.0 Hz)'
                )

            with rx_lock:
                pub_hz = targets_rx_count / stream_elapsed
                assert pub_hz >= 15.0, f'N={num_drones} /vision/targets rate {pub_hz:.1f} Hz < 15.0 Hz'
                assert max_targets_seen == 3, f'N={num_drones} expected 3 targets, saw max {max_targets_seen}'
                assert len(latencies) >= 10
                mean_lat = sum(latencies) / len(latencies)
                assert mean_lat < 0.25, f'N={num_drones} mean latency {mean_lat:.3f}s >= 0.25s'
        finally:
            node._shutdown_flag = True
            executor.shutdown()
            sub_executor.shutdown()
            spin_thread.join(timeout=1.5)
            sub_spin_thread.join(timeout=1.5)
            sub_node.destroy_node()
            node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        drv_proc.join(timeout=5.0)
        if drv_proc.is_alive():
            drv_proc.terminate()
            drv_proc.join(timeout=2.0)
