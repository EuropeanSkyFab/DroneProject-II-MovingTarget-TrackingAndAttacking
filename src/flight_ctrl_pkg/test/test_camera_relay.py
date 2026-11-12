# -*- coding: utf-8 -*-
import time

import rclpy
from sensor_msgs.msg import CameraInfo, Image

from flight_ctrl_pkg.camera_relay_node import CameraRelayNode


def test_camera_relay_aligns_sim_timestamp():
    rclpy.init()
    node = CameraRelayNode()
    try:
        # 模拟 Gazebo 仿真零点开始第 12.5 秒的图像时间戳
        img = Image()
        img.header.stamp.sec = 12
        img.header.stamp.nanosec = 500_000_000
        aligned = node._align_stamp(img.header.stamp)
        now_s = node.get_clock().now().nanoseconds / 1e9
        aligned_s = aligned.sec + aligned.nanosec / 1e9
        assert abs(now_s - aligned_s) < 0.2

        # 真实墙钟经过 0.1s 后，仿真时间第 12.6 秒帧保留 0.1s 的真实时间间隔
        time.sleep(0.1)
        img2 = Image()
        img2.header.stamp.sec = 12
        img2.header.stamp.nanosec = 600_000_000
        aligned2 = node._align_stamp(img2.header.stamp)
        dt = (aligned2.sec - aligned.sec) + (aligned2.nanosec - aligned.nanosec) / 1e9
        assert abs(dt - 0.1) < 1e-3

        # 验证 CameraInfo 同样生效
        info = CameraInfo()
        info.header.stamp.sec = 12
        info.header.stamp.nanosec = 600_000_000
        aligned_info = node._align_stamp(info.header.stamp)
        assert aligned_info.sec == aligned2.sec
    finally:
        node.destroy_node()
        rclpy.shutdown()
