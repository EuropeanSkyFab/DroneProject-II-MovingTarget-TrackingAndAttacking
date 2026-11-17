# -*- coding: utf-8 -*-
"""机载视觉节点标准启动入口：ros2 launch vision_pkg vision_node.launch.py"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('process_mode', default_value='round_robin'),
        DeclareLaunchArgument('per_stream_hz', default_value='15.0'),
        DeclareLaunchArgument('detector_type', default_value='color'),
        DeclareLaunchArgument('ground_z', default_value='5.5'),
        DeclareLaunchArgument('min_area', default_value='24'),
        DeclareLaunchArgument('enable_triangulation', default_value='true'),
        Node(
            package='vision_pkg',
            executable='vision_node',
            name='vision_node',
            output='screen',
            parameters=[{
                'drone_ids': ['drone_1', 'drone_2', 'drone_3'],
                'process_mode': LaunchConfiguration('process_mode'),
                'per_stream_hz': LaunchConfiguration('per_stream_hz'),
                'detector_type': LaunchConfiguration('detector_type'),
                'ground_z': LaunchConfiguration('ground_z'),
                'min_area': LaunchConfiguration('min_area'),
                'enable_triangulation': LaunchConfiguration('enable_triangulation'),
                'camera_mount_rpy': [0.95993109, 0.0, -1.57079633],
                'camera_offset_body': [0.12, 0.0, -0.04],
            }],
        ),
    ])
