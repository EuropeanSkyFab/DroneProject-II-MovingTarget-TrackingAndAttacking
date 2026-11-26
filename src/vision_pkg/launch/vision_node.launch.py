# -*- coding: utf-8 -*-
"""机载视觉节点标准启动入口：ros2 launch vision_pkg vision_node.launch.py"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    params_file = LaunchConfiguration('params_file').perform(context).strip()
    if not params_file:
        pkg_share = get_package_share_directory('vision_pkg')
        params_file = os.path.join(pkg_share, 'config', 'vision_params.yaml')

    typed_args = (
        ('num_drones', int),
        ('process_mode', str),
        ('max_workers', int),
        ('cv2_num_threads', int),
        ('per_stream_hz', float),
        ('target_publish_hz', float),
        ('detector_type', str),
        ('enable_multi_signature', 'bool'),
        ('enable_multi_target', 'bool'),
        ('ground_z', float),
        ('min_area', int),
        ('enable_triangulation', 'bool'),
        ('enable_los_check', 'bool'),
        ('obstacles_yaml_path', str),
    )

    overrides = {}
    for name, conv in typed_args:
        raw = LaunchConfiguration(name).perform(context).strip()
        if not raw:
            continue
        if conv == 'bool':
            overrides[name] = raw.lower() in ('1', 'true', 'yes', 'on')
        else:
            overrides[name] = conv(raw)

    params = [params_file] if not overrides else [params_file, overrides]
    return [
        Node(
            package='vision_pkg',
            executable='vision_node',
            name='vision_node',
            output='screen',
            parameters=params,
        )
    ]


def generate_launch_description():
    pkg_share = get_package_share_directory('vision_pkg')
    default_params_file = os.path.join(pkg_share, 'config', 'vision_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params_file),
        DeclareLaunchArgument('num_drones', default_value=''),
        DeclareLaunchArgument('process_mode', default_value=''),
        DeclareLaunchArgument('max_workers', default_value=''),
        DeclareLaunchArgument('cv2_num_threads', default_value=''),
        DeclareLaunchArgument('per_stream_hz', default_value=''),
        DeclareLaunchArgument('target_publish_hz', default_value=''),
        DeclareLaunchArgument('detector_type', default_value=''),
        DeclareLaunchArgument('enable_multi_signature', default_value=''),
        DeclareLaunchArgument('enable_multi_target', default_value=''),
        DeclareLaunchArgument('ground_z', default_value=''),
        DeclareLaunchArgument('min_area', default_value=''),
        DeclareLaunchArgument('enable_triangulation', default_value=''),
        DeclareLaunchArgument('enable_los_check', default_value=''),
        DeclareLaunchArgument('obstacles_yaml_path', default_value=''),
        OpaqueFunction(function=_launch_setup),
    ])
