# -*- coding: utf-8 -*-
"""路径规划与多机协同制导节点启动入口：ros2 launch planning_pkg planning_node.launch.py"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('drone_ids', default_value='["drone_1", "drone_2", "drone_3"]'),
        DeclareLaunchArgument('auto_discover_drones', default_value='true'),
        DeclareLaunchArgument('obstacles_yaml_path', default_value=''),
        DeclareLaunchArgument('plan_hz', default_value='20.0'),
        DeclareLaunchArgument('track_radius', default_value='6.0'),
        DeclareLaunchArgument('track_alt_offset', default_value='3.5'),
        DeclareLaunchArgument('v_intercept', default_value='10.5'),
        DeclareLaunchArgument('v_attack', default_value='12.2'),
        DeclareLaunchArgument('max_coast_sec', default_value='3.0'),
        DeclareLaunchArgument('confirm_m', default_value='3'),
        DeclareLaunchArgument('confirm_n', default_value='5'),
        Node(
            package='planning_pkg',
            executable='planning_node',
            name='planning_node',
            output='screen',
            parameters=[{
                'drone_ids': LaunchConfiguration('drone_ids'),
                'auto_discover_drones': ParameterValue(
                    LaunchConfiguration('auto_discover_drones'), value_type=bool
                ),
                'obstacles_yaml_path': LaunchConfiguration('obstacles_yaml_path'),
                'plan_hz': ParameterValue(LaunchConfiguration('plan_hz'), value_type=float),
                'track_radius': ParameterValue(LaunchConfiguration('track_radius'), value_type=float),
                'track_alt_offset': ParameterValue(LaunchConfiguration('track_alt_offset'), value_type=float),
                'v_intercept': ParameterValue(LaunchConfiguration('v_intercept'), value_type=float),
                'v_attack': ParameterValue(LaunchConfiguration('v_attack'), value_type=float),
                'max_coast_sec': ParameterValue(LaunchConfiguration('max_coast_sec'), value_type=float),
                'confirm_m': ParameterValue(LaunchConfiguration('confirm_m'), value_type=int),
                'confirm_n': ParameterValue(LaunchConfiguration('confirm_n'), value_type=int),
            }],
        ),
    ])
