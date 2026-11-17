# -*- coding: utf-8 -*-
"""路径规划与多机协同制导节点启动入口：ros2 launch planning_pkg planning_node.launch.py"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('plan_hz', default_value='20.0'),
        DeclareLaunchArgument('track_radius', default_value='6.0'),
        DeclareLaunchArgument('track_alt_offset', default_value='3.5'),
        DeclareLaunchArgument('v_intercept', default_value='10.5'),
        DeclareLaunchArgument('v_attack', default_value='12.2'),
        DeclareLaunchArgument('max_coast_sec', default_value='3.0'),
        Node(
            package='planning_pkg',
            executable='planning_node',
            name='planning_node',
            output='screen',
            parameters=[{
                'drone_ids': ['drone_1', 'drone_2', 'drone_3'],
                'plan_hz': LaunchConfiguration('plan_hz'),
                'track_radius': LaunchConfiguration('track_radius'),
                'track_alt_offset': LaunchConfiguration('track_alt_offset'),
                'v_intercept': LaunchConfiguration('v_intercept'),
                'v_attack': LaunchConfiguration('v_attack'),
                'max_coast_sec': LaunchConfiguration('max_coast_sec'),
            }],
        ),
    ])
