# -*- coding: utf-8 -*-
"""动力学阶跃响应标定启动文件。"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    drone_id = LaunchConfiguration('drone_id')
    cruise_alt = LaunchConfiguration('cruise_alt')
    step_dist = LaunchConfiguration('step_dist')

    return LaunchDescription([
        DeclareLaunchArgument('drone_id', default_value='drone_1'),
        DeclareLaunchArgument('cruise_alt', default_value='8.0'),
        DeclareLaunchArgument('step_dist', default_value='20.0'),
        Node(
            package='flight_ctrl_pkg',
            executable='dynamics_calibrator',
            name='dynamics_calibrator',
            output='screen',
            parameters=[{
                'drone_id': drone_id,
                'cruise_alt': cruise_alt,
                'step_dist': step_dist,
            }],
        ),
    ])
