# -*- coding: utf-8 -*-
"""单机模式启动文件：启动 drone_1 的 uXRCE-DDS 高阶桥接 + 双路相机(导航+PX4云台)转发节点。"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    enable_camera = LaunchConfiguration('enable_camera')
    enable_gimbal = LaunchConfiguration('enable_gimbal')
    min_track_alt = LaunchConfiguration('min_track_alt')
    target_ref_alt = LaunchConfiguration('target_ref_alt')
    use_vel_feedforward = LaunchConfiguration('use_vel_feedforward')
    use_acc_feedforward = LaunchConfiguration('use_acc_feedforward')
    control_hz = LaunchConfiguration('control_hz')

    return LaunchDescription([
        DeclareLaunchArgument('enable_camera', default_value='true'),
        DeclareLaunchArgument('enable_gimbal', default_value='true'),
        DeclareLaunchArgument('min_track_alt', default_value='8.0'),
        DeclareLaunchArgument('target_ref_alt', default_value='5.5'),
        DeclareLaunchArgument('use_vel_feedforward', default_value='true'),
        DeclareLaunchArgument('use_acc_feedforward', default_value='true'),
        DeclareLaunchArgument('control_hz', default_value='50.0'),
        Node(
            package='flight_ctrl_pkg',
            executable='px4_bridge_node',
            name='px4_bridge_drone_1',
            output='screen',
            parameters=[{
                'drone_id': 'drone_1',
                'px4_ns': '',
                'target_system_id': 1,
                'offset_x': 0.0,
                'offset_y': 0.0,
                'control_hz': control_hz,
                'enable_gimbal_cmd': enable_gimbal,
                'min_track_alt': min_track_alt,
                'target_ref_alt': target_ref_alt,
                'use_vel_feedforward': use_vel_feedforward,
                'use_acc_feedforward': use_acc_feedforward,
            }],
        ),
        Node(
            package='flight_ctrl_pkg',
            executable='camera_relay_node',
            name='camera_relay_drone_1',
            output='screen',
            condition=IfCondition(enable_camera),
            parameters=[{
                'drone_id': 'drone_1',
                'source_mode': 'gz_transport',
                'gz_world': 'moving_target_arena',
                'gz_model': 'x500_down_cam_0',
                'gz_link': 'down_cam_link',
                'gz_sensor': 'down_cam',
                'out_namespace': 'camera',
                'restamp_to_wall_clock': True,
            }],
        ),
        Node(
            package='flight_ctrl_pkg',
            executable='camera_relay_node',
            name='gimbal_relay_drone_1',
            output='screen',
            condition=IfCondition(enable_camera),
            parameters=[{
                'drone_id': 'drone_1',
                'source_mode': 'gz_transport',
                'gz_world': 'moving_target_arena',
                'gz_model': 'x500_down_cam_0',
                'gz_link': 'gimbal_pitch_link',
                'gz_sensor': 'gimbal_cam',
                'out_namespace': 'gimbal_cam',
                'restamp_to_wall_clock': True,
            }],
        ),
    ])
