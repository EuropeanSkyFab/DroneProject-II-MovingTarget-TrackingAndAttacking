# -*- coding: utf-8 -*-
"""三机协同模式启动文件：同时启动 drone_1 / drone_2 / drone_3 的 uXRCE-DDS 高阶桥接与相机转发。"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    enable_gimbal_cam_relay = LaunchConfiguration('enable_gimbal_cam_relay')
    fleet = [
        {'drone_id': 'drone_1', 'px4_ns': '', 'sys_id': 1, 'ox': 0.0, 'oy': 0.0, 'gz_model': 'x500_down_cam_0'},
        {'drone_id': 'drone_2', 'px4_ns': 'px4_1', 'sys_id': 2, 'ox': 0.0, 'oy': 2.0, 'gz_model': 'x500_down_cam_1'},
        {'drone_id': 'drone_3', 'px4_ns': 'px4_2', 'sys_id': 3, 'ox': 0.0, 'oy': 4.0, 'gz_model': 'x500_down_cam_2'},
    ]

    nodes = [
        DeclareLaunchArgument('enable_gimbal_cam_relay', default_value='false'),
    ]
    for cfg in fleet:
        d_id = cfg['drone_id']
        nodes.append(
            Node(
                package='flight_ctrl_pkg',
                executable='px4_bridge_node',
                name=f'px4_bridge_{d_id}',
                output='screen',
                parameters=[{
                    'drone_id': d_id,
                    'px4_ns': cfg['px4_ns'],
                    'target_system_id': cfg['sys_id'],
                    'offset_x': cfg['ox'],
                    'offset_y': cfg['oy'],
                    'control_hz': 50.0,
                    'enable_gimbal_cmd': True,
                    'min_track_alt': 8.0,
                    'target_ref_alt': 5.5,
                    'use_vel_feedforward': True,
                    'use_acc_feedforward': True,
                }],
            )
        )
        nodes.append(
            Node(
                package='flight_ctrl_pkg',
                executable='camera_relay_node',
                name=f'camera_relay_{d_id}',
                output='screen',
                parameters=[{
                    'drone_id': d_id,
                    'source_mode': 'gz_transport',
                    'gz_world': 'moving_target_arena',
                    'gz_model': cfg['gz_model'],
                    'gz_link': 'down_cam_link',
                    'gz_sensor': 'down_cam',
                    'out_namespace': 'camera',
                    'restamp_to_wall_clock': True,
                }],
            )
        )
        nodes.append(
            Node(
                package='flight_ctrl_pkg',
                executable='camera_relay_node',
                name=f'gimbal_relay_{d_id}',
                output='screen',
                condition=IfCondition(enable_gimbal_cam_relay),
                parameters=[{
                    'drone_id': d_id,
                    'source_mode': 'gz_transport',
                    'gz_world': 'moving_target_arena',
                    'gz_model': cfg['gz_model'],
                    'gz_link': 'gimbal_pitch_link',
                    'gz_sensor': 'gimbal_cam',
                    'out_namespace': 'gimbal_cam',
                    'restamp_to_wall_clock': True,
                }],
            )
        )

    return LaunchDescription(nodes)
