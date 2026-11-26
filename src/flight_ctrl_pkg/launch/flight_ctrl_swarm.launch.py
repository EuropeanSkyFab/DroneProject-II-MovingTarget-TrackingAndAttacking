# -*- coding: utf-8 -*-
"""可伸缩多机集群桥接与相机转发启动文件：支持 N=1..8+ 无人机，自动对齐 config/obstacles.yaml 起降坪。"""

import os
from typing import Any, Dict, List

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _as_bool(val: str) -> bool:
    return str(val).strip().lower() in ('1', 'true', 'yes', 'on')


def load_fleet_config(num_drones: int, obstacles_yaml: str) -> List[Dict[str, Any]]:
    """Read helipads from obstacles.yaml with deterministic fallback for N=1..8+."""
    pads: Dict[str, Dict[str, float]] = {}
    if obstacles_yaml and os.path.isfile(obstacles_yaml):
        with open(obstacles_yaml, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}
        for pad in data.get('helipads', []):
            d_id = str(pad.get('drone_id', ''))
            if d_id:
                pads[d_id] = {
                    'x': float(pad.get('x', 0.0)),
                    'y': float(pad.get('y', 0.0)),
                }

    fleet: List[Dict[str, Any]] = []
    for i in range(num_drones):
        d_id = f'drone_{i + 1}'
        if d_id in pads:
            ox, oy = pads[d_id]['x'], pads[d_id]['y']
        elif i < 3:
            ox, oy = 0.0, 3.0 * i
        else:
            ox, oy = -3.5 * (i // 3), 3.0 * (i % 3)
        fleet.append({
            'drone_id': d_id,
            'px4_ns': '' if i == 0 else f'px4_{i}',
            'sys_id': i + 1,
            'ox': ox,
            'oy': oy,
            'gz_model': f'x500_down_cam_{i}',
        })
    return fleet


def _launch_setup(context, *args, **kwargs):
    num_drones = max(1, int(LaunchConfiguration('num_drones').perform(context)))
    gz_world = LaunchConfiguration('gz_world').perform(context)
    obstacles_yaml = LaunchConfiguration('obstacles_yaml').perform(context)
    enable_camera = _as_bool(LaunchConfiguration('enable_camera').perform(context))
    enable_gimbal_cam_relay = _as_bool(LaunchConfiguration('enable_gimbal_cam_relay').perform(context))
    gimbal_relay_all_drones = _as_bool(LaunchConfiguration('gimbal_relay_all_drones').perform(context))
    enable_thermal_cam_relay = _as_bool(LaunchConfiguration('enable_thermal_cam_relay').perform(context))
    use_vel_ff = _as_bool(LaunchConfiguration('use_vel_feedforward').perform(context))
    use_acc_ff = _as_bool(LaunchConfiguration('use_acc_feedforward').perform(context))
    control_hz = float(LaunchConfiguration('control_hz').perform(context))

    fleet = load_fleet_config(num_drones, obstacles_yaml)
    actions = []
    for cfg in fleet:
        d_id = cfg['drone_id']
        actions.append(
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
                    'control_hz': control_hz,
                    'enable_gimbal_cmd': True,
                    'min_track_alt': 8.0,
                    'target_ref_alt': 5.5,
                    'use_vel_feedforward': use_vel_ff,
                    'use_acc_feedforward': use_acc_ff,
                }],
            )
        )
        if enable_camera:
            actions.append(
                Node(
                    package='flight_ctrl_pkg',
                    executable='camera_relay_node',
                    name=f'camera_relay_{d_id}',
                    output='screen',
                    parameters=[{
                        'drone_id': d_id,
                        'source_mode': 'gz_transport',
                        'gz_world': gz_world,
                        'gz_model': cfg['gz_model'],
                        'gz_link': 'down_cam_link',
                        'gz_sensor': 'down_cam',
                        'out_namespace': 'camera',
                        'restamp_to_wall_clock': True,
                    }],
                )
            )
        emit_gimbal = (enable_gimbal_cam_relay and d_id == 'drone_1') or gimbal_relay_all_drones
        if emit_gimbal:
            actions.append(
                Node(
                    package='flight_ctrl_pkg',
                    executable='camera_relay_node',
                    name=f'gimbal_relay_{d_id}',
                    output='screen',
                    parameters=[{
                        'drone_id': d_id,
                        'source_mode': 'gz_transport',
                        'gz_world': gz_world,
                        'gz_model': cfg['gz_model'],
                        'gz_link': 'gimbal_pitch_link',
                        'gz_sensor': 'gimbal_cam',
                        'out_namespace': 'gimbal_cam',
                        'restamp_to_wall_clock': True,
                    }],
                )
            )
        if enable_thermal_cam_relay and d_id == 'drone_1':
            actions.append(
                Node(
                    package='flight_ctrl_pkg',
                    executable='camera_relay_node',
                    name=f'thermal_relay_{d_id}',
                    output='screen',
                    parameters=[{
                        'drone_id': d_id,
                        'source_mode': 'gz_transport',
                        'gz_world': gz_world,
                        'gz_model': cfg['gz_model'],
                        'gz_link': 'gimbal_pitch_link',
                        'gz_sensor': 'thermal_cam',
                        'out_namespace': 'thermal_cam',
                        'restamp_to_wall_clock': True,
                    }],
                )
            )
    return actions


def generate_launch_description():
    fc_share = get_package_share_directory('flight_ctrl_pkg')
    default_obstacles = os.path.join(fc_share, 'config', 'obstacles.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('num_drones', default_value='3'),
        DeclareLaunchArgument('gz_world', default_value='moving_target_arena'),
        DeclareLaunchArgument('obstacles_yaml', default_value=default_obstacles),
        DeclareLaunchArgument('enable_camera', default_value='true'),
        DeclareLaunchArgument('enable_gimbal_cam_relay', default_value='false'),
        DeclareLaunchArgument('gimbal_relay_all_drones', default_value='false'),
        DeclareLaunchArgument('enable_thermal_cam_relay', default_value='false'),
        DeclareLaunchArgument('use_vel_feedforward', default_value='true'),
        DeclareLaunchArgument('use_acc_feedforward', default_value='true'),
        DeclareLaunchArgument('control_hz', default_value='50.0'),
        OpaqueFunction(function=_launch_setup),
    ])
