# -*- coding: utf-8 -*-
"""
课题二四包全系统协同总启动文件：
  支持 N=3 向后兼容及 N>=6 可扩展集群配置。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _as_bool(val: str) -> bool:
    return str(val).strip().lower() in ('1', 'true', 'yes', 'on')


def _launch_setup(context, *args, **kwargs):
    num_drones = max(1, int(LaunchConfiguration('num_drones').perform(context)))
    gz_world = LaunchConfiguration('gz_world').perform(context).strip().removesuffix('.sdf')
    obstacles_yaml = LaunchConfiguration('obstacles_yaml').perform(context)
    auto_mission = LaunchConfiguration('auto_mission')
    enable_gimbal_cam_relay = LaunchConfiguration('enable_gimbal_cam_relay')
    enable_thermal_cam_relay = LaunchConfiguration('enable_thermal_cam_relay')
    target_profile = LaunchConfiguration('target_profile').perform(context)
    target_speed_scale = float(LaunchConfiguration('target_speed_scale').perform(context))
    exit_on_complete = _as_bool(LaunchConfiguration('exit_on_complete').perform(context))
    summary_json_path = LaunchConfiguration('summary_json_path').perform(context)
    log_jsonl_path = LaunchConfiguration('log_jsonl_path').perform(context)
    process_mode = LaunchConfiguration('process_mode').perform(context)
    max_workers = int(LaunchConfiguration('max_workers').perform(context))
    min_area = int(LaunchConfiguration('min_area').perform(context))
    enable_multi_signature = _as_bool(LaunchConfiguration('enable_multi_signature').perform(context))
    enable_multi_target = _as_bool(LaunchConfiguration('enable_multi_target').perform(context))
    enable_triangulation = _as_bool(LaunchConfiguration('enable_triangulation').perform(context))
    enable_los_check = _as_bool(LaunchConfiguration('enable_los_check').perform(context))
    auto_discover_drones = _as_bool(LaunchConfiguration('auto_discover_drones').perform(context))

    fc_share = get_package_share_directory('flight_ctrl_pkg')
    drone_ids = [f'drone_{i + 1}' for i in range(num_drones)]

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(fc_share, 'launch', 'flight_ctrl_swarm.launch.py')
            ),
            launch_arguments={
                'num_drones': str(num_drones),
                'gz_world': gz_world,
                'obstacles_yaml': obstacles_yaml,
                'enable_gimbal_cam_relay': enable_gimbal_cam_relay,
                'enable_thermal_cam_relay': enable_thermal_cam_relay,
            }.items(),
        ),
        Node(
            package='flight_ctrl_pkg',
            executable='target_sim_node',
            name='target_sim_node',
            output='screen',
            parameters=[{
                'sim_backend': 'auto',
                'gz_world': gz_world,
                'profile': target_profile,
                'speed_scale': target_speed_scale,
                'enable_radar': True,
                'enable_multi_target': enable_multi_target,
                'obstacles_yaml_path': obstacles_yaml,
            }],
        ),
        Node(
            package='vision_pkg',
            executable='vision_node',
            name='vision_node',
            output='screen',
            parameters=[{
                'drone_ids': drone_ids,
                'num_drones': num_drones,
                'process_mode': process_mode,
                'max_workers': max_workers,
                'cv2_num_threads': 1,
                'per_stream_hz': 15.0,
                'target_publish_hz': 20.0,
                'detector_type': 'color',
                'enable_multi_signature': enable_multi_signature,
                'enable_multi_target': enable_multi_target,
                'ground_z': 5.5,
                'min_area': min_area,
                'enable_triangulation': enable_triangulation,
                'enable_los_check': enable_los_check,
                'obstacles_yaml_path': obstacles_yaml,
                'camera_mount_rpy': [0.95993109, 0.0, -1.57079633],
                'camera_offset_body': [0.12, 0.0, -0.04],
            }],
        ),
        Node(
            package='planning_pkg',
            executable='planning_node',
            name='planning_node',
            output='screen',
            parameters=[{
                'drone_ids': drone_ids,
                'auto_discover_drones': auto_discover_drones,
                'obstacles_yaml_path': obstacles_yaml,
                'plan_hz': 20.0,
                'track_radius': 6.0,
                'track_alt_offset': 3.5,
                'v_intercept': 10.5,
                'v_attack': 12.2,
                'max_coast_sec': 3.0,
                'confirm_m': 3,
                'confirm_n': 5,
            }],
        ),
        Node(
            package='center_node_pkg',
            executable='center_node',
            name='center_node',
            output='screen',
            parameters=[{
                'swarm_ids': drone_ids,
                'takeoff_alt': 9.5,
                'track_safe_dist': 5.0,
                'home_spacing_y': 3.0,
                'obstacles_yaml_path': obstacles_yaml,
            }],
        ),
        Node(
            package='flight_ctrl_pkg',
            executable='mission_commander_node',
            name='mission_commander_node',
            output='screen',
            condition=IfCondition(auto_mission),
            parameters=[{
                'drone_ids': drone_ids,
                'exit_on_complete': exit_on_complete,
                'summary_json_path': summary_json_path,
                'log_jsonl_path': log_jsonl_path,
                'obstacles_yaml_path': obstacles_yaml,
            }],
        ),
    ]


def generate_launch_description():
    fc_share = get_package_share_directory('flight_ctrl_pkg')
    default_obstacles = os.path.join(fc_share, 'config', 'obstacles.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('num_drones', default_value='3'),
        DeclareLaunchArgument('gz_world', default_value='moving_target_arena'),
        DeclareLaunchArgument('obstacles_yaml', default_value=default_obstacles),
        DeclareLaunchArgument('auto_mission', default_value='false'),
        DeclareLaunchArgument('enable_gimbal_cam_relay', default_value='true'),
        DeclareLaunchArgument('enable_thermal_cam_relay', default_value='false'),
        DeclareLaunchArgument('target_profile', default_value='evasive_weave'),
        DeclareLaunchArgument('target_speed_scale', default_value='1.0'),
        DeclareLaunchArgument('exit_on_complete', default_value='false'),
        DeclareLaunchArgument('summary_json_path', default_value='/tmp/project2_mission_summary.json'),
        DeclareLaunchArgument('log_jsonl_path', default_value='/tmp/project2_mission_trace.jsonl'),
        DeclareLaunchArgument('process_mode', default_value='threaded_batch'),
        DeclareLaunchArgument('max_workers', default_value='0'),
        DeclareLaunchArgument('min_area', default_value='24'),
        DeclareLaunchArgument('enable_multi_signature', default_value='true'),
        DeclareLaunchArgument('enable_multi_target', default_value='true'),
        DeclareLaunchArgument('enable_triangulation', default_value='true'),
        DeclareLaunchArgument('enable_los_check', default_value='true'),
        DeclareLaunchArgument('auto_discover_drones', default_value='true'),
        OpaqueFunction(function=_launch_setup),
    ])
