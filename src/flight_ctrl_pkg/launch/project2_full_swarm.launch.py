# -*- coding: utf-8 -*-
"""
课题二四包全系统协同总启动文件：
  一键拉起 飞控桥接(x3) + 相机转发(x3/x6) + 红方三维机动靶与雷达(target_sim_node)
          + 机载立体视觉(vision_pkg) + 预测与协同制导(planning_pkg) + 安全中枢(center_node_pkg)
          + 可选全自主战术指挥官(mission_commander_node)
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    auto_mission = LaunchConfiguration('auto_mission')
    enable_gimbal_cam_relay = LaunchConfiguration('enable_gimbal_cam_relay')
    target_profile = LaunchConfiguration('target_profile')
    target_speed_scale = LaunchConfiguration('target_speed_scale')
    exit_on_complete = LaunchConfiguration('exit_on_complete')
    summary_json_path = LaunchConfiguration('summary_json_path')
    log_jsonl_path = LaunchConfiguration('log_jsonl_path')

    fc_share = get_package_share_directory('flight_ctrl_pkg')
    vis_share = get_package_share_directory('vision_pkg')
    plan_share = get_package_share_directory('planning_pkg')
    center_share = get_package_share_directory('center_node_pkg')

    return LaunchDescription([
        DeclareLaunchArgument('auto_mission', default_value='false'),
        DeclareLaunchArgument('enable_gimbal_cam_relay', default_value='true'),
        DeclareLaunchArgument('target_profile', default_value='evasive_weave'),
        DeclareLaunchArgument('target_speed_scale', default_value='1.0'),
        DeclareLaunchArgument('exit_on_complete', default_value='false'),
        DeclareLaunchArgument('summary_json_path', default_value='/tmp/project2_mission_summary.json'),
        DeclareLaunchArgument('log_jsonl_path', default_value='/tmp/project2_mission_trace.jsonl'),

        # 1. 三机 PX4 uXRCE-DDS 桥接 + 双路相机转发
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(fc_share, 'launch', 'flight_ctrl_swarm.launch.py')
            ),
            launch_arguments={'enable_gimbal_cam_relay': enable_gimbal_cam_relay}.items(),
        ),

        # 2. 红方高机动靶机闭环驱动与外部雷达 (接口 B)
        Node(
            package='flight_ctrl_pkg',
            executable='target_sim_node',
            name='target_sim_node',
            output='screen',
            parameters=[{
                'sim_backend': 'auto',
                'profile': target_profile,
                'speed_scale': target_speed_scale,
                'enable_radar': True,
            }],
        ),

        # 3. 机载视觉检测与多机三维立体三角定位 (接口 A)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(vis_share, 'launch', 'vision_node.launch.py')
            ),
        ),

        # 4. 九维 Singer/CTRV 状态估计与多机协同制导规划 (接口 C)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(plan_share, 'launch', 'planning_node.launch.py')
            ),
        ),

        # 5. 系统集成中枢状态机与三维防撞总闸
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(center_share, 'launch', 'center_node.launch.py')
            ),
        ),

        # 6. 可选：全自主战术指挥官与结构化评测裁判
        Node(
            package='flight_ctrl_pkg',
            executable='mission_commander_node',
            name='mission_commander_node',
            output='screen',
            condition=IfCondition(auto_mission),
            parameters=[{
                'exit_on_complete': exit_on_complete,
                'summary_json_path': summary_json_path,
                'log_jsonl_path': log_jsonl_path,
            }],
        ),
    ])
