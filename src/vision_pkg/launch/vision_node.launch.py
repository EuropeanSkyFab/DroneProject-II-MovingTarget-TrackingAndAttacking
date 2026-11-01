# 机载视觉节点一键启动
# 启动命令（严禁 ros2 run）：ros2 launch vision_pkg vision_node.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vision_pkg',
            executable='vision_node',
            name='vision_node',
            output='screen',
            parameters=[
                # 注：use_sim_time 当前全系统禁用（WSL 时钟兼容问题），不要加！
                # {'use_sim_time': True}

                # 需要改配置时在这里加，例如切换 YOLO 检测器：
                # {'detector_type': 'yolo'},
                # {'yolo_model_path': 'yolov8n.pt'},
            ]
        )
    ])
