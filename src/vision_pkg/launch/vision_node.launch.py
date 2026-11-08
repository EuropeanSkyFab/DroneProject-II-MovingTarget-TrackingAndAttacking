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

                # ===== 多机集群配置（默认值已符合三机协同，一般不用改） =====
                # {'drone_ids': ['drone_1', 'drone_2', 'drone_3']},
                # {'process_mode': 'round_robin'},   # 轮转处理(省算力) / 'parallel'(有显卡时用)
                # {'per_stream_hz': 10.0},           # 轮转模式下每路处理频率

                # ===== 切换 YOLO 检测器时取消注释 =====
                # {'detector_type': 'yolo'},
                # {'yolo_model_path': 'yolov8n.pt'},
            ]
        )
    ])
