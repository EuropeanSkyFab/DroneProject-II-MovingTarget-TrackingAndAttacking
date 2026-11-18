# 规划算法节点一键启动
# 启动命令：ros2 launch algorithm_pkg algorithm.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='algorithm_pkg',
            executable='algorithm_node',
            name='algorithm_node',
            output='screen',
            parameters=[
                # 与 center_node.py 的 swarm_config 保持一致；
                # 三机不一致时算法会接受指令但不发对应飞机的点。
                {'drone_ids': ['drone_1', 'drone_2', 'drone_3']},

                # 飞控默认控制上限：default=0.14×20、intercept=0.38×20、
                # attack=0.62×20。v_max 只作为兜底，不再拿一个值套所有模式。
                {'v_max': 6.0},
                {'v_max_intercept': 7.6},
                {'v_max_attack': 12.4},
                {'t_go_max': 8.0},

                # P1 目标估计窗口；视觉 10Hz 时 1s 内通常有 5~10 帧。
                {'max_age': 1.0},
                # P2 发点门限：超过 0.5s 的预测不再用于发点。
                {'max_pred_age': 0.5},
                {'max_speed': 15.0},

                # 目标过近不输出点；INTERCEPT 不可达时尾随距离。
                {'min_distance': 0.5},
                {'d_tail': 4.0},

                # 与中枢 plan_callback / send_single_target 的边界一致。
                {'min_alt': 3.0},
                {'safe_alt': 5.0},
                {'max_alt': 50.0},
                {'max_xy': 1000.0},
                {'max_step': 0.0},  # 0 = 不限制单帧跳变

                # 没收到 /strategy/cmd 前的默认模式；HOVER 最安全。
                {'default_action': 'HOVER'},
            ],
        )
    ])
