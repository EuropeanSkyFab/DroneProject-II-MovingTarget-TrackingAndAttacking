# 规划算法节点一键启动
# 启动命令：ros2 launch algorithm_pkg algorithm.launch.py
#
# 注意：本 launch 只负责规划节点；中枢、视觉、桥接分别由
# center_node_pkg / vision_pkg / mock_nodes 启动。联调顺序见 README。
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

                # 规划定时器频率；中枢消费是 20Hz，两者不要差太多。
                {'publish_rate_hz': 20.0},

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

                # ! 默认 0.62 = 12.4 m/s × 0.05s，不是拍脑袋的平滑值：
                # ! - 0.30/0.38 作为全局默认会把 ATTACK 压到 6/7.6 m/s，
                # !   ATTACK 最小目标距离从 15.83m 恶化到 50.4/44.8m；
                # ! - 单值 0.62 在 ATTACK 场景盲飞 0.00m（中枢 2s 保留期间），
                # !   与“按模式 0.30/0.38/0.62”相比没有功能损失。
                # ! 0 仅保留为“不限幅”调试入口，联调/交付不要用。
                {'max_step': 0.62},
                {'target_timeout': 0.5},
                {'history_sec': 3.0},

                # 没收到 /strategy/cmd 前的默认模式；HOVER 最安全。
                {'default_action': 'HOVER'},

                # ---------------- P3 异常降级参数（初值，待联调标定） ----------------
                # 目标 0.5s 内正常、0.5~1.0s 外推、超过 1.0s 保持。
                {'t_extrap_max': 1.0},
                # 20Hz 下 n_enter=2 → 0.10s、n_exit=1 → 0.05s、
                # n_recover=3 → 0.15s。
                {'n_enter': 2},
                {'n_exit': 1},
                {'n_recover': 3},
                {'max_step_recover': 0.2},
                {'n_recover_step': 5},
                {'freq_anomaly_enabled': False},
                {'freq_anomaly_threshold': 0.5},
                {'freq_window_size': 20},
                {'hold_point_source': 'last_valid'},
                {'ego_timeout': 1.0},
                {'target_min_alt': 0.0},
            ],
        )
    ])
