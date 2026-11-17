# `flight_ctrl_pkg` — 课题二多旋翼/巡飞弹飞控桥接与全闭环对抗仿真包

本包提供课题二底层 **50Hz uXRCE-DDS 高机动飞控桥接**、**低延迟机载相机桥接**、**红方三维高机动靶机与雷达模拟器**、**全流程自主战略指挥裁判节点** 以及 **160m 综合战术试飞场（Gazebo Harmonic）**，已完成对 **PX4 `v1.17.0` + `px4_msgs` (`release/1.17`) + `Micro-XRCE-DDS-Agent v2.4.3`** 的全链路实测标定。

---

## 一、核心模块与特性

1. **50Hz micro-XRCE-DDS 原生级联前馈桥接 (`px4_bridge_node.py`)**：
   - 支持 `OffboardControlMode` 的 `position + velocity + acceleration` 三自由度并发前馈（Shadow-FF），内置防积分饱和拴绳（Anti-Windup Tether）与分模式动态限幅（`TRACK` / `INTERCEPT` / `ATTACK`）。
   - 针对四旋翼高速机动混控特性，在 `px4-rc.params` 中精细配平推力球水平裕度（`MPC_THR_XY_MARG=0.38`、`MPC_TILTMAX_AIR=36.0°`）与偏航角速率上限（`MC_YAWRATE_MAX=80.0 deg/s`），根除高速急转时对角电机偏航饱和引发的集体升力损失与高度下掉。
   - 原生集成 MAVLink Gimbal Manager v2 协议，实时接收中枢三维俯仰+偏航角设定点，驱动三轴稳像光电吊舱锁死机动目标。
2. **零拷贝相机中继与墙钟平滑对齐 (`camera_relay_node.py`)**：
   - 基于 `gz.transport13` 直连 Gazebo Harmonic 渲染管线，在全系统不启用 `use_sim_time` 的架构下自动完成仿真时间戳到系统墙钟的平滑偏移锁定。
   - 三机集群模式下默认开启 3 路 `down_cam` 前斜导航视觉流 + 主攻机 `gimbal_cam` 长焦特写流，兼顾多机立体定位精度与仿真器实时率（RTF）。
3. **仿真时钟同步红方机动靶与雷达模拟器 (`target_sim_node.py`)**：
   - 实时提取 Gazebo `dynamic_pose/info` 物理仿真时钟驱动三维复合蛇形（`evasive_weave`）、8 字（`figure8`）、盘旋（`circle`）机动轨迹，确保靶机物理速度与角速度不受宿主机实时率波动影响；同步输出 `/sim/target_gt` 真值与带受控高斯噪声的接口 B `/external/target_detected` 雷达引导流。
4. **全自主战略指挥与结构化可观测性裁判 (`mission_commander_node.py`)**：
   - 按物理态势事件驱动 `WAIT_READY -> TAKEOFF -> RESUME_PATROL -> TRACK -> INTERCEPT -> ATTACK -> RETURN -> COMPLETE` 全任务闭环，实时统计立体定位占比、三维定位 RMSE、最小防撞间距、末端脱靶量与返航落点精度，并输出 JSON-Lines 追踪日志。

---

## 二、目录结构

```text
flight_ctrl_pkg/
├── package.xml
├── setup.py
├── README.md
├── flight_ctrl_pkg/
│   ├── coord_utils.py                 # ENU<->NED 位姿/四元数映射、Shadow-FF 前馈与版本自适应 DDS 话题工具
│   ├── px4_bridge_node.py             # 50Hz uXRCE-DDS 飞控桥接、云台伺服、6S 电池遥测与心跳看门狗
│   ├── camera_relay_node.py           # Gazebo Harmonic -> ROS 2 图像/内参转发与时间戳对齐
│   ├── target_sim_node.py             # 红方三维高机动靶机闭环控制器与外部雷达模拟器 (接口 B)
│   ├── mission_commander_node.py      # 全流程自主战术指挥官与 JSONL 结构化评测裁判
│   └── dynamics_calibrator.py         # 阶跃动力学自动化标定工具
├── launch/
│   ├── flight_ctrl_single.launch.py   # 单机桥接启动入口
│   ├── flight_ctrl_swarm.launch.py    # 三机集群桥接启动入口（对齐 H1(0,0), H2(0,3), H3(0,6)）
│   ├── project2_full_swarm.launch.py  # 四包全系统协同总启动入口
│   └── dynamics_calibrator.launch.py  # 动力学标定启动入口
├── scripts/
│   ├── env.sh                         # Conda px4_ros2 / ROS 2 Humble 环境激活与 ROS 1 隔离脚本
│   ├── px4-rc.params                  # PX4 v1.17.0 安全保护、推力分配与三轴云台固化参数表
│   ├── px4-rc.mavlink                 # PX4 启动后置钩子
│   ├── start_px4_swarm.sh             # 一键拉起 MicroXRCEAgent + Gazebo Harmonic + 1/3 架 PX4 SITL
│   └── run_full_mission.sh            # 一键全自主三机四包闭环追踪打击大联考脚本
├── sim/
│   ├── worlds/moving_target_arena.sdf # 160m 综合战术试飞场（含磁力计插件与三座标准停机坪）
│   └── models/                        # x500_down_cam、red_target_uav 与机场设施模型
└── test/
    ├── test_coord_utils.py            # 坐标转换、前馈控制律与 DDS 版本后缀单元测试
    ├── test_camera_relay.py           # 相机时间戳对齐与转发单元测试
    └── test_target_sim.py             # 靶机三维机动参考轨迹边界单元测试
```

---

## 三、快速编译、测试与运行

```bash
cd /home/hkw/flying/ros2_ws
source ./src/flight_ctrl_pkg/scripts/env.sh

# 1. 编译与单元测试
colcon build --packages-select flight_ctrl_pkg
source ./install/setup.bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 colcon test --packages-select flight_ctrl_pkg
colcon test-result --verbose
# 预期输出: 11 tests, 0 errors, 0 failures, 0 skipped

# 2. 一键运行三机四包全自主闭环实弹联考（无头模式）
./src/flight_ctrl_pkg/scripts/run_full_mission.sh 1
```

---

## 四、三机四包 PX4 `v1.17.0` 全闭环实测基准

| 验收指标 | 实测值 | 说明 |
| :--- | :--- | :--- |
| **机载立体三维定位占比** | `90.29%` | `158 / 175` 帧为双机/三机射线三角化真三维解算 |
| **机载视觉全航程定位 RMSE** | `0.6583 m` | 中位数误差 `0.5122 m`，高度 MAE `0.3314 m` |
| **集群全程最小安全间距** | `2.9009 m` | 全程零碰撞风险（高于 `2.0 m` 安全基线） |
| **三维机动靶末端打击脱靶量** | `0.5847 m` | 峰值拦截空速 `10.38 m/s`，零高度失速下掉 |
| **停机坪自主返航落点精度** | `0.159 m / 0.163 m / 0.463 m` | 三机精准降落于 `H1(0,0)`、`H2(0,3)`、`H3(0,6)` 并自动上锁 |

---

## 五、PX4 `v1.17.0` 跨版本兼容要点

1. **动态 DDS 话题版本后缀 (`_vN`)**：`coord_utils.versioned_px4_topic()` 在运行期自动读取 `px4_msgs.msg` 的 `MESSAGE_VERSION`，无缝兼容 `v1.17.0`（`_v1`）与 `v1.15.4`（无后缀）。
2. **参数注入兼容**：`start_px4_swarm.sh` 自动将 `px4-rc.params` 解析导出为 `PX4_PARAM_*` 环境变量注入 `PX4 v1.17.0` 的 `rcS`，同时保留 `PX4_RC_PARAMS_POST` 兼容旧版固件。
3. **版本配对提示**：请确保本地 `PX4-Autopilot` 与 `px4_msgs` 同为 `v1.17.0` / `release/1.17` 成对版本，避免因消息结构体修订号错位导致 DDS 订阅静默失配。
