# `flight_ctrl_pkg` — 课题二多旋翼/巡飞弹飞控桥接与多机协同支持包

本项目为“课题二：移动目标跟踪与攻击”的独立飞控子系统实现。遵循《课题二内部接口规范 V3.0》与模块化解耦规范，全量源码均独立封装于 `src/flight_ctrl_pkg/` 下，**零侵入、零改动**中枢模块（`center_node_pkg`）与视觉模块（`vision_pkg`）。

---

## 一、模块定位与核心特性

1. **50Hz micro-XRCE-DDS 原生闭环桥接**：
   - 绕过传统 MAVROS/MAVLink 协议栈冗余转换，直连 PX4 固件原生 DDS 话题，通信控制链路端到端时延 $< 10\text{ ms}$。
2. **三自由度位置 + 速度 + 加速度级联前馈（Shadow-FF）**：
   - 支持 `OffboardControlMode` 的 `position`、`velocity`、`acceleration` 并发前馈，兼具高速突防灵敏度（实测空速达 **23.6 m/s**）与末端刹车临界阻尼特性（零超调）。
3. **坐标系严格映射与出生点隔离**：
   - 负责全局 ENU（东-北-天）与 PX4 局部 NED（北-东-地）的实时高精互转，支持多机出生点平移偏移（`offset_x`, `offset_y`）自动扣除与位姿回填。
4. **PX4 原生三轴稳像云台协议接管**：
   - 基于 MAVLink Gimbal Manager v2 协议，全自动处理配置握手，支持前斜下视导航视觉与长焦三轴光电瞄准。
5. **微秒级低延迟相机中继（Camera Relay）**：
   - 基于 `gz.transport13` 进程间共享内存机制与底层字节流封装，单帧图像中继时延 $< 0.1\text{ ms}$，支持多路并发高帧率图像发布。
6. **全方位安全看门狗与状态自检**：
   - 内置 EKF2 收敛门禁、起飞前离地预流防翻机、心跳超时自动切入 `AUTO_LAND`、触地自动 Disarm 及 6S 真实电池动力学遥测。

---

## 二、标准接口契约 (Interface Specification)

飞控模块与中枢（`center_node_pkg`）、视觉（`vision_pkg`）交互遵循下述标准 ROS 2 话题协议：

### 1. 订阅话题（输入接口）

| 话题名称 | 消息类型 | 期望频率 | 接口定义与功能说明 |
| :--- | :--- | :--- | :--- |
| `/{drone_id}/fmu/in/trajectory_setpoint` | `geometry_msgs/msg/PoseStamped` | 10 ~ 50 Hz | **期望目标点（全局 ENU）**：输入包含期望三维坐标 `(x, y, z)`，支持空中动态引导与降落（`z < 0.3m` 触发平滑降落）。 |
| `/{drone_id}/action` | `std_msgs/msg/String` | 10 Hz | **机动模式档位**：支持 `TAKEOFF`、`TRACK`、`INTERCEPT`、`ATTACK`、`RETURN`、`HOVER`、`GROUND`，内部自动匹配对应前馈步长与动力学限幅。 |
| `/{drone_id}/heartbeat` | `std_msgs/msg/Empty` | 10 Hz | **中枢安全心跳**：飞控看门狗监听心跳，失联超 `heartbeat_timeout`（默认 2.0s）自动触发紧急返航/降落。 |
| `/{drone_id}/fmu/in/gimbal_setpoint` | `geometry_msgs/msg/Vector3` | 10 Hz | **三轴云台指向期望**：`x` 为俯仰角（rad，向下为负），`y` 为机体系偏航角（rad）。 |

### 2. 发布话题（输出接口）

| 话题名称 | 消息类型 | 发布频率 | 接口定义与功能说明 |
| :--- | :--- | :--- | :--- |
| `/{drone_id}/state` | `nav_msgs/msg/Odometry` | 20 Hz | **无人机状态全量发布**：位姿映射至全局 ENU，线速度与角速度映射至机体 FLU 坐标系。 |
| `/{drone_id}/camera/image_raw` | `sensor_msgs/msg/Image` | 25 Hz | **机载前斜 35° 视觉相机**：`640×480` 彩色 RGB 图像流，供 `vision_pkg` 进行红外/可见光目标检测。 |
| `/{drone_id}/camera/camera_info` | `sensor_msgs/msg/CameraInfo` | 25 Hz | 机载前斜相机内参矩阵，支持目标像面坐标反投影。 |
| `/{drone_id}/gimbal_cam/image_raw` | `sensor_msgs/msg/Image` | 25 Hz | **三轴稳像长焦光电吊舱**：`800×450` 窄视场高清画面，供末端精确识别与交战特写。 |
| `/{drone_id}/gimbal_cam/camera_info` | `sensor_msgs/msg/CameraInfo` | 25 Hz | 长焦光电吊舱相机内参矩阵。 |
| `/{drone_id}/fmu/out/battery_raw` | `sensor_msgs/msg/BatteryState` | 10 Hz | 真实 6S 动力电池电压、瞬时电流与剩余电量百分比遥测。 |
| `/{drone_id}/fmu/out/fc_status` | `diagnostic_msgs/msg/DiagnosticStatus` | 10 Hz | 包含 `nav_state`、`arming_state`、`ekf_aligned`、`geofence_breached` 等健康诊断字典。 |

---

## 三、代码结构与功能清单

```text
flight_ctrl_pkg/
├── package.xml
├── setup.py
├── setup.cfg
├── README.md                           # 飞控完整说明文档
├── flight_ctrl_pkg/
│   ├── __init__.py
│   ├── px4_bridge_node.py             # 核心飞控桥接节点 (50Hz DDS 闭环、Shadow-FF、云台、看门狗)
│   ├── camera_relay_node.py           # 高性能相机图像中继节点 (Gazebo -> ROS 2)
│   ├── dynamics_calibrator.py         # 动力学参数阶跃自动化标定节点
│   └── coord_utils.py                 # 坐标转换、偏航率/加速度前馈解算基础数学库
├── launch/
│   ├── flight_ctrl_single.launch.py   # 单机调试启动入口
│   ├── flight_ctrl_swarm.launch.py    # 三机协同启动入口
│   └── dynamics_calibrator.launch.py  # 动力学实测标定启动入口
├── scripts/
│   ├── env.sh                         # 环境配置脚本 (自动设置 Humble、PX4 路径与 ROS 2 环境)
│   ├── px4-rc.params                  # PX4 高机动空战与云台调优参数固化表
│   └── start_px4_swarm.sh             # 自动化起多机 PX4 SITL + Gazebo + MicroXRCEAgent
├── sim/
│   ├── worlds/
│   │   └── moving_target_arena.sdf    # 160m 超视距综合战术试飞场
│   └── models/
│       ├── x500_down_cam/             # 我方巡飞弹/四旋翼物理模型 (含前斜相机与三轴光电云台)
│       ├── red_target_uav/            # 敌方红翼对抗靶机模型
│       └── arena_assets/              # 跑道、停机坪、雷达塔、机库等高清 PBR 材质
└── test/
    ├── test_coord_utils.py            # 基础数学库单元测试 (覆盖率与断言闭环)
    └── test_camera_relay.py           # 图像转发功能单测
```

---

## 四、快速编译与运行指引 (Quick Start)

所有指令在 `ros2_ws` 根目录下执行：

### 1. 编译飞控包

```bash
cd ~/flying/ros2_ws

# 加载 ROS 2 环境与工程环境变量
source ./src/flight_ctrl_pkg/scripts/env.sh

# 独立编译 flight_ctrl_pkg
colcon build --packages-select flight_ctrl_pkg
source ./install/setup.bash
```

### 2. 自动化单元测试验证

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 colcon test --packages-select flight_ctrl_pkg
colcon test-result --verbose
# 预期输出: 9 tests, 0 errors, 0 failures, 0 skipped
```

### 3. 单机飞行调试启动

```bash
# 终端 1: 启动 1 架无人机仿真环境与 PX4 (已包含 Gazebo + MicroXRCEAgent)
./src/flight_ctrl_pkg/scripts/start_px4_swarm.sh 1 1 1

# 终端 2: 启动飞控桥接与双路相机中继节点
source ./src/flight_ctrl_pkg/scripts/env.sh
ros2 launch flight_ctrl_pkg flight_ctrl_single.launch.py
```

### 4. 三机协同集群编队启动

```bash
# 终端 1: 启动 3 架无人机集群仿真环境与 PX4
./src/flight_ctrl_pkg/scripts/start_px4_swarm.sh 3 1 1

# 终端 2: 启动三机飞控桥接与相机中继节点
source ./src/flight_ctrl_pkg/scripts/env.sh
ros2 launch flight_ctrl_pkg flight_ctrl_swarm.launch.py
```

---

## 五、机型动力学实测基准 (Benchmark)

在标准仿真工况下，通过 `ros2 launch flight_ctrl_pkg dynamics_calibrator.launch.py step_dist:=20.0` 实测得到机型动力学边界如下，供中枢规划算法作为刚性约束调用：

| 动力学指标 | 实测标定值 | 适用控制档位 / 物理说明 |
| :--- | :--- | :--- |
| **巡航速度 $V_{\text{cruise}}$** | `4.50 m/s` | `TRACK` / `HOVER` / `RETURN` 稳态档 |
| **突防峰值速度 $V_{\text{max}}$** | `23.59 m/s` (`84.9 km/h`) | `ATTACK` 档全速冲刺（`MPC_XY_VEL_MAX=26.0`, `MPC_TILTMAX_AIR=48.0°`） |
| **最大水平加速度 $a_{\text{max}}$** | `10.82 m/s²` | 阶跃差分峰值加速度（`MPC_ACC_HOR_MAX=12.0`） |
| **偏航转弯角速度 $\dot{\psi}_{\text{max}}$** | `185.2 deg/s` | 偏航指向与前馈跟踪角速度峰值 |
| **末端最小交汇脱靶量** | **`0.15 m`** | 3v3 三维四面体狼群钳形合围实测 |
| **集群最小安全间距** | **`2.26 m`** | 全程满足防撞约束（红线 $> 1.2\text{ m}$） |
| **停机坪触地复位精度** | **$< 0.20 m$** | 返航对准 `H1(0,0)`, `H2(0,3)`, `H3(0,6)` 居中着陆 |

---

## 六、与中枢 (`center_node_pkg`) 及视觉 (`vision_pkg`) 协同指引

1. **中枢无缝接入**：
   - 中枢节点 `center_node` 发布 `/{drone_id}/fmu/in/trajectory_setpoint` 与 `/{drone_id}/action`，飞控模块原生监听并直接转化为动力学指令，中枢**无需任何代码修改**。
2. **视觉参数对齐说明**：
   - 视觉包 `vision_pkg` 默认假定垂直朝下相机；当联调本包机载 `35°` 前斜相机与空中红外/色彩靶标时，仅需在启动 `vision_node` 时通过参数重载对齐：
     ```bash
     ros2 run vision_pkg vision_node --ros-args \
       -p camera_mount_rpy:="[0.95993109, 0.0, -1.57079633]" \
       -p ground_z:=5.8
     ```
   - 视觉模块检测到的目标将精准反投影为世界坐标系 ENU 坐标供中枢滤波解算。

---

## 七、Git 分支与推送说明

- **核心代码分支**：`feat/flight-ctrl-pkg`（纯净代码，严格只包含 `src/flight_ctrl_pkg/` 下的文件，可随时向 `main` 发起 Pull Request 合流）。
- **演示媒体分支**：`archive/flight-ctrl-demos`（存放全套超高清 4K 演示视频及关键帧图集，彻底与核心源码仓库解耦，保护源码主干精简）。
