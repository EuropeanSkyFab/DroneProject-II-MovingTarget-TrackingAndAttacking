# 课题二：面向动态目标的无人机视觉跟踪与自主打击

本仓库为课题二系统集成与全链路闭环工程主仓库，基于 **ROS 2 Humble**、**PX4 Autopilot (`v1.17.0` / `uXRCE-DDS`)** 与 **Gazebo Harmonic** 构建三机协同立体视觉跟踪、机动轨迹预测与自主拦截打击全套系统。

---

## 一、系统四包协同架构与目录结构

全系统严格遵循《课题二内部接口规范 V3.0》与《课题二-课题三接口规范》，划分为四大正交 ROS 2 功能包，上层模块统一采用全局 **ENU（东-北-天）** 坐标系，仅由底层飞控桥接层执行 **ENU ↔ NED** 精确变换：

```text
ros2_ws/
├── .gitignore
├── README.md
├── 文档/
│   ├── 内部分工细节.md
│   ├── 课题二内部接口规范.md
│   ├── 课题二-课题三-接口规范.md
│   └── 进度报告.md
├── mock_nodes/                           # 跨机无仿真干跑节点与可视化联调台
│   ├── launch_control.py                 # 统一 GUI 联调控制台（支持 ALL 群发与真实规划器切换）
│   ├── mock_mavsdk_bridge_dry.py         # 纯运动学干跑桥接节点（无需 PX4/Gazebo）
│   ├── mock_plan_node.py                 # 基础规划模拟节点
│   ├── mock_vision_node.py               # 基础视觉模拟节点
│   └── fake_camera.py                    # 合成针孔相机流注入器
└── src/
    ├── vision_pkg/                       # 【模块一：机载视觉感知（接口 A 提供方）】
    │   ├── vision_pkg/
    │   │   ├── detector.py               # 抗阳光过曝红度增强色块检测 + YOLOv8 可选后端
    │   │   ├── geo_utils.py              # 35° 前斜外参、机体杆臂补偿、多机异面视线最小二乘立体三角测量
    │   │   └── vision_node.py            # 35 帧里程计延迟对齐缓存、双/三机真 3D 定位与单目自适应高度回退
    │   ├── launch/vision_node.launch.py
    │   └── test/                         # 立体三角化与抗眩光检测单元测试
    ├── planning_pkg/                     # 【模块二：机动目标估计与协同制导规划（接口 C 提供方）】
    │   ├── planning_pkg/
    │   │   ├── estimator.py              # 9 维 Singer-CA + CTRV 曲率卡尔曼滤波、马氏门控与 <=3.0s 丢星外推
    │   │   ├── guidance.py               # 120° 锁相合围、Apollonius 可达时间求解、钳形截击与 3D APN 末端制导
    │   │   └── planning_node.py          # 融合接口 A/B，20Hz 输出 /planning/target_pose 与诊断流
    │   ├── launch/planning_node.launch.py
    │   └── test/                         # 状态估计器与协同制导律单元测试
    ├── center_node_pkg/                  # 【模块三：战术调度与三维安全防撞中枢】
    │   ├── center_node_pkg/
    │   │   └── center_node.py            # 全状态机调度、ALL 广播、NaN/越界/超时门禁、三维切向+垂直避障、云台指向与真实 6S 遥测
    │   ├── launch/center_node.launch.py
    │   └── test/                         # 状态机、防撞、版权/代码规范与跨包混沌故障注入测试
    └── flight_ctrl_pkg/                  # 【模块四：50Hz uXRCE-DDS 飞控桥接与高动态对抗仿真场】
        ├── flight_ctrl_pkg/
        │   ├── px4_bridge_node.py        # 50Hz Pos+Vel+Acc Shadow-FF 桥接、MAVLink v2 三轴云台、心跳看门狗
        │   ├── camera_relay_node.py      # gz.transport13 零拷贝低延迟相机转发与墙钟时间戳平滑对齐
        │   ├── target_sim_node.py        # 红方三维高机动逃逸靶机物理控制器（对齐 Gazebo 仿真时钟）与外部雷达（接口 B）
        │   ├── mission_commander_node.py # 全流程自主战略指挥官与结构化 JSONL 评测裁判节点
        │   └── coord_utils.py            # ENU<->NED 刚体姿态变换与版本自适应 DDS 话题工具
        ├── launch/
        │   ├── flight_ctrl_single.launch.py
        │   ├── flight_ctrl_swarm.launch.py
        │   └── project2_full_swarm.launch.py  # 四包全系统一键协同总启动入口
        ├── scripts/
        │   ├── env.sh                    # 环境隔离与激活脚本（自动屏蔽宿主 ROS 1 干扰）
        │   ├── px4-rc.params             # PX4 v1.17.0 高机动推力球分配、防偏航电机饱和与云台参数表
        │   ├── start_px4_swarm.sh        # 一键启动 MicroXRCEAgent + Gazebo Harmonic + 3x PX4 SITL
        │   └── run_full_mission.sh       # 单命令全自主实弹追踪打击大联考启动脚本
        └── sim/                          # 160m 战术试飞场 SDF 世界、H1/H2/H3 停机坪、x500_down_cam 与红方靶机模型
```

---

## 二、快速编译与全量自动化测试

```bash
cd /home/hkw/flying/ros2_ws

# 1. 加载纯净 ROS 2 Humble + px4_msgs 环境
source ./src/flight_ctrl_pkg/scripts/env.sh

# 2. 编译全部四个功能包
colcon build
source ./install/setup.bash

# 3. 运行全工作空间单元测试、规范检查与跨包故障注入测试
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 colcon test
colcon test-result --verbose
# 验收结果: Summary: 22 tests, 0 errors, 0 failures, 0 skipped
```

---

## 三、一键运行三机四包全闭环动态追踪与自主打击任务

### 方式 A：单命令全自动闭环演示与量化评测（推荐）
自动拉起 `MicroXRCEAgent` + `Gazebo Harmonic` + `3× PX4 v1.17.0 SITL` + 四大功能包全部节点，自动执行 **起飞 (`TAKEOFF`) → 平行巡逻搜索 (`RESUME_PATROL`) → 120° 三机立体视觉合围跟踪 (`TRACK`) → 钳形截击 (`INTERCEPT`) → 末端高速精确打击 (`ATTACK`) → 分层返航精准着陆停机坪 (`RETURN`)**：

```bash
./src/flight_ctrl_pkg/scripts/run_full_mission.sh 1   # 1=无头高速模式, 0=开启 Gazebo 3D 图形窗口
```
任务完成后自动在 `/tmp/project2_mission_summary.json` 输出量化验收指标，并在 `/tmp/project2_mission_trace.jsonl` 生成全链路结构化追踪日志。

### 方式 B：分终端手动控制或接入课题三战略指控 (`/strategy/cmd`)

```bash
# 终端 1：启动底层三机 PX4 v1.17.0 + Gazebo 仿真集群（出生点对齐 H1(0,0), H2(0,3), H3(0,6)）
./src/flight_ctrl_pkg/scripts/start_px4_swarm.sh 3 1 1

# 终端 2：一键拉起四包全系统（不启动内置自动指挥官，等待外部 /strategy/cmd 指令）
source ./src/flight_ctrl_pkg/scripts/env.sh
source ./install/setup.bash
ros2 launch flight_ctrl_pkg project2_full_swarm.launch.py auto_mission:=false

# 终端 3：下发战略指令（支持单机 ID 或 "ALL" 集群广播）
source ./src/flight_ctrl_pkg/scripts/env.sh
ros2 topic pub --once /strategy/cmd std_msgs/msg/String "{data: '{\"drone_id\": \"ALL\", \"action\": \"TAKEOFF\"}'}"
ros2 topic pub --once /strategy/cmd std_msgs/msg/String "{data: '{\"drone_id\": \"ALL\", \"action\": \"TRACK\"}'}"
ros2 topic pub --once /strategy/cmd std_msgs/msg/String "{data: '{\"drone_id\": \"ALL\", \"action\": \"INTERCEPT\"}'}"
ros2 topic pub --once /strategy/cmd std_msgs/msg/String "{data: '{\"drone_id\": \"drone_1\", \"action\": \"ATTACK\"}'}"
ros2 topic pub --once /strategy/cmd std_msgs/msg/String "{data: '{\"drone_id\": \"ALL\", \"action\": \"RETURN\"}'}"
```

### 方式 C：无 PX4/Gazebo 轻量化 GUI 联调台
```bash
source ./src/flight_ctrl_pkg/scripts/env.sh
source ./install/setup.bash
python3 mock_nodes/launch_control.py
```

---

## 四、系统实测性能与受控消融基准

### 1. PX4 `v1.17.0` + Gazebo Harmonic 三机四包全闭环实测指标

| 评测维度 | 实测指标 | 工程技术支撑 |
| :--- | :--- | :--- |
| **立体真三维视觉占比** | **`90.29%`** (`158 / 175` 帧) | 三机异构视角最小二乘视线交会 + `3.4 m` 最小基线与 `11.5°` 视差门控 |
| **机载视觉全航程 3D 定位误差** | **RMSE `0.6583 m`**（中位数 **`0.5122 m`**，高度 MAE **`0.3314 m`**） | 无需假定靶机固定高度，`65 ms` 渲染延迟里程计环形缓存对齐 |
| **集群飞行最小安全间距** | **`2.9009 m`**（安全红线 `> 2.0 m`） | 高度分层合围 + 中枢切向绕飞与径向排斥复合防撞势场 |
| **高机动逃逸靶末端脱靶量** | **`0.5847 m`**（峰值拦截速度 **`10.38 m/s`**） | 9 维 Singer-CTRV 预测 + Apollonius/3D APN 制导 + 防偏航电机饱和推力分配 |
| **三机自主返航停机坪落点误差** | `H1: 0.159 m`、`H2: 0.163 m`、`H3: 0.463 m` | 分层返航走廊 + 停机坪正上方悬停刹停对准后垂直着陆上锁 |

### 2. 受控消融实验基准（`seed=42`）

| 子系统实验组 | 基线方法 1 | 基线方法 2 | 本系统方案 | 核心收益 |
| :--- | :--- | :--- | :--- | :--- |
| **三维视觉定位误差** ($N=600$) | 单目固定高度面：`2.116 ± 0.080 m` | 双机立体三角化：`0.034 ± 0.001 m` | **三机立体三角化：`0.026 ± 0.001 m`** | 消除靶机高度机动引起的单目投影发散（RMSE 降低 **81.4×**） |
| **1.4s 视觉遮挡外推误差** ($N=200$) | 零阶保持 (ZOH)：`5.48 ± 0.03 m` | 6 维线性 CV 滤波：`2.28 ± 0.11 m` | **9 维 Singer-CA + CTRV：`1.96 ± 0.11 m`** | 遮挡期间保持机动曲率外推，误差较 ZOH 降低 **64.3%** |
| **三维机动靶末端脱靶量** ($N=36$) | 纯追踪 (Pure Pursuit)：`0.794 ± 0.104 m` | 线性速度提前量：`0.296 ± 0.048 m` | **Apollonius + 3D APN：`0.172 ± 0.023 m`** | 脱靶量较纯追踪降低 **4.6×** |

---

## 五、核心工程纪律与安全保护机制

1. **坐标系隔离契约**：`vision_pkg`、`planning_pkg`、`center_node_pkg` 内部严禁混入 NED 转换，统一采用 `frame_id="world"` 的全局 ENU 坐标系。
2. **输入防腐门禁**：所有订阅回调执行 `copy.deepcopy`、时间戳保鲜期校验（视觉 `<=0.5s`、规划 `<=0.8s`）及 `NaN`/`Inf`/电子围栏边界过滤。
3. **双层失控安全保护**：中枢心跳丢失超 `1.5s` 桥接节点自动切入 `AUTO_LAND`；PX4 固件层固化 `COM_OBL_RC_ACT=4 (Land)`、`COM_OF_LOSS_T=0.5s` 与 `NAV_DLL_ACT=0`，杜绝进程异常退出导致飞控漂移爬升。
