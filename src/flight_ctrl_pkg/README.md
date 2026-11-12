# `flight_ctrl_pkg` — 课题二飞控调试与 PX4 uXRCE-DDS 全特性桥接模块

严格遵守《课题二内部接口规范 V3.0》与核心开发铁律：本功能包完全独立位于 `src/flight_ctrl_pkg/`，不修改中枢 (`center_node_pkg`) 与视觉 (`vision_pkg`) 源码。

## 功能组成

1. **`px4_bridge_node`**：基于 **uXRCE-DDS (`px4_msgs`)** 的 **50 Hz** 多机高阶飞控桥接节点：
   - **标准接口对齐**：订阅 `/{drone_id}/fmu/in/trajectory_setpoint` (ENU)、`/{drone_id}/action`、`/{drone_id}/heartbeat`、`/{drone_id}/fmu/in/gimbal_setpoint` (`Vector3`)；将 PX4 局部 NED / FRD 位姿与四元数姿态精确转换至全局 ENU / FLU 并发布至 `/{drone_id}/state`。
   - **三自由度联合前馈与敏捷步长牵引**（`compute_shadow_ff_accel`，`control_hz=50.0`、`use_vel_feedforward=True`、`use_acc_feedforward=True`）：协同开启 `OffboardControlMode.position`、`velocity`、`acceleration`，同步下发平滑位置影子点、水平/垂直速度前馈、径向加速度前馈、航向角及偏航角速度前馈（`compute_yawspeed_ned`）；按 `action` 分级设置时间步不变步长上限（`max_step_default=0.14` / `max_step_intercept=0.38` / `max_step_attack=0.62`，内部按参考速率 `20 Hz` 折算等价速度上限）；`TRACK/HOVER` 强制保持 `min_track_alt=8.0m` 俯视占位（`target_ref_alt=5.5m`，`INTERCEPT/ATTACK/RETURN` 解锁立体俯冲与降落）；降落前执行水平刹车，心跳超时自动切入 `AUTO_LAND` 并在触地 `disarm` 后清空内部前馈状态。
   - **PX4 原生三轴稳像云台接管**（`enable_gimbal_cmd=True`）：启动后自动发送 `VEHICLE_CMD_DO_GIMBAL_MANAGER_CONFIGURE`（严格过滤系统 ID/组件 ID 为 `(1, 1)` 的 `VehicleCommandAck` 以排除 Navigator 启动期释放指令的 `TRANSIENT_LOCAL` 伪确认，若收到 `RESULT_DENIED` 自动重配），并按 `action` 持续发送 `VEHICLE_CMD_DO_GIMBAL_MANAGER_PITCHYAW` 控制真实云台伺服跟踪空中目标。
   - **飞控健康监控与 6S 电池遥测**：订阅 `VehicleCommandAck`、`FailsafeFlags`、`EstimatorStatusFlags` 与 `BatteryStatus`，汇总发布至 `/{drone_id}/fmu/out/fc_status` (`DiagnosticStatus`，含 `nav_state`、`arming_state`、`ekf_aligned`、`geofence_breached`、`gimbal_configured`、`acc_ff`) 与 `/{drone_id}/fmu/out/battery_raw`。

2. **`camera_relay_node`**：基于 `gz.transport13` + `gz.msgs10`（兼容 `ros2` 降级模式）的高吞吐相机直连转发节点（支持通过 `out_namespace` 同时桥接多路机载传感器）：
   - 使用 `array.array('B').frombytes()` 实现零逐字节拷贝封装（将单帧 `800x450` 赋值耗时从 ~110 ms 降至 `< 0.1 ms`），在 RTX 3090 硬件渲染下稳定以 `24.0~24.5 Hz` 双路并发发布：
     - 前斜 35° 导航/视觉相机：`/{drone_id}/camera/image_raw`、`/{drone_id}/camera/camera_info`（`640x480 @ 25Hz`）
     - 三轴稳像长焦云台相机：`/{drone_id}/gimbal_cam/image_raw`、`/{drone_id}/gimbal_cam/camera_info`（`800x450 @ 25Hz`、`hfov=0.82`）
   - 在全系统禁用 `use_sim_time` 约束下将仿真时间戳对齐至系统墙钟（实测图像滞后 `< 0.1s`）。

3. **`dynamics_calibrator`**：阶跃响应自动化测试节点，输出仿真机型实测 `V_cruise`、`V_max`、`a_max`、`yaw_rate`、`R_min` 供规划模块对齐。

4. **RTX 3090 硬件渲染训练场与 PX4 参数固化**：
   - **`scripts/start_px4_swarm.sh`**：无头仿真显式取消 `DISPLAY` 并绑定 NVIDIA EGL ICD (`__EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/10_nvidia.json`)，确保 Gazebo Harmonic OGRE2 直接运行在 RTX 3090 硬件管线上（`RTF ~ 0.99`）。
   - **`sim/worlds/moving_target_arena.sdf` + `sim/models/arena_assets` + `sim/models/red_target_uav`**：包含 PBR 纹理草坪地形、黄金时段低仰角斜射阳光（`intensity=1.85`）、沥青跑道标线、专用 `H1(0,0)/H2(0,3)/H3(0,6)` 停机坪、双太阳能屋顶机库、集装箱、双完整四轮 `r1_rover` 无人车（`parked_ugv_west` / `parked_ugv_east`）、4 座边缘发光导航信标立柱、旋转雷达塔台、6 棵多层圆锥松树与丘陵围栏；空中靶机升级为绯红涂装 **Skywalker X8 VTOL 复合飞翼模型**（`red_moving_target`，含前掠主机翼、`x8_elevon`、四轴垂起旋翼 + 尾部推力螺旋桨，初始悬停 `z=5.5m` 位于 `(6.8, 0.5, 5.5)`）。
   - **`sim/models/x500_down_cam`**：保留无侧滚畸变的 **35° 前斜下视视觉相机**（`0.12 0.0 -0.04 0.0 0.61086524 0.0`，`horizontal_fov=1.35`），同时叠加物理可动的 Yaw/Roll/Pitch 三轴云台关节 (`servo_1/2/0`) 与长焦 `gimbal_cam`；配合未修改的 `vision_pkg` 时需显式对齐 `camera_mount_rpy:=[0.95993109, 0.0, -1.57079633]` 与 `ground_z:=5.5`。
   - **`scripts/px4-rc.params`**：
     - 电子围栏与双层保护：`GF_ACTION=2`、`GF_MAX_HOR_DIST=150.0`、`GF_MAX_VER_DIST=45.0`、`COM_OBL_RC_ACT=4` (`AUTO_LAND`)、`COM_OF_LOSS_T=0.5`、`COM_LOW_BAT_ACT=2`。
     - 敏捷多旋翼闭环增益：`MPC_TILTMAX_AIR=55.0`、`MPC_THR_MAX=0.95`、`MPC_XY_P=1.6`、`MPC_Z_P=1.4`、`MPC_XY_VEL_MAX=14.0`、`MPC_XY_CRUISE=6.5`、`MPC_ACC_HOR=8.0`、`MPC_ACC_HOR_MAX=10.5`、`MPC_JERK_AUTO=18.0`、`MPC_YAWRAUTO_MAX=180.0`。
     - 原生三轴云台混控：`MNT_MODE_IN=4`、`MNT_MODE_OUT=0`、`MNT_DO_STAB=1`、`MNT_RANGE_PITCH/YAW/ROLL=114.6`、`SIM_GZ_SV_FUNC1/2/3=421/422/420`。
     - 6S 锂电池仿真模型：`SIM_BAT_ENABLE=1`、`BAT1_N_CELLS=6`、`BAT1_V_CHARGED=4.20`、`BAT1_V_EMPTY=3.50`、`SIM_BAT_DRAIN=600.0`、`SIM_BAT_MIN_PCT=15.0`。

## 接口对齐与协定说明

1. **云台协同（接口 I，`/{drone_id}/fmu/in/gimbal_setpoint`）**：团队接口草稿曾标写 `GimbalManagerSetPitchYaw（暂定）`，实际联调中的 `center_node.py` 发布的是 `geometry_msgs/msg/Vector3`（`x=pitch_rad, y=yaw_ned_rad`）。`px4_bridge_node` 已严格对齐现网中枢 `Vector3`，并在内部换算为 PX4 `VEHICLE_CMD_DO_GIMBAL_MANAGER_PITCHYAW`。
2. **电池与飞控健康（`/{drone_id}/fmu/out/battery_raw`、`fc_status`）**：由于当前 `center_node.py` 自行在 `/{drone_id}/battery` 发布固定占位电量并在 `/{drone_id}/link_status` 发布链路状态且未订阅底层电池，本包将 PX4 6S 真实电池遥测与健康诊断发布至 `/{drone_id}/fmu/out/battery_raw` 与 `/{drone_id}/fmu/out/fc_status`，避免同话题双发布者冲突。
3. **PX4 `v1.15.4` 断联参数说明**：`scripts/px4-rc.params` 中设置 `COM_OBL_RC_ACT=4`（经核对 PX4 `v1.15.4` `commander_params.c`，`3=Return`、`4=Land`、`5=Hold`），确保 Offboard 断联 `0.5s` 内进入 `AUTO_LAND`。

## 实测机型动力学参数（PX4 `v1.15.4` `x500` SITL + 50 Hz 位置/速度/加速度前馈）

通过 `ros2 launch flight_ctrl_pkg dynamics_calibrator.launch.py step_dist:=20.0` 实测得到：

| 参数 | 实测值 | 对应控制档位 / 说明 |
|:---|:---|:---|
| `V_cruise` | `3.73 m/s` | `TRACK/HOVER/RETURN` 巡航档实测速度 |
| `V_max` | `12.98 m/s` | `ATTACK` 高机动前馈冲刺实测峰值速度（`MPC_XY_VEL_MAX=14.0`、`MPC_TILTMAX_AIR=55.0`） |
| `a_max` | `9.01 m/s^2` | 阶跃加速阶段 `0.10 s` 差分窗口实测峰值水平加速度（对齐 `MPC_ACC_HOR_MAX=10.5`） |
| `yaw_rate` | `166.4 deg/s` | 横向机动偏航对准实测角速度峰值（对齐 `MPC_YAWRAUTO_MAX=180.0`） |
| `R_min(估)` | `1.54 m` | 按 `V_cruise^2 / a_max` 估算的最小转弯半径 |

## 快速启动（在 `ros2_ws` 根目录执行）

```bash
# 0. 首次拉取或修改后编译工作空间
cd ~/ros2_ws
colcon build --packages-select flight_ctrl_pkg

# 1. 加载环境（支持源码或 install 目录自动探测工作空间，隔离 ROS 1 并优先兼容 px4_ros2 / Humble）
source ./src/flight_ctrl_pkg/scripts/env.sh

# 2. 启动 PX4 SITL + Gazebo Harmonic + MicroXRCEAgent
./src/flight_ctrl_pkg/scripts/start_px4_swarm.sh 1 1 1
# 三机协同：./src/flight_ctrl_pkg/scripts/start_px4_swarm.sh 3 1 1

# 3. 另开终端启动飞控桥接与双路相机转发（统一使用 ros2 launch）
source ./src/flight_ctrl_pkg/scripts/env.sh
ros2 launch flight_ctrl_pkg flight_ctrl_single.launch.py
# 三机协同：ros2 launch flight_ctrl_pkg flight_ctrl_swarm.launch.py

# 4. 单元测试验收（若环境中存在旧版 launch_testing 与新 pytest 冲突，请禁用自动插件加载）
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 colcon test --packages-select flight_ctrl_pkg
colcon test-result --test-result-base build/flight_ctrl_pkg --verbose
```

> **联调说明（对齐 `x500_down_cam` 与靶机 `z=5.5m`）**：未改动 `vision_pkg` 时，其默认参数 `camera_mount_rpy=[0,0,0]`、`ground_z=0.0` 面向垂直朝下地面靶标；联调本场馆 `35°` 前斜相机与 `z=5.5m` 空中红靶机时，需在视觉端传入 `ground_z:=5.5` 与 `camera_mount_rpy:=[0.95993109, 0.0, -1.57079633]`（如通过 ROS 2 `--ros-args -p` 覆盖参数，或在 `vision_node.launch.py` 本地配置中取消对应参数注释）。
