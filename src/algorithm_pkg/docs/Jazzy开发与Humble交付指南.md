# Docker 开发与 Humble 交付指南

> 文件名沿用旧名，内容已整体改为 Docker-only；原“主机 Jazzy + 容器 Humble”的双环境方案不再使用。
>
> 适用场景：开发机可以是 Ubuntu 24.04、22.04 或 Windows/WSL；**所有 ROS 2 操作都在 Humble Docker 容器里完成**。
>
> 目标：不再维护“主机 Jazzy + 容器 Humble”的双环境。代码在主机编辑，编译、测试、launch、GUI、联调、交付验收全部在同一个 Humble 容器里做。
>
> 原则：**主机只负责编辑和 Git；容器是唯一的 ROS 2 环境。** 不要在主机 Jazzy 环境中 source、编译或启动本项目节点。

---

## 0. 总体做法

| 事项 | 做法 |
|---|---|
| 日常写代码 | 在主机编辑器里改 `src/`、`src/algorithm_pkg/docs/`、`mock_nodes/` |
| 编译 / 测试 / 联调 | 一律进 Humble Docker 容器 |
| 镜像 | 首选本机已有的 `osrf/ros:humble-desktop-full`；没有就 `docker pull` |
| 工作区挂载 | 把本仓库挂到容器 `/ws`，不要用旧容器的 `/root/ros2_ws` 空目录 |
| 网络 / DDS | `--net=host --ipc=host`，团队统一 `ROS_DOMAIN_ID=0` |
| Python 语法 | 以 Python 3.10 为准（Humble 自带 3.10） |
| numpy | 用容器 apt 的 1.x，禁止 `pip install -U numpy` |
| 交付证据 | 容器内 `colcon build`、`colcon test`、`ros2 launch` 的日志/截图 |

---

## 1. 为什么改成 Docker-only

之前“主机 Jazzy 写代码 + Humble 容器验收”的方案有两个问题：

1. **两套环境容易混用**：主机 Jazzy 的 Python 3.12、numpy、cv_bridge 与 Humble 的 Python 3.10、numpy 1.x 不一致，可能在主机环境得到与 Humble 不一致的测试结果。
2. **联调链路必须同发行版**：Jazzy 节点和 Humble 节点混在同一个 `ROS_DOMAIN_ID` 里，DDS 类型哈希、QoS、共享内存都可能出问题，不能作为验收依据。

现在统一为：**主机不装/不 source ROS，所有命令都进 Humble 容器。** 这样只剩一套 Python、一套 numpy、一套 DDS，联调和交付一致。

> 如果主机以前装过 Jazzy，不需要卸载；只要不在本项目的终端里 `source /opt/ros/jazzy/setup.bash` 即可。也不要让 Jazzy 和 Humble 使用同一个 `ROS_DOMAIN_ID` 同时跑节点。

---

## 2. 一次性准备

### 2.1 检查 Docker

```bash
docker --version
docker info
```

如果出现 `permission denied while trying to connect to the Docker daemon socket`：

```bash
sudo usermod -aG docker $USER
# 退出当前终端重新登录，或重启系统
```

### 2.2 准备镜像

本机已有：

```text
osrf/ros:humble-desktop-full
```

没有就拉取：

```bash
docker pull osrf/ros:humble-desktop-full
```

这个镜像包含 ROS 2 Humble、RViz、rqt、`cv_bridge`、`vision_msgs`、`diagnostic_msgs` 等常用组件，适合本项目的纯逻辑联调和 GUI。它**不带 Gazebo/PX4 SITL**；需要 Gazebo 时再单独安装 `ros-humble-gazebo-ros-pkgs`，或换用带 Gazebo 的镜像。

### 2.3 创建开发容器（推荐命令）

先保证宿主机允许容器访问 X11（只做一次，重新登录后可能要再做）：

```bash
xhost +local:root
```

在仓库根目录执行：

```bash
docker run -it \
  --name ros2_humble \
  --net=host \
  --ipc=host \
  -e ROS_DOMAIN_ID=0 \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$HOME/Projects/ros2_ws:/ws" \
  -w /ws \
  osrf/ros:humble-desktop-full bash
```

参数说明：

| 参数 | 作用 |
|---|---|
| `--net=host` | 容器和主机共用网络，简化 DDS 发现 |
| `--ipc=host` | 共享 `/dev/shm`，避免 Fast DDS 共享内存跨容器失败 |
| `-e ROS_DOMAIN_ID=0` | 团队统一 domain；不要和主机 Jazzy 的 domain 混用 |
| `-e DISPLAY` + X11 挂载 | RViz、rqt、Tkinter GUI 需要 |
| `-v "$HOME/Projects/ros2_ws:/ws"` | 代码双向挂载；主机改完容器立即可见 |
| `-w /ws` | 进入容器就在工作空间根目录 |

> **不要沿用旧容器的挂载配置。** 如果执行 `docker inspect ros2_humble_dev --format '{{json .Mounts}}'` 看到的是 `/home/silensky/ros2_ws → /root/ros2_ws`，该目录为空，不是本仓库。继续使用旧容器时，`/root/ros2_ws` 下没有 `src/`，`colcon build` 会显示成功但没有编译任何包。
>
> 处理办法：确认旧容器里没有要保留的东西后删除重建：
>
> ```bash
> docker rm -f ros2_humble_dev
> # 然后按上面的 docker run 重新创建，名字可以继续叫 ros2_humble_dev
> ```
>
> 如果旧容器里有 `apt install` 的依赖，删除后需要重新装；所以长期建议用第 4 节的自定义镜像。

### 2.4 容器内初始化依赖

进入容器后：

```bash
source /opt/ros/humble/setup.bash

apt-get update
apt-get install -y --no-install-recommends \
  python3-pip \
  python3-pytest \
  python3-tk \
  python3-numpy \
  python3-opencv \
  ros-humble-cv-bridge \
  ros-humble-vision-msgs \
  ros-humble-diagnostic-msgs \
  ros-humble-rmw-cyclonedds-cpp

# 只有跑 mock 桥接/视觉 YOLO 时才需要
pip3 install --no-cache-dir mavsdk
# pip3 install --no-cache-dir ultralytics   # 可选，YOLO 后端才装

python3 -c "import numpy; print('numpy', numpy.__version__)"
python3 -c "from cv_bridge import CvBridge; CvBridge(); print('cv_bridge OK')"
```

注意：

- `python3-numpy` 用 apt 的 1.21.x；**不要** `pip install -U numpy`，否则 Humble 的 `cv_bridge` 可能报 `module 'numpy' has no attribute 'float'`。
- `python3-pytest` 不装的话，`colcon test` 和算法单测都跑不了。
- 容器是 root，编译后宿主机上的 `build/install/log` 会变成 root 属主；见 2.5。

### 2.5 处理文件权限

容器里用 root 编译后，在宿主机执行：

```bash
sudo chown -R "$(id -u):$(id -g)" \
  "$HOME/Projects/ros2_ws/build" \
  "$HOME/Projects/ros2_ws/install" \
  "$HOME/Projects/ros2_ws/log"
```

更彻底的做法是在自定义镜像中创建相同 UID/GID 的用户；此时容器内不能直接执行 `apt install`，依赖需要提前写入镜像。团队内部使用 root + `chown` 更简单。

---

## 3. 每天的工作流

### 3.1 启动容器

```bash
docker start -ai ros2_humble
```

如果容器已经删除，按 2.3 重新 `docker run`。

### 3.2 容器内编译和测试

```bash
source /opt/ros/humble/setup.bash
cd /ws

# 首次或依赖变化后
rosdep install --from-paths src --ignore-src -r -y

# 只编算法包（改算法时最快）
colcon build --packages-select algorithm_pkg --symlink-install
source install/setup.bash

# 全量编译（联调前/交付前）
colcon build --symlink-install
source install/setup.bash

# 算法纯函数单测
PYTHONPATH=src/algorithm_pkg python3 -m pytest -q src/algorithm_pkg/test

# 包级测试
colcon test --packages-select algorithm_pkg
colcon test-result --verbose
```

> 切换过 ROS 发行版、Python 版本或依赖后，先删掉 `build/ install/ log/` 再全量编译；容器内不要复用主机 Jazzy 的编译产物。

### 3.3 启动联调

推荐顺序：视觉 → 中枢 → 算法 → 战术指令。

```bash
# 终端 1：视觉（没有相机时可用 mock_vision_node）
ros2 launch vision_pkg vision_node.launch.py

# 终端 2：中枢
ros2 launch center_node_pkg center_node.launch.py

# 终端 3：算法
ros2 launch algorithm_pkg algorithm.launch.py

# 终端 4：看规划输出
ros2 topic echo /planning/target_pose

# 终端 5：起飞，再切跟踪/拦截/攻击
ros2 topic pub --once /strategy/cmd std_msgs/msg/String \
  "{data: '{\"drone_id\":\"drone_1\",\"action\":\"TAKEOFF\"}'}"

ros2 topic pub --once /strategy/cmd std_msgs/msg/String \
  "{data: '{\"drone_id\":\"drone_1\",\"action\":\"TRACK\"}'}"
```

注意：

- `algorithm_node` 默认 `default_action=HOVER`，每帧发布己方当前位置（`reason=hover_hold`）。收到 `/strategy/cmd` 的 `TRACK`、`INTERCEPT` 或 `ATTACK` 后，改为发布规划点。
- 如果算法节点比战术指令晚启动，它可能错过那条指令；重新发一次即可，或把 launch 参数 `default_action` 改成需要的模式（不推荐默认 `ATTACK`）。
- `TAKEOFF/RETURN` 的坐标由中枢生成，算法节点**不发布**这两个模式的点；`HOVER` 由算法节点发布己方当前位置。
- 中枢的 `TRACK` 优先使用 `/vision/target_pose`，规划点只是兜底；`INTERCEPT/ATTACK` 只使用 `/planning/target_pose`。
- 中枢 `plan_callback` 不解析规划 JSON 里的 `action`；真正的模式来自 `/strategy/cmd`。规划里的 `action` 只作为标签/调试信息。
- `ATTACK` 无解时策略层不输出规划点，适配层补发己方当前位置保持点；若完全无点，中枢会保留上一帧最多 2 秒。这是已知的联调风险，不能把“停止发布”视为立即停止。

### 3.4 GUI

GUI 也要在容器里跑，不要在主机 Jazzy 里跑：

```bash
# 主机
xhost +local:root

# 容器
cd /ws
source /opt/ros/humble/setup.bash
python3 mock_nodes/launch_control.py
```

如果容器报 `cannot open display`：

1. 确认启动容器时传了 `-e DISPLAY=$DISPLAY` 和 `-v /tmp/.X11-unix:/tmp/.X11-unix:rw`；
2. 宿主机重新 `xhost +local:root`；
3. 检查 `echo $DISPLAY`、`ls /tmp/.X11-unix`；
4. WSLg 下 `DISPLAY` 通常是 `:0`，不要改成 `localhost:0`。

这里的 GUI 是 mock 联调控制台。Gazebo 3D GUI 已放弃，见 3.5。

### 3.5 P5 SITL（PX4 + Gazebo Harmonic）

P5 的物理仿真在 `p5_sitl` 容器里，不混用主机的 Humble/Jazzy 工作区。依赖：Humble、Gazebo Harmonic 8.15、PX4 v1.17、uXRCE-DDS Agent 2.4.3。PX4 源码和编译产物在 `.p5/`，已 git ignore，不占 `/`。

启动整栈：

```bash
cd .p5/main_ws
./run_p5_theory.sh
```

测试启停用 `.p5/p5_stack.sh start|stop`。理论参数注入三处：

- `px4_runtime_inject.py`：MAVSDK 在运行时改 PX4 参数，不改 main 文件；
- `theory_bridge_params.yaml`：桥接三档 `max_step`，最终 0.225 / 0.38 / 0.74；
- `algorithm.launch.py`：算法参数，最终 4.50 / 7.60 / 14.80。

理论值只作参考；本仿真 V_max 上限约 14.8 m/s，23.59 不可达。3D GUI 已放弃（驱动问题），观察和汇报用 2D 图表。测试数据、四步收口和场景报告见 `算法测试基本方法.md` 第 8～10 章。

---

## 4. 长期方案：自定义 Humble 镜像

如果不想每次重建容器都 `apt install`，在仓库外放一个 `Dockerfile.humble`：

```dockerfile
FROM osrf/ros:humble-desktop-full

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-pytest \
    python3-tk \
    python3-numpy \
    python3-opencv \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-diagnostic-msgs \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*

# numpy 保持 apt 的 1.x；mavsdk 只有 mock 桥接需要
RUN pip3 install --no-cache-dir mavsdk

WORKDIR /ws
```

构建：

```bash
docker build -f Dockerfile.humble -t ros2-humble-dev .
```

运行：

```bash
docker run -it \
  --name ros2_humble \
  --net=host --ipc=host \
  -e ROS_DOMAIN_ID=0 \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "$HOME/Projects/ros2_ws:/ws" \
  -w /ws \
  ros2-humble-dev bash
```

---

## 5. Humble 兼容规则

虽然不再使用主机 Jazzy，代码仍按 Humble 的边界编写，避免在其他环境中运行时出错。

### 5.1 Python 3.10

Humble 是 Python 3.10。不要用 3.11/3.12 才有的语法：

- `tomllib`、`typing.Self`、`ExceptionGroup` / `except*`、`asyncio.TaskGroup`、`enum.StrEnum`、`datetime.UTC`
- `type` 语句、`itertools.batched`、`pathlib.Path.walk`、`typing.override`
- PEP 701 的 f-string 嵌套同类引号

可以放心用：`match/case`、`X | Y` 类型联合、`list[int]`、`dataclass(slots=True)`、`from __future__ import annotations`。

### 5.2 ROS 2 API

只用 Humble 稳定子集：`rclpy.init`、`Node`、`create_publisher`、`create_subscription`、`create_timer`、`get_clock().now().to_msg()`、`spin`、`shutdown`。不要用 `rclpy.experimental.*` 或 Jazzy 新 API。

### 5.3 消息与话题

- 只用标准消息：`std_msgs/String`、`geometry_msgs/PoseStamped`、`nav_msgs/Odometry`、`sensor_msgs/*`、`diagnostic_msgs/*`。
- 包名可以叫 `algorithm_pkg`，但公共话题 `/planning/target_pose` **不能改**。
- 新增自定义 `.msg` 必须在 Humble 容器里重新 `colcon build` 生成，不能复用其它发行版的 `install/`。

### 5.4 numpy / OpenCV / cv_bridge

- Humble：numpy 1.21.x，`cv_bridge` 按 1.x 编译。
- 禁止在 Humble 里 `pip install "numpy>=2"`。
- 避免 numpy 2.x 移除的别名：`np.float_`、`np.int0`、`np.bool8`、`np.product`、`np.alltrue`、`np.sometrue`。
- 不要同时混用 apt 的 `python3-opencv` 和 pip 的 `opencv-python`。

### 5.5 时间

- 当前全系统不启用 `use_sim_time`，launch 里不要加。
- 规划 JSON 的 `stamp` 用**发布时刻**的 ROS 时钟秒数：`self.get_clock().now().nanoseconds / 1e9`；不要用目标消息的旧时间戳。
- 不要混用 `time.time()` 和 ROS clock 做超时判断。

---

## 6. `algorithm_pkg` 与其它包的适配

### 6.1 包结构

```text
src/algorithm_pkg/
├── algorithm_pkg/
│   ├── algorithm_core.py    # 纯算法，不 import rclpy
│   ├── algorithm_node.py    # ROS 2 适配层
│   └── __init__.py
├── launch/
│   └── algorithm.launch.py
├── test/
│   └── test_algorithm_core.py
├── package.xml
├── setup.py
└── setup.cfg
```

### 6.2 接口

`algorithm_node` 订阅：

| 话题 | 类型 | 用途 |
|---|---|---|
| `/vision/target_pose` | `geometry_msgs/PoseStamped` | 视觉目标，优先使用 |
| `/external/target_detected` | `geometry_msgs/PoseStamped` | 外部检测，视觉不新鲜时兜底 |
| `/strategy/cmd` | `std_msgs/String` JSON | `drone_id` + `action`，决定本机模式 |
| `/drone_<id>/state` | `nav_msgs/Odometry` | 己方 ENU 位置 |

发布：

| 话题 | 类型 | 说明 |
|---|---|---|
| `/planning/target_pose` | `std_msgs/String` JSON | 字段 `drone_id`、`stamp`、`x`、`y`、`z`、`action` |

约束：

- `TRACK/INTERCEPT/ATTACK` 发布规划点；`HOVER` 发布己方当前位置保持点；`TAKEOFF/RETURN/RESUME_PATROL` 不发布。
- `TRACK` 发目标预测点，不做 standoff；中枢会在 5 米内自己加环绕偏移，避免双重偏移。
- `INTERCEPT` 不可达时发布目标运动方向后方 `d_tail=4.0m` 的尾随点；`ATTACK` 不可达时不输出尾随点，策略层返回 `None`，由适配层补当前位置保持点。
- 高度：`TRACK/INTERCEPT` 最低 5 米，`ATTACK` 最低 3 米；最高 50 米；XY 限制 ±1000 米。
- `INTERCEPT/ATTACK` 至少需要两帧目标（`velocity_valid=True`）才发点；只有一帧时等待，避免把“当前位置”当成拦截点。
- 输出 `stamp` 是发布时刻；中枢 `max_plan_delay=0.8s`（接口规范写 0.5s），过期点会被丢弃。

### 6.3 与其它包的关系

- 与 `center_node_pkg`：算法只发 `/planning/target_pose`；中枢负责状态机、限幅、防撞、`/{drone_id}/fmu/in/trajectory_setpoint`。算法不直接发飞控话题。
- 与 `vision_pkg`：算法只订阅 `/vision/target_pose`，不直接读相机。
- 与 `flight_ctrl_pkg`：飞控桥接消费中枢的 `/{drone_id}/fmu/in/trajectory_setpoint`、`/{drone_id}/action`，不直接消费算法话题。
- 与 `mock_nodes`：`mock_plan_node.py` 可以继续作为无算法节点时的假规划源；接上 `algorithm_node` 后不要同时启动两个规划源，避免同一话题互相覆盖。

### 6.4 启动参数

`algorithm.launch.py` 已给出与 `center_node.py` 一致的三机默认值。常用参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `drone_ids` | `['drone_1','drone_2','drone_3']` | 必须与中枢 `swarm_config` 一致 |
| `v_max` | 6.0 | 兜底速度，m/s |
| `v_max_intercept` | 6.0 | INTERCEPT 速度上限，m/s |
| `v_max_attack` | 6.0 | ATTACK 速度上限，m/s |
| `t_go_max` | 8.0 | 最大拦截时间，s |
| `max_age` | 1.0 | 目标最大年龄，s |
| `target_timeout` | 0.5 | 视觉/外部源切换阈值，s |
| `ego_timeout` | 1.0 | 己方状态超时，s |
| `default_action` | HOVER | 未收到战术指令时的模式 |
| `max_step` | 0.30 | 单帧最大跳变，m；0 表示不限制 |

---

## 7. 常见故障排查

| 现象 | 原因 | 处理 |
|---|---|---|
| 容器里 `/ws` 没有 `src/` | 挂载到了错误的空目录 | `docker inspect <容器> --format '{{json .Mounts}}'` 检查 Source；重建容器 |
| `ros2: command not found` | 没 source | `source /opt/ros/humble/setup.bash` |
| `Package 'algorithm_pkg' not found` | 没编译或没 source install | `colcon build --packages-select algorithm_pkg`；`source install/setup.bash` |
| `pytest: command not found` | 镜像没装 pytest | `apt-get install -y python3-pytest` |
| `ModuleNotFoundError: No module named 'yaml'` | 在主机 conda Python 里跑了 ROS | 进容器；容器内用 `/usr/bin/python3` |
| `cv_bridge` 导入失败 / numpy 报错 | Humble 里 numpy 被升到 2.x | 重装 apt 的 `python3-numpy`；不要 `pip install -U numpy` |
| 容器和主机互相看不到话题 | DDS 网络/共享内存问题 | `--net=host --ipc=host`；同 `ROS_DOMAIN_ID`；`ros2 daemon stop` |
| GUI 报 `cannot open display` | X11 未授权/未挂载 | `xhost +local:root`、`-e DISPLAY`、`-v /tmp/.X11-unix` |
| `build/install` 权限拒绝 | 容器 root 创建 | 宿主机 `sudo chown -R $(id -u):$(id -g) build install log` |
| `/planning/target_pose` 没有输出 | 没收到 `TRACK/INTERCEPT/ATTACK`、目标过期、己方状态缺失 | 先发 `/strategy/cmd`；检查 `/vision/target_pose` 和 `/drone_1/state` |
| `colcon test` 没有测试 | 包的测试没注册/没装 pytest | 确认 `test/` 存在、`python3-pytest` 已装 |

---

## 8. 交付前验收清单

在 Humble 容器中逐项确认：

- [ ] `docker inspect` 确认工作区挂载到 `/ws`，不是空目录
- [ ] `source /opt/ros/humble/setup.bash` 后 `python3 --version` 是 3.10
- [ ] `rm -rf build install log` 后从零 `colcon build` 成功
- [ ] `rosdep install --from-paths src --ignore-src -r -y` 无报错
- [ ] `PYTHONPATH=src/algorithm_pkg python3 -m pytest -q src/algorithm_pkg/test` 全部通过
- [ ] `colcon test --packages-select algorithm_pkg` 全部通过
- [ ] `ros2 launch algorithm_pkg algorithm.launch.py` 能启动
- [ ] 发 `TAKEOFF` + `TRACK` 后 `/planning/target_pose` 有 JSON，字段/时间戳/z 范围正确
- [ ] 没有写死 `/opt/ros/jazzy`、`/home/xxx` 等主机路径
- [ ] 没有使用 Python 3.11/3.12 语法
- [ ] 保存容器内 build/test 日志和 launch 截图作为交付证据

---

## 9. 命名约定

| 对象 | 名称 | 说明 |
|---|---|---|
| Git 分支 | `algorithm_pkg` | Git 分支允许下划线 |
| ROS 包 | `algorithm_pkg` | 不再使用 `planning_pkg`；ROS 包名不允许连字符 |
| 节点/可执行文件 | `algorithm_node` | 入口点 `algorithm_pkg.algorithm_node:main` |
| launch 文件 | `algorithm.launch.py` | `ros2 launch algorithm_pkg algorithm.launch.py` |
| 输出话题 | `/planning/target_pose` | **公共接口，保持不变**；包名和话题名解耦 |

---

## 10. 参考

- ROS 2 Humble 文档：<https://docs.ros.org/en/humble/>
- ROS 2 Humble Docker 镜像：<https://hub.docker.com/r/osrf/ros>
- `rosdep` 使用：<https://docs.ros.org/en/humble/Tutorials/Beginner-Client-Libraries/Colcon-Tutorial.html>
- REP 2000（发行版平台定义）：<https://www.ros.org/reps/rep-2000.html>

---

*最后更新：2026-11-19*
