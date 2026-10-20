# 课题二：面向动态目标的无人机视觉跟踪与自主打击 (系统集成)

欢迎来到课题二的核心代码仓库！本仓库由系统集成（联调）负责维护主干架构。

为了保证团队协作的顺畅，避免代码冲突和环境混乱，请所有成员**务必严格遵守以下开发规范与目录结构**。

---

## 1. 环境要求
- **操作系统**：Ubuntu 22.04 (WSL2 或原生系统)
- **ROS 2 版本**：Humble
- **编译工具**：`colcon`

---

## 2. 代码存放位置（严禁乱放）

本仓库是一个标准的 ROS 2 工作空间。**所有的功能包（Package）必须且只能放在 `src/` 目录下！**

**绝对不要**把 `build/`、`install/`、`log/` 目录提交到 Git 仓库中（仓库根目录已配置 `.gitignore` 自动忽略这些文件夹）。

### 团队分工与包命名规范
每位成员必须在 `src/` 目录下创建属于自己的独立功能包，**严禁修改他人的功能包**：

- **视觉模块（张嘉芮）**：在 `src/` 下创建 `vision_pkg`
- **路径规划模块（规划同学）**：在 `src/` 下创建 `planning_pkg`
- **飞控桥接模块（底层同学）**：在 `src/` 下创建 `fmu_bridge_pkg`
- **系统集成中枢（赖纳）**：已创建 `center_node_pkg`（请勿修改核心逻辑）

---

## 3. 如何克隆与上传代码（Git 标准流程）

### 首次拉取代码
打开终端，执行以下命令将仓库克隆到本地：
```bash
git clone git@github.com:EuropeanSkyFab/DroneProject-II-MovingTarget-TrackingAndAttacking.git
```

### 日常开发流程（千万别忘）
每次写代码前，先拉取最新代码；写完代码后，推送到云端：
```bash
# 1. 拉取云端最新进度
git pull origin main

# 2. 写你的代码...

# 3. 编译并测试
colcon build
source install/setup.bash

# 4. 提交你的代码
git add .
git commit -m "feat: 完成了视觉模块的假数据发布测试"
git push origin main
```

---

## 4. 标准启动与测试流程

在终端中依次启动各个模块（**请开 4 个不同的终端**）：

### 终端 1：编译并启动中枢节点
```bash
cd ~/你的工作空间路径
colcon build
source install/setup.bash
export ROS_DOMAIN_ID=0
ros2 run center_node_pkg center_node
```

### 终端 2：启动视觉模块 (张嘉芮)
```bash
source install/setup.bash
export ROS_DOMAIN_ID=0
# 替换为你实际的启动命令
ros2 run vision_pkg vision_node
```

### 终端 3：启动路径规划模块
```bash
source install/setup.bash
export ROS_DOMAIN_ID=0
# 替换为你实际的启动命令
ros2 run planning_pkg planning_node
```

### 终端 4：启动飞控底层桥接
```bash
source install/setup.bash
export ROS_DOMAIN_ID=0
# 替换为你实际的启动命令
ros2 run fmu_bridge_pkg bridge_node
```

---

## 5. 内部测试专用假数据指令

在真实模块开发完成前，可以使用以下指令发送假数据测试中枢节点的状态机：

**注意**：执行前必须先执行 `source /opt/ros/humble/setup.bash` 和 `export ROS_DOMAIN_ID=0`。

### 发送规划假数据（全局 ENU 坐标）
```bash
ros2 topic pub /planning/target_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: 'world'}, pose: {position: {x: 10.0, y: 20.0, z: 5.0}}}"
```

### 发送视觉假数据（全局 ENU 坐标）
```bash
ros2 topic pub /vision/target_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: 'world'}, pose: {position: {x: 3.5, y: 2.5, z: -5.0}}}"
```

*(建议在指令末尾加上 `-r 10`，让假数据以 10Hz 持续发送，防止中枢节点触发超时降级)*

---

## 6. 开发者避坑警告（必看）

### 坑 1：坐标系统一
- **全系统内部（中枢、视觉、规划）统一使用 ENU 坐标系（东-北-天，Z轴向上为正）。**
- **严禁在内部模块中写 NED 坐标系转换代码。** 所有的 ENU 到 NED 转换，统一由底层的飞控桥接模块（或 DDS 节点）负责处理。

### 坑 2：深拷贝污染（Python 引用传递）
在写 ROS 2 回调函数时，如果要把收到的 `msg` 存到类变量中，**必须使用 `copy.deepcopy(msg)`**。
```python
import copy

def callback(self, msg):
    # 错误做法：直接赋值，底层回收内存会导致数据乱码
    # self.latest_data = msg 
    
    # 正确做法：深拷贝，物理隔离
    self.latest_data = copy.deepcopy(msg) 
```

### 坑 3：网络隔离
如果在不同电脑上联调（或者多终端），发现收不到数据，**请检查双方的 `ROS_DOMAIN_ID` 是否一致**（建议统一设置为 `export ROS_DOMAIN_ID=0`）。

---
*如果有任何接口疑问或需要新增话题，请联系系统集成（赖纳）更新接口文档。*
