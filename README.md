# 课题二：面向动态目标的无人机视觉跟踪与自主打击

本项目为课题二系统集成主仓库。系统基于 **ROS 2 (Humble)** 与 **PX4 (uXRCE-DDS)** 构建。

## ⚠️ 核心开发铁律（必读）

1. **绝对不要修改别人的包**：每个人在自己的功能包内写代码，严禁跨包修改。
2. **绝对不要提交编译产物**：仓库根目录已经配置了 `.gitignore`，`build/`、`install/`、`log/` 以及 `__pycache__/` 不会被提交。**千万不要用 `git add -f` 强制提交这些文件夹。**
3. **必须遵守接口规范**：所有话题名、消息类型、坐标系必须严格遵守团队内部的《接口文档》。

---

## 📂 仓库目录结构

本仓库是一个标准的 ROS 2 工作空间（Workspace），结构如下：

```text
ros2_ws/
├── .gitignore              # Git 忽略规则（已配置好 ROS 2 规则）
├── README.md               # 本说明文档
└── src/                    # 所有源码放这里
    └── center_node_pkg/    # 【系统集成】中枢节点包（赖纳负责）
        ├── center_node_pkg/
        │   ├── __init__.py
        │   └── center_node.py  # 中枢主控逻辑
        ├── package.xml
        └── setup.py
```

---

## 🚀 成员上手指南（How to use）

### 第一步：克隆仓库
打开你的 WSL Ubuntu 终端，进入你想存放代码的目录（比如 `~/`），执行：
```bash
git clone https://github.com/EuropeanSkyFab/DroneProject-II-MovingTarget-TrackingAndAttacking.git ros2_ws
cd ros2_ws
```

### 第二步：创建自己的功能包
**严禁所有人都在同一个包里写代码！** 请使用以下命令创建属于你自己的功能包（注意修改 `你的名字` 和 `你的负责模块`）：
```bash
cd src
# 示例：机载视觉模块上传
ros2 pkg create --build-type ament_python vision_pkg --dependencies rclpy std_msgs geometry_msgs vision_msgs
# 示例：路径规划模块上传
ros2 pkg create --build-type ament_python planning_pkg --dependencies rclpy std_msgs geometry_msgs
```

### 第三步：编译工作空间
每次修改完代码后，必须回到根目录（`ros2_ws`）进行编译：
```bash
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### 第四步：写代码与注册入口
1. 进入你的包：`src/你的包名/你的包名/`。
2. 新建你的 `.py` 代码文件。
3. 打开同级的 `setup.py`，在 `entry_points` 的 `console_scripts` 里注册你的节点：
```python
    entry_points={
        'console_scripts': [
            'your_node_name = 你的包名.你的代码文件名:main',
        ],
    },
```

---

## ☁️ Git 提交规范（怎么传代码）

**写完代码后，建议及时把代码推送到云端！**

```bash
# 1. 确保你在 ros2_ws 根目录下
cd ~/ros2_ws

# 2. 拉取云端最新代码（防止队友改了代码导致冲突）
git pull origin main

# 3. 查看你改了哪些文件
git status

# 4. 添加修改（. 代表添加所有改动，.gitignore 会自动过滤垃圾文件）
git add .

# 5. 提交并写清楚说明（不要写“update”这种废话）
git commit -m "feat(视觉): 完成了YOLO目标检测并发布3D坐标"

# 6. 推送到云端
git push origin main
```

---

## 🧪 联调测试常用命令

在终端运行任何 ROS 2 节点之前，**请务必确保执行了以下两行**：
```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0  # 保证大家都在同一个局域网
```

**发送假数据测试指令（供联调参考）：**
```bash
# 1. 发送路径规划模块假数据 (10, 20, 5)
ros2 topic pub /planning/target_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: 'world'}, pose: {position: {x: 10.0, y: 20.0, z: 5.0}}}"

# 2. 发送机载视觉模块假数据 (3.5, 2.5, -5)
ros2 topic pub /vision/target_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: 'world'}, pose: {position: {x: 3.5, y: 2.5, z: -5.0}}}"
```

---


*Latest Update：2026-10-20 by EuroepanSkyFab*
