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

## 🚀 上手指南（How to use）

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

## ！！避坑警告！！

### 1：坐标系统一
- **全系统内部（中枢、视觉、规划）统一使用 ENU 坐标系（东-北-天，Z轴向上为正）。**
- **严禁在内部模块中写 NED 坐标系转换代码。** 所有的 ENU 到 NED 转换，统一由底层的飞控桥接模块（或 DDS 节点）负责处理。

### 2：深拷贝污染（Python 引用传递）
由于Python的特性，可变对象变量在赋值给其他变量的时候不会重新指定地址，而是直接把两个变量的指针挂在同一个地址。因此在写 ROS 2 回调函数时，如果要把收到的 `msg` 存到类变量中，**必须使用 `copy.deepcopy(msg)`**，以免后续变量底层回收内存时数据乱码。
```python
import copy

def callback(self, msg):
    # 错误做法：直接赋值，底层回收内存会导致数据乱码
    # self.latest_data = msg 
    
    # 正确做法：深拷贝，物理隔离
    self.latest_data = copy.deepcopy(msg) 
```

### 3：网络隔离
如果在不同电脑上联调（或者多终端），发现收不到数据，**请检查双方的 `ROS_DOMAIN_ID` 是否一致**（建议统一设置为 `export ROS_DOMAIN_ID=0`）。

### 4：WSL 渲染与高频打印导致定时器降频（系统卡顿）
*   **现象**：在跑 Gazebo 3D 画面并进行多节点联调时，Gazebo的3D渲染和WSl终端的高频输出会占用大量算力导致CPU满载，进而导致 ROS 2 定时器严重降频。表现为状态机超时判定延迟（如本该 2 秒切回 GUIDANCE，实际等了 5-10 秒才触发）。
*   **解决方案**：
    1.  **启动无头模式**：联调时推荐使用 `make px4_sitl gz_x500 HEADLESS=1` 启动 PX4 仿真，关闭 3D 渲染以释放 CPU 算力。
    2.  **日志节流**：在节点代码中，所有高频触发的日志打印，须加上节流参数，例如：`self.get_logger().info("msg", throttle_duration_sec=1.0)`，防止终端渲染阻塞线程。
    3.  **频率监控**：用 `ros2 topic hz /fmu/in/trajectory_setpoint` 监控实际发布频率，确认底层数据流是否因 CPU 瓶颈而卡顿。

### 5：中枢/桥接进程崩溃导致无人机无限爬升（致命 Failsafe 缺失）
*   **现象**：在飞行过程中，如果直接强杀（`Ctrl+C` 或关闭终端）中枢节点或桥接节点，飞控会失去外部指令流。由于 PX4 默认的 Offboard 失控动作为“悬停（Hold）”，在 Gazebo 物理引擎中会因积分漂移导致无人机一直向上爬升直至坠毁。
*   **解决方案**：
    1.  **修改飞控底层参数（关键）**：在 QGC 参数设置或 PX4 终端中，将 `COM_OBL_RC_ACT`（Offboard 失控动作）强制设为 **`3 (Land)`**。同时将 `COM_OF_LOSS_T`（失控判定时间）设为 **`0.5`** 秒。
    2.  **代码持久化**：若在终端用 `param set` 修改，务必执行 `param save` 保存。如果是团队开发，建议负责底层的同学直接将该参数写死在 PX4 源码的 `rcS` 启动脚本中，实现“出厂即保护”。
    3.  **安全验收**：联调时，务必进行“直接杀进程”的极限测试，验证飞控是否能在 0.5 秒内接管并安全降落。
---

*Latest Update：2026-10-23 by EuroepanSkyFab*
