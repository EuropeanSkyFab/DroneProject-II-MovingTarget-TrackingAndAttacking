# algorithm_pkg

课题二规划算法包：纯算法核心 + ROS 2 适配节点。

## 组成

- `algorithm_pkg/algorithm_core.py`：目标预测、拦截求解、安全约束；不依赖 ROS。
- `algorithm_pkg/algorithm_node.py`：订阅视觉/外部目标、己方状态和战术指令，发布 `/planning/target_pose`。
- `launch/algorithm.launch.py`：三机默认参数的一键启动。
- `test/test_algorithm_core.py`：纯算法单测。

## 编译与测试（Humble 容器内）

```bash
source /opt/ros/humble/setup.bash
cd /ws
colcon build --packages-select algorithm_pkg --symlink-install
source install/setup.bash

PYTHONPATH=src/algorithm_pkg python3 -m pytest -q src/algorithm_pkg/test
colcon test --packages-select algorithm_pkg
colcon test-result --verbose
```

## 运行

```bash
ros2 launch algorithm_pkg algorithm.launch.py
```

节点默认 `HOVER`，收到 `/strategy/cmd` 的 `TRACK/INTERCEPT/ATTACK` 后才发点。
典型顺序：先 `TAKEOFF`，再 `TRACK` 或 `INTERCEPT/ATTACK`。

## 接口

订阅：

- `/vision/target_pose`：`geometry_msgs/PoseStamped`，优先目标源；
- `/external/target_detected`：`geometry_msgs/PoseStamped`，视觉不新鲜时兜底；
- `/strategy/cmd`：`std_msgs/String` JSON，字段 `drone_id`、`action`；
- `/drone_<id>/state`：`nav_msgs/Odometry`。

发布：

- `/planning/target_pose`：`std_msgs/String` JSON，
  字段 `drone_id`、`stamp`、`x`、`y`、`z`、`action`。

`stamp` 是发布时刻的 ROS 时钟秒数。话题名是公共契约，不随包名变化。

## 约束

- 只对 `TRACK/INTERCEPT/ATTACK` 发点；`TAKEOFF/HOVER/RETURN` 由中枢生成。
- `TRACK` 发目标预测点，不做 standoff；中枢在近距离自己加环绕偏移。
- `INTERCEPT` 不可达发尾随点；`ATTACK` 不可达不发点。
- `TRACK/INTERCEPT` 最低 5m，`ATTACK` 最低 3m，最高 50m，XY ±1000m。
- `INTERCEPT/ATTACK` 至少两帧目标（`velocity_valid=True`）才发点。
