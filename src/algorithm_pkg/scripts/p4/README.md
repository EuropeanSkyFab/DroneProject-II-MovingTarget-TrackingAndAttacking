# ROS 联调脚本

这些脚本不参与 `colcon test`，用于验证 ROS 节点之间的话题、字段、时间戳和多机隔离。全部在 Humble 容器中运行，宿主机只编辑。

```bash
docker run --rm -it --net=host \
  -v "$HOME/Projects/ros2_ws:/ws" -w /ws \
  osrf/ros:humble-desktop bash

source /opt/ros/humble/setup.bash
export PYTHONPATH=/ws/src/algorithm_pkg:$PYTHONPATH
export ROS_DOMAIN_ID=0
```

在 `p5_sitl` 容器里时，直接进容器，source `/root/ros2_ws/.p5/main_ws/install/setup.bash` 后运行。

| 脚本 | 用途 |
|---|---|
| `smoke_node.py` | 单节点冒烟：手动发 A/B/D/E，断言接口 C 字段、时间戳、动作、ATTACK 保持点 |
| `fault_injection.py` | 7 类失效注入，断言 L1/L2/L3 与恢复；日志写到 `$INTEGRATION_LOG`（默认 `algo.log`） |
| `three_drone.py` | 三机隔离：1 个节点、3 个 drone_id，断言不同飞机之间互不影响 |

`smoke_node.py` 会自行启动被测节点，不需要额外终端。`fault_injection.py`、`three_drone.py` 需要先启动被测节点，前置条件见各脚本顶部注释。测试分层见 [算法测试基本方法](../../docs/算法测试基本方法.md)。
