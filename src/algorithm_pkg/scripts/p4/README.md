# P4 联调脚本

这些脚本不参与 `colcon test`，用于 `文档/P4联调记录.md` 里的四类取证。
全部在 Humble 容器里运行（主机只编辑）：

```bash
docker run --rm --user $(id -u):$(id -g) --net=host --ipc=host \
  -e ROS_DOMAIN_ID=88 -e HOME=/tmp \
  -v "$HOME/Projects/ros2_ws:/ws" -w /ws \
  osrf/ros:humble-desktop-full bash
source /opt/ros/humble/setup.bash
export PYTHONPATH=/ws/src/algorithm_pkg:$PYTHONPATH
```

| 脚本 | 用途 |
|---|---|
| `smoke_node.py` | 单节点冒烟：手动发 A/B/D/E，断言接口 C 字段/时间戳/动作/ATTACK 保持点 |
| `closed_loop.py` | 单机闭环：真实 center + algorithm + dry bridge，断言六阶段与转发误差 |
| `fault_injection.py` | 7 类失效注入，断言 L1/L2/L3 与恢复；日志写到 `$P4_LOG`（默认 `algo.log`） |
| `three_drone.py` | 三机隔离：1 节点 3 个 drone_id，断言不串机；日志同上 |
| `calibrate_speed.py` | 直接驱动 dry bridge，测桥接实际限幅速度（10/6/3 m/s） |

`smoke_node.py` / `closed_loop.py` 自己拉起节点，不需要额外终端；
`fault_injection.py` / `three_drone.py` / `calibrate_speed.py` 需要先起被测节点，
具体命令见 `文档/P4联调记录.md` 第 4～8 节。
