# `center_node_pkg` — 课题二多机战术调度与安全总闸中枢包

本模块实现《课题二内部接口规范 V3.0》与《课题二-课题三双向接口规范 V2.0》中的**系统集成与安全中枢子系统**。

## 一、核心职责与工程加固

1. **单层并发战术状态机与指令安检门**：
   - 支持个体与集群群发指令（`"drone_id": "ALL"`），覆盖全部标准动作枚举：`TAKEOFF`、`HOVER`、`RETURN`、`TRACK`、`INTERCEPT`、`ATTACK`、`RESUME_PATROL`。
   - 地面防误触拦截（`GROUND` 态仅响应 `TAKEOFF`）与起飞水平锚点锁定。
2. **输入防腐层（Anti-Corruption Layer）与三维电子围栏**：
   - 全量拦截 `NaN` / `Inf`、超时延迟帧及越界坐标；常规模式执行最低高度防撞地保护，末端 `ATTACK` 模式允许俯冲打击低空目标。
3. **动态优先级三维切向与拉升避障**：
   - `ATTACK` 突击机享有最高路权；僚机根据相对高度与垂直速度自适应切换战术拉升或水平切向滑步；返航阶段采用分层高度走廊（`7.5m / 8.7m / 9.9m`）对准专属停机坪 `H1(0,0)`、`H2(0,3)`、`H3(0,6)` 垂直着陆。
4. **真实 6S 电池转发、自回环免疫链路健康度与云台三维指向**：
   - 消除 `/{drone_id}/state` 自发自收假死回环，真实检测桥接断连（`GOOD` / `WEAK` / `LOST`），输出完整 `delay_ms` 与 `rssi`，转发真实 PX4 6S 电池遥测并聚合发布 `/fleet/state`。

## 二、编译、测试与启动

```bash
colcon build --packages-select center_node_pkg
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 colcon test --packages-select center_node_pkg
colcon test-result --verbose

ros2 launch center_node_pkg center_node.launch.py
```
