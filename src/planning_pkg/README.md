# `planning_pkg` — 课题二机动目标状态估计与多机协同拦截制导包

本模块实现《课题二内部接口规范 V3.0》中的**路径规划与协同制导子系统（接口 C 提供方）**，全部坐标运算严格基于全局 **ENU（东-北-天）** 坐标系。

## 一、核心算法架构

1. **九维 Singer-CA / CTRV 异构融合状态估计器 (`estimator.py`)**：
   - 状态向量 $\mathbf{x} = [p_x, p_y, p_z, v_x, v_y, v_z, a_x, a_y, a_z]^\top \in \mathbb{R}^9$。
   - 自适应融合高精度机载视觉（`/vision/target_pose`）与广域外部雷达引导（`/external/target_detected`），内置 $\chi^2$ 马氏距离门控拦截异常跳变野值，采用 Joseph 对称正定协方差更新。
   - 实时估计水平瞬时转弯角速度 $\hat{\omega}$，在视觉短暂遮挡/丢失窗口（$\le 3.0\text{ s}$）内执行精确 CTRV 圆弧积分外推，实现无感降级锁定。
2. **多机协同制导与战术轨迹规划器 (`guidance.py`)**：
   - **`TRACK`（120° 锁相合围伴飞）**：通过全排列代价寻优与滞回锁定，动态分配三机均匀相位槽位并保持相对靶机 $+3.5\text{ m}$ 最佳俯视高度差（精确匹配 35° 前斜相机光轴中心）。
   - **`INTERCEPT`（Apollonius 可达域钳形截击）**：二分迭代求解考虑初始惯性滞后的最早可达时间 $t_{go}^*$，实现中路迎头截击与左右双翼包抄封堵；超出速度极限时自动退化为内侧尾随追击。
   - **`ATTACK`（三维增广比例导引 3D APN + 狼群突防）**：融合视线旋转率与目标法向机动补偿解算最优碰撞交汇点，附带末端冲击轴向穿透提前量与多机微张角四面体防追尾解耦。
   - **`RESUME_PATROL`（三机协同搜索巡逻）**：生成覆盖试飞场核心空域的三机异相 8 字形搜索航线。

## 二、启动与测试

```bash
# 编译与单元测试
colcon build --packages-select planning_pkg
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 colcon test --packages-select planning_pkg
colcon test-result --verbose

# 标准 Launch 启动
ros2 launch planning_pkg planning_node.launch.py
```
