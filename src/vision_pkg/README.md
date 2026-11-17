# `vision_pkg` — 课题二多机异构视角机载视觉与三维立体定位包

本模块实现《课题二内部接口规范 V3.0》中的**机载视觉感知子系统（接口 A 提供方）**，严格输出全局 **ENU (`world`)** 坐标并保留图像原始采集时间戳。

## 一、核心特性

1. **多机异面视线三维最小二乘三角测量 (`triangulate_rays`)**：
   - 当 2 架及以上无人机在同步时间窗（`0.16s`）内同时捕获目标且基线视差角 $\ge 3^\circ$ 时，直接求解异面光线正交距离平方和最小化问题，**无需预设靶机飞行高度即可解算三维机动靶真值坐标 `(x, y, z)`**。
   - 内置正深度校验（Cheirality Check）与均方根异面残差审计，自动更新单机回退高度先验 `_adaptive_target_z`。
2. **室外抗强光红度色差增强检测 (`ColorBlobDetector` + `YoloDetector`)**：
   - 融合 HSV 双红区阈值与红度色差门限（`R - max(G, B) >= 35`），消除阳光高光反射斑误检，同时将最小有效检测面积下探至 `24 px²`，有效探测距离扩展至 `40m+`。

## 二、编译、测试与启动

```bash
colcon build --packages-select vision_pkg
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 colcon test --packages-select vision_pkg
colcon test-result --verbose

ros2 launch vision_pkg vision_node.launch.py
```
