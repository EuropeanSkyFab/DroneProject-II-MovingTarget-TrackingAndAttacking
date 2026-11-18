# algorithm_pkg

课题二规划算法包：纯算法核心 + ROS 2 适配节点。

## 组成

- `algorithm_pkg/algorithm_core.py`：P1 目标预测、拦截求解、安全约束；不依赖 ROS。
- `algorithm_pkg/plan_types.py`：P2 `PlanResult` 和动作分组。
- `algorithm_pkg/plan_params.py`：P2 `PlanParams`，按动作选择速度。
- `algorithm_pkg/plan_policy.py`：P2 `plan()`，决定一帧发不发点。
- `algorithm_pkg/algorithm_node.py`：ROS 适配层；调用 P2 的 `plan()`，只负责消息、参数、日志和发布。
- `launch/algorithm.launch.py`：三机默认参数的一键启动。
- `test/test_algorithm_core.py`：P1 纯算法单测。
- `test/test_plan_*.py`：P2 参数、策略、模式切换测试。
- `文档/P2模式策略契约.md`：P2 的 7 动作表和边界说明。

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

P2 在每帧发点前还会做一次 0.5 秒的预测年龄门限（`max_pred_age`）；
超过门限的旧目标不会继续被用来算拦截点。`ATTACK` 无解或目标失效时，
适配层会发布“当前位置保持点”，避免中枢继续使用上一帧攻击点。

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
- `INTERCEPT` 不可达发尾随点；`ATTACK` 不可达时 P2 不发点，适配层改发
  “当前位置保持点”，避免中枢继续使用旧攻击点。
- `TRACK/INTERCEPT` 最低 5m，`ATTACK` 最低 3m，最高 50m，XY ±1000m。
- `INTERCEPT/ATTACK` 至少两帧目标（`velocity_valid=True`）且预测年龄不超过
  `max_pred_age`（默认 0.5s）才发点。

## 上游更新策略

`algorithm_pkg` 是独立算法包，不跟随 `origin/main` 自动合并。算法只依赖课题二中枢/视觉的公共接口；`flight_ctrl_pkg` 是中枢下游的飞控包，算法不直接订阅它。

### 原则

1. 不把 `origin/main` 合进 `algorithm_pkg`，也不为了追新而 rebase；本地分支只保留算法改动。
2. 每次上游更新先看 diff，确认是否改了算法依赖的接口：
   - `center_node_pkg` 的 `/planning/target_pose` 契约、`plan_callback` 限幅、`/{drone_id}/state`；
   - `vision_pkg` 的 `/vision/target_pose`；
   - `/strategy/cmd` 的字段和动作集合。
   只改 `flight_ctrl_pkg`、PX4 版本、相机安装参数时，算法不需要改。
3. 不 push、不合并 `main`、不上传 GitHub；需要保留时只做本地 commit。
4. 如果要在本地跑最新全栈，用独立 worktree 取 `origin/main`，不要把它并进算法分支：

   ```bash
   git worktree add /tmp/ros2_ws_upstream origin/main
   ```

   算法包可临时复制到该 worktree 联调，联调完不提交。

### 检查命令

```bash
git fetch origin main
BASE=$(git merge-base HEAD origin/main)

# 中枢/视觉是否改了算法依赖的接口
git diff --stat "$BASE"..origin/main -- src/center_node_pkg src/vision_pkg

# 上游是否也动了 algorithm_pkg
git diff --stat "$BASE"..origin/main -- src/algorithm_pkg
```

判断规则：

- 前一条为空、后一条也为空：只新增了下游包，算法不改；
- 前一条有变化：逐项核对 topic 名、JSON 字段、坐标系和限幅，再决定是否适配；
- 后一条有变化：先停止合并，比较上游算法实现和本地实现，避免覆盖本地算法。

### 当前对齐记录（2026-11-17）

| 项 | 当前值 | 上游依据 | 处理 |
|---|---|---|---|
| 上游 `main` | `ba401d5` | 新增 `flight_ctrl_pkg` 与 PX4 v1.17 适配 | 不合并，仅记录 |
| `INTERCEPT` 速度上限 | 7.6 m/s | `max_step_intercept=0.38 × 20Hz` | 已设为节点默认参数 |
| `ATTACK` 速度上限 | 12.4 m/s | `max_step_attack=0.62 × 20Hz` | 已设为节点默认参数 |
| 兜底 `v_max` | 6.0 m/s | 旧接口文档 | 仅作兜底 |
| state 新鲜度 | `header.stamp` 优先 | center 会把 `/{drone_id}/state` 转发到同一 topic | 旧帧不覆盖新帧；无时间戳时退回接收时刻 |
| `TRACK` 高度 | 算法最低 5m | 飞控 `min_track_alt=8.0` | 实际可能被飞控抬到 8m，不是算法 bug |

本记录只表示检查时状态；后续以 `git log` 和实际接口 diff 为准。
