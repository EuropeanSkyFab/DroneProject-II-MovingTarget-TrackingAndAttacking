# algorithm_pkg

课题二规划算法包：**纯算法核心（P1/P2/P3）+ 纯逻辑适配层 + ROS 2 外壳**。

它不控制中枢，只回答一个问题：**这一帧该不该发点、发哪个点**。

当前状态：P1~P4 已完成，203 项 pytest；Humble 下 `colcon build/test/launch` 通过。

---

## 一分钟看懂架构

```text
/vision/target_pose ───────┐
/external/target_detected ─┤
/strategy/cmd ─────────────┤        ┌────────────────────────────────┐
/drone_<id>/state ─────────┘  ───▶  │ algorithm_node.py（ROS 外壳）   │
                                    │ 收消息、20Hz 定时、发消息、日志  │
                                    └───────────────┬────────────────┘
                                                    │ 纯 Python 数据
                                    ┌───────────────▼────────────────┐
                                    │ adapter_core.py（翻译官）       │
                                    │ 时间戳门禁、目标缓存、每机状态、 │
                                    │ ATTACK 保持点、JSON payload      │
                                    └───────────────┬────────────────┘
                                    ┌───────────────▼────────────────┐
                                    │ degrade_policy.py（调度员）     │
                                    │ 按 L0~L3 决定这一帧怎么算        │
                                    │  ├ validators.py（体检）        │
                                    │  ├ degrade_fsm.py（记忆/滞回）  │
                                    │  ├ safety_guard.py（安全员）    │
                                    │  └ plan_policy.py（七动作合同） │
                                    │       └ algorithm_core.py（公式）│
                                    └────────────────────────────────┘
```

一句话分工：

- **外壳**只搬运，不做判断；
- **翻译官**把 ROS 消息变成算法能吃的纯数据，并守住时间戳；
- **调度员**决定正常 / 外推 / 保持 / 不发；
- **合同层**决定七种动作各输出什么；
- **公式层**只算几何和运动学。

这样拆的好处：外壳可以很薄，翻译官能脱离 ROS 单测，公式层永远是纯函数。

---

## 一帧数据是怎么走的

1. **收消息**：视觉/外部目标进 `TargetBuffer`，零时间戳、未来时间戳、倒退时间戳、NaN 帧直接丢；己方 state 存快照；`/strategy/cmd` 只改“这架飞机当前想干什么”。
2. **20Hz tick**：`choose_target()` 选目标——视觉新鲜就用视觉，否则用外部，再否则用视觉的预测值。
3. **算一帧**：`plan_drone()` 调 `plan_with_degrade()`——先体检（目标/己方/指令），再过降级状态机，再按级别规划，最后过安全护栏。
4. **发或者不发**：有点就 `payload()` 生成 `{drone_id, stamp, x, y, z, action}`，`stamp` 用**发布时刻**；没点就一帧不发。
5. **中枢接手**：中枢按自己的模式决定怎么飞。接口 C 里的 `action` 只是标签，不控制中枢模式。

---

## 每个文件是干什么的

| 文件 | 层 | 一句话作用 |
|---|---|---|
| `algorithm_pkg/algorithm_core.py` | P1 | 公式内核：目标预测、拦截求解、单帧约束。不依赖 ROS。 |
| `algorithm_pkg/plan_types.py` | P2 | 定义 `PlanResult`、七种动作分组、降级级别。所有层共用的“结果形状”。 |
| `algorithm_pkg/plan_params.py` | P2/P3 | 所有可调参数集中在这里（`PlanParams`），带默认值和合法性校验；`v_max_for()` 按动作选速度。 |
| `algorithm_pkg/plan_policy.py` | P2 | 七动作合同：`plan()` 决定这一帧发不发点、发哪个点。 |
| `algorithm_pkg/validators.py` | P3 | 体检：把目标/己方/指令分成“正常、过期、非法、倒退、丢失”等状态。 |
| `algorithm_pkg/degrade_fsm.py` | P3 | 记忆：L0 正常 / L1 外推 / L2 保持 / L3 无计划，带滞回和恢复计数。每架飞机一个实例。 |
| `algorithm_pkg/degrade_policy.py` | P3 | 调度：`plan_with_degrade()` 把体检结果 + 状态机 + 分级规划 + 护栏串起来。 |
| `algorithm_pkg/safety_guard.py` | P3 | 安全员：NaN、XY/高度边界、跳变、最小距离、时间戳单调，按顺序裁剪。 |
| `algorithm_pkg/adapter_core.py` | P4 | 翻译官（纯逻辑）：时间戳门禁、目标缓冲、目标选择、每机 `DroneRuntime`、ATTACK 无解保持点、JSON payload。不 import rclpy。 |
| `algorithm_pkg/algorithm_node.py` | P4 | ROS 外壳：订阅 A/B/D/E，20Hz 调 `plan_drone()`，发布 C，打日志。每架飞机复用同一个 FSM。 |
| `launch/algorithm.launch.py` | P4 | 一键启动：31 个参数，默认三机 `drone_1/2/3`。 |
| `scripts/p4/` | P4 | 联调脚本：单节点冒烟、单机闭环、失效注入、三机隔离、速度标定。 |
| `package.xml` / `setup.py` | — | 依赖、节点入口、launch 注册。 |
| `test/` | — | 203 项 pytest；`simulate_p3.py` 的时间推进仿真由 `test_scenario_p3.py` 固化，图表不参与断言。 |

---

## 谁负责什么（P4 定下的边界）

这张表是理解代码的关键：同一件事只有一个地方做。

| 事项 | 归谁 |
|---|---|
| 模式/动作从哪来 | 中枢 `/strategy/cmd`；节点只贴标签，不改中枢 |
| 时间戳是否有问题 | P3 `validators`；节点原样搬运，**不改写** |
| 视觉还是外部目标 | 适配层 `choose_target()` |
| 降级到哪一级 | P3 `DegradeFSM`（每机一个实例） |
| L1/L2 保持点 | P3 `degrade_policy` |
| ATTACK 正常但无解时的保持点 | 适配层 `attack_hold()`（唯一的特例） |
| 安全裁剪 | P3 `safety_guard`（P2 还会按己方位置限一次 `max_step`） |
| 发布时刻的 `stamp` | 节点；用发布瞬间的 ROS clock |
| 参数默认值 | `launch` → `PlanParams` |

---

## P4 相对 P3 改了什么

1. **新增纯逻辑适配层** `adapter_core.py`：时间戳门禁、目标缓冲、每机 Runtime、ATTACK 保持点、payload 都搬到这里，可以脱离 ROS 单测。
2. **节点重写**：从“只调 P2 + 自己改写时间戳 + 自己补保持点”改成“每机一个 `DegradeFSM` + 调 `plan_with_degrade()`”；节点只负责消息、参数、日志、发布。
3. **三处搬迁**：
   - 时间戳改写删除（以前把零/未来 stamp 改成 now，导致 P3 的“倒退帧”分支永远触发不了）；
   - 保持点归 P3（L1 外推、L2 保持），节点只保留 ATTACK 正常无解这一个特例；
   - 参数统一进 `launch`，共 31 个。
4. **`max_step` 默认 0.62 m/帧**：12.4 m/s × 0.05s。实验依据见 `文档/P4联调记录.md`：0.30/0.38 会把 ATTACK 压到追不上（最小目标距退化到 50/44m）。
5. **测试**：新增 `test_node_adapter.py` 13 项、`test_scenario_p3.py` 5 项；P1/P2 补强 23 项；总数 203。

---

## 运行

```bash
ros2 launch algorithm_pkg algorithm.launch.py
```

节点默认 `HOVER`，收到 `/strategy/cmd` 的 `TRACK/INTERCEPT/ATTACK` 后才发点。

典型顺序：先 `TAKEOFF`，再 `TRACK` 或 `INTERCEPT/ATTACK`。

`no_plan` 本身不发接口 C。ATTACK 无解时，适配层补“己方当前位置保持点”
（`reason=hold_attack_no_solution`），避免中枢继续用上一帧攻击点。

参数都在 launch 里，常用几个：

| 参数 | 默认 | 含义 |
|---|---|---|
| `v_max` | 6.0 | 兜底速度 |
| `v_max_intercept` | 7.6 | INTERCEPT 速度上限 |
| `v_max_attack` | 12.4 | ATTACK 速度上限 |
| `max_step` | 0.62 | 单帧最大跳变（米） |
| `max_pred_age` | 0.5 | 目标预测年龄上限（秒） |
| `ego_timeout` | 1.0 | 己方状态超时（秒） |
| `d_tail` | 4.0 | INTERCEPT 尾随距离（米） |
| `min_alt` / `safe_alt` / `max_alt` | 3 / 5 / 50 | 高度边界（米） |
| `max_xy` | 1000 | 水平边界（米） |

---

## 测试

```bash
PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q src/algorithm_pkg/test
# 或 Humble 容器内
colcon test --packages-select algorithm_pkg
colcon test-result --verbose
```

| 层 | 文件 | 数量 |
|---|---|---|
| P1 | `test_algorithm_core.py` | 33 |
| P2 | `test_plan_params.py`、`test_plan_policy.py`、`test_plan_replay.py` | 61 |
| P3 | `test_validators.py`、`test_degrade_fsm.py`、`test_degrade_policy.py`、`test_safety_guard.py`、`test_recovery.py`、`test_replay_anomaly.py` | 91 |
| P4 | `test_node_adapter.py` | 13 |
| 场景 | `test_scenario_p3.py` | 5 |

`simulate_p3.py` 是三个复杂场景的时间推进仿真，A/B/C 已由 `test_scenario_p3.py` 固化为 pytest；
也可单独跑 CLI 看迁移过程、导出 CSV/JSON 和图，用法见 `文档/P3调试方法.md`。

---

## 接口

订阅：

- `/vision/target_pose`：`geometry_msgs/PoseStamped`，优先目标源；
- `/external/target_detected`：`geometry_msgs/PoseStamped`，视觉不新鲜时兜底；
- `/strategy/cmd`：`std_msgs/String` JSON，字段 `drone_id`、`action`；
- `/drone_<id>/state`：`nav_msgs/Odometry`。

发布：

- `/planning/target_pose`：`std_msgs/String` JSON，
  字段 `drone_id`、`stamp`、`x`、`y`、`z`、`action`。

`stamp` 是发布时刻的 ROS 时钟秒数；话题名是公共契约，不随包名变化。
接口 C 是**单 topic**，多机靠 JSON 里的 `drone_id` 区分。

---

## 已知问题（P4 联调记录在案，不在本包修）

- **no_plan 不等于停飞机**：中枢最多保留旧点约 2.0s（实测 1.98s），之后才停发。
  验收口径改为“保留时长 + `max_step` 量化盲飞距离”。
- **TRACK 优先视觉**：中枢在视觉新鲜时直接用视觉原始目标，P3 的 TRACK 点可能不被执行。
  这是中枢策略，不是算法 bug。
- **RESUME_PATROL 无分支**：中枢会写入该模式，但 timer 里没有对应分支，该机此后不再收到 setpoint。
- **E/F 自回灌**：`/drone_<id>/state` 中枢既订阅又发布；节点按 `header.stamp` 判新鲜，
  旧 stamp 帧不会覆盖新帧，但回灌会影响“断流”判断的语义。
- **mock 速度上限低于配置**：dry bridge 是 10/6/3 m/s，配置是 12.4/7.6/6.0，
  标定表按实测记录，真机需重测。
- **接口 C 的 action 不是控制输入**：中枢 `plan_callback` 不解析它；它只是标签。

---

## 文档导航

- `文档/算法原理与策略.md`：P1 的预测/拦截/尾随原理与系统约定。
- `文档/P2模式策略契约.md`：七动作合同与边界。
- `文档/P3降级策略.md`：L0/L1/L2/L3 策略表、护栏顺序、待确认参数。
- `文档/P3调试方法.md`：单函数、单帧、场景、系统四层调试方法与图表读法。
- `文档/P3三个复杂场景仿真.md`：A/B/C 三个场景的结果与读法。
- `文档/P4联调记录.md`：冒烟、闭环、失效、三机、标定、Humble、裁决实验与已知问题。
- `文档/课题二内部接口规范.md`、`文档/课题二-课题三-接口规范.md`：上下游接口。
- `文档/Jazzy开发与Humble交付指南.md`：容器与交付约定。
- `文档/代码注释规范.md`：注释风格（Google docstring + `# !` 等标签）。

---

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
