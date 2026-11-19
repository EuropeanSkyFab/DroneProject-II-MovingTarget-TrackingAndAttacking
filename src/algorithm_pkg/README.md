# algorithm_pkg 算法包说明

本包计算无人机编队的规划点：接收目标位置、己方状态和战术指令，输出一个 ENU 点和动作标签，发布到 `/planning/target_pose`。本包不控制飞行器，不处理避障、姿态、油门和速度控制。

当前状态：P1～P4 已完成，P5 参数化已加入，212 项 pytest（原 206 + 6 个参数用例）；Jazzy 下 `colcon build` 与 `colcon test` 通过。

阅读建议：

- 需要运行或联调：见“运行”与“接口”。
- 需要排查异常：见“外部系统限制与已知问题”，以及 [文档/算法测试基本方法.md](../../文档/算法测试基本方法.md)。
- 需要修改代码：先看“代码分层与职责边界”。

---

## 1. 系统定位与数据流

本包在系统中的职责是计算规划点，不承担飞行控制。中枢负责状态机、限幅和指令下发；飞控负责动力学执行。

输入与输出：

| 项目 | 来源 / 去向 | 说明 |
|---|---|---|
| 目标位置 | `/vision/target_pose` 或 `/external/target_detected` | 用于速度估计和位置外推 |
| 己方位置 | `/drone_<id>/state` | 用于计算距离和方向 |
| 战术指令 | `/strategy/cmd` | 决定本帧使用的动作 |
| 规划点 | `/planning/target_pose` | ENU 坐标（东-北-天，米）与动作标签 |

ENU 坐标系的原点由上游约定。本包不做坐标变换，也不定义原点，只要求目标位置和己方位置处于同一坐标系。

本包不处理以下事项：避障、姿态、油门、中枢状态机、中枢模式切换、飞控加速度限制。上述职责属于其他模块或飞控。

### 数据流

```text
 视觉目标          己方状态          战术指令
 /vision/...       /drone_1/state    /strategy/cmd
     │                  │                 │
     └────────┬─────────┴─────────┬───────┘
              ▼                   ▼
     algorithm_node.py ……………… ROS 适配层
       订阅消息 · 20 Hz 定时 · 发布 JSON · 记录日志
              │  纯 Python 数据，不含 ROS 类型
              ▼
     adapter_core.py ………………… 适配逻辑
       时间戳过滤 · 目标缓存 · 视觉/外部选源 · 每机运行状态
              ▼
     degrade_policy.py …………… 降级调度
       判断本帧输入是否可用，确定 L0/L1/L2/L3
              ▼
     plan_policy.py ………………… 动作策略
       判断本帧是否发点、发哪个点、给出原因
              ▼
     algorithm_core.py …………… 几何计算
       速度估计 · 位置外推 · 拦截方程求解 · 边界裁剪
              ▼
     safety_guard.py ……………… 输出约束
       非有限值 · XY 边界 · 高度 · 跳变 · 最小距离 · 时间戳单调
              ▼
     /planning/target_pose  →  规划点消息
```

P3 在流程中出现两次：入口处判断输入是否可用，出口处执行输出约束。该顺序是固定的，原因见第 4 节。

P1～P4 是内部分层编号：P1 几何计算、P2 动作策略、P3 异常降级与安全约束、P4 ROS 适配与联调。

### 单帧处理流程

1. **接收**：目标帧先经过时间戳过滤，再进入缓存。时间戳为零、超前于当前时刻、倒退或含非有限值的帧被丢弃，不进入缓存，也不参与降级计数。
2. **20 Hz 定时器**：选择目标源。视觉源在 `target_timeout=0.5 s` 内有新帧时优先使用视觉，否则使用外部引导。
3. **计算**：每架飞机执行一次 `plan_drone()`：输入校验 → 更新降级级别 → 按级别规划 → 输出约束。
4. **发布**：`point` 非空时发布 JSON，`stamp` 取发布时刻的 ROS 时钟；`point` 为空时不发布。
5. **中枢处理**：中枢按自身模式和限幅执行。接口 C 中的 `action` 只是标签，中枢不解析。

### 分层原因

- 只有 `algorithm_node.py` 依赖 ROS，其余模块可在无 ROS 环境下运行 pytest。
- 几何计算、动作策略、异常降级职责独立，修改其中一项不会影响其他两项。
- 参数集中在 `plan_params.py` 并在启动时校验，配置错误在启动阶段暴露。

---

## 2. 七种动作的处理规则

`action` 共 7 个枚举值，来自 `/strategy/cmd`。下表是本包对七种动作的处理约定。

| 动作 | 使用场景 | 处理方式 | 是否发点 |
|---|---|---|---|
| `TAKEOFF` | 起飞 | 坐标由中枢生成 | 否 |
| `RETURN` | 返航 | 坐标由中枢生成 | 否 |
| `HOVER` | 原地悬停 | 发布己方当前位置 | 是 |
| `TRACK` | 跟踪目标 | 发布目标预测点，不加偏移 | 是 |
| `INTERCEPT` | 拦截目标 | 发布拦截点；不可达时发布尾随点 | 是 |
| `ATTACK` | 撞击目标 | 发布拦截点；不可达时不发布规划点，由适配层补保持点 | 视解算结果 |
| `RESUME_PATROL` | 恢复巡逻 | 接口未定义坐标，不推断 | 否 |
| 未知动作 | 拼写错误、空值 | 拒绝并记录告警，不修改当前模式 | 否 |

### 2.1 `TAKEOFF` / `RETURN`

这两个动作的坐标由中枢生成。本包返回 `takeoff_center_generated` / `return_center_generated`，不发布规划点。

不在本包生成坐标的原因：起降点取决于中枢的停机坪配置和状态机阶段，本包无法获取这些状态，重复计算会产生第二个坐标来源。

### 2.2 `HOVER`

《内部分工细节》将 `HOVER` 定义为“指令无人机在当前坐标原地保持不动”。本包直接发布己方当前位置，`reason=hover_hold`，`reachable=True`。

这样处理的依据：中枢在收不到新规划点时，会保留上一帧最多 2 s。HOVER 期间每帧发布当前位置，可持续覆盖中枢保留的旧点。中枢自身的 HOVER 分支也会发布当前位置，两者不冲突。

边界条件：

- 己方状态未到达时没有当前位置，本帧不发布，`reason=own_invalid`。
- 己方位置超出包线（`z < min_alt`、`z > max_alt`、`|x|/|y| > max_xy`）时不发布，`reason=hover_out_of_envelope`，避免中枢丢弃该点并产生告警日志。

### 2.3 `TRACK`

目标身份尚未确认，或尚未到攻击时机。处理方式是保持跟踪，维持距离。

约定：**TRACK 只发布目标的预测位置，不做 standoff 偏移**。原因是中枢的 TRACK 分支在水平距离小于 5 m 时会自行执行环绕（`center_node.py:604`，`TRACK_SAFE_DIST=5.0`）。若本包再减去一次安全距离，会产生双重偏移。

本包不发布速度指令。TRACK 的速度由中枢按自身限幅决定。`v_max` 运行值是 4.50（学长理论；代码兜底 6.0），TRACK 的点不经过该参数。

### 2.4 `INTERCEPT`

计算目标未来位置，而不是当前位置。做法是求解二次方程，取 `(0, t_go_max]` 内的最早正根，输出拦截点。无解时输出尾随点：目标运动方向后方 `d_tail=4 m` 处。尾随点 `reachable=False`，但仍需发布，因为目标下一帧改变方向后可能重新可达。

`reachable=False` 不代表不发布，这一点容易混淆。

常见无解情况：目标速度更快、目标横穿、目标距离超出 `t_go_max` 可达范围。判定方式是解方程，不使用“目标更快”作为前置条件。

### 2.5 `ATTACK`

与 INTERCEPT 使用同一方程，速度上限取 `v_max_attack`，launch 为 23.59 m/s（学长理论；代码兜底 6.0）。

区别只在无解时的处理：INTERCEPT 退化为尾随点，ATTACK 不退化为尾随点。原因是中枢收到 ATTACK 后会全速飞行，若本包给出目标后方的尾随点，飞行器会全速冲向目标后方，动作与意图不一致。

ATTACK 无解时，策略层返回 `point=None, reason=attack_no_solution`；适配层补发“己方当前位置”的保持点（`reason=hold_attack_no_solution`），避免中枢继续使用上一帧的攻击点。

“撞击前解除限幅”由中枢和飞控负责，不在本包。接口 C 的 `action` 不参与中枢控制逻辑，中枢 `plan_callback` 不解析该字段。

### 2.6 `RESUME_PATROL`

中枢会下发该模式，但接口未定义对应坐标。本包返回 `resume_patrol_unspecified`，不发布规划点，也不推断坐标。

需要注意：中枢状态机没有 `RESUME_PATROL` 分支，进入该模式后该架飞机不再收到 setpoint。该行为属于中枢实现，不在本包范围内。

### 2.7 未知动作

未知动作（拼写错误、空字符串、`None`）一律返回 `unknown_action` 并记录 warn 日志。

不得将未知动作修正为 HOVER。修正后飞行器会继续执行旧模式，地面无法察觉异常。

### 2.8 动作策略的判定顺序

以下 14 步是 `plan_policy.plan()` 的内部顺序，在 P3 确定降级级别之后执行。P3 的输入校验（包括“己方异常优先”）在更早一层，不在本表中。排查时先确认级别，再对照本表。

```text
 1. 归一化：strip + upper
 2. 不在 7 个枚举内         → unknown_action
 3. HOVER 且有己方位置       → 当前位置，reason=hover_hold
 4. HOVER 且无己方位置/越界  → own_invalid / hover_out_of_envelope
 5. TAKEOFF / RETURN        → *_center_generated
 6. RESUME_PATROL           → resume_patrol_unspecified
 7. 无目标                   → target_missing
 8. 目标无效                 → target_invalid
 9. 预测年龄 > max_pred_age  → target_too_old
10. 己方位置非法             → own_invalid
11. INTERCEPT/ATTACK 无速度  → velocity_unavailable
12. 拦截方程无解             → 尾随点（INTERCEPT）或 attack_no_solution（ATTACK）
13. 边界裁剪后无有效点       → constraint_rejected
14. 输出约束拒绝             → guard_* 原因
```

两个年龄参数含义不同：

- `max_age=1.0 s`：目标采样超过该年龄后不再用于速度估计（P1）。
- `max_pred_age=0.5 s`：预测点超过该年龄后不再发布（P2）。

目标年龄在 0.5～1.0 s 之间时进入 L1 外推：L1 将目标按匀速模型外推到当前时刻，构造当前时刻的目标快照后交给 P2，因此 `max_pred_age` 在该层不再拦截。无法外推时（无速度或无采样）退化为保持点或 L3。

---

## 3. 代码分层与职责边界

原则：同一职责只在一个模块中实现。重复实现会产生第二个事实来源。

| 文件 | 职责 | 边界 |
|---|---|---|
| `algorithm_node.py` | 订阅、20 Hz 定时、发布 JSON、记录日志 | 不包含判断逻辑 |
| `adapter_core.py` | 时间戳过滤、目标缓存、视觉/外部选源、每机运行状态、payload 生成 | 不执行降级和裁剪 |
| `degrade_policy.py` | 串联输入校验、状态机、分级规划、输出约束 | 不定义动作语义 |
| `validators.py` | 将输入分类为新鲜、过期、非法、倒退、丢失 | 不决定降级级别 |
| `degrade_fsm.py` | L0～L3 迁移、滞回、恢复计数 | 不发布点 |
| `safety_guard.py` | 六步输出约束，顺序固定 | 不修改 action，不重新规划 |
| `plan_policy.py` | 七种动作的发布规则与原因 | 不执行几何计算 |
| `plan_types.py` | `PlanResult` 类型与动作分组 | 不包含业务判断 |
| `plan_params.py` | 参数、默认值、合法性校验 | 其他模块不得硬编码参数 |
| `algorithm_core.py` | 预测、拦截求解、边界裁剪，纯函数 | 不依赖 ROS，不包含动作策略 |
| `launch/algorithm.launch.py` | 33 个启动参数 | 不得隐藏参数 |

### ROS 依赖约束

在 `src/algorithm_pkg/algorithm_pkg/` 下，`import rclpy` 只允许出现在 `algorithm_node.py`。其余模块为纯 Python，单元测试不需要 ROS 环境。

检查命令：

```bash
grep -rn "import rclpy" src/algorithm_pkg/algorithm_pkg/
```

`scripts/p4/` 为联调脚本，会启动 ROS 节点，不受该约束限制。

该约束可直接验证：设置 `PYTHONPATH=src/algorithm_pkg` 后，`pytest` 可运行全部 212 项测试，不需要 launch，也不需要中枢节点在线。

### 修改指引

| 需要修改的内容 | 修改位置 | 注意事项 |
|---|---|---|
| 速度上限 | `plan_params.py`、`launch`、节点 `declare_parameter`、测试断言 | 四处必须同步，否则语义不一致 |
| 某动作是否发布规划点 | `plan_policy.py` | 不在节点中增加条件判断 |
| L1/L2 保持逻辑 | `degrade_policy.py` | 不修改 `degrade_fsm` 的计数逻辑 |
| 进级/恢复帧数 | `plan_params.py` | 不硬编码 2/3 |
| 输出约束顺序 | `safety_guard.py` | 顺序不可调整 |
| 新增话题 | `algorithm_node.py`、`adapter_core.py` | 不将解析逻辑放入节点回调 |
| 拦截计算 | `algorithm_core.py` | 不在此模块判断动作模式 |

---

## 4. 异常降级与安全约束

目标丢失或己方状态中断时，本包按降级级别决定是否发布规划点。

| 级别 | 触发条件 | 输出 | 说明 |
|---|---|---|---|
| L0 `NORMAL` | 输入新鲜且合法 | 正常规划点 | 正常追踪 |
| L1 `EXTRAPOLATING` | 目标短期丢失，时长 ≤ `t_extrap_max` | 外推点，沿用原动作语义 | 目标可能被遮挡 |
| L2 `HOLDING` | 外推超时，己方状态正常 | 保持点（上一有效点或己方当前位置） | 目标丢失，保持位置 |
| L3 `NO_PLAN` | 己方状态中断、输入非法、无法降级 | 不发布规划点 | 规划失效，由中枢接管 |

只有 L3 不输出规划点。L1、L2 仍发布规划点，但发布的是降级点。恢复时不会直接回到 L0。

L0 下也可能本帧没有可发布的点，例如输出约束拒绝（`guard_*`），或 ATTACK 无解（由适配层补保持点）。这属于单帧拒绝，不是降级。

### 状态迁移

```text
                连续 n_enter=2 帧异常
  NORMAL ───────────────────────────► EXTRAPOLATING
     ▲  │                                   │
     │  │ 连续 2 帧异常且丢失 > 1.0s          │ 丢失 > t_extrap_max=1.0s
     │  └──────────────────────────► HOLDING ◄┘
     │                                   │
     │                                   │ 己方断流 / 目标非法
     │ 连续 n_recover=3 帧有效             ▼
     └──────────────────────────── NO_PLAN
         HOLDING ◄── 连续 n_exit=1 帧有效 ──┘
```

规则：

1. 进入更差级别需要连续 `n_enter=2` 帧异常，单帧异常不改变级别。
2. L1 设有硬上限 `t_extrap_max=1.0 s`，超时立即进入 L2。
3. 恢复不可跳级：L3 → L2 需要 `n_exit=1` 帧有效，L2 → L0 需要 `n_recover=3` 帧有效。
4. 回到 L0 的第一帧，跳变上限临时收紧为 `max_step_recover=0.2 m`，随后用 `n_recover_step=5` 帧逐步放开。

时间戳倒退的帧整帧丢弃，不参与计数，也不改变级别。单帧时间戳异常不应影响滞回状态。

己方异常优先于目标异常：没有己方位置时，即使目标仍然新鲜，也不计算拦截点。

### 关键参数

| 参数 | 默认值 | 单位 | 含义 | 来源 |
|---|---|---|---|---|
| `v_max_intercept` | 7.60 | m/s | 拦截速度上限 | 学长桥接理论（0.38×20）；不是本机实测 |
| `v_max_attack` | 23.59 | m/s | 攻击速度上限 | 学长桥接 `V_max`；理论值 |
| `a_max` | 10.82 | m/s² | 最大水平加速度 | 学长桥接理论；不是本机实测 |
| `yaw_rate_max` | 3.232 | rad/s | 最大偏航角速度 | 学长 185.2°/s；理论值 |
| `v_max` | 4.50 | m/s | 兜底/巡航速度 | 学长 `V_cruise`；理论值 |
| `max_step` | 1.1795 | m/帧 | 单帧最大跳变 | 学长 `V_max=23.59` × 0.05 s；单一值未拆，交付前必须拆 |
| `max_step_recover` | 0.2 | m/帧 | 恢复第一帧的跳变上限 | 未单独标定 |
| `t_go_max` | 8.0 | s | 最大拦截时间 | 初值，未标定 |
| `max_age` | 1.0 | s | 目标采样最大年龄 | 初值，待视觉实测帧率后复核 |
| `max_pred_age` | 0.5 | s | 预测点最大年龄 | 超过后进入 L1 |
| `t_extrap_max` | 1.0 | s | L1 外推上限 | 与 `max_age` 对齐 |
| `ego_timeout` | 1.0 | s | 己方状态最大年龄 | 替代旧参数 `state_timeout` |
| `min_distance` | 0.5 | m | 目标最小距离 | 见“未确认 / 待决策” |
| `d_tail` | 4.0 | m | 尾随距离 | 场景 B 实测可用 |
| `min_alt` / `safe_alt` / `max_alt` | 3 / 5 / 50 | m | 高度边界 | 中枢 `plan_callback` |
| `max_xy` | 1000 | m | 水平边界 | 中枢 `MAX_XY_DIST` |

> # ! 上表的 V_max / a_max / yaw 来自学长桥接模块，只是理论参考，不是本机实测。
> # ! P5 本机实测：默认参数 V_max 13.00；运行时注入 PX4 26/48/12 后 x500 14.82、x500_down_cam 14.69，仍到不了 23.59。
> # ! 23.59 m/s 在本机/真机能否达到未验证，不要拿它当交付能力。

`max_step=1.1795` 是学长理论值 `23.59 m/s × 0.05 s`；单一 `max_step` 无法同时表示 4.5 / 7.6 / 23.59 三档，按动作拆分前不要直接交付。main 桥接的注入参数在 `.p5/main_ws/theory_bridge_params.yaml`，三档是 0.225 / 0.38 / 1.1795。

历史配置：P4 曾按飞控步长用 12.4/0.62；P5 按负责人要求把 launch 改成学长理论值 4.50 / 7.60 / 23.59、10.82、185.2°/s，并标注“理论值，不是本机实测”。`PlanParams` 代码兜底仍是 6.0 / 2.0 / 90。

### 理论参数注入

main 的 `px4-rc.params` 和桥接默认值不改，用 `.p5/main_ws` 下的注入文件覆盖：

- `theory_bridge_params.yaml`：桥接 `max_step_default/intercept/attack = 0.225 / 0.38 / 1.1795`，`max_yaw_rate_deg=185.2`。
- `px4-rc.theory.params` + `start_px4_theory.sh`：PX4 `MPC_XY_VEL_MAX=26`、`MPC_TILTMAX_AIR=48`、`MPC_ACC_HOR_MAX=12`、`MPC_XY_CRUISE=4.5`、`MPC_YAWRAUTO_MAX=185.2`。
- `run_p5_theory.sh`：一键拉起 PX4 + Gazebo + 桥接 + 算法 + 中枢。

首次验证：桥接层注入生效，PX4 参数用 MAVSDK 运行时 set 到 26/48/12/4.5/185.2 并读回一致；但 `x500_down_cam` 实测 `V_max=14.69`、`x500` 实测 14.82，仍不到理论 23.59。说明卡点不是参数没注入，而是 main 的模型/控制器上限；main 的 README/参数表/桥接步长仍然对不上。

这些值仍是学长理论值，不是本机实测；跑完要用 `dynamics_calibrator` 复核。

### 时间戳过滤

适配层在入口拦截三类帧：

| 情况 | 原因 | 处理 |
|---|---|---|
| `stamp <= 0` | 发布端未填写时间戳 | 丢弃，`stamp_zero` |
| `stamp > now + 1e-6` | 时钟不同步 | 丢弃，`stamp_future` |
| `stamp < 上一帧` | 乱序或回灌 | 丢弃，`stamp_reversed` |

被丢弃的帧不进入缓存，也不参与降级计数。己方状态例外：没有时间戳时退回接收时刻判断新鲜度，并记录 warn 日志，因为 E/F 回灌的旧帧会使断流无法检测。

---

## 5. 外部系统限制与已知问题

以下问题的修复位置不在本包，本包在能力范围内做了规避。上真机前应逐项确认。

### 5.1 中枢保留旧规划点最多 2 秒

现象：本包停止发布后，飞行器仍继续飞行约 2 秒。

原因：中枢仅在收到新 JSON 时更新 `latest_plan_targets`，超过 2 s 未更新才清空（`center_node.py:455`）。实测保留 1.98 s。

本包的处理：L3 期间不发布规划点；ATTACK 无解时补保持点；`max_step=0.30` 限制保留期间的飞行距离。仿真中 ATTACK 场景的保留期间飞行距离为 0。

影响：`no_plan` 不等于立即停止。验收应按“保留时长 + 量化飞行距离”计算，不能按“立即停止”验收。

### 5.2 `/drone_<id>/state` 存在回灌

现象：桥接中断后，己方状态仍显示为新鲜。

原因：中枢每 20 Hz 将缓存的 `/{drone_id}/state` 重新发布到同名话题（`center_node.py:651`）。本包订阅同名话题，可能收到中枢转发的旧帧。

本包的处理：按 `header.stamp` 判断新鲜度，旧时间戳不覆盖新帧；时间戳为 0 时退回接收时刻并记录告警。

影响：时间戳为 0 的回灌帧无法区分新旧，这是原理性限制，不能通过配置消除。

### 5.3 TRACK 模式下中枢优先使用视觉目标

现象：TRACK 下本包发布的规划点未被执行，飞行器跟随视觉目标。

原因：中枢 TRACK 分支中，`latest_vision_target` 非空时直接使用视觉目标，不读取规划点（`center_node.py:592`）。实测偏差 14.14 m，仿真最大 19.7 m。

本包的处理：无。该行为由中枢策略决定，本包只保证规划点计算正确。

影响：TRACK 场景的验收只能验证本包的输出，不能以飞行器轨迹作为算法验收依据。

### 5.4 `RESUME_PATROL` 与未知动作导致后续无 setpoint

现象：飞行器进入某模式后不再响应。

原因：中枢会记录这两个动作并发布 `/action`，但状态机没有对应分支，此后不再发布 setpoint。

本包的处理：不推断坐标，返回 `no_plan` 并记录告警。

影响：需要中枢负责人补充分支或明确处理规则。

### 5.5 接口 C 的 `action` 不是控制输入

现象：规划消息中的 `action` 不影响中枢限幅。

原因：中枢 `plan_callback` 不解析 `action`，模式仅由 `/strategy/cmd` 驱动，限幅走中枢自身的接口 H。

影响：`action` 目前仅用于选择速度、记录日志和后续扩展。若需要中枢按动作调整限幅，需要修改中枢接口。

### 5.6 两层 `max_step` 同时生效

现象：己方静止时，TRACK 点始终位于 `own + 0.30`，看似未更新。

原因：P2 的 `apply_constraints` 以己方当前位置为基准限制一次；P3 的输出约束再以“上一帧发布的点”为基准限制一次。

本包的处理：两次约束都取更保守的结果，不互相放宽。

影响：这是预期行为。闭环中己方位置持续变化，点会随之推进。排查抖动时需先确认是哪一层在限制。

### 5.7 规划延迟阈值不一致

接口规范第 5 节规定“规划延迟 > 0.5 s 丢弃”，中枢实现为 `max_plan_delay=0.8 s`（`center_node.py:165`）。中枢使用 `time.time()` 判断规划延迟，使用 ROS clock 生成时间戳。

本包的处理：按 0.5 s 设计，预留 0.3 s 余量。

影响：跨机联调前需要先对时，否则两个时间基准会冲突。

### 5.8 mock 环境的速度上限低于配置

dry bridge 的实际限幅为 10/6/3 m/s；规划 launch 现在按学长理论值 4.50/7.60/23.59，代码兜底 6.0。本机 P5 实测最高 14.82 m/s（运行时注入 PX4 26/48/12 后），不是 23.59。

影响：mock 环境实测的 9.82/5.88/2.94 m/s 是链路能力，不代表飞行器能力，真机需重新标定。

---

## 运行

```bash
ros2 launch algorithm_pkg algorithm.launch.py
```

启动后默认 `HOVER`，每帧发布己方当前位置（`reason=hover_hold`）。收到 `/strategy/cmd` 的 `TRACK` / `INTERCEPT` / `ATTACK` 后改为发布规划点；`TAKEOFF` / `RETURN` 不发布规划点。典型顺序：先 `TAKEOFF`，再 `TRACK` 或 `INTERCEPT` / `ATTACK`，最后 `RETURN`。

全部参数在 launch 文件中，共 33 个，均有注释。修改速度、边界和降级阈值应修改 launch，不应修改代码中的默认值。

## 接口

订阅：

| 话题 | 类型 | 说明 |
|---|---|---|
| `/vision/target_pose` | `geometry_msgs/PoseStamped` | 视觉目标，优先 |
| `/external/target_detected` | `geometry_msgs/PoseStamped` | 外部引导，视觉不新鲜时启用 |
| `/strategy/cmd` | `std_msgs/String` | JSON：`{"drone_id", "action"}` |
| `/drone_<id>/state` | `nav_msgs/Odometry` | 己方 ENU 位置 |

发布：

| 话题 | 类型 | 字段 |
|---|---|---|
| `/planning/target_pose` | `std_msgs/String` | `drone_id`、`stamp`、`x`、`y`、`z`、`action` |

单话题，多机通过 `drone_id` 区分。每架飞机各有一个 publisher，均发布到同一话题。接口 C 即该话题；接口 H 是中枢的限幅接口，本包不涉及。

`stamp` 取发布时刻的 ROS 时钟，不取目标消息的时间戳。目标时间戳仅用于计算年龄。

JSON 仅包含上述 6 个字段。`reason`、`level`、`reachable`、`guard_notes` 只出现在节点日志中，不随消息发布。未发布点时，通过日志 `本帧不发点：level=… reason=…` 查询原因。

## 测试

```bash
# 主机，纯 Python
PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q src/algorithm_pkg/test

# Jazzy / Humble 容器
colcon test --packages-select algorithm_pkg
colcon test-result --verbose
```

共 212 项，按层分布：

| 层 | 文件 | 数量 |
|---|---|---|
| P1 几何 | `test_algorithm_core.py` | 33 |
| P2 策略 | `test_plan_params.py`、`test_plan_policy.py`、`test_plan_replay.py` | 70 |
| P3 降级 | `test_validators.py`、`test_degrade_fsm.py`、`test_degrade_policy.py`、`test_safety_guard.py`、`test_recovery.py`、`test_replay_anomaly.py` | 91 |
| P4 适配 | `test_node_adapter.py` | 13 |
| 场景回归 | `test_scenario_p3.py` | 5 |

测试方法、单帧回放、场景仿真的用法见 [文档/算法测试基本方法.md](../../文档/算法测试基本方法.md)。

## 文档

- 本文件：算法包的接口、策略、分层与已知问题。
- [文档/算法测试基本方法.md](../../文档/算法测试基本方法.md)：测试分层、调试方法、必做场景和排查流程。
- [文档/课题二内部接口规范.md](../../文档/课题二内部接口规范.md)、[课题二-课题三-接口规范.md](../../文档/课题二-课题三-接口规范.md)：上下游接口。
- [文档/内部分工细节.md](../../文档/内部分工细节.md)：团队分工与物理参数依据。

## 未确认 / 待决策（P5 更新）

- 接口 C 的 `stamp`：规范写“源头时间戳”，本包仍按发布时刻。P5 实测接口 C 的 DDS 传输 p99 0.89 ms；真正的大头是规划只有约 10 Hz、出现 0.5～0.86 s 空档。0.8 s 阈值能用但没余量；按源头时间戳的 0.5 s 在 p95 就会超。规范措辞和阈值仍需 main 统一。
- `V_max`：launch 现在按学长桥接理论值 4.50 / 7.60 / 23.59、`a_max=10.82`、`yaw=185.2°/s`；`PlanParams` 代码兜底仍是 6.0 / 2.0 / 90。理论值不是本机实测；P5 默认参数实测 `V_max=13.00`，运行时注入 PX4 26/48/12 后实测 `x500=14.82`、`x500_down_cam=14.69`。23.59 在这套仿真不可达。
- `min_distance=0.5 m`：P5 ATTACK 固定目标最近 0.082 m，没有抽搐或过冲；暂保持 0.5。
- 输出约束第 5 步的 `guard_target_too_close` 在 P5 ATTACK 未误拒；慢速目标风险仍在，保留。
- `max_step_recover=0.2 m/帧`：P5 做了 L2 保持后恢复（目标 15 s 恢复），恢复窗口规划最大跳变 0.275 m，首帧按 0.2 限幅；没有 L3 后恢复，单测 `test_recovery.py` 覆盖，保留 0.2。
- 真机可达速度：P5 数据只覆盖 Gazebo 的 `x500`/`x500_down_cam`，不是真机；V_max 14.8 是仿真模型数据，23.59 是理论值。
- 规划频率：P5 观测到 `/planning/target_pose` 约 10 Hz（launch 写 20 Hz），原因未定位；这是 0.8 s 没余量的主因，后续要查 CPU、定时器或丢帧。
- `a_max`、`yaw_rate_max` 已进 `PlanParams` 和 launch；launch 用学长理论 10.82 / 185.2°/s，`PlanParams` 代码兜底 2.0 / 90°/s。P5 本机实测 10.13 / 168.6 与理论值不同。
- `max_step=1.1795` 是学长理论值；单一值不能覆盖三档，per-action 拆分前不要交付。main 桥接已用注入参数 0.225 / 0.38 / 1.1795，算法侧仍是单值。
- `rosdep install` 未执行成功（镜像依赖齐全，源站超时）。
