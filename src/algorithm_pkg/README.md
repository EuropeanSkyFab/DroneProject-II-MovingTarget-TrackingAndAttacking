# algorithm_pkg 算法包

本包计算无人机编队的规划点。输入目标位置、己方状态和战术指令，输出一个 ENU 点和动作标签，发布到 `/planning/target_pose`。本包不控制飞行器，不处理避障、姿态、油门和速度闭环。

测试：267 项通过（unit 239、scenario 20、sim 8），生成 25 张证据图。Humble 容器 `colcon build` 与 `colcon test` 通过：267 tests，0 errors，0 failures，0 skipped。真实视觉链路和多机 Gazebo 联调不在本包验收范围，见“已知限制”。

## 1. 快速开始

### 依赖

- ROS 2 Humble，用于节点和 launch。
- Python 3.10 或更高版本。
- 纯 Python 测试需要 `pytest`、`matplotlib`。图表中文需要 `fonts-noto-cjk`；缺失时图表会生成，但中文会缺字形。
- 从仓库根目录执行命令，包路径为 `src/algorithm_pkg`。

### 构建与运行

```bash
cd ~/ros2_ws
colcon build --packages-select algorithm_pkg
source install/setup.bash
ros2 launch algorithm_pkg algorithm.launch.py
```

本 launch 只启动算法节点。中枢、视觉、桥接和飞控由其它包启动，整栈顺序见仓库根目录的 README。

节点启动后默认 `HOVER`。收到 `/strategy/cmd` 的 `TRACK`、`INTERCEPT` 或 `ATTACK` 后发布规划点；`TAKEOFF`、`RETURN` 不发布规划点。典型顺序是先 `TAKEOFF`，再 `TRACK` 或 `INTERCEPT` / `ATTACK`，最后 `RETURN`。

### 测试

纯 Python 测试不需要 ROS，不需要 launch，也不需要中枢在线。下面的 `/usr/bin/python3` 是本项目主机的系统解释器；换机器时，换成你环境里装了 pytest 的 `python3` 或 venv 的 `python`。

```bash
# 全部
PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q src/algorithm_pkg/test

# 单元测试，不出图
PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
    src/algorithm_pkg/test/unit

# 情景测试，出图
PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
    src/algorithm_pkg/test/scenario \
    --scenario-chart-dir /tmp/algorithm_scenario_charts

# 多帧仿真，出图
PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
    src/algorithm_pkg/test/sim \
    --scenario-chart-dir /tmp/algorithm_scenario_charts

# 单条用例
PYTHONPATH=src/algorithm_pkg /usr/bin/python3 -m pytest -q \
    src/algorithm_pkg/test/unit/test_safety_guard.py::test_step_limited
```

测试结束后终端会打印“算法实测结果”和证据图路径。默认图目录是 `/tmp/algorithm_scenario_charts`，可以用 `--scenario-chart-dir` 改。

## 2. 系统定位

本包只负责一件事：根据输入算出一个规划点。中枢负责状态机、限幅和下发，飞控负责动力学执行，视觉负责目标定位。

| 项目 | 来源或去向 | 说明 |
|---|---|---|
| 目标位置 | `/vision/target_pose_cov`、`/vision/target_pose` 或 `/external/target_detected` | 用于速度估计和位置外推 |
| 己方位置 | `/drone_<id>/state` | 用于计算距离、方向和锚点 |
| 战术指令 | `/strategy/cmd` | 决定本帧使用的动作 |
| 规划点 | `/planning/target_pose` | ENU 坐标和动作标签 |

ENU 原点由上游约定。本包不做坐标变换，也不定义原点，只要求目标位置和己方位置在同一坐标系。

本包不处理避障、姿态、油门、中枢状态机、中枢模式切换和飞控加速度限制。

## 3. 数据流与单帧处理

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
       时间戳过滤 · 目标缓存 · 每机运行状态 · 锚点
              ▼
     degrade_policy.py …………… 降级调度
       判断输入是否可用，确定 L0/L1/L2/L3
              ▼
     plan_policy.py ………………… 动作策略
       判断本帧是否发点、发哪个点、给出原因
              ▼
     algorithm_core.py …………… 几何计算
       速度估计 · 位置外推 · 拦截求解 · 边界裁剪
              ▼
     safety_guard.py ……………… 输出约束
       非有限值 · XY 边界 · 高度 · 跳变 · 最小距离 · 时间戳
              ▼
     /planning/target_pose  →  规划点消息
```

单帧流程：

1. 接收目标帧。先过滤时间戳，再进入缓存。时间戳为零、超前、倒退或含非有限值的帧被丢弃，不进入缓存，也不参与降级计数。
2. 20 Hz 定时器选择目标源。视觉在 `target_timeout=0.5 s` 内有新帧时优先使用视觉，否则使用外部引导。带协方差话题优先，无协方差话题在带协方差话题停 0.5 s 后回退。
3. 每架飞机执行一次 `plan_drone()`：输入校验、更新降级级别、按级别规划、输出约束。
4. `point` 非空时发布 JSON。`stamp` 取该帧所用数据的源头时刻；没有来源时回落到发布时刻。
5. 中枢按自身模式和限幅执行。接口 C 的 `action` 只用于选择速度和 HOVER 点筛选，不驱动中枢模式。

只有 `algorithm_node.py` 依赖 ROS，其余模块可以在无 ROS 环境运行 pytest。

## 4. 代码分层

| 文件 | 职责 | 边界 |
|---|---|---|
| `algorithm_node.py` | 订阅、20 Hz 定时、发布 JSON、记录日志 | 不包含判断逻辑 |
| `adapter_core.py` | 时间戳过滤、目标缓存、视觉/外部选源、每机运行状态、锚点、payload | 不执行降级和裁剪 |
| `degrade_policy.py` | 串联输入校验、状态机、分级规划、输出约束 | 不定义动作语义 |
| `validators.py` | 将输入分类为新鲜、过期、非法、倒退、丢失 | 不决定降级级别 |
| `degrade_fsm.py` | L0～L3 迁移、滞回、恢复计数 | 不发布点 |
| `safety_guard.py` | 六步输出约束，顺序固定 | 不修改 action，不重新规划 |
| `plan_policy.py` | 七种动作的发布规则与原因 | 不执行几何计算 |
| `plan_types.py` | `PlanResult` 类型与动作分组 | 不包含业务判断 |
| `plan_params.py` | 参数、默认值、合法性校验 | 其它模块不得硬编码参数 |
| `algorithm_core.py` | 预测、拦截求解、转弯半径判定、边界裁剪，纯函数 | 不依赖 ROS，不包含动作策略 |
| `vision_gate.py` | 视觉协方差、测距模式和话题回退 | 不修改坐标 |
| `launch/algorithm.launch.py` | 41 个启动参数 | 不得隐藏参数 |

ROS 依赖约束：`src/algorithm_pkg/algorithm_pkg/` 下，`import rclpy` 只允许出现在 `algorithm_node.py`。检查命令：

```bash
grep -rn "import rclpy" src/algorithm_pkg/algorithm_pkg/
```

`scripts/p4/`、`scripts/p5/`、`scripts/p7/` 是联调和仿真脚本，可以启动 ROS，不受该约束。

修改参数时的四处同步点：`plan_params.py`、`launch/algorithm.launch.py`、节点 `declare_parameter`、测试断言。只改一处会造成语义不一致。

## 5. 七种动作

`action` 有 7 个枚举值，来自 `/strategy/cmd`。

| 动作 | 处理方式 | 是否发点 |
|---|---|---|
| `TAKEOFF` | 坐标由中枢生成 | 否 |
| `RETURN` | 坐标由中枢生成 | 否 |
| `HOVER` | 进入时捕获己方位置，之后持续发布固定锚点 | 是 |
| `TRACK` | 发布目标预测点，不加偏移 | 是 |
| `INTERCEPT` | 发布拦截点；不可达时发布尾随点 | 是 |
| `ATTACK` | 发布拦截点；无解时由适配层补保持点 | 视解算结果 |
| `RESUME_PATROL` | 保持当前模式，不切模式 | 沿用原动作 |
| 未知动作 | 拒绝并记录告警，不修改当前模式 | 否 |

### HOVER

适配层在进入 HOVER 的第一帧捕获己方位置，之后每帧发同一个点，`reason=hover_hold`，`reachable=True`。锚点高度按 `[min_alt, max_alt]` 收边：低于 3 m 抬到 3 m，高于 50 m 压到 50 m。切出 HOVER 时清空锚点，下次进入重新捕获。

这样做的原因：中枢收不到新规划点时会保留上一帧最多 2 s。HOVER 持续发固定点可以覆盖中枢保留的上一帧点。中枢侧需要支持 HOVER 读规划点；本包不含中枢代码，联调前确认系统侧已实现。

### TRACK

目标身份未确认，或还没到攻击时机。TRACK 只发布目标的预测位置，不做 standoff 偏移。

原因：中枢的 TRACK 分支在水平距离小于 5 m 时会自行环绕（`center_node.py` 的 `TRACK_SAFE_DIST=5.0`）。本包再减一次安全距离会产生双重偏移。

TRACK 的速度由中枢按自身限幅决定。`v_max=4.50` 只作兜底和巡航参考，TRACK 的点不经过该参数。

### INTERCEPT

计算目标未来位置。做法是解二次方程，取 `(0, t_go_max]` 内的正根，从早到晚逐个检查动力学可达性，输出第一个可达的拦截点。早根转不过来时继续试晚根。没有任何可达根时输出尾随点：目标运动方向后方 `d_tail=4 m` 处。尾随点 `reachable=False`，也要发布，因为目标下一帧改变方向后可能重新可达。

动力学可达性判定：

```text
R_min(v) = max(v² / a_max, v / omega_max)          # 最小转弯半径
R_req    = |d| / (2·sin(θ/2))                      # 到拦截点的等半径圆弧
R_req < R_min  →  动力学不可达
```

θ 是己方当前速度方向到拦截方向的转向角，`d` 是到拦截点的位移。所有根都不可达时按无解处理，输出尾随点，`reason=dynamic_infeasible_tail`。

边界条件：

- 调用方没传己方速度时跳过判定。
- 己方速度小于 0.5 m/s 时跳过判定。低速时 `R_min` 趋 0，强行判定会放过所有转弯。
- 判定只对 INTERCEPT 生效。

### ATTACK

与 INTERCEPT 使用同一方程，速度上限取 `v_max_attack`。区别在无解时的处理：INTERCEPT 退化为尾随点，ATTACK 不退化为尾随点。原因是中枢收到 ATTACK 后会全速飞行，给出目标后方的尾随点会让动作与意图不一致。

ATTACK 不做动力学可达性判定。末段语义（撞击还是贴近后保持）尚未定案，详见“已知限制”。

ATTACK 无解时，策略层返回 `point=None, reason=attack_no_solution`；适配层补发己方当前位置的保持点，`reason=hold_attack_no_solution`，避免中枢继续使用上一帧的攻击点。

### RESUME_PATROL 与未知动作

`RESUME_PATROL` 在接口里有名字，但没有定义坐标和行为。中枢收到它会告警并保持当前模式，本包也保持原模式，不改模式、不停止发点。

未知动作（拼写错误、空字符串、`None`）返回 `unknown_action` 并记录 warn 日志。不得把未知动作修正为 HOVER。

### 判定顺序

`plan_policy.plan()` 的内部顺序如下，在降级级别确定之后执行：

```text
 1. 归一化：strip + upper
 2. 不在 7 个枚举内         → unknown_action
 3. HOVER 且有己方位置       → 固定锚点，reason=hover_hold
 4. HOVER 且无己方位置/越界  → own_invalid / hover_out_of_envelope
 5. TAKEOFF / RETURN        → *_center_generated
 6. RESUME_PATROL           → 节点层保持原模式
 7. 无目标                   → target_missing
 8. 目标无效                 → target_invalid
 9. 预测年龄 > max_pred_age  → target_too_old
10. 己方位置非法             → own_invalid
11. INTERCEPT/ATTACK 无速度  → velocity_unavailable
12. 拦截方程无解或不可达     → 尾随点（INTERCEPT）或 attack_no_solution（ATTACK）
13. 边界裁剪后无有效点       → constraint_rejected
14. 输出约束拒绝             → guard_* 原因
```

两个年龄参数含义不同：

- `max_age=1.0 s`：目标采样超过该年龄后不用于速度估计。
- `max_pred_age=0.45 s`：预测点超过该年龄后不发布。

目标年龄在 0.45～1.0 s 之间时进入 L1 外推。L1 把目标按匀速模型外推到当前时刻，再把当前时刻的目标快照交给策略层。

## 6. 降级与安全约束

| 级别 | 触发条件 | 输出 | 说明 |
|---|---|---|---|
| L0 `NORMAL` | 输入新鲜且合法 | 正常规划点 | 正常追踪 |
| L1 `EXTRAPOLATING` | 目标短期丢失，时长不超过 `t_extrap_max` | 外推点，沿用原动作 | 目标可能被遮挡 |
| L2 `HOLDING` | 外推超时，己方状态正常 | 固定锚点 | 停在进入 L2 时的位置 |
| L3 `NO_PLAN` | 己方状态中断、输入非法、无法降级 | 默认不发布；目标坏且己方有效时补固定保持点 | 规划失效，由中枢接管 |

L1 继续发外推点。L2 在进入时捕获己方位置，之后发同一个固定点，不跟目标点。L3 默认不发布；如果异常来自目标且己方状态有效，适配层补一个固定保持点，用来覆盖中枢保留的上一帧攻击或拦截点。己方状态失效时不补。

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

恢复不可跳级。回到 L0 的第一帧，跳变上限临时收紧为 `max_step_recover=0.2 m`，随后用 `n_recover_step=5` 帧逐步放开。

己方异常优先于目标异常：没有己方位置时，即使目标新鲜，也不计算拦截点。

时间戳过滤：

| 情况 | 原因 | 处理 |
|---|---|---|
| `stamp <= 0` | 发布端未填写时间戳 | 丢弃，`stamp_zero` |
| `stamp > now + 1e-6` | 时钟不同步 | 丢弃，`stamp_future` |
| `stamp < 上一帧` | 乱序或回灌 | 丢弃，`stamp_reversed` |

被丢弃的帧不进入缓存，也不参与降级计数。己方状态例外：没有时间戳时退回接收时刻判断新鲜度，并记录 warn。

## 7. 参数

全部参数在 `launch/algorithm.launch.py`，共 41 个。下表是关键运行值。

| 参数 | 默认值 | 单位 | 含义 | 来源 |
|---|---:|---|---|---|
| `v_max` | 4.50 | m/s | 兜底和巡航速度 | 本机实测 5.49，保守取 4.50 |
| `v_max_intercept` | 7.60 | m/s | 拦截速度上限 | 桥接 INTERCEPT 档 |
| `v_max_attack` | 12.40 | m/s | 攻击速度上限 | 桥接 ATTACK 档 |
| `a_max` | 9.79 | m/s² | 最大水平加速度 | 本机实测 |
| `yaw_rate_max` | 3.009 | rad/s | 最大偏航角速度 | 本机实测 172.4°/s |
| `t_go_max` | 8.0 | s | 最大拦截时间 | 初值 |
| `max_age` | 1.0 | s | 目标采样最大年龄 | 初值 |
| `max_pred_age` | 0.45 | s | 预测点最大年龄 | 0.5 s 边界会抖进 L1，取 0.45 |
| `t_extrap_max` | 1.0 | s | L1 外推上限 | 与 `max_age` 对齐 |
| `ego_timeout` | 1.0 | s | 己方状态最大年龄 | 己方状态超时参数 |
| `min_distance` | 0.5 | m | 目标最小距离 | 实测无抽搐或过冲 |
| `d_tail` | 4.0 | m | 尾随距离 | 实测可用 |
| `min_alt` / `safe_alt` / `max_alt` | 3 / 5 / 50 | m | 高度边界 | 与中枢一致 |
| `max_xy` | 1000 | m | 水平边界 | 与中枢 `MAX_XY_DIST` 一致 |
| `max_step` | 0.30 | m/帧 | 单帧位移兜底 | 动作专用值优先 |
| `max_step_track` | 0.14 | m/帧 | TRACK 单帧位移 | 桥接常规档 |
| `max_step_intercept` | 0.38 | m/帧 | INTERCEPT 单帧位移 | 桥接 INTERCEPT 档 |
| `max_step_attack` | 0.62 | m/帧 | ATTACK 单帧位移 | 桥接 ATTACK 档 |
| `max_step_recover` | 0.2 | m/帧 | 恢复第一帧的跳变上限 | 实测保持 0.2 |
| `vision_use_covariance` | true | - | 订阅带协方差的视觉话题 | 视觉新话题 |
| `vision_cov_topic` | `/vision/target_pose_cov` | - | 新视觉话题名 | 与视觉包一致 |
| `vision_cov_max_var` | 100.0 | m² | 允许的最大位置方差 | 超过视为不可信 |
| `vision_cov_min_z` | 0.0 | m | 目标最低高度 | 地面目标可为 0 |
| `vision_legacy_timeout` | 0.5 | s | 带协方差话题停多久后回退无协方差话题 | 兼容回退窗口 |

代码兜底值和 launch 不同：`PlanParams` 的 `v_max` / `v_max_intercept` / `v_max_attack` 兜底是 6.0，`a_max` 是 2.0，`yaw_rate_max` 是 90°/s。这些只用于单元测试和未加载 launch 的调试，不要拿来验收。

单帧位移按动作分档：TRACK 0.14、INTERCEPT 0.38、ATTACK 0.62，`max_step=0.30` 只作兜底。层级顺序是：动作专用值优先，没有时用 `max_step`。`0` 表示不限幅，只用于调试。

## 8. 接口

订阅：

| 话题 | 类型 | 说明 |
|---|---|---|
| `/vision/target_pose_cov` | `geometry_msgs/PoseWithCovarianceStamped` | 新视觉目标，优先；带协方差和测距模式 |
| `/vision/target_pose` | `geometry_msgs/PoseStamped` | 无协方差视觉目标；带协方差话题超时后回退 |
| `/external/target_detected` | `geometry_msgs/PoseStamped` | 外部引导，视觉不新鲜时启用 |
| `/strategy/cmd` | `std_msgs/String` | JSON：`{"drone_id", "action"}` |
| `/drone_<id>/state` | `nav_msgs/Odometry` | 己方 ENU 位置和速度 |

视觉消息约定：

- `orientation.w=1.0` 表示地面投影，`0.5` 表示尺寸估距。
- 位置方差超过 `vision_cov_max_var`、NaN、未知模式直接丢弃。
- 带协方差话题只要还在发，无协方差话题就不覆盖；带协方差话题停 `vision_legacy_timeout` 后才回退无协方差话题。
- 带协方差话题刚接上时清空无协方差话题采样，防止第一帧缓存点混进来。
- 地面目标 `z=0` 可以进算法；发点由 `apply_constraints` 按动作最低高度处理。

发布：

| 话题 | 类型 | 字段 |
|---|---|---|
| `/planning/target_pose` | `std_msgs/String` | `drone_id`、`stamp`、`x`、`y`、`z`、`action` |

单话题，多机通过 `drone_id` 区分。每架飞机各有一个 publisher，均发布到同一话题。接口 C 即该话题。

`stamp` 取该帧所用数据的源头时刻：

- 跟踪、拦截、攻击用目标采样的 `stamp`。
- HOVER、L2 固定点、L3 保持点和 ATTACK 保持点用当前己方状态的 `stamp`。
- 没有来源时回落到发布时刻。

JSON 只包含上述 6 个字段。`reason`、`level`、`reachable`、`guard_notes` 只出现在节点日志，不随消息发布。未发点时，通过日志 `本帧不发点：level=… reason=…` 查原因。

## 9. 已知限制与未决项

### 外部系统限制

1. 中枢收不到新规划点时会保留上一帧最多 2 s，实测 1.98 s。本包的 L3 保持点只能覆盖上一帧点，不能取消中枢的保留逻辑。`no_plan` 不等于立即停止。
2. 中枢 TRACK 分支在视觉目标新鲜时优先使用视觉，不读取规划点。TRACK 场景的飞行轨迹不能作为算法验收依据。
3. 接口 C 的 `action` 不驱动中枢模式，也不影响中枢限幅。中枢只在 HOVER 时用它筛选规划点。
4. 两层 `max_step` 同时生效：策略层以己方当前位置为基准限制一次，输出约束再以上一帧发布的点为基准限制一次。两次取更保守的结果。
5. mock 环境的实际限幅是 10 / 6 / 3 m/s，低于配置值。mock 实测速度不代表飞行器能力。

### 跨模块未决项

| ID | 状态 | 说明 |
|---|---|---|
| W-02 | 未合入 | 中枢侧需要支持 HOVER 读规划点；本包不含中枢代码，联调前确认系统侧已实现。 |
| W-05 | 未统一 | 断联 failsafe 时间有 0.5、1.0、1.5、3.0 s 多个口径。 |
| W-06 | 未统一 | 规范写返航回起飞点，中枢代码飞各自停机坪。 |
| W-08 | 部分 | 算法支持带协方差的视觉话题；真实视觉未交付，目标高度方案未定。 |
| W-09 | 部分 | 中枢有动作白名单和 `RESUME_PATROL` 保护；算法没有 GROUND 状态门。 |
| W-11 | 部分 | `v_max_attack`、分档步长与桥接一致；`a_max` / `yaw_rate_max` 用本机实测 9.79 / 172.4，接口规范给的是理论值 10.82 / 185.2，权威值待团队确认。 |
| W-12 | 部分 | L2 发固定锚点；`action` 不切 HOVER，恢复时由谁切回原动作未定。 |

### 不在本包范围

- camera relay：因 protobuf 7.x 与 `gz.msgs10` 不兼容，接口 J/K 未交付。本包不修。
- 6 m/s 横穿目标：追不上。原因是策略层裁剪语义和桥接控制回路的组合，不是 `a_max` / `R_min` 判定。本包不单独修。
- ATTACK 末段语义未定：撞击还是贴近后保持。定案前不对 ATTACK 加动力学判定。
- 多机 Gazebo 联调由系统集成方执行。本包只保证同一目标下三机的模式、状态、FSM、锚点和发布互不串。
- 不做风模型、风速估计和风扰补偿。没有强风测试数据。
- 不恢复 Gazebo GUI。数据有三个方向的变化时，用 3D 路径图和 x-y 顶视图；Gazebo 截图不能代替过程图。

## 10. 测试

测试分五层，详细命令、图表规范和排障方法见 [docs/算法测试基本方法.md](docs/算法测试基本方法.md)。

| 层 | 数量 | 内容 |
|---|---:|---|
| 单元测试 | 239 | 几何、策略、降级、护栏、适配、参数、视觉门 |
| 情景测试 | 20 | 单点行为 7、场景定义 8、视觉门 5 |
| 多帧仿真 | 8 | 追逃降级、L3 保持、3 条多帧场景、级别覆盖汇总、3D 爬升、双机主攻/警戒 |
| 联调脚本 | 3 个 | 节点冒烟、失效注入、多机隔离，需要 ROS |
| 物理仿真 | 4 个场景 + 探针 | PX4/Gazebo 闭环，需要容器 |

情景和多帧测试通过 `scenario_chart` 生成证据图，unit 测试不加载 matplotlib。图型选择、标题、标注和 3D 规则见测试方法第 5 节。测试基线是 `v_max=4.50`、`v_max_intercept=7.60`、`v_max_attack=14.80`、`max_step=0.74`。生产 launch 用 12.40 和分档步长。这条差异是有意的：单元测试覆盖生产参数。

视觉门禁用 `test/support/vision_mock.py` 注入延迟、噪声和丢帧。mock 参数是假设值，不能把 mock 通过写成真实视觉通过。

## 11. 文档与本地复现材料

- 本文件：架构、接口、参数、已知限制。
- [docs/算法测试基本方法.md](docs/算法测试基本方法.md)：测试分层、调试、图表、Gazebo 验收。
- [docs/代码注释规范.md](docs/代码注释规范.md)：中文注释、术语和边界写法。
- [scripts/p4/README.md](scripts/p4/README.md)：ROS 联调脚本。
- [文档/课题二内部接口规范.md](../../文档/课题二内部接口规范.md)、[课题二-课题三-接口规范.md](../../文档/课题二-课题三-接口规范.md)：上下游接口。
- [文档/内部分工细节.md](../../文档/内部分工细节.md)：团队分工和物理参数依据。
- [docs/Jazzy开发与Humble交付指南.md](docs/Jazzy开发与Humble交付指南.md)：Docker/Humble 环境和 PX4 SITL。
- `docs/figures/`：4 张参考图，手工出图，不参与测试。

`.p5/` 是本地仿真工作区，不随包交付。当前证据图在 `.p5/evidence/charts`，验收日志在 `.p5/evidence/clean_env`，旧材料归档在 `.p5/archive`。文档里的 `.p5/` 路径只用于本机复现；拿到包后按交付版命令准备环境。

## 12. 未完成事项

1. **最新 main 尚未合入。** 当前包单独验收；main 还有中枢 HOVER 读规划点、带协方差视觉、防撞消抖、桥接断链 RTL 和三机启动器。合并后需要重新跑中枢/视觉联调。另有 `px4_msgs` 与本地 PX4 v1.17 的字段差异：直接用 latest main 的 `px4_msgs` 配 v1.17 会出现 RTPS 长度错误，合并时一并处理。
2. **双机正式配置未落库。** 中枢、桥接和算法仍按 `drone_1..3` 启动；双机基线是用临时 2 架 PX4、2 路桥接和 3 机节点（drone_3 静默）跑通的，还没有正式的 2 机 launch 和双机探针。
3. **TRACK 分工未定。** 中枢 TRACK 使用全局视觉点并加切向环绕，算法规划点不参与 TRACK 控制；双机同目标需要先定主攻/警戒或环绕归属。当前双机测试只覆盖“主攻 ATTACK + 警戒 HOVER”。
4. **真实视觉未打通。** camera relay（P5-01）仍因 protobuf 与 `gz.msgs10` 不兼容而阻塞；视觉门测试用 mock，真实视觉下的双机不能验收。
5. **三机/多机 Gazebo 不在本包范围。** 三机由中枢负责；本包只保证同一目标下各机的模式、状态、FSM、锚点和发布互不串。
