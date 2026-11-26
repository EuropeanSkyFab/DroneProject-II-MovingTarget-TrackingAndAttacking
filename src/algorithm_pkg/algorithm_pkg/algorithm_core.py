"""规划算法纯函数核心：目标预测、拦截求解、安全约束。

本模块刻意不依赖 ROS，便于在宿主机和 Humble 容器里直接跑 pytest。

P1 假设：
- 目标做匀速直线运动（CV 模型）。
- 己方在拦截点方向上的速度上限是 v_max；可选传入己方当前速度，
  此时用最大加速度 a_max 与最大偏航角速度 omega_max 做一次动力学
  可达性判定（转弯半径）。不传速度时不做判定。
  判定只对 INTERCEPT 生效；ATTACK 的末段语义尚未定案，不在此改变。
- 位置、速度都使用右手 ENU 坐标系，单位米 / 米每秒。

动力学判定的口径（规范见《课题二内部接口规范》§4）：
- 最小转弯半径 R_min(v) = max(v² / a_max, v / omega_max)；
- 从己方当前位置到拦截点的等半径圆弧所需半径 R_req = |d| / (2·sin(θ/2))，
  θ 是己方当前速度方向到拦截方向的转向角；
- R_req < R_min 视为动力学不可达。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, isfinite, pi, sin, sqrt
from typing import Optional, Sequence

Vec3 = tuple[float, float, float]

_EPS = 1e-9

# 己方速度低于该值时没有可信朝向，动力学判定跳过，单位米/秒。
# 这个阈值同时用作 R_min 计算的速度下限：低速时 R_min→0，
# 强行判定会把所有转弯都放过，不如显式跳过。
_MIN_TURN_SPEED = 0.5

# 代码兜底的动力学参数（宪章值）。launch 会传实测值覆盖。
_CHARTER_A_MAX = 2.0
_CHARTER_OMEGA_MAX = pi / 2.0


@dataclass(frozen=True)
class TargetSample:
    """一次目标位置采样。

    Attributes:
        stamp: ROS 时间戳，单位秒；必须和 now 使用同一个时钟。
        x: ENU x 坐标，单位米。
        y: ENU y 坐标，单位米。
        z: ENU z 坐标，单位米。
    """

    stamp: float
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class TargetState:
    """predict_target 在 now 时刻估计出的目标状态。

    Attributes:
        position: now 时刻的预测位置 (x, y, z)，ENU 米。
        velocity: 估计速度 (vx, vy, vz)，ENU 米/秒。
        stamp: 参与拟合的最新采样时间，秒。
        age: now - stamp，单位秒。
        valid: 目标是否还在 max_age 内。
        velocity_valid: 是否由至少两帧、且时间跨度足够的数据估出速度；
            单帧或时间跨度过小时为 False。
    """

    position: Vec3
    velocity: Vec3
    stamp: float
    age: float
    valid: bool
    velocity_valid: bool


@dataclass(frozen=True)
class InterceptSolution:
    """solve_intercept 的返回值。

    Attributes:
        action: 本次求解对应的战术模式。
        point: 建议发布的 ENU 点；None 表示本帧不应发布。
        reachable: 是否真正解出了拦截点。
        t_go: 预计到达时间，单位秒；尾随或跟踪时为 None。
        reason: 调试原因，便于日志定位。

    Note:
        ATTACK 无解时 point 一定是 None，不会退化为尾随点。
    """

    action: str
    point: Optional[Vec3]
    reachable: bool
    t_go: Optional[float]
    reason: str


def _finite(*values: float) -> bool:
    """判断所有值都是有限数。"""
    return all(isfinite(value) for value in values)


def _dot(a: Vec3, b: Vec3) -> float:
    """三维向量点积。"""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(v: Vec3) -> float:
    """三维向量长度。"""
    return sqrt(_dot(v, v))


def _unit(v: Vec3) -> Optional[Vec3]:
    """返回单位向量；向量长度接近 0 时返回 None。"""
    length = _norm(v)
    if length <= _EPS:
        return None
    return (v[0] / length, v[1] / length, v[2] / length)


def _clamp(value: float, low: float, high: float) -> float:
    """把 value 限制在 [low, high] 内。"""
    return max(low, min(high, value))


def _tail_point(target: TargetState, d_tail: float) -> Vec3:
    """返回目标运动方向后方的尾随点。

    目标速度接近 0 时没有“后方”方向，尾随点退化为目标位置本身。
    """
    direction = _unit(target.velocity)
    if direction is None:
        return target.position
    return (
        target.position[0] - direction[0] * d_tail,
        target.position[1] - direction[1] * d_tail,
        target.position[2] - direction[2] * d_tail,
    )


def _min_turn_radius(speed: float, a_max: float, omega_max: float) -> float:
    """按速度算最小转弯半径 R_min(v) = max(v²/a_max, v/omega_max)，单位米。

    v²/a_max 是横向加速度约束，v/omega_max 是偏航角速度约束；
    分界速度是 a_max/omega_max，低于它由角速度主导，高于它由加速度主导。
    """
    return max(speed * speed / a_max, speed / omega_max)


def _turn_infeasible(
    own: Vec3,
    own_velocity: Vec3,
    point: Vec3,
    a_max: float,
    omega_max: float,
) -> bool:
    """判断“当前位置到拦截点”的转弯是否超出动力学能力。

    几何：过起点和终点的等半径圆弧，弦长 |d|，偏转角 θ（等于圆心角），
    满足 |d| = 2·R·sin(θ/2)，所以所需半径 R_req = |d| / (2·sin(θ/2))。
    θ→0 时 R_req→∞（直线），θ=180° 时 R_req=|d|/2（半圆），无奇点。

    Returns:
        True 表示动力学上转不过来，应当放弃拦截点。
    """
    delta = (
        point[0] - own[0],
        point[1] - own[1],
        point[2] - own[2],
    )
    chord = _norm(delta)
    if chord <= _EPS:
        return False

    speed = _norm(own_velocity)
    if speed < _MIN_TURN_SPEED:
        # 没有可信朝向，不做判定。
        return False

    unit_delta = _unit(delta)
    unit_velocity = _unit(own_velocity)
    if unit_delta is None or unit_velocity is None:
        return False

    # 浮点误差可能让点积略超 ±1，acos 会返回 nan。
    cosine = _clamp(_dot(unit_velocity, unit_delta), -1.0, 1.0)
    turn_angle = acos(cosine)

    # 转向角接近 0：直线飞行，任何 R_min 都不构成约束。
    if turn_angle <= 1e-3:
        return False

    # ! 速度下限用 _MIN_TURN_SPEED：低速时 R_min→0，判定会失去意义。
    # ! 这个下限让 0.5 m/s 附近的 R_min 很小，判定偏宽松，不会误拒。
    effective_speed = max(speed, _MIN_TURN_SPEED)
    required_radius = chord / (2.0 * sin(turn_angle / 2.0))
    return required_radius < _min_turn_radius(effective_speed, a_max, omega_max)



def predict_target(
    samples: Sequence[TargetSample],
    now: float,
    *,
    max_age: float = 1.0,
    max_speed: float = 15.0,
) -> Optional[TargetState]:
    """根据最近的目标位置采样，估计 now 时刻的位置和速度。

    对 x/y/z 三个方向分别做一次最小二乘直线拟合，再把直线外推到 now。
    只有一帧可用，或时间跨度太小时，位置取最新帧，速度设为 0，
    并令 velocity_valid=False。

    Args:
        samples: 目标位置采样序列；顺序可以乱，函数内部会按 stamp 排序。
        now: 当前时刻，单位秒；必须和 samples 使用同一个时钟。
        max_age: 允许的最大数据年龄，单位秒；超过后 valid=False。
        max_speed: 允许的最大目标速度，单位米/秒；超过则返回 None。

    Returns:
        估计出的 TargetState；没有任何可用采样、出现 NaN/Inf，
        或速度超过 max_speed 时返回 None。

    Raises:
        ValueError: max_age 为负数，或 max_speed 不是正数。

    Note:
        返回 valid=False 时仍然会返回状态，方便调用方记录日志；
        但调用方不应再基于它规划。
    """
    if not _finite(now):
        return None
    if max_age < 0.0:
        raise ValueError("max_age must be non-negative")
    if max_speed <= 0.0:
        raise ValueError("max_speed must be positive")

    clean: list[TargetSample] = []

    # 按时间排序，并丢掉未来时间戳、数值非法、以及同一时刻重复的采样。
    for sample in sorted(samples, key=lambda item: item.stamp):
        if not _finite(sample.stamp, sample.x, sample.y, sample.z):
            continue
        if sample.stamp > now + _EPS:
            continue
        if clean and sample.stamp - clean[-1].stamp <= _EPS:
            # 同一时间戳只保留最后一条；后到的通常是最新修正值。
            clean[-1] = sample
        else:
            clean.append(sample)

    if not clean:
        return None

    latest = clean[-1]
    age = max(0.0, now - latest.stamp)

    if len(clean) == 1:
        # 只有一帧时估不出速度，只能把速度当成 0。
        # velocity_valid=False 提醒调用方：这不是“目标静止”的结论。
        return TargetState(
            position=(latest.x, latest.y, latest.z),
            velocity=(0.0, 0.0, 0.0),
            stamp=latest.stamp,
            age=age,
            valid=age <= max_age,
            velocity_valid=False,
        )

    # t_ref 是最新采样时刻。用相对时间做拟合，
    # 避免大时间戳相减时损失浮点精度。
    t_ref = latest.stamp
    times = [sample.stamp - t_ref for sample in clean]

    # count 是参与拟合的样本数；
    # mean_t 是相对时间的平均值；
    # denominator 是最小二乘斜率公式的分母，单位是秒²。
    count = len(clean)
    mean_t = sum(times) / count
    denominator = sum((t - mean_t) ** 2 for t in times)

    if denominator <= _EPS:
        # 时间跨度太小时，斜率会非常不稳定，
        # 退化为“位置取最新采样、速度设为 0”。
        velocity = (0.0, 0.0, 0.0)
        position_at_ref = (latest.x, latest.y, latest.z)
        velocity_valid = False
    else:
        # 把 x、y、z 三个方向拆开，每个方向单独拟合一条直线。
        coords = (
            [sample.x for sample in clean],
            [sample.y for sample in clean],
            [sample.z for sample in clean],
        )

        slopes = []
        intercepts = []
        for values in coords:
            # mean_value 是该方向位置的平均值。
            mean_value = sum(values) / count

            # slope 是最小二乘直线的斜率，物理意义是速度分量，单位 m/s。
            # 分子是“时间偏差 × 位置偏差”的和，
            # 分母是“时间偏差平方和”。
            slope = sum(
                (t - mean_t) * (value - mean_value)
                for t, value in zip(times, values)
            ) / denominator

            # intercept 是直线在 t=0（最新采样时刻）的位置，单位米。
            # 注意：它不是“拦截点”，只是拟合位置。
            slopes.append(slope)
            intercepts.append(mean_value - slope * mean_t)

        velocity = (slopes[0], slopes[1], slopes[2])
        position_at_ref = (intercepts[0], intercepts[1], intercepts[2])
        velocity_valid = True

    speed = _norm(velocity)

    # ! 速度超过上限通常意味着检测跳变；直接丢帧，避免规划被带飞。
    if not isfinite(speed) or speed > max_speed:
        return None

    # 把拟合结果从 t_ref 外推到 now。
    position_now = (
        position_at_ref[0] + velocity[0] * age,
        position_at_ref[1] + velocity[1] * age,
        position_at_ref[2] + velocity[2] * age,
    )
    if not _finite(*position_now, *velocity):
        return None

    return TargetState(
        position=position_now,
        velocity=velocity,
        stamp=latest.stamp,
        age=age,
        valid=age <= max_age,
        velocity_valid=velocity_valid,
    )


def solve_intercept(
    own: Vec3,
    target: Optional[TargetState],
    v_max: float,
    *,
    mode: str = "INTERCEPT",
    t_go_max: float = 8.0,
    min_distance: float = 0.5,
    d_tail: float = 4.0,
    own_velocity: Optional[Vec3] = None,
    a_max: float = _CHARTER_A_MAX,
    omega_max: float = _CHARTER_OMEGA_MAX,
) -> InterceptSolution:
    """求解匀速目标的最优拦截点。

    拦截方程：

        |p_t + v_t * t - p_o| = v_max * t

    展开为二次方程：

        A*t^2 + B*t + C = 0
        A = |v_t|^2 - v_max^2
        B = 2 * (p_t - p_o) · v_t
        C = |p_t - p_o|^2

    Args:
        own: 己方当前位置，ENU 米。
        target: predict_target 的输出；None 或 valid=False 时不求解。
        v_max: 己方最大速度，米/秒。
        mode: TRACK、INTERCEPT 或 ATTACK，大小写不敏感。
        t_go_max: 允许的最大拦截时间，秒；超过视为不可达。
        min_distance: 小于该距离时视为太近，不输出点，米。
        d_tail: INTERCEPT 不可达时，尾随点距目标多远，米。
        own_velocity: 己方当前速度，ENU 米/秒；``None`` 表示调用方没提供，
            此时跳过动力学判定。只有 INTERCEPT/ATTACK 用得到。
        a_max: 己方最大水平加速度，米/秒²；代码兜底 2.0，launch 用 9.79。
        omega_max: 己方最大偏航角速度，弧度/秒；代码兜底 pi/2，
            launch 用 172.4°/s。

    Returns:
        InterceptSolution。point=None 表示本帧不要发布点。

    Raises:
        ValueError: v_max/t_go_max/a_max/omega_max 非正，
            或 min_distance/d_tail 为负。

    Note:
        TRACK 只返回目标预测点，不做 standoff 偏移；偏移由中枢负责。
        ATTACK 不可达时返回 None，绝不退化为尾随点。
        INTERCEPT 的动力学不可达（转弯半径不够）按“不可达”处理，
        走尾随点；ATTACK 不做动力学判定，原因见模块说明与 README。
    """
    action = str(mode).strip().upper()

    # ! 上游约定 action 用大写，但联调时可能传小写或带空格。
    # ! 这里先归一化，避免模式匹配失败后静默返回 None。
    if action not in ("TRACK", "INTERCEPT", "ATTACK"):
        return InterceptSolution(action, None, False, None, "unsupported_mode")
    if target is None or not target.valid:
        return InterceptSolution(action, None, False, None, "target_invalid")
    if not _finite(*own, *target.position, *target.velocity, v_max):
        return InterceptSolution(action, None, False, None, "non_finite_input")
    if own_velocity is not None and not _finite(*own_velocity):
        return InterceptSolution(action, None, False, None, "non_finite_input")
    if v_max <= 0.0:
        raise ValueError("v_max must be positive")
    if t_go_max <= 0.0:
        raise ValueError("t_go_max must be positive")
    if min_distance < 0.0 or d_tail < 0.0:
        raise ValueError("min_distance and d_tail must be non-negative")
    if a_max <= 0.0:
        raise ValueError("a_max must be positive")
    if omega_max <= 0.0:
        raise ValueError("omega_max must be positive")

    if action == "TRACK":
        # ? 中枢的 TRACK 分支在距离小于 5 米时会自己加环绕偏移。
        # ? 这里只给目标预测点，避免规划再叠一层 standoff 造成双重偏移。
        return InterceptSolution(
            action=action,
            point=target.position,
            reachable=True,
            t_go=None,
            reason="track_target_point",
        )

    delta_pos = (
        target.position[0] - own[0],
        target.position[1] - own[1],
        target.position[2] - own[2],
    )
    distance = _norm(delta_pos)

    # 距离太近时不发点，避免把“拦截”变成“撞机”。
    if distance <= min_distance:
        return InterceptSolution(action, None, False, None, "target_too_close")

    vx, vy, vz = target.velocity
    a = vx * vx + vy * vy + vz * vz - v_max * v_max
    b = 2.0 * _dot(delta_pos, target.velocity)
    c = _dot(delta_pos, delta_pos)

    candidates: list[float] = []

    # ! A≈0：目标速度和己方极限速度相同，二次方程退化成一次方程。
    # ! 直接除以 A 会除零，必须单独分支。
    if abs(a) <= _EPS:
        if abs(b) > _EPS:
            t_linear = -c / b
            if _EPS < t_linear <= t_go_max + _EPS:
                candidates.append(t_linear)
    else:
        discriminant = b * b - 4.0 * a * c

        # ? 判别式在 0 附近会被浮点误差算成 -1e-16。
        # ? 用 -_EPS 容差，避免把“刚好相切”误判成不可达。
        if discriminant >= -_EPS:
            discriminant = max(0.0, discriminant)
            root = sqrt(discriminant)
            for t_root in (
                (-b - root) / (2.0 * a),
                (-b + root) / (2.0 * a),
            ):
                if _finite(t_root) and _EPS < t_root <= t_go_max + _EPS:
                    candidates.append(t_root)

    if candidates:
        # 只保留 (0, t_go_max] 内的正根，按时间从早到晚排。
        ordered = sorted(candidates)

        # 没给己方速度，或动作不是 INTERCEPT 时，不判动力学，取最早根。
        if action != "INTERCEPT" or own_velocity is None:
            t_go = ordered[0]
            point = (
                target.position[0] + target.velocity[0] * t_go,
                target.position[1] + target.velocity[1] * t_go,
                target.position[2] + target.velocity[2] * t_go,
            )
            if not _finite(*point):
                return InterceptSolution(
                    action, None, False, None, "non_finite_solution"
                )
            return InterceptSolution(
                action, point, True, t_go, "intercept_point"
            )

        # 几何有解不等于飞得到：掉头、急转所需半径可能小于 R_min。
        # 早根转不过来时继续试晚根，不能直接退尾随点。
        finite_roots = 0
        for t_go in ordered:
            point = (
                target.position[0] + target.velocity[0] * t_go,
                target.position[1] + target.velocity[1] * t_go,
                target.position[2] + target.velocity[2] * t_go,
            )
            if not _finite(*point):
                continue
            finite_roots += 1
            if _turn_infeasible(own, own_velocity, point, a_max, omega_max):
                continue
            return InterceptSolution(
                action, point, True, t_go, "intercept_point"
            )

        if finite_roots == 0:
            return InterceptSolution(
                action, None, False, None, "non_finite_solution"
            )

        tail = _tail_point(target, d_tail)
        if not _finite(*tail):
            return InterceptSolution(
                action, None, False, None, "non_finite_solution"
            )
        return InterceptSolution(
            action, tail, False, None, "dynamic_infeasible_tail"
        )

    # ! ATTACK 无解时不能退回尾随点。
    # ! 否则飞机会按 ATTACK 全速飞向目标后方，动作和意图完全错位。
    if action == "ATTACK":
        return InterceptSolution(
            action, None, False, None, "attack_no_solution"
        )

    # INTERCEPT 无解时退化为尾随点：卡在目标运动方向后方 d_tail 米。
    tail = _tail_point(target, d_tail)
    if not _finite(*tail):
        return InterceptSolution(
            action, None, False, None, "non_finite_solution"
        )
    return InterceptSolution(action, tail, False, None, "fallback_tail")


def apply_constraints(
    point: Optional[Vec3],
    mode: str,
    *,
    own: Optional[Vec3] = None,
    min_alt: float = 3.0,
    safe_alt: float = 5.0,
    max_alt: float = 50.0,
    max_xy: float = 1000.0,
    max_step: Optional[float] = None,
) -> Optional[Vec3]:
    """把规划点裁剪到当前飞行安全包线内。

    这是纯函数：不修改传入的 point，返回一个新的 (x, y, z)。
    只处理 TRACK/INTERCEPT/ATTACK；其他模式返回 None，因为
    TAKEOFF/HOVER/RETURN 的坐标由中枢生成。

    Args:
        point: 待裁剪的 ENU 点，单位米；None 表示本帧没有点。
        mode: 当前战术模式，大小写不敏感。
        own: 己方当前位置；只有同时给 max_step 时才限制步长。
        min_alt: ATTACK 的最低高度，默认 3 米。
        safe_alt: TRACK/INTERCEPT 的最低高度，默认 5 米。
        max_alt: 最高高度，默认 50 米。
        max_xy: 水平边界，x/y 限制在 ±max_xy，默认 1000 米。
        max_step: 每个控制周期允许的最大位移，单位米；None 表示不限。

    Returns:
        裁剪后的 (x, y, z)；无法或不应发布时返回 None。

    Raises:
        ValueError: 高度/边界参数矛盾，或 max_step 不是正数。

    Note:
        中枢的 plan_callback 只接受 z ∈ [3, 50]；这里先裁。
        max_step 是设定值跳变限制，不是刹车距离。
    """
    action = str(mode).strip().upper()

    # ! 上游约定 action 用大写，但联调时可能传小写或带空格。
    # ! 这里先归一化，避免模式匹配失败后静默返回 None。
    if action not in ("TRACK", "INTERCEPT", "ATTACK"):
        return None

    # 没有可用的点，直接返回 None。
    if point is None:
        return None

    # 点的 x/y/z 里如果有 NaN 或 Inf，也返回 None，
    # 避免把非法坐标发布给飞控。
    if not _finite(*point):
        return None

    # 检查高度参数是否合理：
    # 最低高度不能是负数，最高高度不能小于最低高度。
    if min_alt < 0.0 or max_alt < 0.0 or max_alt < min_alt:
        raise ValueError("altitude bounds are invalid")

    # max_xy 是水平方向的最大允许距离，必须是正数。
    if max_xy <= 0.0:
        raise ValueError("max_xy must be positive")

    # max_step 是每个控制周期允许的最大位移。
    # 如果传了 max_step，它必须是正数。
    if max_step is not None and max_step <= 0.0:
        raise ValueError("max_step must be positive when provided")

    # 如果传了 own，own 也必须是有效坐标。
    if own is not None and not _finite(*own):
        return None

    # ? ATTACK 可能打低空目标，最低高度用 min_alt=3 米。
    # ? TRACK/INTERCEPT 用 safe_alt=5 米，防止撞地和下洗气流。
    # max() 是为了防止 safe_alt 被配得比 min_alt 还低。
    z_min = min_alt if action == "ATTACK" else max(min_alt, safe_alt)

    # 如果最低高度已经超过最高高度，说明参数矛盾，直接报错。
    if z_min > max_alt:
        raise ValueError("safe_alt is above max_alt")

    # 把 x、y 限制在 ±max_xy，把 z 限制在 [z_min, max_alt]。
    # 例如 max_xy=1000 时，x=2000 会变成 1000。
    x = _clamp(point[0], -max_xy, max_xy)
    y = _clamp(point[1], -max_xy, max_xy)
    z = _clamp(point[2], z_min, max_alt)
    clamped: Vec3 = (x, y, z)

    # 只有同时给了当前位置 own 和最大步长 max_step，
    # 才需要限制“这一帧最多移动多远”。
    if own is not None and max_step is not None:
        # 计算从当前位置 own 指向裁剪后点 clamped 的三维向量。
        step = (
            clamped[0] - own[0],
            clamped[1] - own[1],
            clamped[2] - own[2],
        )

        # length 是这个向量的长度，也就是这一帧需要飞行的距离。
        length = _norm(step)

        # 如果这一帧要飞的距离超过 max_step，
        # 就把向量按比例缩短到 max_step。
        # 例如 length=20、max_step=2 时，scale=0.1。
        if length > max_step:
            scale = max_step / length
            clamped = (
                own[0] + step[0] * scale,
                own[1] + step[1] * scale,
                own[2] + step[2] * scale,
            )

            # ! 缩短之后，x/y/z 可能又超出了安全范围。
            # ! 所以这里再裁剪一次，安全边界优先于 max_step。
            clamped = (
                _clamp(clamped[0], -max_xy, max_xy),
                _clamp(clamped[1], -max_xy, max_xy),
                _clamp(clamped[2], z_min, max_alt),
            )

    # max_step 由调用方按 v_max*dt 给初值。
    # a_max/R_min 只用于 solve_intercept 的转弯半径判定，还没用于 max_step 的自适应标定。
    return clamped
