#!/usr/bin/env python3
"""统一场景执行器：按 ``scenario_defs.SCENARIOS`` 跑 Gazebo 场景并出结论。

本文件只做执行，场景定义、目标轨迹、判定逻辑都在 ``scenario_defs.py``
（纯 Python，可在宿主机 pytest）。这里是唯一 import ``rclpy`` 的地方。

容器内用法：

```bash
# 列场景
python3 -m algorithm_pkg.scripts.p5.scenario_suite --list

# 跑一个场景（假定整栈已启动）
python3 .../scenario_suite.py --run intercept_cross_4 --out /root/ros2_ws/.p5/final_run/p6

# 一键编排：每个场景自己 start / stop 整栈
python3 .../scenario_suite.py --orchestrate --run all
```

约定：
- 目标只从 ``/vision/target_pose`` 注入，不碰真实视觉链路（P5-01 未交付）。
- 每个场景跑完自动发 ``HOVER``，再写 CSV。
- 不生成 3D 图；观察用 CSV 与终端汇总表。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Empty, String

from algorithm_pkg.scenario_defs import (
    SCENARIOS,
    Scenario,
    evaluate,
    get_scenario,
    verdict,
)

DEFAULT_OUT = os.environ.get("P6_OUT_DIR", "/root/ros2_ws/.p5/final_run/p6")
STACK_SCRIPT = os.environ.get("P5_STACK", "/root/ros2_ws/.p5/p5_stack.sh")

# 起飞与切模式的边界。
TAKEOFF_AT = 0.5
TAKEOFF_ALT = 6.0
TAKEOFF_TIMEOUT = 35.0


class ScenarioRunner(Node):
    """把 `Scenario` 跑成一次真实的闭环观测。"""

    def __init__(self, scenario: Scenario) -> None:
        super().__init__(f"p5_scenario_{scenario.id}")
        self.scenario = scenario
        drone = scenario.drone
        self.vision_pub = self.create_publisher(PoseStamped, "/vision/target_pose", 20)
        self.cmd_pub = self.create_publisher(String, "/strategy/cmd", 10)
        self.hb_pub = self.create_publisher(Empty, f"/{drone}/heartbeat", 10)
        self.create_subscription(Odometry, f"/{drone}/state", self._on_state, 20)
        self.create_subscription(String, "/planning/target_pose", self._on_plan, 50)
        self.create_timer(0.05, self._tick)

        self.start = time.monotonic()
        self.state: List[tuple] = []
        self.plans: List[tuple] = []
        self.z = 0.0
        self.mode_time: Optional[float] = None
        self.switch_time: Optional[float] = None
        self.sent_takeoff = False
        self.sent_mode = False
        self.sent_switch = False
        self.finished = False
        # ! armed=False 期间只发目标和心跳，不发任何战术指令：
        # ! 整栈还没出数据就发 TAKEOFF 会丢令，场景直接跑偏。
        self.armed = False

    # ------------------------------------------------------------------
    def now(self) -> float:
        return time.monotonic() - self.start

    def _throttle(self) -> float:
        """按场景最大速度给一个保守的限速空档，避免目标飞太快。"""
        return 0.0

    def _publish_cmd(self, action: str) -> None:
        msg = String()
        msg.data = json.dumps({"drone_id": self.scenario.drone, "action": action})
        self.cmd_pub.publish(msg)
        self.get_logger().info(f"{self.now():.1f}s 发 {action}")

    def _publish_target(self, t: float) -> None:
        x, y, z = self.scenario.target_at(t)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0
        self.vision_pub.publish(msg)

    def _tick(self) -> None:
        t = self.now()
        self._publish_target(t)
        self.hb_pub.publish(Empty())

        if not self.armed:
            return

        if t > TAKEOFF_AT and not self.sent_takeoff:
            self.sent_takeoff = True
            self._publish_cmd("TAKEOFF")

        if not self.sent_mode and self.z > TAKEOFF_ALT:
            self.sent_mode = True
            self.mode_time = t
            self._publish_cmd(self.scenario.mode)

        if (
            self.sent_mode
            and self.scenario.switch_mode
            and self.scenario.switch_dist
            and not self.sent_switch
            and self.state
        ):
            tx, ty, tz = self.scenario.target_at(t)
            _, x, y, z, _speed = self.state[-1]
            dist = math.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)
            if dist < self.scenario.switch_dist:
                self.sent_switch = True
                self.switch_time = t
                self._publish_cmd(self.scenario.switch_mode)

        if self.mode_time is not None and t - self.mode_time > self.scenario.run_sec:
            if not self.finished:
                self.finished = True
                self._publish_cmd("HOVER")

        if t > TAKEOFF_TIMEOUT and not self.sent_mode:
            # 没起飞成功就别一直等，直接收尾。
            self.finished = True

    # ------------------------------------------------------------------
    def _on_state(self, msg: Odometry) -> None:
        position = msg.pose.pose.position
        linear = msg.twist.twist.linear
        self.z = float(position.z)
        self.state.append(
            (
                self.now(),
                float(position.x),
                float(position.y),
                float(position.z),
                math.hypot(float(linear.x), float(linear.y)),
            )
        )

    def _on_plan(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self.plans.append(
                (
                    self.now(),
                    float(data.get("x", 0.0)),
                    float(data.get("y", 0.0)),
                    float(data.get("z", 0.0)),
                    str(data.get("action", "")),
                )
            )
        except (ValueError, TypeError):
            return

    # ------------------------------------------------------------------
    def write_csv(self, out_dir: str) -> Dict[str, str]:
        os.makedirs(out_dir, exist_ok=True)
        state_path = os.path.join(out_dir, f"{self.scenario.id}_state.csv")
        plan_path = os.path.join(out_dir, f"{self.scenario.id}_plan.csv")
        with open(state_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["t", "drone_x", "drone_y", "drone_z", "speed",
                 "target_x", "target_y", "target_z", "dist"]
            )
            for t, x, y, z, speed in self.state:
                tx, ty, tz = self.scenario.target_at(t)
                dist = math.sqrt((x - tx) ** 2 + (y - ty) ** 2 + (z - tz) ** 2)
                writer.writerow([f"{t:.3f}", x, y, z, speed, tx, ty, tz, dist])
        with open(plan_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["t", "x", "y", "z", "action"])
            for row in self.plans:
                writer.writerow([f"{row[0]:.3f}", row[1], row[2], row[3], row[4]])
        return {"state": state_path, "plan": plan_path}


def wait_ready(node: "ScenarioRunner", timeout: float = 40.0) -> bool:
    """等到 ``/drone_1/state`` 真的有数据，再让场景时钟起跑。

    # ! 只等“主题存在”不够：``p5_stack.sh start`` 用的是 ``ros2 topic list``，
    # ! 上一套栈的残留主题可能让它在整栈真正出数据前就返回。实测会白等 14 s，
    # ! 目标先跑出去几十米，场景直接判负——这不是算法问题，是观测没对齐。
    """
    deadline = time.monotonic() + timeout
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if len(node.state) >= 10:
            return True
    return False


def reset_clock(node: "ScenarioRunner") -> None:
    """把场景时钟、缓存和发令标志全部归零。"""
    node.start = time.monotonic()
    node.state.clear()
    node.plans.clear()
    node.z = 0.0
    node.mode_time = None
    node.switch_time = None
    node.sent_takeoff = False
    node.sent_mode = False
    node.sent_switch = False
    node.finished = False
    node.armed = True


def run_scenario(scenario: Scenario, out_dir: str) -> Dict[str, object]:
    """跑一个场景，返回汇总行。调用前整栈必须已就绪。"""
    node = ScenarioRunner(scenario)
    node.get_logger().info(f"场景 {scenario.id}：{scenario.title}")
    try:
        if not wait_ready(node):
            raise RuntimeError("40 s 内没有收到 /drone_1/state，整栈未就绪")
        reset_clock(node)
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.now() > TAKEOFF_TIMEOUT + scenario.run_sec + 40.0:
                break
            if node.finished and node.mode_time is not None:
                # 收尾后再多收 3 秒状态，方便看过冲。
                if node.now() - node.mode_time > scenario.run_sec + 3.0:
                    break
            if node.finished and node.mode_time is None:
                break
        paths = node.write_csv(out_dir)
        start_t = node.mode_time if node.mode_time is not None else 0.0
        v_max = 14.8 if scenario.mode == "ATTACK" else 7.6
        metrics = evaluate(
            scenario, node.state, node.plans, start_t=start_t, v_max=v_max
        )
        passed, detail = verdict(scenario, metrics)
    finally:
        node.destroy_node()
    return {
        "id": scenario.id,
        "title": scenario.title,
        "expect": scenario.expect,
        "passed": passed,
        "detail": detail,
        "metrics": metrics,
        "paths": paths,
        "mode_time": node.mode_time,
        "switch_time": node.switch_time,
    }


def _print_summary(rows: List[Dict[str, object]]) -> int:
    ok = 0
    print("\n场景汇总")
    print(f"{'场景':<20} {'期望':<11} {'实测':<34} {'判定'}")
    print("-" * 80)
    for row in rows:
        metrics = row["metrics"]
        detail = (
            f"{row['detail']} | 过冲 {metrics.overshoot:.2f} m"
            f" | Vmax {metrics.max_speed:.2f} m/s"
        )
        mark = "PASS" if row["passed"] else "FAIL"
        ok += 1 if row["passed"] else 0
        print(f"{row['id']:<20} {row['expect']:<11} {detail:<34} {mark}")
    print("-" * 80)
    print(f"合计 {ok}/{len(rows)} PASS")
    return 0 if ok == len(rows) else 1


def _stack(action: str) -> bool:
    try:
        result = subprocess.run(
            ["bash", STACK_SCRIPT, action], capture_output=True, text=True
        )
    except OSError as exc:
        print(f"整栈 {action} 失败：{exc}", file=sys.stderr)
        return False
    print((result.stdout or "").strip()[:400])
    return result.returncode == 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="P5/P6 统一场景执行器")
    parser.add_argument("--list", action="store_true", help="列出全部场景")
    parser.add_argument("--run", default="", help="场景 id，或 all，或逗号分隔")
    parser.add_argument("--out", default=DEFAULT_OUT, help="CSV 输出目录")
    parser.add_argument(
        "--orchestrate",
        action="store_true",
        help="每个场景自行启停整栈（容器内运行）",
    )
    args = parser.parse_args(argv)

    if args.list or not args.run:
        print(f"{'场景':<20} {'期望':<11} 说明")
        print("-" * 78)
        for scenario in SCENARIOS:
            print(f"{scenario.id:<20} {scenario.expect:<11} {scenario.title}")
            print(f"{'':<20} {'':<11} 目的：{scenario.purpose}")
        return 0

    if args.run == "all":
        chosen = list(SCENARIOS)
    else:
        chosen = [get_scenario(item.strip()) for item in args.run.split(",") if item.strip()]

    rclpy.init()
    rows: List[Dict[str, object]] = []
    try:
        for scenario in chosen:
            if args.orchestrate and not _stack("start"):
                print(f"{scenario.id}: 整栈未就绪，跳过", file=sys.stderr)
                continue
            try:
                rows.append(run_scenario(scenario, args.out))
            finally:
                if args.orchestrate:
                    _stack("stop")
    finally:
        rclpy.shutdown()

    if not rows:
        print("没有完成任何场景", file=sys.stderr)
        return 1
    code = _print_summary(rows)
    for row in rows:
        print(f"CSV: {row['paths']['state']} / {row['paths']['plan']}")
    return code


if __name__ == "__main__":
    sys.exit(main())
