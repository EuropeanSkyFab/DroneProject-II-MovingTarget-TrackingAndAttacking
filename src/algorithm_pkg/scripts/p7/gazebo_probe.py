#!/usr/bin/env python3
"""Gazebo 多帧探针：起飞后按场景发目标/指令，记录状态、规划点和事件。

只用于联调取证，不参与飞控。默认跑单机 drone_1，无头 Gazebo。

用法（容器内，整栈已启动）：
    python3 gazebo_probe.py --scenario hover --out /root/ros2_ws/.p5/evidence/runs/p7
    python3 gazebo_probe.py --scenario l2 --out /root/ros2_ws/.p5/evidence/runs/p7
    python3 gazebo_probe.py --scenario resume --out /root/ros2_ws/.p5/evidence/runs/p7
    python3 gazebo_probe.py --scenario speed --mode ATTACK --out /root/ros2_ws/.p5/evidence/runs/p7
    python3 gazebo_probe.py --scenario delay --delay 0.5 --out /root/ros2_ws/.p5/evidence/runs/p7
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from typing import List, Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Empty, String

TAKEOFF_AT = 0.5
TAKEOFF_ALT = 6.0
TAKEOFF_TIMEOUT = 30.0
SCENARIOS = ("hover", "l2", "resume", "speed", "delay")


class GazeboProbe(Node):
    def __init__(self, args) -> None:
        super().__init__(f"p7_probe_{args.scenario}")
        self.args = args
        self.start = time.monotonic()
        self.mode_time: Optional[float] = None
        self.target_stopped = False
        self.hover_sent = False
        self.resume_sent = False
        self.takeoff_sent = False
        self.finished = False
        self.armed = False
        self.z = 0.0
        self.state: List[tuple] = []
        self.plans: List[tuple] = []
        self.targets: List[tuple] = []
        self.events: List[tuple] = []

        self.vision_pub = self.create_publisher(PoseStamped, "/vision/target_pose", 20)
        self.cmd_pub = self.create_publisher(String, "/strategy/cmd", 10)
        self.hb_pub = self.create_publisher(Empty, f"/{args.drone}/heartbeat", 10)
        self.create_subscription(Odometry, f"/{args.drone}/state", self._on_state, 20)
        self.create_subscription(String, "/planning/target_pose", self._on_plan, 50)
        self.create_timer(0.05, self._tick)

    def now(self) -> float:
        return time.monotonic() - self.start

    def _event(self, text: str) -> None:
        self.events.append((self.now(), text))
        self.get_logger().info(f"{self.now():.1f}s {text}")

    def _cmd(self, action: str) -> None:
        self.cmd_pub.publish(
            String(data=json.dumps({"drone_id": self.args.drone, "action": action}))
        )
        self._event(f"发 {action}")

    def _truth(self, t: float) -> tuple[float, float, float]:
        # t 是相对节点启动的秒数，不要再减 self.start。
        return (
            self.args.target_x + self.args.target_speed * t,
            self.args.target_y,
            self.args.target_z,
        )

    def _publish_target(self, now: float) -> None:
        if self.args.scenario == "l2" and self.target_stopped:
            return
        if self.args.scenario == "l2" and self.mode_time is not None:
            if now - self.mode_time >= self.args.stop_after:
                self.target_stopped = True
                self._event("目标流停止")
                return
        source = now - self.args.delay if self.args.scenario == "delay" else now
        if source < 0.0:
            return
        x, y, z = self._truth(source)
        msg = PoseStamped()
        # ! 延迟场景保留源头时间戳；重打 stamp 测不出真实延迟。
        ros_now = self.get_clock().now().nanoseconds * 1e-9
        stamp_s = ros_now - self.args.delay if self.args.scenario == "delay" else ros_now
        msg.header.stamp.sec = int(stamp_s)
        msg.header.stamp.nanosec = int((stamp_s - int(stamp_s)) * 1e9)
        msg.header.frame_id = "world"
        msg.pose.position.x = float(x)
        msg.pose.position.y = float(y)
        msg.pose.position.z = float(z)
        msg.pose.orientation.w = 1.0
        self.vision_pub.publish(msg)
        self.targets.append((self.now(), stamp_s, x, y, z, ros_now - stamp_s))

    def _tick(self) -> None:
        now = self.now()
        self.hb_pub.publish(Empty())

        if not self.armed:
            return

        self._publish_target(now)

        if now > TAKEOFF_AT and not self.takeoff_sent:
            self.takeoff_sent = True
            self._cmd("TAKEOFF")

        if self.mode_time is None and self.z > TAKEOFF_ALT:
            self.mode_time = now
            self._cmd(self.args.mode)
        if self.mode_time is None:
            if now > TAKEOFF_TIMEOUT:
                self._event("起飞超时，退出")
                self._shutdown()
            return

        elapsed = now - self.mode_time
        if self.args.scenario == "hover" and not self.hover_sent and elapsed >= self.args.hover_after:
            self.hover_sent = True
            self._cmd("HOVER")
        if self.args.scenario == "resume" and not self.resume_sent and elapsed >= self.args.resume_after:
            self.resume_sent = True
            self._cmd("RESUME_PATROL")

        if elapsed >= self.args.run_after and not self.finished:
            self.finished = True
            self._cmd("HOVER")
            self.create_timer(1.0, self._shutdown)

    def _shutdown(self) -> None:
        self._write_outputs()
        # 探针是一次性脚本；rclpy.spin 在定时器里调用 shutdown 有时不返回，
        # 写完结果直接退出，避免 docker exec 一直挂着。
        os._exit(0)

    def _on_state(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self.z = float(p.z)
        self.state.append(
            (
                self.now(),
                self.mode_time is not None and self.now() - self.mode_time or 0.0,
                float(p.x),
                float(p.y),
                float(p.z),
                math.hypot(float(v.x), float(v.y)),
            )
        )
        self.armed = True

    def _on_plan(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        self.plans.append(
            (
                self.now(),
                self.mode_time is not None and self.now() - self.mode_time or 0.0,
                float(data.get("x", 0.0)),
                float(data.get("y", 0.0)),
                float(data.get("z", 0.0)),
                str(data.get("action", "")),
            )
        )

    def _write_outputs(self) -> None:
        os.makedirs(self.args.out, exist_ok=True)
        prefix = f"{self.args.scenario}_{self.args.mode.lower()}"
        state_path = os.path.join(self.args.out, f"{prefix}_state.csv")
        plan_path = os.path.join(self.args.out, f"{prefix}_plan.csv")
        target_path = os.path.join(self.args.out, f"{prefix}_target.csv")
        summary_path = os.path.join(self.args.out, f"{prefix}_summary.json")

        with open(state_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "t_mode", "x", "y", "z", "speed"])
            w.writerows(self.state)
        with open(plan_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "t_mode", "x", "y", "z", "action"])
            w.writerows(self.plans)
        with open(target_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "stamp", "x", "y", "z", "age"])
            w.writerows(self.targets)

        summary = self._summary()
        with open(summary_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        self.get_logger().info(f"结果: {summary_path}")

    def _summary(self) -> dict:
        result = {
            "scenario": self.args.scenario,
            "mode": self.args.mode,
            "delay": self.args.delay,
            "state_count": len(self.state),
            "plan_count": len(self.plans),
            "target_count": len(self.targets),
            "events": self.events,
        }
        if self.mode_time is None:
            result["error"] = "未起飞"
            return result

        after_mode = [s for s in self.state if s[1] > 0.5]
        if after_mode:
            result["max_speed"] = max(s[5] for s in after_mode)
            result["mean_speed"] = sum(s[5] for s in after_mode) / len(after_mode)
        plan_after = [p for p in self.plans if p[1] > 0.5]
        if plan_after:
            jumps = [
                math.hypot(b[2] - a[2], b[3] - a[3])
                for a, b in zip(plan_after, plan_after[1:])
            ]
            result["plan_jump_max"] = max(jumps) if jumps else 0.0
            result["plan_rate_hz"] = len(plan_after) / max(
                1e-9, plan_after[-1][0] - plan_after[0][0]
            )
        # 固定点场景：看事件之后规划点和己方的变化范围。
        event_text = {"hover": "发 HOVER", "l2": "目标流停止", "resume": "发 RESUME_PATROL"}
        if self.args.scenario in event_text:
            event_t = next(
                (e[0] for e in self.events if event_text[self.args.scenario] in e[1]),
                None,
            )
            if event_t is not None:
                window = [
                    p for p in self.plans if p[0] >= event_t + 0.3
                ]
                if window:
                    result["plan_span_x"] = max(p[2] for p in window) - min(p[2] for p in window)
                    result["plan_span_y"] = max(p[3] for p in window) - min(p[3] for p in window)
                own_window = [s for s in self.state if s[0] >= event_t + 0.3]
                if own_window:
                    result["own_span_x"] = max(s[2] for s in own_window) - min(s[2] for s in own_window)
                    result["own_span_y"] = max(s[3] for s in own_window) - min(s[3] for s in own_window)
                result["event_t"] = event_t
        if self.args.scenario == "resume":
            event_t = next(
                (e[0] for e in self.events if "RESUME_PATROL" in e[1]), None
            )
            if event_t is not None:
                before = [p for p in self.plans if event_t - 2.0 <= p[0] < event_t]
                after = [p for p in self.plans if event_t <= p[0] < event_t + 2.0]
                result["plan_before_resume"] = len(before)
                result["plan_after_resume"] = len(after)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Gazebo 多帧探针")
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--mode", default="TRACK")
    parser.add_argument("--drone", default="drone_1")
    parser.add_argument("--out", default="/root/ros2_ws/.p5/evidence/runs/p7")
    parser.add_argument("--target-x", type=float, default=30.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=10.0)
    parser.add_argument("--target-speed", type=float, default=2.0)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--hover-after", type=float, default=3.0)
    parser.add_argument("--resume-after", type=float, default=3.0)
    parser.add_argument("--stop-after", type=float, default=4.0)
    parser.add_argument("--run-after", type=float, default=10.0)
    args = parser.parse_args()

    rclpy.init()
    node = GazeboProbe(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node._write_outputs()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
