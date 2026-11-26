"""单节点冒烟：假 state/视觉/指令，观察 /planning/target_pose。

场景：TAKEOFF -> TRACK -> ATTACK（目标 20m/s 远离）-> 非法指令 -> RESUME_PATROL。
预期：TRACK 发预测点；ATTACK 无解发己方保持点；非法指令被忽略；RESUME 后停发。
"""

from __future__ import annotations

import json
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

PHASE = [
    (0.30, "TAKEOFF"),
    (0.80, "TRACK"),
    (2.30, "ATTACK"),
    (3.80, "BOGUS"),
    (4.30, "RESUME_PATROL"),
]
END_AT = 5.50


class Smoke(Node):
    def __init__(self) -> None:
        super().__init__("p4_smoke")
        self.state_pub = self.create_publisher(Odometry, "/drone_1/state", 10)
        self.vision_pub = self.create_publisher(
            PoseStamped, "/vision/target_pose", 10
        )
        self.cmd_pub = self.create_publisher(String, "/strategy/cmd", 10)
        self.create_subscription(
            String, "/planning/target_pose", self._on_plan, 50
        )
        self.messages: list[tuple[float, float, dict]] = []
        self.start = time.monotonic()
        self.sent: set[int] = set()
        self.create_timer(0.05, self._tick)

    def _now_msg_stamp(self):
        return self.get_clock().now().to_msg()

    def _tick(self) -> None:
        elapsed = time.monotonic() - self.start

        state = Odometry()
        state.header.stamp = self._now_msg_stamp()
        state.header.frame_id = "world"
        state.pose.pose.position.x = 0.0
        state.pose.pose.position.y = 0.0
        state.pose.pose.position.z = 10.0
        self.state_pub.publish(state)

        target = PoseStamped()
        target.header.stamp = self._now_msg_stamp()
        target.header.frame_id = "world"
        target.pose.position.x = 10.0 + 14.0 * elapsed
        target.pose.position.y = 0.0
        target.pose.position.z = 10.0
        self.vision_pub.publish(target)

        for index, (at, action) in enumerate(PHASE):
            if elapsed >= at and index not in self.sent:
                self.sent.add(index)
                cmd = String()
                cmd.data = json.dumps(
                    {"drone_id": "drone_1", "action": action}
                )
                self.cmd_pub.publish(cmd)

    def _on_plan(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        self.messages.append((time.monotonic() - self.start, time.time(), data))


def window(msgs, low, high):
    return [(t, w, d) for t, w, d in msgs if low <= t <= high]


def main() -> int:
    rclpy.init()
    node = Smoke()
    end = time.monotonic() + END_AT
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.02)
    node.destroy_node()
    rclpy.shutdown()

    checks: list[tuple[str, bool, str]] = []

    track = window(node.messages, 0.9, 2.2)
    checks.append(
        (
            "TRACK 发受限预测点(own+0.62)",
            len(track) >= 5
            and all(d.get("action") == "TRACK" for _, _, d in track)
            and all(abs(float(d.get("x", 0.0)) - 0.62) < 0.05 for _, _, d in track)
            and all(abs(float(d.get("z", 0.0)) - 10.0) < 0.05 for _, _, d in track),
            f"count={len(track)}, sample={track[-1][2] if track else None}",
        )
    )

    attack = window(node.messages, 2.4, 3.7)
    ok_attack = len(attack) >= 5 and all(
        d.get("action") == "ATTACK"
        and abs(float(d.get("x", 999.0))) < 1.0
        and abs(float(d.get("y", 999.0))) < 1.0
        and abs(float(d.get("z", 0.0)) - 10.0) < 0.5
        for _, _, d in attack
    )
    checks.append(
        (
            "ATTACK 无解发己方保持点",
            ok_attack,
            f"count={len(attack)}, sample={attack[-1][2] if attack else None}",
        )
    )

    bogus = window(node.messages, 3.85, 4.25)
    checks.append(
        (
            "非法指令被忽略且节点不退出",
            len(bogus) >= 2
            and all(d.get("action") == "ATTACK" for _, _, d in bogus),
            f"count={len(bogus)}",
        )
    )

    after_resume = window(node.messages, 4.45, END_AT)
    checks.append(
        (
            "RESUME_PATROL 后停发",
            len(after_resume) == 0,
            f"count={len(after_resume)}",
        )
    )

    stamps_ok = True
    for t, wall, d in node.messages:
        if abs(float(d.get("stamp", 0.0)) - wall) > 2.0:
            stamps_ok = False
            break
    checks.append(
        (
            "stamp 接近发布时刻",
            stamps_ok,
            f"messages={len(node.messages)}",
        )
    )

    print("=== 单节点冒烟 ===")
    failed = 0
    for name, ok, evidence in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {evidence}")
        failed += 0 if ok else 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
