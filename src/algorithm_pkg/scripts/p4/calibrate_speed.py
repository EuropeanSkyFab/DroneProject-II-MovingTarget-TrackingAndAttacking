"""P4 速度标定：直接驱动 mock_mavsdk_bridge_dry，测桥接实际限幅速度。

只验“桥接链路能达到的速度”，不代表真机可达速度（裁决 1 A）。
"""
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

END_AT = 8.0


class CalProbe(Node):
    def __init__(self):
        super().__init__("p4_cal_probe")
        self.sp = self.create_publisher(
            PoseStamped, "/drone_1/fmu/in/trajectory_setpoint", 10
        )
        self.act = self.create_publisher(String, "/drone_1/action", 10)
        self.samples = []  # (t, x, y, z)
        self.create_subscription(Odometry, "/drone_1/state", self.on_state, 10)
        self.start = time.monotonic()
        self.phase = None
        self.timer = self.create_timer(0.05, self.tick)

    def now(self):
        return time.monotonic() - self.start

    def tick(self):
        t = self.now()
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.pose.position.x = 50.0
        msg.pose.position.y = 0.0
        msg.pose.position.z = 10.0
        msg.pose.orientation.w = 1.0
        self.sp.publish(msg)

        action = None
        if t >= 0.5:
            action = "ATTACK" if t < 3.0 else "INTERCEPT" if t < 5.0 else "TRACK"
        if action != self.phase:
            self.phase = action
            if action:
                out = String()
                out.data = action
                self.act.publish(out)

    def on_state(self, msg):
        self.samples.append(
            (
                self.now(),
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            )
        )


def speed(samples, low, high):
    window = [s for s in samples if low <= s[0] <= high]
    if len(window) < 5:
        return None
    dx = window[-1][1] - window[0][1]
    dy = window[-1][2] - window[0][2]
    dt = window[-1][0] - window[0][0]
    return (dx * dx + dy * dy) ** 0.5 / dt


def main():
    rclpy.init()
    node = CalProbe()
    while node.now() < END_AT:
        rclpy.spin_once(node, timeout_sec=0.02)

    windows = {
        "ATTACK(0.5-3.0)": (1.0, 2.8, 10.0),
        "INTERCEPT(3.0-5.0)": (3.5, 4.8, 6.0),
        "TRACK(5.0-8.0)": (5.5, 7.8, 3.0),
    }
    print(f"state 样本 {len(node.samples)}")
    ok = True
    for name, (low, high, expected) in windows.items():
        measured = speed(node.samples, low, high)
        if measured is None:
            print(f"[FAIL] {name}: 样本不足")
            ok = False
            continue
        ratio = measured / expected
        passed = 0.85 <= ratio <= 1.15
        ok = ok and passed
        print(
            f"[{'PASS' if passed else 'FAIL'}] {name}: 实测 {measured:.2f} m/s，"
            f"桥接标称 {expected:.1f} m/s，比值 {ratio:.2f}"
        )
    node.destroy_node()
    rclpy.shutdown()
    print("结果：" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
