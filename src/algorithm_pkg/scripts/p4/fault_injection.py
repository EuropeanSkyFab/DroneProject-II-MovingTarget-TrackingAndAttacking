"""P4 失效注入探针：只跑 algorithm_node，注入 7 类异常，读日志断言降级。

场景（本机时间轴，20Hz）：
  0.5  TRACK
  2.0-2.6  目标短断流 0.6s        -> NORMAL->EXTRAPOLATING
  4.0-6.2  目标长断流 2.2s        -> EXTRAPOLATING->HOLDING（保持点继续发）
  6.5  目标时间戳回退 -5s         -> 丢弃外部目标帧：stamp_reversed
  7.7  目标未来时间戳 +5s         -> 丢弃外部目标帧：stamp_future
  8.9  目标零时间戳               -> 丢弃外部目标帧：stamp_zero
  8.0-9.5  己方 state 断流 1.5s   -> 进 NO_PLAN（no_plan_ego_lost），停发
  10.5 己方 state 时间戳回退 -3s  -> frame_discarded_time_reversed
  11.5 非法指令 JSON              -> 忽略非法指令
"""
import json
import os
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

END_AT = 12.5


class FaultProbe(Node):
    def __init__(self):
        super().__init__("p4_fault_probe")
        self.state_pub = self.create_publisher(Odometry, "/drone_1/state", 10)
        self.target_pub = self.create_publisher(
            PoseStamped, "/external/target_detected", 10
        )
        self.cmd_pub = self.create_publisher(String, "/strategy/cmd", 10)

        self.plans = []  # (t, data)
        self.create_subscription(String, "/planning/target_pose", self.on_plan, 10)

        self.start = time.monotonic()
        self.track_sent = False
        self.one_shots = set()
        self.timer = self.create_timer(0.05, self.tick)

    def now(self):
        return time.monotonic() - self.start

    def stamp(self, offset=0.0):
        seconds = self.get_clock().now().nanoseconds / 1e9 + offset
        sec = int(seconds)
        nanosec = int(round((seconds - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec, nanosec = sec + 1, nanosec - 1_000_000_000
        return sec, nanosec

    def publish_state(self, t):
        if 8.0 <= t < 9.5:
            return
        msg = Odometry()
        if abs(t - 10.5) < 0.06:
            msg.header.stamp.sec, msg.header.stamp.nanosec = self.stamp(-3.0)
        else:
            msg.header.stamp.sec, msg.header.stamp.nanosec = self.stamp()
        msg.header.frame_id = "world"
        msg.pose.pose.position.x = 0.0
        msg.pose.pose.position.y = 0.0
        msg.pose.pose.position.z = 10.0
        msg.pose.pose.orientation.w = 1.0
        self.state_pub.publish(msg)

    def publish_target(self, t):
        if 2.0 <= t < 2.6 or 4.0 <= t < 6.2:
            return
        msg = PoseStamped()
        msg.header.frame_id = "world"
        if abs(t - 6.5) < 0.06:
            msg.header.stamp.sec, msg.header.stamp.nanosec = self.stamp(-5.0)
        elif abs(t - 7.7) < 0.06:
            msg.header.stamp.sec, msg.header.stamp.nanosec = self.stamp(5.0)
        elif abs(t - 8.9) < 0.06:
            msg.header.stamp.sec, msg.header.stamp.nanosec = 0, 0
        else:
            msg.header.stamp.sec, msg.header.stamp.nanosec = self.stamp()
        msg.pose.position.x = 10.0 + 3.0 * t
        msg.pose.position.y = 0.0
        msg.pose.position.z = 10.0
        msg.pose.orientation.w = 1.0
        self.target_pub.publish(msg)

    def tick(self):
        t = self.now()
        self.publish_state(t)
        self.publish_target(t)

        if t >= 0.5 and not self.track_sent:
            self.track_sent = True
            msg = String()
            msg.data = json.dumps({"drone_id": "drone_1", "action": "TRACK"})
            self.cmd_pub.publish(msg)

        if t >= 11.5 and "bad" not in self.one_shots:
            self.one_shots.add("bad")
            msg = String()
            msg.data = "{not json"
            self.cmd_pub.publish(msg)

    def on_plan(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        self.plans.append((self.now(), data))


def window(items, low, high):
    return [item for item in items if low <= item[0] <= high]


def main():
    rclpy.init()
    node = FaultProbe()
    while node.now() < END_AT:
        rclpy.spin_once(node, timeout_sec=0.05)

    log = open(os.environ.get("P4_LOG", "algo.log"), "r", encoding="utf-8", errors="replace").read()
    checks = []

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    track_plans = window(node.plans, 0.7, 1.9)
    short_gap_plans = window(node.plans, 2.1, 2.6)
    hold_plans = window(node.plans, 5.2, 6.1)
    l3_plans = window(node.plans, 9.1, 9.45)
    after_cmd_plans = window(node.plans, 11.7, END_AT)

    add("正常 TRACK 发点", len(track_plans) >= 5,
        f"count={len(track_plans)}, last={track_plans[-1][1] if track_plans else None}")
    add("短断流进 EXTRAPOLATING", "NORMAL->EXTRAPOLATING" in log,
        "log 含 NORMAL->EXTRAPOLATING")
    add("短断流仍发外推点", len(short_gap_plans) >= 5,
        f"count={len(short_gap_plans)}")
    add("长断流进 HOLDING", "->HOLDING" in log, "log 含 ->HOLDING")
    add("长断流保持点继续发", len(hold_plans) >= 5,
        f"count={len(hold_plans)}")
    add("时间戳回退被丢弃", "stamp_reversed" in log, "log 含 stamp_reversed")
    add("未来时间戳被丢弃", "stamp_future" in log, "log 含 stamp_future")
    add("零时间戳被丢弃", "stamp_zero" in log, "log 含 stamp_zero")
    add("己方断流进 NO_PLAN", "no_plan_ego_lost" in log, "log 含 no_plan_ego_lost")
    add("NO_PLAN 期间停发", len(l3_plans) == 0, f"l3_plans={len(l3_plans)}")
    add("恢复经 HOLDING 回 NORMAL",
        "NO_PLAN->HOLDING" in log and "HOLDING->NORMAL" in log,
        "log 含 NO_PLAN->HOLDING 与 HOLDING->NORMAL")
    add("己方时间戳回退按帧丢弃", "frame_discarded_time_reversed" in log,
        "log 含 frame_discarded_time_reversed")
    add("非法指令被忽略", "忽略非法指令" in log, "log 含 忽略非法指令")
    add("非法指令后仍正常发点", len(after_cmd_plans) >= 3,
        f"count={len(after_cmd_plans)}")

    failures = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"planning={len(node.plans)}")
    print(f"结果：{len(checks) - len(failures)}/{len(checks)} PASS")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
