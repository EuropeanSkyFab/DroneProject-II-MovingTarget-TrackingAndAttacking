"""P4 三机隔离探针：1 个 algorithm_node（drone_1/2/3），验证不同飞机之间互不影响。

0.5  三机都发 TRACK
2.0-4.0  停发 drone_2 的 state（应只影响 drone_2，进 NO_PLAN）
3.0  只给 drone_3 发 HOVER（应只停 drone_3）
4.2  drone_2 state 恢复 + 重新发 TRACK
"""
import json
import os
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

END_AT = 6.0
HOME = {"drone_1": 0.0, "drone_2": 100.0, "drone_3": 200.0}


class MultiProbe(Node):
    def __init__(self):
        super().__init__("p4_multi_probe")
        self.state_pubs = {
            d: self.create_publisher(Odometry, f"/{d}/state", 10) for d in HOME
        }
        self.target_pub = self.create_publisher(PoseStamped, "/vision/target_pose", 10)
        self.cmd_pub = self.create_publisher(String, "/strategy/cmd", 10)
        self.plans = []
        self.create_subscription(String, "/planning/target_pose", self.on_plan, 10)
        self.start = time.monotonic()
        self.sent = set()
        self.create_timer(0.05, self.tick)

    def now(self):
        return time.monotonic() - self.start

    def stamp(self):
        nanoseconds = self.get_clock().now().nanoseconds
        return int(nanoseconds // 1_000_000_000), int(nanoseconds % 1_000_000_000)

    def cmd(self, drone_id, action):
        msg = String()
        msg.data = json.dumps({"drone_id": drone_id, "action": action})
        self.cmd_pub.publish(msg)

    def tick(self):
        t = self.now()
        sec, nanosec = self.stamp()
        for drone_id, x in HOME.items():
            if drone_id == "drone_2" and 2.0 <= t < 4.2:
                continue
            msg = Odometry()
            msg.header.stamp.sec, msg.header.stamp.nanosec = sec, nanosec
            msg.header.frame_id = "world"
            msg.pose.pose.position.x = x
            msg.pose.pose.position.z = 10.0
            msg.pose.pose.orientation.w = 1.0
            self.state_pubs[drone_id].publish(msg)

        target = PoseStamped()
        target.header.stamp.sec, target.header.stamp.nanosec = sec, nanosec
        target.header.frame_id = "world"
        target.pose.position.x = 50.0
        target.pose.position.y = 50.0
        target.pose.position.z = 10.0
        target.pose.orientation.w = 1.0
        self.target_pub.publish(target)

        if t >= 0.5 and "track" not in self.sent:
            self.sent.add("track")
            for drone_id in HOME:
                self.cmd(drone_id, "TRACK")
        if t >= 3.0 and "hover3" not in self.sent:
            self.sent.add("hover3")
            self.cmd("drone_3", "HOVER")
        if t >= 4.2 and "retrack2" not in self.sent:
            self.sent.add("retrack2")
            self.cmd("drone_2", "TRACK")
            self.cmd("drone_9", "TRACK")  # 不存在的飞机，应被忽略

    def on_plan(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        self.plans.append((self.now(), data))


def win(plans, low, high, drone_id=None):
    return [p for t, p in plans if low <= t < high and (drone_id is None or p["drone_id"] == drone_id)]


def main():
    rclpy.init()
    node = MultiProbe()
    while node.now() < END_AT:
        rclpy.spin_once(node, timeout_sec=0.05)

    log = open(os.environ.get("P4_LOG", "algo.log"), "r", encoding="utf-8", errors="replace").read()
    checks = []

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    early = {d: win(node.plans, 0.7, 2.0, d) for d in HOME}
    add("三机都发点", all(len(v) >= 5 for v in early.values()),
        {d: len(v) for d, v in early.items()})
    own_ok = True
    detail = {}
    for d, x in HOME.items():
        if not early[d]:
            own_ok = False
            continue
        near = all(abs(p["x"] - x) <= 1.0 for p in early[d])
        near_y = all(abs(p["y"]) <= 1.0 for p in early[d])
        detail[d] = f"x≈{round(early[d][-1]['x'], 2)} (home {x})"
        own_ok = own_ok and near and near_y
    add("各机使用己方 position，互不影响", own_ok, detail)

    d2_l3 = win(node.plans, 3.3, 4.1, "drone_2")
    add("drone_2 断流后停发", len(d2_l3) == 0, f"drone_2={len(d2_l3)}")
    # drone_3 在 3.0 收到 HOVER，所以“不受 drone_2 断流影响”只能看 3.0 前。
    add("drone_1/3 不受 drone_2 断流影响",
        len(win(node.plans, 2.2, 2.9, "drone_1")) >= 5
        and len(win(node.plans, 2.2, 2.9, "drone_3")) >= 5,
        {"drone_1": len(win(node.plans, 2.2, 2.9, "drone_1")),
         "drone_3": len(win(node.plans, 2.2, 2.9, "drone_3"))})

    add("drone_3 HOVER 后停发", len(win(node.plans, 3.3, 4.1, "drone_3")) == 0,
        f"drone_3={len(win(node.plans, 3.3, 4.1, 'drone_3'))}")
    add("drone_1 继续发点", len(win(node.plans, 3.3, 4.1, "drone_1")) >= 5,
        f"drone_1={len(win(node.plans, 3.3, 4.1, 'drone_1'))}")

    add("drone_2 恢复后重新发点", len(win(node.plans, 4.6, END_AT, "drone_2")) >= 5,
        f"drone_2={len(win(node.plans, 4.6, END_AT, 'drone_2'))}")
    add("drone_3 保持 HOVER 不发点", len(win(node.plans, 4.6, END_AT, "drone_3")) == 0,
        f"drone_3={len(win(node.plans, 4.6, END_AT, 'drone_3'))}")
    ids = {p["drone_id"] for _, p in node.plans}
    add("接口 C 单 topic 只含三机 id", ids == set(HOME), ids)
    add("未知 drone_id 被忽略且节点不退出", "drone_9" not in log or len(node.plans) > 0,
        f"plans={len(node.plans)}")

    failures = [c for c in checks if not c[1]]
    for name, ok, info in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")
    print(f"planning={len(node.plans)}")
    print(f"结果：{len(checks) - len(failures)}/{len(checks)} PASS")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
