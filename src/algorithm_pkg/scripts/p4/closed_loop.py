"""P4 单机闭环探针：center + algorithm + dry bridge + 假目标源。

断言：六阶段动作、规划点、中枢转发、HOVER/RETURN 停发规划、返航落地。
不修改仓库文件；仅用于 P4 联调取证。
"""
import json
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

COMMANDS = [
    (0.5, "TAKEOFF"),
    (2.0, "TRACK"),
    (4.0, "INTERCEPT"),
    (5.5, "ATTACK"),
    (7.0, "HOVER"),
    (8.0, "RETURN"),
]
END_AT = 24.0


class Probe(Node):
    def __init__(self):
        super().__init__("p4_closed_loop_probe")
        self.external_pub = self.create_publisher(
            PoseStamped, "/external/target_detected", 10
        )
        self.cmd_pub = self.create_publisher(String, "/strategy/cmd", 10)

        self.plans = []      # (t, dict)
        self.setpoints = []  # (t, x, y, z)
        self.actions = []    # (t, action)
        self.states = []     # (t, x, y, z, stamp)

        self.create_subscription(String, "/planning/target_pose", self.on_plan, 10)
        self.create_subscription(
            PoseStamped,
            "/drone_1/fmu/in/trajectory_setpoint",
            self.on_setpoint,
            10,
        )
        self.create_subscription(String, "/drone_1/action", self.on_action, 10)
        self.create_subscription(Odometry, "/drone_1/state", self.on_state, 10)

        self.start = time.monotonic()
        self.sent = set()
        self.timer = self.create_timer(0.05, self.tick)
        self.get_logger().info("闭环探针启动")

    def now(self):
        return time.monotonic() - self.start

    def tick(self):
        t = self.now()
        target = PoseStamped()
        target.header.stamp = self.get_clock().now().to_msg()
        target.header.frame_id = "world"
        target.pose.position.x = 10.0 + 3.0 * max(0.0, t - 2.0)
        target.pose.position.y = 0.0
        target.pose.position.z = 10.0
        target.pose.orientation.w = 1.0
        self.external_pub.publish(target)

        for when, action in COMMANDS:
            if t >= when and when not in self.sent:
                self.sent.add(when)
                msg = String()
                msg.data = json.dumps({"drone_id": "drone_1", "action": action})
                self.cmd_pub.publish(msg)
                self.get_logger().info(f"{t:.2f}s 发送 {action}")

    def on_plan(self, msg):
        try:
            data = json.loads(msg.data)
        except ValueError:
            return
        self.plans.append((self.now(), data))

    def on_setpoint(self, msg):
        p = msg.pose.position
        self.setpoints.append((self.now(), p.x, p.y, p.z))

    def on_action(self, msg):
        self.actions.append((self.now(), msg.data.strip().upper()))

    def on_state(self, msg):
        p = msg.pose.pose.position
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.states.append((self.now(), p.x, p.y, p.z, stamp))


def window(items, low, high):
    return [item for item in items if low <= item[0] <= high]


def main():
    rclpy.init()
    node = Probe()
    while node.now() < END_AT:
        rclpy.spin_once(node, timeout_sec=0.05)

    checks = []

    def add(name, ok, detail):
        checks.append((name, bool(ok), detail))

    track_plans = window(node.plans, 2.2, 4.0)
    intercept_plans = window(node.plans, 4.2, 5.5)
    attack_plans = window(node.plans, 5.7, 7.0)
    hover_plans = window(node.plans, 7.3, 8.0)
    return_plans = window(node.plans, 8.2, END_AT)

    add("TRACK 有规划点", len(track_plans) >= 5,
        f"count={len(track_plans)}, last={track_plans[-1][1] if track_plans else None}")
    add("INTERCEPT 有规划点", len(intercept_plans) >= 5,
        f"count={len(intercept_plans)}, last={intercept_plans[-1][1] if intercept_plans else None}")
    add("ATTACK 有规划点", len(attack_plans) >= 5,
        f"count={len(attack_plans)}, last={attack_plans[-1][1] if attack_plans else None}")
    add("TRACK 动作标签正确", all(d.get("action") == "TRACK" for _, d in track_plans),
        f"actions={sorted({d.get('action') for _, d in track_plans})}")
    add("INTERCEPT 动作标签正确",
        all(d.get("action") == "INTERCEPT" for _, d in intercept_plans),
        f"actions={sorted({d.get('action') for _, d in intercept_plans})}")
    add("ATTACK 动作标签正确",
        all(d.get("action") == "ATTACK" for _, d in attack_plans),
        f"actions={sorted({d.get('action') for _, d in attack_plans})}")

    add("HOVER 后算法停发规划点", len(hover_plans) == 0,
        f"hover_plans={len(hover_plans)}")
    add("RETURN 后算法停发规划点", len(return_plans) == 0,
        f"return_plans={len(return_plans)}")

    seen_actions = [a for _, a in node.actions]
    for expected in ("TAKEOFF", "TRACK", "INTERCEPT", "ATTACK", "HOVER", "RETURN"):
        add(f"中枢动作 {expected}", expected in seen_actions, f"seen={seen_actions[:12]}")

    intercept_sps = window(node.setpoints, 4.2, 5.5)
    attack_sps = window(node.setpoints, 5.7, 7.0)
    add("INTERCEPT 中枢转发规划点", len(intercept_sps) >= 5,
        f"count={len(intercept_sps)}")
    add("ATTACK 中枢转发规划点", len(attack_sps) >= 5,
        f"count={len(attack_sps)}")

    if intercept_sps and intercept_plans:
        _, sx, sy, sz = intercept_sps[-1]
        plan = intercept_plans[-1][1]
        err = max(abs(sx - plan["x"]), abs(sy - plan["y"]), abs(sz - plan["z"]))
        add("INTERCEPT 设定值≈规划点", err < 0.6, f"err={err:.3f}m")
    if attack_sps and attack_plans:
        _, sx, sy, sz = attack_sps[-1]
        plan = attack_plans[-1][1]
        err = max(abs(sx - plan["x"]), abs(sy - plan["y"]), abs(sz - plan["z"]))
        add("ATTACK 设定值≈规划点", err < 0.6, f"err={err:.3f}m")

    hover_sps = window(node.setpoints, 7.3, 8.0)
    add("HOVER 中枢仍在发设定值", len(hover_sps) >= 5, f"count={len(hover_sps)}")

    if node.states:
        _, x, y, z, _ = node.states[-1]
        add("RETURN 后落在原点附近", abs(x) < 3.0 and abs(y) < 3.0 and z < 1.5,
            f"final=({x:.2f},{y:.2f},{z:.2f})")
    else:
        add("RETURN 后落在原点附近", False, "no state")

    failures = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"planning={len(node.plans)} setpoints={len(node.setpoints)} "
          f"states={len(node.states)} actions={len(node.actions)}")
    print(f"结果：{len(checks) - len(failures)}/{len(checks)} PASS")
    node.destroy_node()
    rclpy.shutdown()
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
