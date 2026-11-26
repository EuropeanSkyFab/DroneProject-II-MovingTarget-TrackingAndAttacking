#!/usr/bin/env python3
"""三机指令序列：按固定时间线向 /strategy/cmd 发动作。

用途：三机 Gazebo 联调时替代临时手敲命令，保证三架飞机的动作顺序一致。
只发指令，不注入目标、不改算法。目标由视觉或 mock 提供。

用法：
    python3 three_drone_commander.py
    python3 three_drone_commander.py --step 10 --dry-run
"""

from __future__ import annotations

import argparse
import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

SEQUENCE = (
    (0.0, "TAKEOFF"),
    (8.0, "TRACK"),
    (18.0, "INTERCEPT"),
    (30.0, "ATTACK"),
    (40.0, "HOVER"),
    (50.0, "RETURN"),
)


class ThreeDroneCommander(Node):
    def __init__(self, drone_ids: list[str], step: float, dry_run: bool) -> None:
        super().__init__("p7_three_drone_commander")
        self._drone_ids = drone_ids
        self._step = step
        self._dry_run = dry_run
        self._pub = self.create_publisher(String, "/strategy/cmd", 10)
        self._start = time.monotonic()
        self._next = 0
        self.create_timer(0.05, self._tick)

    def _tick(self) -> None:
        elapsed = time.monotonic() - self._start
        while self._next < len(SEQUENCE):
            at, action = SEQUENCE[self._next]
            at = at * self._step / 10.0
            if elapsed < at:
                return
            for drone_id in self._drone_ids:
                payload = json.dumps({"drone_id": drone_id, "action": action})
                if self._dry_run:
                    self.get_logger().info(f"{at:.1f}s 计划 {payload}")
                else:
                    self._pub.publish(String(data=payload))
                    self.get_logger().info(f"{at:.1f}s 发 {payload}")
            self._next += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="三机战术指令序列")
    parser.add_argument("--drone-ids", nargs="+", default=["drone_1", "drone_2", "drone_3"])
    parser.add_argument("--step", type=float, default=10.0, help="序列整体缩放；10 表示原时间")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rclpy.init()
    node = ThreeDroneCommander(args.drone_ids, args.step, args.dry_run)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
