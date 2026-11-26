#!/usr/bin/env python3
"""延迟中继：把消息晚固定秒数转发，保留源头时间戳。

用途：给 Gazebo 联调注入 0.5 s 延迟，观察算法和中枢在“消息晚到”时的
行为。只在测试时运行，不放进正式 launch。

用法：
    # 视觉目标晚 0.5 s；输出到 /vision/target_pose_delayed
    python3 delay_relay.py --kind pose --topic /vision/target_pose --delay 0.5

    # 规划 JSON 晚 0.5 s
    python3 delay_relay.py --kind string --topic /planning/target_pose --delay 0.5

注意：重打 header.stamp 或 JSON 里的 stamp 会改变语义，本脚本不改。
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String


class _DelayRelay(Node):
    def __init__(self, kind: str, topic: str, out_topic: str, delay: float) -> None:
        super().__init__("p7_delay_relay")
        self.delay = delay
        self._queue: deque = deque()
        self._received = 0
        self._sent = 0

        if kind == "pose":
            self.create_subscription(PoseStamped, topic, self._on_msg, 100)
            self._pub = self.create_publisher(PoseStamped, out_topic, 100)
        else:
            self.create_subscription(String, topic, self._on_msg, 100)
            self._pub = self.create_publisher(String, out_topic, 100)

        self.get_logger().info(
            f"延迟中继：{topic} -> {out_topic}，延迟 {delay:.3f}s（保留 stamp）"
        )
        self.create_timer(0.01, self._flush)

    def _on_msg(self, msg) -> None:
        self._queue.append((time.monotonic(), msg))
        self._received += 1

    def _flush(self) -> None:
        now = time.monotonic()
        while self._queue and now - self._queue[0][0] >= self.delay:
            _, msg = self._queue.popleft()
            self._pub.publish(msg)
            self._sent += 1
            if self._sent % 100 == 0:
                self.get_logger().info(
                    f"已转发 {self._sent} 条，待转发 {len(self._queue)} 条"
                )

    def destroy_node(self) -> bool:
        self.get_logger().info(
            f"退出：收到 {self._received} 条，转发 {self._sent} 条，"
            f"未转发 {len(self._queue)} 条"
        )
        return super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser(description="固定延迟中继（保留源头时间戳）")
    parser.add_argument("--kind", choices=("pose", "string"), required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--out-topic", default=None)
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()

    if args.delay < 0.0:
        raise SystemExit("--delay 不能为负")
    out_topic = args.out_topic or f"{args.topic}_delayed"

    rclpy.init()
    node = _DelayRelay(args.kind, args.topic, out_topic, args.delay)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
