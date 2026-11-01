import rclpy
import json
import threading
import tkinter as tk
from tkinter import scrolledtext
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry


class MockTopic3(Node):
    def __init__(self, gui_callback):
        super().__init__('mock_topic3')
        self.gui_callback = gui_callback

        # 1. 发布战术指令
        self.cmd_pub = self.create_publisher(String, '/strategy/cmd', 10)
        
        # 2. 订阅无人机状态
        self.state_sub = self.create_subscription(
            Odometry, '/drone_1/state', self.state_callback, 10)

    def send_command(self, action):
        msg = String()
        data = {
            "drone_id": "drone_1",
            "action": action
        }
        msg.data = json.dumps(data)
        self.cmd_pub.publish(msg)
        self.get_logger().info(f'发送指令: {action}')

    def state_callback(self, msg):
        # 提取位置信息
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        # 把数据丢给 GUI 线程更新
        self.gui_callback(f"X: {x:8.2f} m | Y: {y:8.2f} m | Z: {z:8.2f} m")


# ==================== GUI 界面 ====================
class Topic3GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("课题三：战略指挥中枢 (Mock)")
        self.root.geometry("500x650")
        self.root.configure(bg="#2b2b2b") # 暗黑背景

        # 字体设置
        self.font_title = ("Arial", 16, "bold")
        self.font_btn = ("Arial", 12, "bold")

        # 标题
        tk.Label(self.root, text="UAV Tactical Command", fg="white", bg="#2b2b2b", font=self.font_title).pack(pady=15)

        # 按钮容器 (使用 Grid 布局)
        btn_frame = tk.Frame(self.root, bg="#2b2b2b")
        btn_frame.pack(pady=10)

        # 按钮样式配置
        btn_conf = {
            "width": 12, "height": 2, "font": self.font_btn, "bg": "#4a4a4a", "fg": "white",
            "activebackground": "#6a6a6a", "activeforeground": "white", "relief": "flat", "bd": 0
        }

        # 放置 6 个按钮 (2列 x 3行)
        self.create_btn(btn_frame, "TAKEOFF", 0, 0, btn_conf)
        self.create_btn(btn_frame, "HOVER", 0, 1, btn_conf)
        self.create_btn(btn_frame, "RETURN", 1, 0, btn_conf)
        self.create_btn(btn_frame, "TRACK", 1, 1, btn_conf)
        self.create_btn(btn_frame, "INTERCEPT", 2, 0, btn_conf)
        self.create_btn(btn_frame, "ATTACK", 2, 1, {**btn_conf, "bg": "#8b0000", "activebackground": "#b22222"}) # 攻击按钮变成红色

        # 数据显示区
        tk.Label(self.root, text="无人机实时位姿", fg="white", bg="#2b2b2b", font=("Arial", 12)).pack(pady=(20, 5))
        
        self.text_area = scrolledtext.ScrolledText(self.root, width=50, height=15, bg="#1e1e1e", fg="#00ff00", font=("Consolas", 12))
        self.text_area.pack(padx=20, pady=5)

        # ROS 2 节点初始化 (放在子线程)
        self.node = None
        threading.Thread(target=self.init_ros, daemon=True).start()

    def create_btn(self, frame, action, row, col, config):
        btn = tk.Button(frame, text=action, command=lambda: self.on_btn_click(action), **config)
        btn.grid(row=row, column=col, padx=10, pady=10)

    def on_btn_click(self, action):
        if self.node:
            self.node.send_command(action)

    def update_text(self, text):
        # 把状态信息打印到文本框最上面，并保留历史
        self.text_area.insert("1.0", f"{text}\n")
        # 限制最多显示 50 行，防止内存溢出
        if int(self.text_area.index('end-1c').split('.')[0]) > 50:
            self.text_area.delete("50.0", "end")

    def init_ros(self):
        rclpy.init()
        self.node = MockTopic3(self.update_text)
        rclpy.spin(self.node)


def main():
    root = tk.Tk()
    app = Topic3GUI(root)
    # 关闭窗口时优雅退出
    def on_closing():
        if app.node:
            app.node.destroy_node()
            rclpy.shutdown()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == '__main__':
    main()