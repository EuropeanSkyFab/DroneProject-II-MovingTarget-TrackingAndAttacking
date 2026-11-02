import rclpy
import json
import threading
import subprocess
import os
import queue
import tkinter as tk
from tkinter import scrolledtext
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry


class MockTopic3(Node):
    def __init__(self, gui_queue):
        super().__init__('mock_topic3')
        self.gui_queue = gui_queue  # 用于跨线程发送 UI 更新请求

        # 1. 发布战术指令
        self.cmd_pub = self.create_publisher(String, '/strategy/cmd', 10)
        
        # 2. 订阅无人机状态
        self.state_sub = self.create_subscription(
            Odometry, '/drone_1/state', self.state_callback, 10)

    def send_command(self, action):
        msg = String()
        data = {"drone_id": "drone_1", "action": action}
        msg.data = json.dumps(data)
        self.cmd_pub.publish(msg)
        self.get_logger().info(f'发送指令: {action}')

    def state_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        # 把数据扔进队列，让主线程去更新 UI
        self.gui_queue.put(("state_update", f"X: {x:8.2f} m | Y: {y:8.2f} m | Z: {z:8.2f} m"))


# ==================== 主 GUI 界面 ====================
class GroundStationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("课题二/三 联合测试指挥中心")
        self.root.geometry("1100x700")
        self.root.configure(bg="#2b2b2b")
        
        # 状态数据
        self.processes = {}
        self.log_queues = {}
        self.ui_queue = queue.Queue()  # 全局 UI 更新队列
        
        self.nodes_config = [
            ("桥接 (Bridge)", "mock_mavsdk_bridge.py"),
            ("规划 (Planning)", "mock_plan_node.py"),
            ("视觉 (Vision)", "mock_vision_node.py")
        ]
        
        # 字体
        self.font_title = ("WenQuanYi Micro Hei", 14, "bold")
        self.font_btn = ("WenQuanYi Micro Hei", 11, "bold")
        self.font_mono = ("Consolas", 11)

        self.setup_layout()
        self.init_ros()
        
        # 启动 UI 轮询定时器（处理日志和状态更新）
        self.root.after(100, self.process_queues)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_layout(self):
        # ================= 左侧：中控 =================
        left_frame = tk.Frame(self.root, bg="#2b2b2b", width=520)
        left_frame.pack(side="left", fill="y", padx=15, pady=15)
        left_frame.pack_propagate(False) # 固定宽度

        tk.Label(left_frame, text="UAV Tactical Command", fg="white", bg="#2b2b2b", font=self.font_title).pack(pady=15)

        btn_frame = tk.Frame(left_frame, bg="#2b2b2b")
        btn_frame.pack(pady=10)

        btn_conf = {"width": 12, "height": 2, "font": self.font_btn, "bg": "#4a4a4a", "fg": "white",
                    "activebackground": "#6a6a6a", "activeforeground": "white", "relief": "flat", "bd": 0}
        
        self.create_btn(btn_frame, "TAKEOFF", 0, 0, btn_conf)
        self.create_btn(btn_frame, "HOVER", 0, 1, btn_conf)
        self.create_btn(btn_frame, "RETURN", 1, 0, btn_conf)
        self.create_btn(btn_frame, "TRACK", 1, 1, btn_conf)
        self.create_btn(btn_frame, "INTERCEPT", 2, 0, btn_conf)
        self.create_btn(btn_frame, "ATTACK", 2, 1, {**btn_conf, "bg": "#8b0000", "activebackground": "#b22222"})

        tk.Label(left_frame, text="无人机实时位姿", fg="white", bg="#2b2b2b", font=self.font_title).pack(pady=(25, 5))
        
        self.text_state = scrolledtext.ScrolledText(left_frame, width=40, height=12, bg="#1e1e1e", fg="#00ff00", font=self.font_mono)
        self.text_state.pack(padx=5, pady=5, fill="both", expand=True)

        # ================= 右侧：进程控制与日志 =================
        right_frame = tk.Frame(self.root, bg="#2b2b2b")
        right_frame.pack(side="right", fill="both", expand=True, padx=(0, 15), pady=15)

        for i, (name, script) in enumerate(self.nodes_config):
            # 1. 配置右侧 Frame 的网格权重，让三行平分垂直空间
            right_frame.grid_rowconfigure(i, weight=1)
            right_frame.grid_columnconfigure(1, weight=1)
            
            # 每行一个水平框架（这次用 grid 放进 right_frame）
            row_frame = tk.Frame(right_frame, bg="#2b2b2b")
            row_frame.grid(row=i, column=0, columnspan=2, sticky="nsew", pady=5)
            
            # 给 row_frame 内部也设置权重，让日志框填满
            row_frame.grid_rowconfigure(0, weight=1)
            row_frame.grid_columnconfigure(1, weight=1)
            
            # 控制按钮区
            ctrl_frame = tk.Frame(row_frame, bg="#2b2b2b", width=120)
            ctrl_frame.grid(row=0, column=0, sticky="ns", padx=5)
            ctrl_frame.grid_propagate(False) # 固定宽度
            
            tk.Label(ctrl_frame, text=name, fg="white", bg="#2b2b2b", font=("WenQuanYi Micro Hei", 10, "bold")).pack(pady=2)
            
            btn_start = tk.Button(ctrl_frame, text="开启", bg="#a0e0a0", font=("WenQuanYi Micro Hei", 9),
                                  command=lambda n=name, s=script: self.start_node(n, s))
            btn_start.pack(pady=2, fill="x")
            
            btn_stop = tk.Button(ctrl_frame, text="关闭", bg="#f0a0a0", state="disabled", font=("WenQuanYi Micro Hei", 9),
                                 command=lambda n=name: self.stop_node(n))
            btn_stop.pack(pady=2, fill="x")

            # 日志框
            txt = scrolledtext.ScrolledText(row_frame, bg="#1e1e1e", fg="#d4d4d4", font=self.font_mono, state="disabled")
            txt.grid(row=0, column=1, sticky="nsew", padx=5)
            
            # 存储组件引用
            if not hasattr(self, 'node_btns_start'): self.node_btns_start, self.node_btns_stop, self.node_log_txts = {}, {}, {}
            self.node_btns_start[name] = btn_start
            self.node_btns_stop[name] = btn_stop
            self.node_log_txts[name] = txt
            self.log_queues[name] = queue.Queue()

    def create_btn(self, frame, action, row, col, config):
        btn = tk.Button(frame, text=action, command=lambda: self.on_btn_click(action), **config)
        btn.grid(row=row, column=col, padx=10, pady=10)

    def on_btn_click(self, action):
        if hasattr(self, 'ros_node') and self.ros_node:
            self.ros_node.send_command(action)
            # 在左侧状态框提示一下发了指令
            self.text_state.insert("1.0", f"Sent: {action}\n")

    # ---------------- ROS 2 部分 ----------------
    def init_ros(self):
        threading.Thread(target=self._run_ros, daemon=True).start()

    def _run_ros(self):
        rclpy.init()
        self.ros_node = MockTopic3(self.ui_queue)
        rclpy.spin(self.ros_node)

    # ---------------- 进程管理部分 ----------------
    def start_node(self, name, script_name):
        if name in self.processes: return
        
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
        if not os.path.exists(script_path):
            self.append_log(name, f"❌ 找不到文件: {script_name}\n")
            return

        self.append_log(name, f"--- 启动 {script_name} ---\n")
        try:
            proc = subprocess.Popen(["python3", script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=os.path.dirname(script_path))
            self.processes[name] = proc
            threading.Thread(target=self._read_output, args=(name, proc, self.log_queues[name]), daemon=True).start()
            
            self.node_btns_start[name].config(state="disabled")
            self.node_btns_stop[name].config(state="normal")
        except Exception as e:
            self.append_log(name, f"❌ 启动失败: {e}\n")

    def _read_output(self, name, proc, q):
        for line in iter(proc.stdout.readline, ''):
            q.put(line)
        proc.stdout.close()
        q.put(f"--- 进程退出 ---\n")

    def stop_node(self, name):
        if name not in self.processes: return
        proc = self.processes.pop(name)
        self.append_log(name, "--- 正在关闭... ---\n")
        proc.terminate()
        def force_kill(p, n):
            try: p.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                p.kill()
                self.log_queues[n].put("--- 强制击杀 ---\n")
        threading.Thread(target=force_kill, args=(proc, name), daemon=True).start()
        self.node_btns_start[name].config(state="normal")
        self.node_btns_stop[name].config(state="disabled")

    def append_log(self, name, text):
        txt = self.node_log_txts[name]
        txt.config(state="normal")
        txt.insert(tk.END, text)
        txt.see(tk.END)
        txt.config(state="disabled")

    # ---------------- 队列轮询 (统一更新 UI) ----------------
    def process_queues(self):
        # 1. 处理 ROS 状态队列
        while not self.ui_queue.empty():
            msg_type, data = self.ui_queue.get()
            if msg_type == "state_update":
                self.text_state.insert("1.0", f"{data}\n")
                # 限制行数
                if int(self.text_state.index('end-1c').split('.')[0]) > 50:
                    self.text_state.delete("50.0", "end")
        
        # 2. 处理进程日志队列
        for name, q in self.log_queues.items():
            while not q.empty():
                line = q.get_nowait()
                self.append_log(name, line)
                if "进程退出" in line and name in self.processes:
                    self.processes.pop(name)
                    self.node_btns_start[name].config(state="normal")
                    self.node_btns_stop[name].config(state="disabled")

        self.root.after(100, self.process_queues)

    def on_close(self):
        for name in list(self.processes.keys()):
            self.processes[name].terminate()
        if hasattr(self, 'ros_node'):
            self.ros_node.destroy_node()
            rclpy.shutdown()
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = GroundStationGUI(root)
    root.mainloop()