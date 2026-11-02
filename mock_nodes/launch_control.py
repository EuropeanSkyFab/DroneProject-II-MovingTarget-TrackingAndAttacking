import rclpy
import json
import threading
import subprocess
import os
import queue
import time
import tkinter as tk
from tkinter import scrolledtext
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

# ==================== ROS 2 节点 ====================
class MockTopic3(Node):
    def __init__(self, gui_queue):
        super().__init__('mock_topic3')
        self.gui_queue = gui_queue

        # 1. 发布战术指令给中枢
        self.cmd_pub = self.create_publisher(String, '/strategy/cmd', 10)
        
        # 2. 发送给假节点（内部注入话题）
        # 注意：如果组员换成了真视觉，可能不认这个 inject 话题。
        # 但通过这个注入区，你可以直接测真节点是否正常。
        self.vision_inject_pub = self.create_publisher(PoseStamped, '/mock_vision_node/vision_inject', 10)
        self.planning_inject_pub = self.create_publisher(String, '/mock_plan_node/planning_inject', 10)
        
        # 3. 订阅无人机状态
        self.state_sub = self.create_subscription(
            Odometry, '/drone_1/state', self.state_callback, 10)

    def send_command(self, action):
        msg = String()
        data = {"drone_id": "drone_1", "action": action}
        msg.data = json.dumps(data)
        self.cmd_pub.publish(msg)
        self.get_logger().info(f'发送指令: {action}')

    def send_vision_target(self, x, y, z):
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        self.vision_inject_pub.publish(msg)

    def send_planning_target(self, x, y, z):
        msg = String()
        data = {"x": x, "y": y, "z": z, "action": "TRACK"}
        msg.data = json.dumps(data)
        self.planning_inject_pub.publish(msg)

    def state_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        self.gui_queue.put(("state_update", f"X: {x:8.2f} m | Y: {y:8.2f} m | Z: {z:8.2f} m"))

# ==================== 主 GUI 界面 ====================
class GroundStationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("课题二通用联调界面")
        self.root.geometry("1350x800") # 稍微加宽一点
        self.root.configure(bg="#2b2b2b")
        
        self.processes = {}
        self.log_queues = {}
        self.ui_queue = queue.Queue()
        
        # 扫描当前文件夹下所有的 py 文件供下拉框选择
        self.available_scripts = self.scan_scripts()
        
        # 3个测试插槽的默认配置
        self.slots_config = [
            {"name": "桥接", "default": "mock_mavsdk_bridge.py"},
            {"name": "规划", "default": "mock_plan_node.py"},
            {"name": "视觉", "default": "mock_vision_node.py"}
        ]
        
        self.font_title = ("WenQuanYi Micro Hei", 14, "bold")
        self.font_btn = ("WenQuanYi Micro Hei", 11, "bold")
        self.font_mono = ("Consolas", 11)

        self.setup_layout()
        self.init_ros()
        
        self.root.after(100, self.process_queues)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def scan_scripts(self):
        # 获取当前目录下所有的 py 文件，并排除自己
        current_dir = os.path.dirname(os.path.abspath(__file__))
        scripts = [f for f in os.listdir(current_dir) if f.endswith('.py') and f != os.path.basename(__file__)]
        scripts.sort()
        return scripts

    def setup_layout(self):
        # ================= 左侧：中控与数据注入 =================
        left_frame = tk.Frame(self.root, bg="#2b2b2b", width=700)
        left_frame.pack(side="left", fill="y", padx=15, pady=15)
        left_frame.pack_propagate(False)

        tk.Label(left_frame, text="UAV Tactical Command", fg="white", bg="#2b2b2b", font=self.font_title).pack(pady=10)
        btn_frame = tk.Frame(left_frame, bg="#2b2b2b")
        btn_frame.pack(pady=5)
        btn_conf = {"width": 12, "height": 2, "font": self.font_btn, "bg": "#4a4a4a", "fg": "white",
                    "activebackground": "#6a6a6a", "activeforeground": "white", "relief": "flat", "bd": 0}
        
        self.create_btn(btn_frame, "TAKEOFF", 0, 0, btn_conf)
        self.create_btn(btn_frame, "HOVER", 0, 1, btn_conf)
        self.create_btn(btn_frame, "RETURN", 1, 0, btn_conf)
        self.create_btn(btn_frame, "TRACK", 1, 1, btn_conf)
        self.create_btn(btn_frame, "INTERCEPT", 2, 0, btn_conf)
        self.create_btn(btn_frame, "ATTACK", 2, 1, {**btn_conf, "bg": "#8b0000", "activebackground": "#b22222"})

        # 视觉数据注入区
        inject_vis_frame = tk.LabelFrame(left_frame, text="视觉坐标注入 (ENU) ", fg="#87ceeb", bg="#2b2b2b", font=("WenQuanYi Micro Hei", 11))
        inject_vis_frame.pack(fill="x", pady=5, padx=10)
        vis_input_row = tk.Frame(inject_vis_frame, bg="#2b2b2b")
        vis_input_row.pack(pady=5, anchor="center")
        tk.Label(vis_input_row, text="X:", fg="white", bg="#2b2b2b").grid(row=0, column=0)
        self.entry_vis_x = tk.Entry(vis_input_row, width=6, font=self.font_mono)
        self.entry_vis_x.insert(0, "15.0")
        self.entry_vis_x.grid(row=0, column=1, padx=5)
        tk.Label(vis_input_row, text="Y:", fg="white", bg="#2b2b2b").grid(row=0, column=2)
        self.entry_vis_y = tk.Entry(vis_input_row, width=6, font=self.font_mono)
        self.entry_vis_y.insert(0, "15.0")
        self.entry_vis_y.grid(row=0, column=3, padx=5)
        tk.Label(vis_input_row, text="Z:", fg="white", bg="#2b2b2b").grid(row=0, column=4)
        self.entry_vis_z = tk.Entry(vis_input_row, width=6, font=self.font_mono)
        self.entry_vis_z.insert(0, "10.0")
        self.entry_vis_z.grid(row=0, column=5, padx=5)
        tk.Button(inject_vis_frame, text="发布视觉坐标", bg="#4682b4", fg="white", font=("WenQuanYi Micro Hei", 9),
                  command=lambda: self.inject_data('vision')).pack(pady=5)

        # 规划数据注入区
        inject_plan_frame = tk.LabelFrame(left_frame, text="规划坐标注入 (ENU) ", fg="#9370db", bg="#2b2b2b", font=("WenQuanYi Micro Hei", 11))
        inject_plan_frame.pack(fill="x", pady=5, padx=10)
        plan_input_row = tk.Frame(inject_plan_frame, bg="#2b2b2b")
        plan_input_row.pack(pady=5, anchor="center")
        tk.Label(plan_input_row, text="X:", fg="white", bg="#2b2b2b").grid(row=0, column=0)
        self.entry_plan_x = tk.Entry(plan_input_row, width=6, font=self.font_mono)
        self.entry_plan_x.insert(0, "20.0")
        self.entry_plan_x.grid(row=0, column=1, padx=5)
        tk.Label(plan_input_row, text="Y:", fg="white", bg="#2b2b2b").grid(row=0, column=2)
        self.entry_plan_y = tk.Entry(plan_input_row, width=6, font=self.font_mono)
        self.entry_plan_y.insert(0, "20.0")
        self.entry_plan_y.grid(row=0, column=3, padx=5)
        tk.Label(plan_input_row, text="Z:", fg="white", bg="#2b2b2b").grid(row=0, column=4)
        self.entry_plan_z = tk.Entry(plan_input_row, width=6, font=self.font_mono)
        self.entry_plan_z.insert(0, "10.0")
        self.entry_plan_z.grid(row=0, column=5, padx=5)
        tk.Button(inject_plan_frame, text="发布规划坐标", bg="#6a5acd", fg="white", font=("WenQuanYi Micro Hei", 9),
                  command=lambda: self.inject_data('planning')).pack(pady=5)

        # 状态反馈区
        tk.Label(left_frame, text="无人机实时位姿", fg="white", bg="#2b2b2b", font=self.font_title).pack(pady=(10, 5))
        self.text_state = scrolledtext.ScrolledText(left_frame, width=40, height=6, bg="#1e1e1e", fg="#00ff00", font=self.font_mono)
        self.text_state.pack(padx=5, pady=5, fill="both", expand=True)

        # ================= 右侧：动态插件管理区 =================
        right_frame = tk.Frame(self.root, bg="#2b2b2b")
        right_frame.pack(side="right", fill="both", expand=True, padx=(0, 15), pady=15)

        for i, slot in enumerate(self.slots_config):
            right_frame.grid_rowconfigure(i, weight=1, uniform="log_rows")
            right_frame.grid_columnconfigure(1, weight=1)
            
            row_frame = tk.Frame(right_frame, bg="#2b2b2b")
            row_frame.grid(row=i, column=0, columnspan=2, sticky="nsew", pady=5)
            row_frame.grid_rowconfigure(0, weight=1)
            row_frame.grid_columnconfigure(1, weight=1)
            
            # 控制区
            ctrl_frame = tk.Frame(row_frame, bg="#2b2b2b", width=180) # 稍微加宽放下拉框
            ctrl_frame.grid(row=0, column=0, sticky="nsew", padx=5)
            ctrl_frame.grid_propagate(False)
            
            tk.Label(ctrl_frame, text=f"Slot {i+1}: {slot['name']}", fg="#a9a9a9", bg="#2b2b2b", font=("WenQuanYi Micro Hei", 10, "bold")).pack(pady=(5,0))
            if i == 0:
                tk.Label(ctrl_frame, text="提示：若无PX4+Gazebo环境请切\n换为 mock_mavsdk_bridge_dry.py", fg="#ffa500", bg="#2b2b2b", font=("WenQuanYi Micro Hei", 9)).pack()
            
            # ★ 核心：下拉框动态选择脚本
            var = tk.StringVar(value=slot['default'])
            # 如果默认脚本不在列表里（被删了），就设为列表第一个
            if slot['default'] not in self.available_scripts and self.available_scripts:
                var.set(self.available_scripts[0])
            
            option_menu = tk.OptionMenu(ctrl_frame, var, *self.available_scripts)
            option_menu.config(width=20, bg="#4a4a4a", fg="white", highlightthickness=0)
            option_menu["menu"].config(bg="#4a4a4a", fg="white")
            option_menu.pack(pady=2)
            
            btn_row = tk.Frame(ctrl_frame, bg="#2b2b2b")
            btn_row.pack(pady=2)
            
            btn_start = tk.Button(btn_row, text="开启", bg="#a0e0a0", width=8,
                                  command=lambda n=f"Slot_{i}", v=var: self.start_node(n, v.get()))
            btn_start.pack(side="left", padx=2)
            
            btn_stop = tk.Button(btn_row, text="关闭", bg="#EE1B1B", state="disabled", width=8,
                                 command=lambda n=f"Slot_{i}": self.stop_node(n))
            btn_stop.pack(side="left", padx=2)

            # 日志区
            txt = scrolledtext.ScrolledText(row_frame, bg="#1e1e1e", fg="#d4d4d4", font=self.font_mono, state="disabled")
            txt.grid(row=0, column=1, sticky="nsew", padx=5)
            
            if not hasattr(self, 'node_btns_start'): self.node_btns_start, self.node_btns_stop, self.node_log_txts = {}, {}, {}
            self.node_btns_start[f"Slot_{i}"] = btn_start
            self.node_btns_stop[f"Slot_{i}"] = btn_stop
            self.node_log_txts[f"Slot_{i}"] = txt
            self.log_queues[f"Slot_{i}"] = queue.Queue()

    def create_btn(self, frame, action, row, col, config):
        btn = tk.Button(frame, text=action, command=lambda: self.on_btn_click(action), **config)
        btn.grid(row=row, column=col, padx=5, pady=5)

    def on_btn_click(self, action):
        if hasattr(self, 'ros_node') and self.ros_node:
            self.ros_node.send_command(action)
            self.text_state.insert("1.0", f"Sent CMD: {action}\n")

    def inject_data(self, data_type):
        if not hasattr(self, 'ros_node') or not self.ros_node:
            return
        if data_type == 'vision':
            try:
                x = float(self.entry_vis_x.get())
                y = float(self.entry_vis_y.get())
                z = float(self.entry_vis_z.get())
                self.ros_node.send_vision_target(x, y, z)
                self.text_state.insert("1.0", f"👁️ 注入视觉: {x}, {y}, {z}\n")
            except ValueError:
                self.text_state.insert("1.0", "⚠️ 视觉坐标输入错误！\n")
        elif data_type == 'planning':
            try:
                x = float(self.entry_plan_x.get())
                y = float(self.entry_plan_y.get())
                z = float(self.entry_plan_z.get())
                self.ros_node.send_planning_target(x, y, z)
                self.text_state.insert("1.0", f"🧠 注入规划: {x}, {y}, {z}\n")
            except ValueError:
                self.text_state.insert("1.0", "⚠️ 规划坐标输入错误！\n")

    def init_ros(self):
        threading.Thread(target=self._run_ros, daemon=True).start()

    def _run_ros(self):
        rclpy.init()
        self.ros_node = MockTopic3(self.ui_queue)
        rclpy.spin(self.ros_node)

    # ---------------- 进程管理 ----------------
    def start_node(self, slot_name, script_name):
        if slot_name in self.processes: return
        
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
        if not os.path.exists(script_path):
            self.append_log(slot_name, f"❌ 找不到文件: {script_name}\n")
            return

        self.append_log(slot_name, f"--- 启动 {script_name} ---\n")
        try:
            proc = subprocess.Popen(["python3", script_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=os.path.dirname(script_path))
            self.processes[slot_name] = proc
            threading.Thread(target=self._read_output, args=(slot_name, proc, self.log_queues[slot_name]), daemon=True).start()
            
            self.node_btns_start[slot_name].config(state="disabled")
            self.node_btns_stop[slot_name].config(state="normal")
        except Exception as e:
            self.append_log(slot_name, f"❌ 启动失败: {e}\n")

    def _read_output(self, slot_name, proc, q):
        for line in iter(proc.stdout.readline, ''):
            q.put(line)
        proc.stdout.close()
        q.put(f"--- 进程退出 ---\n")

    def stop_node(self, slot_name):
        if slot_name not in self.processes: return
        proc = self.processes.pop(slot_name)
        self.append_log(slot_name, "--- 正在关闭... ---\n")
        proc.terminate()
        def force_kill(p, n):
            try: p.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                p.kill()
                self.log_queues[n].put("--- 强制击杀 ---\n")
        threading.Thread(target=force_kill, args=(proc, slot_name), daemon=True).start()
        self.node_btns_start[slot_name].config(state="normal")
        self.node_btns_stop[slot_name].config(state="disabled")

    def append_log(self, slot_name, text):
        txt = self.node_log_txts[slot_name]
        txt.config(state="normal")
        txt.insert(tk.END, text)
        txt.see(tk.END)
        txt.config(state="disabled")

    def process_queues(self):
        while not self.ui_queue.empty():
            msg_type, data = self.ui_queue.get()
            if msg_type == "state_update":
                self.text_state.insert("1.0", f"{data}\n")
                if int(self.text_state.index('end-1c').split('.')[0]) > 50:
                    self.text_state.delete("50.0", "end")
        
        for slot_name, q in self.log_queues.items():
            while not q.empty():
                line = q.get_nowait()
                self.append_log(slot_name, line)
                if "进程退出" in line and slot_name in self.processes:
                    self.processes.pop(slot_name)
                    self.node_btns_start[slot_name].config(state="normal")
                    self.node_btns_stop[slot_name].config(state="disabled")

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
    
