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

        # 1. 发布战术指令
        self.cmd_pub = self.create_publisher(String, '/strategy/cmd', 10)
        
        # 2. 发送给假节点
        self.vision_inject_pub = self.create_publisher(PoseStamped, '/mock_vision_node/vision_inject', 10)
        self.planning_inject_pub = self.create_publisher(String, '/mock_plan_node/planning_inject', 10)
        
        # 3. ★ 关键修复：监听三架无人机
        self.swarm_config = ["drone_1", "drone_2", "drone_3"] 
        for drone_id in self.swarm_config:
            self.create_subscription(
                Odometry, 
                f'/{drone_id}/state', 
                lambda msg, d_id=drone_id: self.state_callback(msg, d_id), 
                10
            )

    def state_callback(self, msg, drone_id):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        
        # ★ 关键修复：必须推入 "pose_update"，且带上坐标元组
        self.gui_queue.put(("pose_update", (drone_id, x, y, z)))

    def send_command(self, action, target_drone_id):
        msg = String()
        data = {"drone_id": target_drone_id, "action": action} # ★ 使用传入的 ID
        msg.data = json.dumps(data)
        self.cmd_pub.publish(msg)
        self.get_logger().info(f'发送指令给 {target_drone_id}: {action}')

    def send_vision_target(self, x, y, z):
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        self.vision_inject_pub.publish(msg)

    def send_planning_target(self, x, y, z, target_drone_id):
        msg = String()
        data = {
            "drone_id": target_drone_id,  # ★ 必须把 ID 带上
            "x": x, 
            "y": y, 
            "z": z, 
            "action": "TRACK"
        }
        msg.data = json.dumps(data)
        self.planning_inject_pub.publish(msg)


# ==================== 主 GUI 界面 ====================
class GroundStationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("课题二通用联调界面")
        self.root.geometry("1920x1080") # 稍微加宽一点
        self.root.configure(bg="#2b2b2b")
        
        self.processes = {}
        self.log_queues = {}
        self.ui_queue = queue.Queue()
        
        # 扫描当前文件夹下所有的 py 文件供下拉框选择
        self.available_scripts = self.scan_scripts()
        
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
        left_frame = tk.Frame(self.root, bg="#2b2b2b", width=600)
        left_frame.pack(side="left", fill="y", padx=15, pady=15)
        left_frame.pack_propagate(False)

        tk.Label(left_frame, text="UAV Tactical Command", fg="white", bg="#2b2b2b", font=self.font_title).pack(pady=10)
        
        # 无人机选择
        target_frame = tk.Frame(left_frame, bg="#2b2b2b")
        target_frame.pack(pady=5)
        tk.Label(target_frame, text="当前控制目标:", fg="#a9a9a9", bg="#2b2b2b", font=("WenQuanYi Micro Hei", 10)).pack(side="left")
        
        self.target_drone_var = tk.StringVar(value="drone_1")
        target_menu = tk.OptionMenu(target_frame, self.target_drone_var, "drone_1", "drone_2", "drone_3")
        target_menu.config(bg="#4a4a4a", fg="white", width=10, relief="flat", highlightthickness=0)
        target_menu.pack(side="left", padx=5)

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

        # (视觉/规划注入区，你自己保留之前的代码即可)
                # ★ 将注入区插在按钮区和指令输出终端之间
        # ================= 视觉数据注入区 =================
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

        # ================= 规划数据注入区 =================
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

        # ================= 指令输出终端 =================
        tk.Label(left_frame, text="指令输出终端", fg="white", bg="#2b2b2b", font=self.font_title).pack(pady=(10, 5))
        self.text_state = scrolledtext.ScrolledText(left_frame, width=40, height=6, bg="#1e1e1e", fg="#00ff00", font=self.font_mono)
        self.text_state.pack(padx=5, pady=5, fill="both", expand=True)
        
        # ================= 左下角：三机实时位姿面板 =================
        tk.Label(left_frame, text="无人机实时位姿", fg="white", bg="#2b2b2b", font=self.font_title).pack(pady=(15, 5))
        
        pose_frame = tk.Frame(left_frame, bg="#1e1e1e", bd=1, relief="solid")
        pose_frame.pack(padx=10, fill="x")
        
        self.pose_labels = {}
        for i in range(1, 4):
            d_id = f"drone_{i}"
            # 使用 Label 来固定显示，字体用 Consolas
            lbl = tk.Label(pose_frame, text=f"[{d_id}] X: 0.00 m | Y: 0.00 m | Z: 0.00 m", 
                           fg="#00ff00", bg="#1e1e1e", font=self.font_mono, anchor="w")
            lbl.pack(fill="x", pady=2, padx=5)
            self.pose_labels[d_id] = lbl
       
        # ================= 右侧：多机监控大盘 (2x3 网格) =================
        right_frame = tk.Frame(self.root, bg="#2b2b2b")
        right_frame.pack(side="right", fill="both", expand=True, padx=(0, 15), pady=15)

        right_frame.grid_rowconfigure(0, weight=1, uniform="row")
        right_frame.grid_rowconfigure(1, weight=1, uniform="row")
        right_frame.grid_columnconfigure(0, weight=1, uniform="col")
        right_frame.grid_columnconfigure(1, weight=1, uniform="col")
        right_frame.grid_columnconfigure(2, weight=1, uniform="col")

        monitors = [
            {"id": "bridge_1", "title": "1号桥接", "default": "mock_mavsdk_bridge.py", "row": 0, "col": 0},
            {"id": "bridge_2", "title": "2号桥接", "default": "mock_mavsdk_bridge.py", "row": 0, "col": 1},
            {"id": "bridge_3", "title": "3号桥接", "default": "mock_mavsdk_bridge.py", "row": 0, "col": 2},
            {"id": "vision",   "title": "视觉模块", "default": "/home/europeanskyfab/ros2_ws/src/vision_pkg/launch/vision_node.launch.py", "row": 1, "col": 0},
            {"id": "planning", "title": "规划模块", "default": "mock_plan_node.py", "row": 1, "col": 1},
            {"id": "center",   "title": "中枢模块", "default": "/home/europeanskyfab/ros2_ws/src/center_node_pkg/launch/center_node.launch.py", "row": 1, "col": 2},
        ]

        self.monitor_vars = {}
        self.monitor_txts = {}
        self.processes = {}
        self.log_queues = {}

        for m in monitors:
            # 外层框架
            slot_frame = tk.Frame(right_frame, bg="#2b2b2b", highlightbackground="#444", highlightthickness=1)
            slot_frame.grid(row=m["row"], column=m["col"], sticky="nsew", padx=5, pady=5)
            slot_frame.grid_rowconfigure(1, weight=1) # 日志区权重
            slot_frame.grid_columnconfigure(0, weight=1)

            # 顶部控制栏 (保持不变)
            top_bar = tk.Frame(slot_frame, bg="#2b2b2b")
            top_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=2)
            
            tk.Label(top_bar, text=m["title"], fg="white", bg="#2b2b2b", font=("WenQuanYi Micro Hei", 10, "bold")).pack(side="left")

            # 按钮组
            btn_stop = tk.Button(top_bar, text="关闭", bg="#f41b1b", state="disabled", width=4, relief="flat",
                                 command=lambda k=m["id"]: self.stop_node(k))
            btn_stop.pack(side="right", padx=2)
            
            btn_start = tk.Button(top_bar, text="开启", bg="#a0e0a0", width=4, relief="flat",
                                  command=lambda k=m["id"]: self.start_node(k))
            btn_start.pack(side="right", padx=2)

            # 路径显示/选择
            var = tk.StringVar(value=m["default"])
            self.monitor_vars[m["id"]] = var
            
            entry_path = tk.Entry(top_bar, textvariable=var, width=12, state="readonly", bg="#4a4a4a", fg="white", font=("Consolas", 9))
            entry_path.pack(side="right", padx=2)
            
            btn_browse = tk.Button(top_bar, text="浏览", bg="#4a4a4a", fg="white", width=2, relief="flat",
                                   command=lambda v=var: self.browse_file(v))
            btn_browse.pack(side="right", padx=2)

            # ★ 核心改造：区分视觉模块和其他模块的布局
            if m["id"] == "vision":
                # 如果是视觉模块，把区域分成上下的 Frame
                content_frame = tk.Frame(slot_frame, bg="#2b2b2b")
                content_frame.grid(row=1, column=0, sticky="nsew")
                content_frame.grid_rowconfigure(0, weight=4) # 视觉日志占 4 份
                content_frame.grid_rowconfigure(1, weight=1) # 假相机按钮占 1 份
                content_frame.grid_columnconfigure(0, weight=1)

                # 1. 视觉日志框
                txt = scrolledtext.ScrolledText(content_frame, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9), state="disabled")
                txt.grid(row=0, column=0, sticky="nsew", padx=5, pady=(5, 0))
                
                # 2. 假相机控制区
                fake_cam_frame = tk.Frame(content_frame, bg="#1e1e1e")
                fake_cam_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
                
                tk.Label(fake_cam_frame, text="假相机注入器", fg="#888888", bg="#1e1e1e", font=("WenQuanYi Micro Hei", 9)).pack(side="left", padx=5)
                
                btn_fake_stop = tk.Button(fake_cam_frame, text="关闭", bg="#f41b1b", state="disabled", width=6, relief="flat",
                                          command=lambda: self.stop_node("fake_camera"))
                btn_fake_stop.pack(side="right", padx=5, pady=5)
                
                btn_fake_start = tk.Button(fake_cam_frame, text="开启", bg="#a0e0a0", width=6, relief="flat",
                                           command=lambda: self.start_fake_camera())
                btn_fake_start.pack(side="right", padx=5, pady=5)

                # 把假相机的按钮存到字典里，方便状态切换
                self.node_btns_start["fake_camera"] = btn_fake_start
                self.node_btns_stop["fake_camera"] = btn_fake_stop
                self.log_queues["fake_camera"] = queue.Queue() # 占位，防止退出时报错
                
                # ★ 核心修复：让假相机的日志也输出到视觉那个黑框里
                self.monitor_txts["fake_camera"] = txt

            else:
                # 其他模块保持原样，直接满屏日志
                txt = scrolledtext.ScrolledText(slot_frame, bg="#1e1e1e", fg="#d4d4d4", font=("Consolas", 9), state="disabled")
                txt.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

            self.monitor_txts[m["id"]] = txt
                
            # 保存按钮引用用于启停控制 (常规模块)
            if not hasattr(self, 'node_btns_start'): self.node_btns_start, self.node_btns_stop = {}, {}
            self.node_btns_start[m["id"]] = btn_start
            self.node_btns_stop[m["id"]] = btn_stop
            self.log_queues[m["id"]] = queue.Queue()

    def create_btn(self, frame, action, row, col, config):
        btn = tk.Button(frame, text=action, command=lambda: self.on_btn_click(action), **config)
        btn.grid(row=row, column=col, padx=5, pady=5)

    def on_btn_click(self, action):
        if hasattr(self, 'ros_node') and self.ros_node:
            # ★ 获取下拉框当前选中的无人机
            target_id = self.target_drone_var.get()
            self.ros_node.send_command(action, target_id)
            self.text_state.insert("1.0", f"Sent CMD to {target_id}: {action}\n")

    def inject_data(self, data_type):
        if not hasattr(self, 'ros_node') or not self.ros_node:
            return
        if data_type == 'vision':
            try:
                x = float(self.entry_vis_x.get())
                y = float(self.entry_vis_y.get())
                z = float(self.entry_vis_z.get())
                self.ros_node.send_vision_target(x, y, z)
                self.text_state.insert("1.0", f"注入视觉: {x}, {y}, {z}\n")
            except ValueError:
                self.text_state.insert("1.0", "视觉坐标输入错误！\n")
        elif data_type == 'planning':
            try:
                x = float(self.entry_plan_x.get())
                y = float(self.entry_plan_y.get())
                z = float(self.entry_plan_z.get())
                # ★ 获取当前下拉框选中的无人机
                target_id = self.target_drone_var.get()
                self.ros_node.send_planning_target(x, y, z, target_id) # 传给函数
                self.text_state.insert("1.0", f"注入规划给 {target_id}: {x}, {y}, {z}\n")
            except ValueError:
                self.text_state.insert("1.0", "规划坐标输入错误！\n")

    def init_ros(self):
        threading.Thread(target=self._run_ros, daemon=True).start()

    def _run_ros(self):
        rclpy.init()
        self.ros_node = MockTopic3(self.ui_queue)
        rclpy.spin(self.ros_node)

    # ---------------- 进程管理 ----------------
    def start_node(self, m_id):
        if m_id in self.processes: return
        script_name = self.monitor_vars[m_id].get()
        
        # 兼容绝对路径和相对路径
        if os.path.isabs(script_name):
            script_path = script_name
        else:
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), script_name)
            
        if not os.path.exists(script_path):
            self.append_log(m_id, f"❌ 找不到文件: {script_name}\n")
            return

        self.append_log(m_id, f"--- 启动 {script_name} ---\n")
        
        try:
            cmd = []
            # ★ 核心改造：判断是不是 launch 文件
            if script_name.endswith('.launch.py'):
                # 如果是 launch 文件，使用 ros2 launch 启动
                # 注意：ros2 launch 只需要包名和 launch 文件名，不需要绝对路径
                # 假设你的包都在 src 目录下，我们可以从路径中提取包名
                # 比如 /home/.../src/vision_pkg/launch/vision_node.launch.py
                parts = script_path.split(os.sep)
                try:
                    # 找到 'src' 后面那个文件夹，就是包名
                    pkg_name = parts[parts.index('src') + 1]
                    # 获取 launch 文件名
                    launch_file = os.path.basename(script_path)
                    cmd = ["ros2", "launch", pkg_name, launch_file]
                except ValueError:
                    self.append_log(m_id, "❌ 无法解析包名，请确保 launch 文件在 src/<pkg_name>/launch/ 目录下\n")
                    return
            else:
                # 如果是普通的 .py 脚本，还是用 python3 直接跑
                cmd = ["python3", script_path]
                # 加上之前的动态参数
                if m_id == "bridge_1":
                    cmd.extend(["--ros-args", "-p", "drone_id:=drone_1", "-p", "port:=50052", "-p", "mav_port:=14540"])
                elif m_id == "bridge_2":
                    cmd.extend(["--ros-args", "-p", "drone_id:=drone_2", "-p", "port:=50053", "-p", "mav_port:=14541"])
                elif m_id == "bridge_3":
                    cmd.extend(["--ros-args", "-p", "drone_id:=drone_3", "-p", "port:=50054", "-p", "mav_port:=14542"])

            # 启动进程
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=os.path.dirname(script_path))
            self.processes[m_id] = proc
            threading.Thread(target=self._read_output, args=(m_id, proc, self.log_queues[m_id]), daemon=True).start()
            
            self.node_btns_start[m_id].config(state="disabled")
            self.node_btns_stop[m_id].config(state="normal")
            
        except Exception as e:
            self.append_log(m_id, f"❌ 启动失败: {e}\n")

    def start_fake_camera(self):
        m_id = "fake_camera"
        if m_id in self.processes: return
        
        # 假设 fake_camera.py 跟 GUI 在同一个文件夹
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fake_camera.py")
        if not os.path.exists(script_path):
            # 如果在其他目录，你可以手动改一下相对路径
            self.append_log("vision", f"❌ 找不到假相机文件: {script_path}\n")
            return

        self.append_log("vision", "--- 启动假相机注入器 ---\n")
        try:
            # 不需要传参数，直接拉起
            cmd = ["python3", script_path]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=os.path.dirname(script_path))
            self.processes[m_id] = proc
            
            # 假相机没有日志输出需求，但我们还是用一个线程把它的输出吃掉，防止 pipe 阻塞
            threading.Thread(target=self._read_output, args=(m_id, proc, self.log_queues[m_id]), daemon=True).start()
            
            self.node_btns_start[m_id].config(state="disabled")
            self.node_btns_stop[m_id].config(state="normal")
        except Exception as e:
            self.append_log("vision", f"❌ 假相机启动失败: {e}\n")
    
    def stop_node(self, m_id):
        if m_id not in self.processes: return
        proc = self.processes.pop(m_id)
        self.append_log("vision", f"--- 正在关闭 {m_id} ---\n") # 日志统一打在视觉框里
        
        # 暴力尝试杀掉整个进程组
        try:
            proc.terminate()
        except:
            pass
            
        def force_kill(p, k):
            try: 
                p.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait() # 确保彻底死透
                self.log_queues[k].put("--- 强制击杀 ---\n")
            except Exception:
                pass
                
        threading.Thread(target=force_kill, args=(proc, m_id), daemon=True).start()
        
        self.node_btns_start[m_id].config(state="normal")
        self.node_btns_stop[m_id].config(state="disabled")

    def append_log(self, m_id, text):
        txt = self.monitor_txts[m_id]
        txt.config(state="normal")
        txt.insert(tk.END, text)
        txt.see(tk.END)
        txt.config(state="disabled")

    def _read_output(self, slot_name, proc, q):
        for line in iter(proc.stdout.readline, ''):
            q.put(line)
        proc.stdout.close()
        q.put(f"--- 进程退出 ---\n")

    def process_queues(self):
        # 1. 处理 ROS 传来的队列
        while not self.ui_queue.empty():
            msg_type, data = self.ui_queue.get()
            
            # ★ 关键修复：处理坐标更新
            if msg_type == "pose_update":
                drone_id, x, y, z = data
                if drone_id in self.pose_labels:
                    self.pose_labels[drone_id].config(
                        text=f"[{drone_id}] X: {x:6.2f} m | Y: {y:6.2f} m | Z: {z:6.2f} m"
                    )
        
        # 2. 处理右侧进程日志队列（保持你原来的不变）
        for m_id, q in self.log_queues.items():
            while not q.empty():
                line = q.get_nowait()
                self.append_log(m_id, line)
                if "进程退出" in line and m_id in self.processes:
                    self.processes.pop(m_id)
                    self.node_btns_start[m_id].config(state="normal")
                    self.node_btns_stop[m_id].config(state="disabled")

        # 循环调用，每 100ms 刷新一次
        self.root.after(100, self.process_queues)

    def on_close(self):
        print("正在关闭所有子进程...")
        for m_id, proc in self.processes.items():
            try:
                # 不要用 terminate 了，直接用 kill 强杀
                proc.kill()
            except:
                pass
                
        if hasattr(self, 'ros_node'):
            self.ros_node.destroy_node()
            rclpy.shutdown()
            
        self.root.destroy()
        # ★ 加上这一句，直接退出 Python 脚本，防止有线程卡住导致窗口关不掉
        import os
        os._exit(0) 
        
    def browse_file(self, var):
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="选择要运行的节点脚本",
            filetypes=[("Python Files", "*.py"), ("All Files", "*.*")]
        )
        if file_path:
            var.set(file_path)

if __name__ == '__main__':
    root = tk.Tk()
    app = GroundStationGUI(root)
    root.mainloop()
    
