#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三机 PX4 SITL 启动器（GUI + CLI 双模式）

界面：三列日志窗，一列对应一架无人机，完整显示该机的 PX4 输出。
顶部按钮：启动 / 停止 / 清空日志 / 环境自检；状态栏显示 Agent、Gazebo 实例数。

在你原本的三条命令基础上补齐三件事（否则桥接连不上）：
  1. 自动拉起 MicroXRCEAgent（udp 8888）。没有它 ROS 2 里没有任何 /fmu 话题。
  2. 2/3 号机设置 PX4_UXRCE_DDS_NS=px4_1 / px4_2，与桥接的 px4_ns 一一对应。
  3. 等 1 号机把 Gazebo 世界拉起来，再启动 2/3 号机。

用法：
  python3 start_swarm_sim.py                 # 图形界面
  python3 start_swarm_sim.py --cli           # 纯终端模式（无显示器/远程时用）
  python3 start_swarm_sim.py --check         # 只做环境自检，不启动
  python3 start_swarm_sim.py --drones 1      # 只起一架
  python3 start_swarm_sim.py --no-agent      # Agent 已手动起过
"""
import argparse
import os
import queue
import signal
import subprocess
import sys
import threading
import time

# ==================== 通用工具 ====================

def pgrep(pattern):
    try:
        out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        return {int(x) for x in out.stdout.split() if x.strip().isdigit()}
    except Exception:
        return set()


def find_agent_bin():
    for cand in (os.path.expanduser("~/ros2_ws/install/microxrcedds_agent/bin/MicroXRCEAgent"),
                 "/usr/local/bin/MicroXRCEAgent",
                 "/usr/bin/MicroXRCEAgent"):
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    found = subprocess.run(["bash", "-lc", "command -v MicroXRCEAgent"],
                           capture_output=True, text=True).stdout.strip()
    return found or None


def load_px4_params(path):
    """解析 px4-rc.params，转成 PX4_PARAM_* 环境变量字典。

    PX4 v1.17+ 的 rcS 不再自动加载参数文件，改为在启动时读取
    PX4_PARAM_{name}={value} 环境变量（见 rcS:247-252）。
    其中最关键的是 NAV_DLL_ACT=0 —— 机身默认是 2（失去数据链触发动作），
    纯 ROS 2 无地面站时会让预检永远报 "No connection to the GCS"。
    """
    env = {}
    if not path or not os.path.isfile(path):
        return env
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 4 and parts[0] == "param" and parts[1] in ("set", "set-default"):
                env[f"PX4_PARAM_{parts[2]}"] = " ".join(parts[3:])
    return env


class Config:
    def __init__(self, drones, px4_dir, log_dir, agent_port, use_agent, use_ns, pose_step,
                 params_file=None, standalone=False, gz_world="default", gz_world_file=None,
                 gz_keep_display=False, init_pose_z=0.5, model_uri="x500"):
        self.drones = drones
        self.px4_dir = os.path.abspath(os.path.expanduser(px4_dir))
        self.log_dir = os.path.abspath(os.path.expanduser(log_dir))
        self.agent_port = agent_port
        self.use_agent = use_agent
        self.use_ns = use_ns
        self.pose_step = pose_step
        self.params_file = os.path.abspath(os.path.expanduser(params_file)) if params_file else None
        self.param_env = load_px4_params(self.params_file) if self.params_file else {}
        # ---- standalone 模式 ----
        self.standalone = standalone
        self.gz_world = gz_world
        self.gz_world_file = (os.path.abspath(os.path.expanduser(gz_world_file))
                              if gz_world_file else None)
        self.gz_keep_display = gz_keep_display
        self.init_pose_z = init_pose_z
        self.model_uri = model_uri

    @property
    def px4_bin(self):
        return os.path.join(self.px4_dir, "build/px4_sitl_default/bin/px4")


# ==================== 受管子进程 ====================

class Managed:
    """带日志文件、逐行回调的受管子进程。"""

    def __init__(self, name, cmd, env, cwd, log_path, emit, quiet=False):
        self.name = name
        self.emit = emit
        self.quiet = quiet
        self.log_path = log_path
        self._logf = open(log_path, "w", encoding="utf-8", buffering=1)
        self._logf.write(f"# cwd={cwd}\n# cmd={' '.join(cmd)}\n")
        self.proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, start_new_session=True,
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            for line in iter(self.proc.stdout.readline, ""):
                self._logf.write(line)
                if not self.quiet:
                    self.emit("log", (self.name, line.rstrip("\n")))
        except Exception:
            pass
        finally:
            try:
                self.proc.stdout.close()
            except Exception:
                pass
            self.emit("log", (self.name, f"--- 进程退出（code={self.proc.poll()}）---"))

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self, timeout=8):
        if not self.alive():
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                os.killpg(os.getpgid(self.proc.pid), sig)
            except Exception:
                pass
            try:
                self.proc.wait(timeout=timeout / 2)
                break
            except subprocess.TimeoutExpired:
                continue
        else:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
        try:
            self._logf.close()
        except Exception:
            pass


# ==================== 启动器核心（与界面无关） ====================

class SwarmRunner:
    def __init__(self, cfg, emit):
        self.cfg = cfg
        self.emit = emit
        self.children = []
        self.agent_proc = None
        self.gz_proc = None
        self.our_gz = set()
        self.gz_before = set()
        self._stop_flag = threading.Event()
        self._started = False

    # ---------- 自检 ----------
    def preflight(self):
        c = self.cfg
        problems = []
        if not (os.path.isfile(c.px4_bin) and os.access(c.px4_bin, os.X_OK)):
            problems.append(f"找不到可执行的 PX4: {c.px4_bin}")
        if c.use_agent and find_agent_bin() is None:
            problems.append("找不到 MicroXRCEAgent（ROS 2 里不会有 /fmu 话题）")
        if not c.params_file or not os.path.isfile(c.params_file):
            problems.append(f"找不到 PX4 参数文件: {c.params_file}"
                            f"（缺它的话 NAV_DLL_ACT 保持默认 2，预检会一直报 "
                            f"'No connection to the GCS'）")
        elif not c.param_env:
            problems.append(f"参数文件解析为空: {c.params_file}")
        if pgrep("px4_sitl_default/bin/px4"):
            problems.append("已有 PX4 在跑，实例号会冲突，请先 pkill -f 'px4_sitl_default/bin/px4'")
        if pgrep("gz sim"):
            problems.append("已有 Gazebo 在跑，模型可能被建到旧世界，请先 pkill -f 'gz sim'")
        return problems

    # ---------- 启动 ----------
    def start(self):
        c = self.cfg
        if self._started:
            return
        self._started = True
        self.gz_before = pgrep("gz sim")
        os.makedirs(c.log_dir, exist_ok=True)

        # 1) Agent
        if c.use_agent:
            running = pgrep("MicroXRCEAgent")
            if running:
                self.emit("status", f"Agent 已在运行（PID {sorted(running)}），复用")
            else:
                agent_bin = find_agent_bin()
                if agent_bin is None:
                    self.emit("status", "✘ 找不到 MicroXRCEAgent")
                    self._started = False
                    return
                self.agent_proc = subprocess.Popen(
                    [agent_bin, "udp4", "-p", str(c.agent_port)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                time.sleep(1.5)
                if self.agent_proc.poll() is not None:
                    self.emit("status", "✘ MicroXRCEAgent 启动即退出（端口被占用？）")
                    self.agent_proc = None
                    self._started = False
                    return
                self.emit("status", f"✔ Agent 已启动（udp {c.agent_port}）")

        # 2) 启动（两种模式）
        if c.param_env:
            self.emit("status", f"注入 {len(c.param_env)} 个 PX4 参数"
                                f"（NAV_DLL_ACT={c.param_env.get('PX4_PARAM_NAV_DLL_ACT', '未设置')}）")
        else:
            self.emit("status", "⚠ 未注入任何 PX4 参数，预检可能因 GCS 未连接而失败")

        if c.standalone:
            self._launch_standalone()
        else:
            self._launch_native()

        # 3) 运行期自检
        self._post_check()

    # ---------- 启动细节 ----------
    def _drone_env(self, i):
        """每架机共用的环境：参数注入 + 机型 + 命名空间。"""
        c = self.cfg
        env = os.environ.copy()
        env.update(c.param_env)              # ★ PX4_PARAM_* 注入
        env["PX4_SYS_AUTOSTART"] = "4001"
        env["PX4_GZ_MODEL"] = c.model_uri
        if i > 0 and c.use_ns:
            env["PX4_UXRCE_DDS_NS"] = f"px4_{i}"
        return env

    def _spawn_px4(self, i, env):
        c = self.cfg
        name = f"drone_{i + 1}"
        extra = ""
        if "PX4_GZ_MODEL_NAME" in env:
            extra += f"  MODEL={env['PX4_GZ_MODEL_NAME']}"
        if "PX4_GZ_MODEL_POSE" in env:
            extra += f"  POSE={env['PX4_GZ_MODEL_POSE']}"
        self.emit("log", (name, f"# 启动: -i {i}  NS={env.get('PX4_UXRCE_DDS_NS', '(root)')}{extra}"))
        try:
            m = Managed(name, [c.px4_bin, "-i", str(i)], env, c.px4_dir,
                        os.path.join(c.log_dir, f"px4_{name}.log"), self.emit)
        except Exception as e:
            self.emit("status", f"✘ {name} 启动失败：{e}")
            return None
        self.children.append(m)
        self.emit("state", {m.name: m.proc.pid})
        return m

    def _launch_native(self):
        """原生模式：1 号机的 PX4 自己拉起 Gazebo，2/3 号机靠 POSE 追加模型。

        风险：多个 PX4 各自拉 gz 时会形成多个同名世界（模型被建到旧世界）。
        本模式只做监测告警，不干预；要确定性请用 --standalone。
        """
        c = self.cfg
        for i in range(c.drones):
            if self._stop_flag.is_set():
                break
            env = self._drone_env(i)
            if i > 0:
                env["PX4_GZ_MODEL_POSE"] = f"0,{c.pose_step * i:g}"
            if self._spawn_px4(i, env) is None:
                break

            if i == 0:
                self.emit("status", "等待 Gazebo 世界启动…")
                t0 = time.time()
                ok = False
                while time.time() - t0 < 45 and not self._stop_flag.is_set():
                    if pgrep("gz sim"):
                        ok = True
                        break
                    time.sleep(1)
                if ok:
                    self.our_gz |= (pgrep("gz sim") - self.gz_before)
                    self.emit("status", f"✔ Gazebo 已就绪（{time.time() - t0:.1f}s）")
                else:
                    self.emit("status", "⚠ 未检测到 Gazebo，后续机型可能失败")
                time.sleep(2)

    def _gz_env(self):
        c = self.cfg
        env = os.environ.copy()
        models_dir = os.path.join(c.px4_dir, "Tools/simulation/gz/models")
        worlds_dir = os.path.join(c.px4_dir, "Tools/simulation/gz/worlds")
        old = env.get("GZ_SIM_RESOURCE_PATH", "")
        env["GZ_SIM_RESOURCE_PATH"] = ":".join([p for p in (models_dir, worlds_dir, old) if p])
        if not c.gz_keep_display:
            env.pop("DISPLAY", None)          # 与学长脚本一致：无头
        return env

    def _start_gz(self):
        """standalone：由启动器自己起唯一的 Gazebo 世界。"""
        c = self.cfg
        world_file = c.gz_world_file or os.path.join(
            c.px4_dir, "Tools/simulation/gz/worlds", f"{c.gz_world}.sdf")
        if not os.path.isfile(world_file):
            self.emit("status", f"✘ 找不到世界文件：{world_file}")
            return False

        env = self._gz_env()
        self.emit("status", f"启动 Gazebo（standalone）：{os.path.basename(world_file)}")
        self.gz_proc = Managed("gazebo", ["gz", "sim", "-r", "-s", world_file], env,
                               c.px4_dir, os.path.join(c.log_dir, "gazebo.log"),
                               self.emit, quiet=True)
        self.our_gz = {self.gz_proc.proc.pid}

        # 等 /world/<name>/clock 出现，而不是干等固定时长
        t0 = time.time()
        while time.time() - t0 < 60 and not self._stop_flag.is_set():
            r = subprocess.run(["gz", "topic", "-l"], capture_output=True, text=True, env=env)
            if f"/world/{c.gz_world}/clock" in (r.stdout or ""):
                self.emit("status", f"✔ Gazebo 就绪（{time.time() - t0:.1f}s）")
                return True
            time.sleep(1)
        self.emit("status", "⚠ Gazebo 60s 内未就绪，仍继续尝试（看 logs/gazebo.log）")
        return True

    def _create_model(self, i, pose):
        """在 Gazebo 世界里生成第 i 个机型，名字必须是 x500_<i>。"""
        c = self.cfg
        name = f"{c.model_uri}_{i}"
        req = ('name: "%s", allow_renaming: false, sdf: "<sdf version=\'1.9\'>'
               '<include><uri>model://%s</uri><pose>%s</pose></include></sdf>"'
               % (name, c.model_uri, pose))
        r = subprocess.run(
            ["gz", "service", "-s", f"/world/{c.gz_world}/create",
             "--reqtype", "gz.msgs.EntityFactory", "--reptype", "gz.msgs.Boolean",
             "--timeout", "5000", "--req", req],
            capture_output=True, text=True, env=self._gz_env())
        out = (r.stdout or "") + (r.stderr or "")
        ok = "data: true" in out
        self.emit("log", (f"drone_{i + 1}",
                          f"# 生成模型 {name} @({pose}) -> " + ("成功" if ok else f"失败 {out.strip()[:100]}")))
        return ok

    def _launch_standalone(self):
        """确定性模式：一个 Gazebo + 三个模型建在同一世界 + 三架 PX4 只做桥接。

        初始坐标与原生模式保持一致：x=0，y = pose_step*i（默认 0/2/4），z = init_pose_z（默认 0.5）。
        """
        c = self.cfg
        if not self._start_gz():
            return
        world_ok = True
        for i in range(c.drones):
            if self._stop_flag.is_set() or not world_ok:
                break
            pose = f"{0:g} {c.pose_step * i:g} {c.init_pose_z:g} 0 0 0"
            if not self._create_model(i, pose):
                self.emit("status", f"✘ 模型 {c.model_uri}_{i} 创建失败，停止后续启动")
                break
            time.sleep(1.5)

            env = self._drone_env(i)
            env["PX4_GZ_STANDALONE"] = "1"
            env["PX4_GZ_MODEL_NAME"] = f"{c.model_uri}_{i}"
            env["PX4_GZ_WORLD"] = c.gz_world
            if self._spawn_px4(i, env) is None:
                break
            time.sleep(2)

    def _post_check(self):
        c = self.cfg
        time.sleep(2)
        self.our_gz |= (pgrep("gz sim") - self.gz_before)
        gz_now = pgrep("gz sim")
        alive = len([m for m in self.children if m.alive()])
        if len(gz_now) > 1:
            self.emit("status", f"⚠ 检测到 {len(gz_now)} 个 Gazebo 实例，可能建到旧世界！")
        else:
            self.emit("status", f"运行中：{alive} 架，Gazebo {len(gz_now)} 个"
                                + ("（standalone）" if c.standalone else "（原生）"))

    # ---------- 停止 ----------
    def stop(self):
        self._stop_flag.set()
        if not self.children and self.agent_proc is None:
            return
        self.emit("status", "正在停止无人机…")
        for m in reversed(self.children):
            m.stop()
        if self.our_gz:
            self.emit("status", f"结束本脚本拉起的 Gazebo {sorted(self.our_gz)}")
            for pid in self.our_gz:
                try:
                    os.kill(pid, signal.SIGINT)
                except Exception:
                    pass
            time.sleep(2)
            for pid in self.our_gz:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
        if self.agent_proc is not None:
            self.emit("status", "停止 Agent…")
            try:
                os.killpg(os.getpgid(self.agent_proc.pid), signal.SIGINT)
                self.agent_proc.wait(timeout=3)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.agent_proc.pid), signal.SIGKILL)
                except Exception:
                    pass
            self.agent_proc = None
        self.children = []
        self._started = False
        self.emit("status", "✔ 已停止")


# ==================== 图形界面 ====================

class SimGUI:
    BG = "#2b2b2b"
    FG = "#dddddd"
    LOG_BG = "#1a1a1a"
    ACCENT = "#a0e0a0"
    DANGER = "#f41b1b"

    def __init__(self, root, cfg):
        import tkinter as tk
        from tkinter import scrolledtext
        self.tk = tk
        self.scrolledtext = scrolledtext
        self.root = root
        self.cfg = cfg
        self.q = queue.Queue()
        self.panes = {}
        self.runner = SwarmRunner(cfg, self._emit_sink)
        self.worker = None

        root.title("三机仿真控制台 — PX4 SITL")
        root.geometry("1700x900")
        root.configure(bg=self.BG)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build()
        self.root.after(100, self._drain)

    # ---------- 界面 ----------
    def _build(self):
        tk = self.tk
        font_btn = ("WenQuanYi Micro Hei", 10, "bold")
        font_mono = ("Consolas", 9)

        top = tk.Frame(self.root, bg=self.BG)
        top.pack(side="top", fill="x", padx=10, pady=8)

        self.btn_start = tk.Button(top, text="启动三机", width=10, bg=self.ACCENT,
                                   relief="flat", font=font_btn, command=self.on_start)
        self.btn_start.pack(side="left", padx=4)

        self.btn_stop = tk.Button(top, text="停止", width=8, bg=self.DANGER, fg="white",
                                  relief="flat", font=font_btn, state="disabled",
                                  command=self.on_stop)
        self.btn_stop.pack(side="left", padx=4)

        self.btn_clear = tk.Button(top, text="清空日志", width=10, bg="#4a4a4a", fg="white",
                                   relief="flat", font=font_btn, command=self.on_clear)
        self.btn_clear.pack(side="left", padx=4)

        self.btn_check = tk.Button(top, text="环境自检", width=10, bg="#4a4a4a", fg="white",
                                   relief="flat", font=font_btn, command=self.on_check)
        self.btn_check.pack(side="left", padx=4)

        self.var_standalone = tk.BooleanVar(value=self.cfg.standalone)
        tk.Checkbutton(top, text="standalone（单 Gazebo）",
                       variable=self.var_standalone,
                       bg=self.BG, fg=self.FG, selectcolor="#4a4a4a",
                       activebackground=self.BG, activeforeground="#ffffff",
                       font=("WenQuanYi Micro Hei", 9)).pack(side="left", padx=10)

        self.lbl_status = tk.Label(top, text="就绪", fg="#87ceeb", bg=self.BG,
                                   font=("WenQuanYi Micro Hei", 10))
        self.lbl_status.pack(side="right", padx=10)

        # 三列日志
        body = tk.Frame(self.root, bg=self.BG)
        body.pack(side="top", fill="both", expand=True, padx=10, pady=(0, 6))
        for i in range(self.cfg.drones):
            name = f"drone_{i + 1}"
            body.grid_columnconfigure(i, weight=1, uniform="col")
            frame = tk.LabelFrame(body, text=f" {name} ", fg="#87ceeb", bg=self.BG,
                                  font=("WenQuanYi Micro Hei", 10, "bold"))
            frame.grid(row=0, column=i, sticky="nsew", padx=4)
            body.grid_rowconfigure(0, weight=1)
            txt = self.scrolledtext.ScrolledText(frame, bg=self.LOG_BG, fg="#d4d4d4",
                                                 font=font_mono, state="disabled",
                                                 wrap="none")
            txt.pack(fill="both", expand=True, padx=3, pady=3)
            self.panes[name] = txt

        # 底部提示
        self.lbl_hint = tk.Label(
            self.root,
            text="启动后另开终端：source /opt/ros/humble/setup.bash && "
                 "source ~/ros2_ws/install/setup.bash && "
                 "ros2 launch flight_ctrl_pkg flight_ctrl_swarm.launch.py",
            fg="#888888", bg=self.BG, font=("Consolas", 9), anchor="w")
        self.lbl_hint.pack(side="bottom", fill="x", padx=12, pady=(0, 8))

    # ---------- 线程 → UI ----------
    def _emit_sink(self, kind, payload):
        self.q.put((kind, payload))

    def _drain(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "log":
                    self._append(*payload)
                elif kind == "status":
                    self.lbl_status.config(text=payload)
                elif kind == "state":
                    self._update_titles(payload)
                elif kind == "titles":
                    self.lbl_status.config(text=payload)
        except queue.Empty:
            pass
        self.root.after(100, self._drain)

    def _append(self, drone_name, line):
        txt = self.panes.get(drone_name)
        if txt is None:
            return
        txt.config(state="normal")
        txt.insert("end", line + "\n")
        # 单窗最多保留 4000 行，防止长时间运行卡顿
        if int(txt.index("end-1c").split(".")[0]) > 4000:
            txt.delete("1.0", "500.0")
        txt.see("end")
        txt.config(state="disabled")

    def _update_titles(self, pids):
        for name, txt in self.panes.items():
            parent = txt.master
            pid = pids.get(name)
            try:
                parent.config(text=f" {name} " + (f"— PID {pid} " if pid else ""))
            except Exception:
                pass

    # ---------- 按钮 ----------
    def on_start(self):
        # 勾选框决定本次启动用哪种模式（runner 持有同一个 cfg 引用）
        self.cfg.standalone = bool(self.var_standalone.get())
        problems = self.runner.preflight()
        if problems:
            self.lbl_status.config(text="✘ " + problems[0])
            for p in problems:
                self._append("drone_1", f"[预检] {p}")
            return
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.lbl_status.config(text="启动中…")
        self.worker = threading.Thread(target=self._start_worker, daemon=True)
        self.worker.start()

    def _start_worker(self):
        self.runner.start()
        self.q.put(("status", "启动流程结束"))

    def on_stop(self):
        self.btn_stop.config(state="disabled")
        self.lbl_status.config(text="停止中…")
        threading.Thread(target=self._stop_worker, daemon=True).start()

    def _stop_worker(self):
        self.runner.stop()
        self.q.put(("status", "已停止"))

    def on_clear(self):
        for txt in self.panes.values():
            txt.config(state="normal")
            txt.delete("1.0", "end")
            txt.config(state="disabled")

    def on_check(self):
        self.cfg.standalone = bool(self.var_standalone.get())
        problems = self.runner.preflight()
        if problems:
            for p in problems:
                self._append("drone_1", f"[自检] ✘ {p}")
            self.lbl_status.config(text="✘ 自检发现问题")
        else:
            self.lbl_status.config(text="✔ 自检通过，可以启动")
            self._append("drone_1", "[自检] ✔ PX4 与 Agent 路径正常，无残留进程")

    def on_close(self):
        try:
            self.runner.stop()
        except Exception:
            pass
        self.root.destroy()
        os._exit(0)


# ==================== CLI 模式 ====================

def run_cli(cfg):
    def emit(kind, payload):
        if kind == "log":
            name, line = payload
            print(f"[{name}] {line}", flush=True)
        elif kind == "status":
            print(f"== {payload}", flush=True)
        elif kind == "state":
            for n, pid in payload.items():
                print(f"   {n}: PID {pid}", flush=True)

    runner = SwarmRunner(cfg, emit)
    problems = runner.preflight()
    if problems:
        print("预检发现问题：")
        for p in problems:
            print("  ✘", p)
        return 2

    threading.Thread(target=runner.start, daemon=True).start()
    print("按 Ctrl+C 结束全部\n")
    try:
        while True:
            time.sleep(1)
            if runner.children and all(not m.alive() for m in runner.children):
                print("所有机型已退出")
                break
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
    return 0


# ==================== 入口 ====================

def main():
    ap = argparse.ArgumentParser(description="三机 PX4 SITL 启动器（GUI + CLI）")
    ap.add_argument("--cli", action="store_true", help="纯终端模式，不弹窗")
    ap.add_argument("--check", action="store_true", help="只做环境自检")
    ap.add_argument("--drones", type=int, default=3)
    ap.add_argument("--px4-dir", default="~/PX4-Autopilot")
    ap.add_argument("--log-dir", default="~/ros2_ws/logs")
    ap.add_argument("--agent-port", type=int, default=8888)
    ap.add_argument("--no-agent", action="store_true")
    ap.add_argument("--no-ns", action="store_true")
    ap.add_argument("--pose-step", type=float, default=2.0)
    ap.add_argument("--params-file",
                    default="~/ros2_ws/src/flight_ctrl_pkg/scripts/px4-rc.params",
                    help="PX4 参数文件，会被转成 PX4_PARAM_* 环境变量注入（默认 flight_ctrl_pkg 里的那份）")
    ap.add_argument("--no-params", action="store_true", help="不注入 PX4 参数（预检大概率过不了）")
    ap.add_argument("--standalone", action="store_true",
                    help="确定性模式：启动器自己起唯一一个 Gazebo，三架 PX4 只做桥接，"
                         "模型由 gz 服务建到同一世界（避免多世界竞态）")
    ap.add_argument("--gz-world", default="default", help="Gazebo 世界名（standalone 用），默认 default")
    ap.add_argument("--gz-world-file", default=None, help="世界文件路径（默认 PX4 的 worlds/<世界名>.sdf）")
    ap.add_argument("--gz-keep-display", action="store_true",
                    help="启动 Gazebo 时保留 DISPLAY（默认剥掉，与学长脚本一致）")
    ap.add_argument("--init-z", type=float, default=0.5, help="初始高度（米），默认 0.5")
    args = ap.parse_args()

    cfg = Config(args.drones, args.px4_dir, args.log_dir, args.agent_port,
                 not args.no_agent, not args.no_ns, args.pose_step,
                 params_file=None if args.no_params else args.params_file,
                 standalone=args.standalone,
                 gz_world=args.gz_world,
                 gz_world_file=args.gz_world_file,
                 gz_keep_display=args.gz_keep_display,
                 init_pose_z=args.init_z)

    if args.check:
        runner = SwarmRunner(cfg, lambda k, p: None)
        problems = runner.preflight()
        print("=== 环境自检 ===")
        print(f"  PX4            : {cfg.px4_bin}")
        agent = find_agent_bin()
        print(f"  MicroXRCEAgent : {agent or '未找到'}")
        print(f"  PX4 参数文件   : {cfg.params_file or '(未指定)'}")
        print(f"  注入参数数量   : {len(cfg.param_env)}"
              + (f"  NAV_DLL_ACT={cfg.param_env.get('PX4_PARAM_NAV_DLL_ACT')}"
                 if 'PX4_PARAM_NAV_DLL_ACT' in cfg.param_env else ""))
        print(f"  启动模式       : "
              + ("standalone（单 Gazebo + gz 服务建模型）" if cfg.standalone
                 else "原生（1 号机 PX4 自拉 Gazebo）"))
        if cfg.standalone:
            print(f"  Gazebo 世界    : {cfg.gz_world}")
            print(f"  初始坐标       : "
                  + ", ".join(f"drone_{i+1}=(0, {cfg.pose_step*i:g}, {cfg.init_pose_z:g})"
                              for i in range(cfg.drones)))
        print(f"  运行中的 PX4   : {sorted(pgrep('px4_sitl_default/bin/px4')) or '无'}")
        print(f"  运行中的 Gazebo: {sorted(pgrep('gz sim')) or '无'}")
        if problems:
            for p in problems:
                print("  ✘", p)
            print("=== 自检结束：存在问题 ===")
            return 5
        print("=== 自检结束：可以启动 ✔ ===")
        return 0

    if args.cli:
        return run_cli(cfg)

    try:
        import tkinter as tk
        from tkinter import scrolledtext  # noqa: F401
    except Exception as e:
        print(f"无法加载 tkinter（{e}），改用终端模式。")
        return run_cli(cfg)

    root = tk.Tk()
    SimGUI(root, cfg)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
