#!/usr/bin/env python3
"""Verify deterministic GPU-3 EGL & CUDA exclusion for Gazebo Harmonic / Ogre2."""

import argparse
import ctypes
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set, Tuple


GPU3_UUID = 'GPU-617ae266-eed3-ef41-ae0f-5f25c730f5d5'
GPU3_BUS_ID = '0000:D9:00.0'
GPU2_BUS_ID = '0000:B1:00.0'


def _resolve_world_path(world_arg: str) -> str:
    if os.path.isabs(world_arg) and os.path.isfile(world_arg):
        return world_arg
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        world_arg,
        os.path.abspath(os.path.join(script_dir, '..', 'sim', 'worlds', world_arg)),
        os.path.abspath(os.path.join(script_dir, '..', 'sim', 'worlds', f'{world_arg}.sdf')),
    ]
    for cand in candidates:
        if os.path.isfile(cand):
            return cand
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = get_package_share_directory('flight_ctrl_pkg')
        for name in (world_arg, f'{world_arg}.sdf'):
            cand = os.path.join(share_dir, 'sim', 'worlds', name)
            if os.path.isfile(cand):
                return cand
    except Exception:
        pass
    raise FileNotFoundError(f'Cannot find world SDF: {world_arg}')


def _find_filter_so() -> str:
    ld_preload = os.environ.get('LD_PRELOAD', '')
    for entry in ld_preload.split(':'):
        if entry and os.path.basename(entry) == 'libegl_gpu_filter.so' and os.path.isfile(entry):
            return entry
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for cand in (
        os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'lib', 'libegl_gpu_filter.so')),
        os.path.join(script_dir, 'libegl_gpu_filter.so'),
    ):
        if os.path.isfile(cand):
            return cand
    raise FileNotFoundError('libegl_gpu_filter.so not found in LD_PRELOAD or expected paths')


def _check_guarded_env() -> str:
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cvd is None:
        raise AssertionError('CUDA_VISIBLE_DEVICES is unset; source gpu3_guard.sh first')
    tokens = [t.strip() for t in cvd.split(',') if t.strip()]
    if '3' in tokens or GPU3_UUID in tokens:
        raise AssertionError(f'GPU 3 found in CUDA_VISIBLE_DEVICES={cvd!r}')
    vendor = os.environ.get('__EGL_VENDOR_LIBRARY_FILENAMES', '')
    if '10_nvidia.json' not in vendor:
        raise AssertionError(f'Unexpected __EGL_VENDOR_LIBRARY_FILENAMES={vendor!r}')
    filter_so = _find_filter_so()
    if filter_so not in os.environ.get('LD_PRELOAD', '').split(':'):
        raise AssertionError(f'{filter_so} is not active in LD_PRELOAD={os.environ.get("LD_PRELOAD", "")!r}')
    return filter_so


def _ctypes_egl_probe(filter_so: str, cuda_visible_devices: str) -> Tuple[int, List[int]]:
    code = r"""
import ctypes, os, sys
so_path = sys.argv[1]
ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL("libEGL.so.1", mode=ctypes.RTLD_GLOBAL)
rtld_default = ctypes.CDLL(None)
rtld_default.eglGetProcAddress.argtypes = [ctypes.c_char_p]
rtld_default.eglGetProcAddress.restype = ctypes.c_void_p

fn_qdev_addr = rtld_default.eglGetProcAddress(b"eglQueryDevicesEXT")
fn_qattr_addr = rtld_default.eglGetProcAddress(b"eglQueryDeviceAttribEXT")
assert fn_qdev_addr, "eglQueryDevicesEXT is NULL"
assert fn_qattr_addr, "eglQueryDeviceAttribEXT is NULL"

QDEV_T = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))
QATTR_T = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_long))
qdev = QDEV_T(fn_qdev_addr)
qattr = QATTR_T(fn_qattr_addr)

n = ctypes.c_int(-1)
ok = qdev(0, None, ctypes.byref(n))
assert ok == 1, f"count query failed: ok={ok}"
devs = (ctypes.c_void_p * max(1, n.value))()
m = ctypes.c_int(-1)
ok2 = qdev(n.value, devs, ctypes.byref(m))
assert ok2 == 1 and m.value == n.value, f"list query failed: ok2={ok2} m={m.value} n={n.value}"
cuda_ids = []
for i in range(m.value):
    val = ctypes.c_long(-1)
    has_cuda = qattr(devs[i], 0x323A, ctypes.byref(val))
    cuda_ids.append(int(val.value) if has_cuda == 1 else -1)
print(f"{m.value}|" + ",".join(str(x) for x in cuda_ids))
"""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices
    env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    env['LD_PRELOAD'] = filter_so
    env['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    env.pop('DISPLAY', None)
    res = subprocess.run(
        [sys.executable, '-c', code, filter_so],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    line = res.stdout.strip().splitlines()[-1]
    cnt_str, ids_str = line.split('|', 1)
    ids = [int(x) for x in ids_str.split(',') if x]
    return int(cnt_str), ids


def _query_nvidia_smi_gpu_procs(gpu_idx: int) -> Dict[int, Tuple[str, int]]:
    """Return {pid: (proc_name, used_mib)} on the given GPU index from nvidia-smi."""
    procs: Dict[int, Tuple[str, int]] = {}
    res = subprocess.run(
        ['nvidia-smi'],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    in_procs = False
    for line in res.stdout.splitlines():
        if '|  Processes:' in line:
            in_procs = True
            continue
        if not in_procs:
            continue
        # Match rows like: |    3   N/A  N/A      3216      G   /usr/lib/xorg/Xorg        4MiB |
        m = re.match(
            r'^\|\s*(\d+)\s+\S+\s+\S+\s+(\d+)\s+\S+\s+(.*?)\s+(\d+)MiB\s*\|$',
            line.strip(),
        )
        if m and int(m.group(1)) == gpu_idx:
            pid = int(m.group(2))
            name = m.group(3).strip()
            mem = int(m.group(4))
            procs[pid] = (name, mem)

    # Also query compute-apps explicitly for GPU index
    comp = subprocess.run(
        [
            'nvidia-smi',
            f'--id={gpu_idx}',
            '--query-compute-apps=pid,process_name,used_memory',
            '--format=csv,noheader,nounits',
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    for line in comp.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith('[Not Supported]'):
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 3 and parts[0].isdigit():
            pid = int(parts[0])
            name = parts[1]
            mem = int(float(parts[2])) if parts[2].replace('.', '', 1).isdigit() else 0
            procs[pid] = (name, max(mem, procs.get(pid, ('', 0))[1]))
    return procs


def _build_gz_resource_path(world_path: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_sim_models = os.path.abspath(os.path.join(script_dir, '..', 'sim', 'models'))
    world_models = os.path.abspath(os.path.join(os.path.dirname(world_path), '..', 'models'))
    parts: List[str] = []
    for p in (pkg_sim_models, world_models):
        if os.path.isdir(p) and p not in parts:
            parts.append(p)
    try:
        from ament_index_python.packages import get_package_share_directory
        share_models = os.path.join(get_package_share_directory('flight_ctrl_pkg'), 'sim', 'models')
        if os.path.isdir(share_models) and share_models not in parts:
            parts.append(share_models)
    except Exception:
        pass
    px4_dir = os.environ.get('PX4_AUTOPILOT_DIR', '/home/hkw/flying/third_party/PX4-Autopilot')
    px4_models = os.path.join(px4_dir, 'Tools', 'simulation', 'gz', 'models')
    if os.path.isdir(px4_models) and px4_models not in parts:
        parts.append(px4_models)
    existing = os.environ.get('GZ_SIM_RESOURCE_PATH', '')
    for p in existing.split(':'):
        if p and p not in parts:
            parts.append(p)
    return ':'.join(parts)


def _terminate_pg(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def run_verification(world_path: str, iterations: int, gz_cvd: str = '2') -> None:
    if shutil.which('gz') is None:
        raise RuntimeError('`gz` CLI not found in PATH; activate px4_ros2 environment first')
    if shutil.which('nvidia-smi') is None:
        raise RuntimeError('`nvidia-smi` not found in PATH')

    filter_so = _check_guarded_env()
    print(f'[OK] Guarded environment verified. Filter SO: {filter_so}')
    print(f'[INFO] World SDF: {world_path}')

    # 1. Direct ctypes EGL probe checks
    cnt_single, ids_single = _ctypes_egl_probe(filter_so, '2')
    if cnt_single != 1 or ids_single != [0]:
        raise AssertionError(
            f'ctypes EGL probe under CUDA_VISIBLE_DEVICES=2 failed: count={cnt_single}, cuda_ids={ids_single}'
        )
    print(f'[OK] ctypes EGL probe (CUDA_VISIBLE_DEVICES=2): count={cnt_single}, cuda_ids={ids_single}')

    cnt_multi, ids_multi = _ctypes_egl_probe(filter_so, '0,1,2')
    if cnt_multi != 3 or sorted(ids_multi) != [0, 1, 2]:
        raise AssertionError(
            f'ctypes EGL probe under CUDA_VISIBLE_DEVICES=0,1,2 failed: count={cnt_multi}, cuda_ids={ids_multi}'
        )
    print(f'[OK] ctypes EGL probe (CUDA_VISIBLE_DEVICES=0,1,2): count={cnt_multi}, cuda_ids={ids_multi}')

    # 2. Baseline GPU 3 process & memory snapshot
    baseline_gpu3 = _query_nvidia_smi_gpu_procs(3)
    print(f'[INFO] Baseline GPU 3 ({GPU3_BUS_ID}) processes: {baseline_gpu3}')

    ogre_log_path = os.path.expanduser('~/.gz/rendering/ogre2.log')
    os.makedirs(os.path.dirname(ogre_log_path), exist_ok=True)

    gz_cvd_tokens = [t.strip() for t in gz_cvd.split(',') if t.strip()]
    if not gz_cvd_tokens or '3' in gz_cvd_tokens or GPU3_UUID in gz_cvd_tokens:
        raise AssertionError(f'Invalid --cuda-visible-devices={gz_cvd!r}')

    gz_env = os.environ.copy()
    gz_env.pop('DISPLAY', None)
    gz_env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    gz_env['CUDA_VISIBLE_DEVICES'] = gz_cvd
    gz_env['LD_PRELOAD'] = filter_so
    gz_env['__EGL_VENDOR_LIBRARY_FILENAMES'] = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
    gz_env['GZ_SIM_RESOURCE_PATH'] = _build_gz_resource_path(world_path)

    # 3. N-iteration live headless Gazebo test
    for it in range(1, iterations + 1):
        # Truncate/remove ogre2.log before each run so stale logs cannot pass
        if os.path.exists(ogre_log_path):
            with open(ogre_log_path, 'w', encoding='utf-8') as f:
                f.truncate(0)

        cmd = ['gz', 'sim', '-r', '-s', '--headless-rendering', world_path]
        proc = subprocess.Popen(
            cmd,
            env=gz_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        seen_egl_line = False
        ogre_text = ''
        live_gpu3_samples: List[Dict[int, Tuple[str, int]]] = []
        live_gpu2_samples: List[Dict[int, Tuple[str, int]]] = []
        try:
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    break
                if os.path.isfile(ogre_log_path):
                    with open(ogre_log_path, 'r', encoding='utf-8', errors='replace') as f:
                        ogre_text = f.read()
                    if 'Found Num EGL Devices:' in ogre_text:
                        seen_egl_line = True
                        # Sample live nvidia-smi while gz sim is running
                        live_gpu3_samples.append(_query_nvidia_smi_gpu_procs(3))
                        live_gpu2_samples.append(_query_nvidia_smi_gpu_procs(2))
                        time.sleep(0.6)
                        live_gpu3_samples.append(_query_nvidia_smi_gpu_procs(3))
                        live_gpu2_samples.append(_query_nvidia_smi_gpu_procs(2))
                        break
                time.sleep(0.25)

            if not seen_egl_line:
                if os.path.isfile(ogre_log_path):
                    with open(ogre_log_path, 'r', encoding='utf-8', errors='replace') as f:
                        ogre_text = f.read()
                _terminate_pg(proc)
                out, err = proc.communicate(timeout=5)
                raise AssertionError(
                    f'Iteration {it}/{iterations}: ogre2.log never reported "Found Num EGL Devices:".\n'
                    f'ogre2.log:\n{ogre_text[-1200:]}\nstdout:\n{out[-600:]}\nstderr:\n{err[-600:]}'
                )

            matches = re.findall(r'Found Num EGL Devices:\s*(\d+)', ogre_text)
            if not matches or any(int(m) != 1 for m in matches):
                raise AssertionError(
                    f'Iteration {it}/{iterations}: expected Found Num EGL Devices: 1 only, got {matches}.\n'
                    f'ogre2.log:\n{ogre_text}'
                )

            # Verify zero new PIDs, zero gz/ruby/python/ogre processes, and zero VRAM delta on GPU 3
            forbidden_names = ('gz', 'ruby', 'python', 'ogre', 'px4')
            for sample in live_gpu3_samples:
                for pid, (pname, mem) in sample.items():
                    low_name = pname.lower()
                    if any(tok in low_name for tok in forbidden_names):
                        raise AssertionError(
                            f'Iteration {it}/{iterations}: forbidden process on GPU 3: pid={pid}, name={pname!r}, mem={mem}MiB'
                        )
                    if pid not in baseline_gpu3:
                        raise AssertionError(
                            f'Iteration {it}/{iterations}: new PID appeared on GPU 3 during gz sim: pid={pid}, name={pname!r}, mem={mem}MiB'
                        )
                    base_mem = baseline_gpu3[pid][1]
                    if mem > base_mem + 2:
                        raise AssertionError(
                            f'Iteration {it}/{iterations}: GPU 3 VRAM increased for pid={pid} ({pname}): {base_mem}MiB -> {mem}MiB'
                        )

            gpu2_summary = live_gpu2_samples[-1] if live_gpu2_samples else {}
            print(
                f'[PASS] Iter {it}/{iterations}: ogre2 Found Num EGL Devices={matches}, '
                f'GPU3 new_pids=0, GPU2 active_pids={len(gpu2_summary)}'
            )
        finally:
            _terminate_pg(proc)

    print(f'[SUCCESS] All {iterations} live headless Gazebo EGL & GPU-3 isolation checks passed.')


def main() -> int:
    parser = argparse.ArgumentParser(description='Verify GPU-3 EGL & CUDA isolation for Gazebo Harmonic')
    parser.add_argument(
        '--world',
        default='urban_counter_swarm_arena.sdf',
        help='World SDF path or filename under sim/worlds (default: urban_counter_swarm_arena.sdf)',
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=5,
        help='Number of live headless gz sim iterations (default: 5)',
    )
    parser.add_argument(
        '--cuda-visible-devices',
        default='2',
        help='CUDA_VISIBLE_DEVICES value used for live gz sim runs (default: 2)',
    )
    args = parser.parse_args()
    world_path = _resolve_world_path(args.world)
    run_verification(world_path, args.iterations, gz_cvd=args.cuda_visible_devices)
    return 0


if __name__ == '__main__':
    sys.exit(main())
