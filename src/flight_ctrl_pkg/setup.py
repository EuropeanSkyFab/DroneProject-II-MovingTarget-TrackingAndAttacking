from glob import glob
import os
import shutil
import subprocess

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py as _build_py

package_name = 'flight_ctrl_pkg'


def _ensure_egl_gpu_filter_so() -> None:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    src_c = os.path.join(pkg_dir, 'scripts', 'egl_gpu_filter.c')
    out_so = os.path.join(pkg_dir, 'scripts', 'libegl_gpu_filter.so')
    if not os.path.isfile(src_c):
        raise FileNotFoundError(f'Missing required C source: {src_c}')
    needs_build = (not os.path.isfile(out_so)) or (
        os.path.getmtime(src_c) > os.path.getmtime(out_so)
    )
    if needs_build:
        gcc = shutil.which('gcc')
        if not gcc:
            if os.path.isfile(out_so):
                return
            raise RuntimeError('gcc not found and prebuilt libegl_gpu_filter.so is missing')
        cmd = [
            gcc,
            '-shared',
            '-fPIC',
            '-O2',
            '-Wall',
            '-Wextra',
            '-Werror',
            '-D_GNU_SOURCE',
            '-DEGL_EGLEXT_PROTOTYPES',
            src_c,
            '-o',
            out_so,
            '-ldl',
            '-lpthread',
        ]
        subprocess.check_call(cmd)


class BuildPyWithEglFilter(_build_py):
    def run(self):
        _ensure_egl_gpu_filter_so()
        super().run()


_ensure_egl_gpu_filter_so()

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    cmdclass={'build_py': BuildPyWithEglFilter},
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('lib', [os.path.join('scripts', 'libegl_gpu_filter.so')]),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.yaml'))),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'sim', 'worlds'),
            glob(os.path.join('sim', 'worlds', '*.sdf'))),
        (os.path.join('share', package_name, 'sim', 'config'),
            glob(os.path.join('sim', 'config', '*.yaml'))),
        (os.path.join('share', package_name, 'sim', 'models', 'x500_down_cam'),
            glob(os.path.join('sim', 'models', 'x500_down_cam', '*'))),
        (os.path.join('share', package_name, 'sim', 'models', 'x500_down_cam_lite'),
            glob(os.path.join('sim', 'models', 'x500_down_cam_lite', '*'))),
        (os.path.join('share', package_name, 'sim', 'models', 'red_target_uav'),
            glob(os.path.join('sim', 'models', 'red_target_uav', '*'))),
        (os.path.join('share', package_name, 'sim', 'models', 'amber_target_uav'),
            glob(os.path.join('sim', 'models', 'amber_target_uav', '*'))),
        (os.path.join('share', package_name, 'sim', 'models', 'slate_target_uav'),
            glob(os.path.join('sim', 'models', 'slate_target_uav', '*'))),
        (os.path.join('share', package_name, 'sim', 'models', 'arena_assets'),
            glob(os.path.join('sim', 'models', 'arena_assets', 'model.*'))),
        (os.path.join('share', package_name, 'sim', 'models', 'arena_assets', 'materials', 'textures'),
            glob(os.path.join('sim', 'models', 'arena_assets', 'materials', 'textures', '*.png'))),
        (os.path.join('share', package_name, 'scripts'),
            sorted(set(
                glob(os.path.join('scripts', '*.sh'))
                + glob(os.path.join('scripts', '*.py'))
                + glob(os.path.join('scripts', '*.c'))
                + glob(os.path.join('scripts', '*.so'))
                + glob(os.path.join('scripts', '*.params'))
                + glob(os.path.join('scripts', 'px4-rc.*'))
            ))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hkw',
    maintainer_email='hkw@todo.todo',
    description='课题二飞控调试与 PX4 uXRCE-DDS 桥接功能包',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'px4_bridge_node = flight_ctrl_pkg.px4_bridge_node:main',
            'camera_relay_node = flight_ctrl_pkg.camera_relay_node:main',
            'dynamics_calibrator = flight_ctrl_pkg.dynamics_calibrator:main',
            'target_sim_node = flight_ctrl_pkg.target_sim_node:main',
            'mission_commander_node = flight_ctrl_pkg.mission_commander_node:main',
        ],
    },
)
