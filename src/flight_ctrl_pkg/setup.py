import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'flight_ctrl_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'sim', 'worlds'),
            glob(os.path.join('sim', 'worlds', '*.sdf'))),
        (os.path.join('share', package_name, 'sim', 'config'),
            glob(os.path.join('sim', 'config', '*.yaml'))),
        (os.path.join('share', package_name, 'sim', 'models', 'x500_down_cam'),
            glob(os.path.join('sim', 'models', 'x500_down_cam', '*'))),
        (os.path.join('share', package_name, 'sim', 'models', 'red_target_uav'),
            glob(os.path.join('sim', 'models', 'red_target_uav', '*'))),
        (os.path.join('share', package_name, 'sim', 'models', 'arena_assets'),
            glob(os.path.join('sim', 'models', 'arena_assets', 'model.*'))),
        (os.path.join('share', package_name, 'sim', 'models', 'arena_assets', 'materials', 'textures'),
            glob(os.path.join('sim', 'models', 'arena_assets', 'materials', 'textures', '*.png'))),
        (os.path.join('share', package_name, 'scripts'),
            glob(os.path.join('scripts', '*.sh'))
            + glob(os.path.join('scripts', '*.params'))),
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
        ],
    },
)
