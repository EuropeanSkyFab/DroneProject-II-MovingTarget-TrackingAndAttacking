from setuptools import find_packages, setup

package_name = 'vision_pkg'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/launch', ['launch/vision_node.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='EuropeanSkyFab',
    maintainer_email='3224896257@qq.com',
    description='课题二 机载视觉模块：多机异构视角三维三角化与目标检测',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vision_node = vision_pkg.vision_node:main',
        ],
    },
)
