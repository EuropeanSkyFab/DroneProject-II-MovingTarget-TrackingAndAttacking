from setuptools import setup

package_name = 'vision_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 注册 launch 文件（不注册 colcon build 会忽略）
        ('share/' + package_name + '/launch', ['launch/vision_node.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vision',
    maintainer_email='user@todo.todo',
    description='课题二 机载视觉模块',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'vision_node = vision_pkg.vision_node:main',
        ],
    },
)
