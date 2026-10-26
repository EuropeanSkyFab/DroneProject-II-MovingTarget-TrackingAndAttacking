from setuptools import find_packages, setup

package_name = 'center_node_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/center_node.launch.py']), 
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='europeanskyfab',
    maintainer_email='europeanskyfab@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'center_node = center_node_pkg.center_node:main',
            
            # 下面三个是发假数据用的测试节点
            'mock_mavsdk_bridge = center_node_pkg.mock_mavsdk_bridge:main',    # 假桥接节点（需要自己装gazebo才能用）
            'mock_plan_node = center_node_pkg.mock_plan_node:main',    # 假规划节点
            'mock_vision_node = center_node_pkg.mock_vision_node:main',    # 假视觉节点
        ],
    },
)



