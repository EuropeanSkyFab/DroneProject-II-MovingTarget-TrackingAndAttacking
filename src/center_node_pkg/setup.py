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
            #'mock_mavsdk_bridge = center_node_pkg.mock_mavsdk_bridge:main',
        ],
    },
)

# 上面的mock_mavsdk_bridge是我自己测试时用的
