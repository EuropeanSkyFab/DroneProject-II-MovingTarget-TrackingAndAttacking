from setuptools import setup

package_name = "algorithm_pkg"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            ["resource/" + package_name],
        ),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            ["launch/algorithm.launch.py"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="silensky",
    maintainer_email="silensky@todo.todo",
    description="课题二 规划算法核心与 ROS 2 适配节点",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "algorithm_node = algorithm_pkg.algorithm_node:main",
        ],
    },
)
