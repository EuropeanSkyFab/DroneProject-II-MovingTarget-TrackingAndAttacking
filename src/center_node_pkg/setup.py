# Copyright 2026 EuropeanSkyFab.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup configuration for center_node_pkg."""

from setuptools import find_packages, setup

package_name = 'center_node_pkg'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/launch', ['launch/center_node.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='EuropeanSkyFab',
    maintainer_email='3224896257@qq.com',
    description='课题二 系统集成与战术调度安全中枢模块',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'center_node = center_node_pkg.center_node:main',
        ],
    },
)
