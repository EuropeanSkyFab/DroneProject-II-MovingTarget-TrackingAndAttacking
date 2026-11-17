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

"""Launch configuration for center_node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for the central integration node."""
    return LaunchDescription([
        DeclareLaunchArgument('takeoff_alt', default_value='9.5'),
        DeclareLaunchArgument('track_safe_dist', default_value='5.0'),
        DeclareLaunchArgument('home_spacing_y', default_value='3.0'),
        Node(
            package='center_node_pkg',
            executable='center_node',
            name='center_node',
            output='screen',
            parameters=[{
                'takeoff_alt': LaunchConfiguration('takeoff_alt'),
                'track_safe_dist': LaunchConfiguration('track_safe_dist'),
                'home_spacing_y': LaunchConfiguration('home_spacing_y'),
            }],
        ),
    ])
