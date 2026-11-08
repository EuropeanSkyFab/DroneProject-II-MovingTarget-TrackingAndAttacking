from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='vision_pkg',
            executable='vision_node',
            output='screen'
        )
    ])