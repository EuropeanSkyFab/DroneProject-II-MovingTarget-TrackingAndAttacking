from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    center_node = Node(
        package='center_node_pkg',
        executable='center_node',
        name='center_node',
        output='screen',
        
        # 启用use_sim_time将会强制使用gazebo原生的`clock`话题以统一时间，但是目前出现了底层bug会导致定时器假死，因此先弃用
        #parameters=[
        #    {'use_sim_time': True}
        #]
        
    )

    return LaunchDescription([
        center_node
    ])