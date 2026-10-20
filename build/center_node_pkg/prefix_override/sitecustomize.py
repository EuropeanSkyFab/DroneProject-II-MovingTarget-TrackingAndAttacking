import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/europeanskyfab/ros2_ws/install/center_node_pkg'
