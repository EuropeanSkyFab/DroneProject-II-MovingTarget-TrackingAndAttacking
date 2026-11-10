import rclpy
import signal
import sys
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

class FakeCamera(Node):
    def __init__(self):
        super().__init__('fake_camera')
        self.pub_img = self.create_publisher(Image, '/drone_1/camera/image_raw', 10)
        self.pub_info = self.create_publisher(CameraInfo, '/drone_1/camera/camera_info', 10)
        self.timer = self.create_timer(0.1, self.timer_callback) # 10Hz
        self.get_logger().info('假相机已启动，正在发送带红色靶标的图像与内参...')

    def timer_callback(self):
        now = self.get_clock().now().to_msg()

        # ================= 1. 发送相机内参 =================
        info = CameraInfo()
        info.header.stamp = now
        info.header.frame_id = 'camera_link'
        info.height = 480
        info.width = 640
        info.distortion_model = 'plumb_bob'
        # 内参：fx, fy, cx, cy (这里用的是标准 640x480 相机的经典内参)
        info.k = [500.0, 0.0, 320.0, 0.0, 500.0, 240.0, 0.0, 0.0, 1.0]
        self.pub_info.publish(info)

        # ================= 2. 生成 480x640 的图像 =================
        # 先创建一个全黑背景 (BGR 格式，每个像素3个字节)
        img_data = bytearray([0, 0, 0] * (480 * 640))
        
        # 在中下方画一个 60x60 的红色正方形 (模拟地面上的红色靶标)
        # 这样反投影射线是朝下的，能算出正数的高度和距离
        for v in range(300, 360):
            for u in range(300, 360):
                idx = (v * 640 + u) * 3
                img_data[idx] = 0       # B
                img_data[idx + 1] = 0   # G
                img_data[idx + 2] = 255 # R

        # ================= 3. 发送图像数据 =================
        img = Image()
        img.header.stamp = now
        img.header.frame_id = 'camera_link'
        img.height = 480
        img.width = 640
        img.encoding = 'bgr8'
        img.step = 1920 # 640 * 3
        img.data = bytes(img_data)
        self.pub_img.publish(img)

def main(args=None):
    rclpy.init(args=args)
    node = FakeCamera()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    rclpy.init(args=None)
    node = FakeCamera()
    
    # ★ 新增：接收 Ctrl+C 或 terminate 信号时，直接强制退出
    def signal_handler(sig, frame):
        node.get_logger().info('假相机收到终止信号，正在退出...')
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass