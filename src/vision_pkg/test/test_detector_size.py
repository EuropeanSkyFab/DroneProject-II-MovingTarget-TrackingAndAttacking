"""颜色检测器：尺寸估距需要包围盒宽度。"""

import cv2
import numpy as np

from vision_pkg.detector import ColorBlobDetector


def test_color_detector_returns_center_and_width():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (300, 200), (340, 230), (0, 0, 255), -1)

    detector = ColorBlobDetector(min_area=200)
    result = detector.detect_with_size(image)
    assert result is not None
    u, v, w_px = result
    assert abs(u - 320.0) < 3.0
    assert abs(v - 215.0) < 3.0
    assert abs(w_px - 40.0) < 3.0
