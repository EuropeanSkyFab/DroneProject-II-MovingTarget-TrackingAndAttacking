# -*- coding: utf-8 -*-
"""
detector.py —— 目标检测后端

两个后端，可切换（vision_node 参数 detector_type）：
  - "color" : ColorBlobDetector，纯 OpenCV HSV 颜色块检测，零模型依赖，开箱即可联调。
              适合 Gazebo 里放一个纯色（默认红色）靶机的场景。
  - "yolo"  : YoloDetector，基于 ultralytics，适合真实复杂场景。
              需要 pip install ultralytics，并通过参数给出模型权重路径。

两个后端统一接口：
    detect(bgr_image) -> (u, v) 或 None
        返回图像中目标的像素中心坐标；未检测到返回 None。
"""

import cv2
import numpy as np


class ColorBlobDetector:
    """HSV 颜色块检测：双区间（兼容红色跨 0° 的特性）+ 形态学去噪 + 最大轮廓质心。"""

    def __init__(self,
                 hsv_lower1=(0, 120, 70), hsv_upper1=(10, 255, 255),
                 hsv_lower2=(170, 120, 70), hsv_upper2=(180, 255, 255),
                 min_area=200):
        self.lower1 = np.array(hsv_lower1, dtype=np.uint8)
        self.upper1 = np.array(hsv_upper1, dtype=np.uint8)
        self.lower2 = np.array(hsv_lower2, dtype=np.uint8)
        self.upper2 = np.array(hsv_upper2, dtype=np.uint8)
        self.min_area = min_area
        self._kernel = np.ones((5, 5), np.uint8)

    def detect(self, bgr_image):
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower1, self.upper1)
        mask |= cv2.inRange(hsv, self.lower2, self.upper2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        biggest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(biggest) < self.min_area:
            return None
        m = cv2.moments(biggest)
        if abs(m["m00"]) < 1e-9:
            return None
        return (m["m10"] / m["m00"], m["m01"] / m["m00"])


class YoloDetector:
    """YOLOv8 检测：取置信度最高且类别匹配的目标框中心。未安装 ultralytics 时优雅降级。"""

    def __init__(self, model_path="yolov8n.pt", conf_thres=0.4, target_class=-1):
        self.conf_thres = conf_thres
        self.target_class = target_class  # -1 表示不限类别
        self._model = None
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
        except Exception:
            # 交给节点层打日志提示，此处保持静默降级
            self._model = None

    @property
    def available(self):
        return self._model is not None

    def detect(self, bgr_image):
        if self._model is None:
            return None
        results = self._model.predict(bgr_image, verbose=False, conf=self.conf_thres)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes
        best_u, best_v, best_conf = None, None, -1.0
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if self.target_class >= 0 and cls_id != self.target_class:
                continue
            conf = float(boxes.conf[i].item())
            if conf > best_conf:
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                best_u, best_v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                best_conf = conf
        if best_u is None:
            return None
        return (best_u, best_v)
