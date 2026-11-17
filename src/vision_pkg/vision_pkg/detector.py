# -*- coding: utf-8 -*-
"""
detector.py —— 抗强光多后端机载视觉目标检测器

支持两种统一接口后端：
  - "color" : ColorBlobDetector —— HSV 双红区 + 红度色差抑制室外日照镜面反光，支持超视距小目标与近距大目标检测。
  - "yolo"  : YoloDetector —— 基于 ultralytics YOLOv8，未安装依赖时自动优雅降级。

完全保持原有接口兼容：
  - detect(bgr_image) -> (u, v) | None
  - detect_all(bgr_image) -> [(u, v), ...]
新增高阶接口：
  - detect_with_meta(bgr_image) -> (u, v, area_px, bbox_span_px) | None
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


class ColorBlobDetector:
    """HSV 双区间 + 红度优势度滤波色块检测器（抗日照高光白斑误检，兼顾 40m 超视距小靶标）。"""

    def __init__(
        self,
        hsv_lower1: Tuple[int, int, int] = (0, 110, 55),
        hsv_upper1: Tuple[int, int, int] = (12, 255, 255),
        hsv_lower2: Tuple[int, int, int] = (168, 110, 55),
        hsv_upper2: Tuple[int, int, int] = (180, 255, 255),
        min_area: float = 24.0,
        enable_glare_suppression: bool = True,
    ) -> None:
        self.lower1 = np.array(hsv_lower1, dtype=np.uint8)
        self.upper1 = np.array(hsv_upper1, dtype=np.uint8)
        self.lower2 = np.array(hsv_lower2, dtype=np.uint8)
        self.upper2 = np.array(hsv_upper2, dtype=np.uint8)
        self.min_area = float(min_area)
        self.enable_glare_suppression = bool(enable_glare_suppression)
        self._kernel_small = np.ones((3, 3), np.uint8)
        self._kernel_close = np.ones((5, 5), np.uint8)

    def _extract_candidates(self, bgr_image: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """返回 [(u, v, area, bbox_max_span), ...]，按面积降序排列。"""
        if bgr_image is None or bgr_image.size == 0:
            return []

        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower1, self.upper1)
        mask |= cv2.inRange(hsv, self.lower2, self.upper2)

        if self.enable_glare_suppression:
            # 室外强光抑制：要求红色通道显著高于绿/蓝通道 (R - max(G, B) >= 35)，排除暖白阳光反射面
            b_ch, g_ch, r_ch = cv2.split(bgr_image)
            gb_max = np.maximum(g_ch, b_ch)
            red_excess = cv2.subtract(r_ch, gb_max)
            _, chroma_mask = cv2.threshold(red_excess, 35, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_and(mask, chroma_mask)

        # 使用 3x3 轻量化开运算保留远距小目标 (>=24 px^2)，再用 5x5 闭运算连通机翼分离区域
        if self.min_area >= 120.0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_close)
        else:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results: List[Tuple[float, float, float, float]] = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < self.min_area:
                continue
            m = cv2.moments(c)
            if abs(m["m00"]) < 1e-9:
                continue
            u = float(m["m10"] / m["m00"])
            v = float(m["m01"] / m["m00"])
            _, _, bw, bh = cv2.boundingRect(c)
            span = float(max(bw, bh))
            results.append((u, v, area, span))

        results.sort(key=lambda r: r[2], reverse=True)
        return results

    def detect_all(self, bgr_image: np.ndarray) -> List[Tuple[float, float]]:
        """返回所有面积超过 min_area 的色块质心列表。"""
        return [(u, v) for u, v, _, _ in self._extract_candidates(bgr_image)]

    def detect(self, bgr_image: np.ndarray) -> Optional[Tuple[float, float]]:
        """返回最大色块的质心 (u, v)。"""
        cands = self._extract_candidates(bgr_image)
        return (cands[0][0], cands[0][1]) if cands else None

    def detect_with_meta(self, bgr_image: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        """返回最大色块的 (u, v, area_px, bbox_span_px)。"""
        cands = self._extract_candidates(bgr_image)
        return cands[0] if cands else None


class YoloDetector:
    """YOLOv8 检测后端：未安装 ultralytics 时优雅降级。"""

    def __init__(self, model_path: str = "yolov8n.pt", conf_thres: float = 0.4, target_class: int = -1) -> None:
        self.conf_thres = float(conf_thres)
        self.target_class = int(target_class)
        self._model = None
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_path)
        except Exception:
            self._model = None

    @property
    def available(self) -> bool:
        return self._model is not None

    def _extract_hits(self, bgr_image: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        if self._model is None or bgr_image is None or bgr_image.size == 0:
            return []
        results = self._model.predict(bgr_image, verbose=False, conf=self.conf_thres)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return []
        boxes = results[0].boxes
        hits: List[Tuple[float, float, float, float, float]] = []
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if self.target_class >= 0 and cls_id != self.target_class:
                continue
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            bw = max(1.0, float(x2 - x1))
            bh = max(1.0, float(y2 - y1))
            hits.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0, conf, bw * bh, max(bw, bh)))
        hits.sort(key=lambda h: h[2], reverse=True)
        return hits

    def detect_all(self, bgr_image: np.ndarray) -> List[Tuple[float, float]]:
        return [(u, v) for u, v, _, _, _ in self._extract_hits(bgr_image)]

    def detect(self, bgr_image: np.ndarray) -> Optional[Tuple[float, float]]:
        hits = self._extract_hits(bgr_image)
        return (hits[0][0], hits[0][1]) if hits else None

    def detect_with_meta(self, bgr_image: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        hits = self._extract_hits(bgr_image)
        if not hits:
            return None
        u, v, _, area, span = hits[0]
        return u, v, area, span
