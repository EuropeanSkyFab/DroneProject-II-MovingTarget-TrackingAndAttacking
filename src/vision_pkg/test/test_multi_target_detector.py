# -*- coding: utf-8 -*-
"""F4 多目标、多特征谱系目标检测器单元测试与性能验证。"""

import time

import cv2
import numpy as np

from vision_pkg.detector import (
    ColorBlobDetector,
    DEFAULT_TARGET_PROFILES,
    Detection,
    MultiSignatureDetector,
    SignatureProfile,
    YoloDetector,
    profiles_from_json,
)


def _make_cluttered_scene(width: int = 640, height: int = 480) -> np.ndarray:
    """构造含草地、沥青跑道、土黄集装箱、黄色停机坪与太阳镜面高光的干扰背景。"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (55, 120, 70)  # meadow BGR
    img[180:300, :] = (62, 56, 54)  # asphalt runway BGR
    # Ochre container clutter (BGR=(38,146,194), LAB a*=136)
    img[60:110, 80:140] = (38, 146, 194)
    # Yellow helipad clutter (BGR=(40,195,240), LAB a*=130)
    img[340:400, 100:160] = (40, 195, 240)
    # Warm-white specular glare patch (BGR=(220,225,250))
    img[40:120, 480:590] = (220, 225, 250)
    return img


def test_multi_signature_simultaneous_detection_and_clutter_rejection():
    det = MultiSignatureDetector(min_area=24.0, enable_glare_suppression=True)
    empty_cluttered = _make_cluttered_scene(640, 480)
    assert det.detect_targets(empty_cluttered) == []
    assert det.detect_all(empty_cluttered) == []
    assert det.detect(empty_cluttered) is None

    scene = empty_cluttered.copy()
    # 1. crimson_uav (BGR=(22, 22, 232)) on meadow
    scene[120:136, 260:276] = (22, 22, 232)
    # 2. amber_uav (BGR=(20, 120, 245), LAB a*=171) on meadow
    scene[350:366, 360:376] = (20, 120, 245)
    # 3. slate_uav (BGR=(255, 46, 199), electric-violet dorsal canopy) on meadow
    scene[125:141, 410:426] = (255, 46, 199)

    dets = det.detect_targets(scene)
    assert len(dets) == 3
    assert all(isinstance(d, Detection) for d in dets)

    by_cls = {d.class_id: d for d in dets}
    assert set(by_cls.keys()) == {'crimson_uav', 'amber_uav', 'slate_uav'}
    assert abs(by_cls['crimson_uav'].u - 267.5) < 1.5 and abs(by_cls['crimson_uav'].v - 127.5) < 1.5
    assert abs(by_cls['amber_uav'].u - 367.5) < 1.5 and abs(by_cls['amber_uav'].v - 357.5) < 1.5
    assert abs(by_cls['slate_uav'].u - 417.5) < 1.5 and abs(by_cls['slate_uav'].v - 132.5) < 1.5
    assert all(0.45 <= d.conf <= 0.99 for d in dets)

    # detect_all() 保持兼容返回 [(u, v), ...]
    uv_all = det.detect_all(scene)
    assert len(uv_all) == 3
    assert all(isinstance(pt, tuple) and len(pt) == 2 for pt in uv_all)


def test_nearby_and_touching_targets_separation():
    det = MultiSignatureDetector(min_area=24.0, enable_glare_suppression=True)
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    img[:, :] = (55, 120, 70)

    # 1) 同类目标相距 5 px (3x3 闭运算不可误合并)
    img[100:112, 100:112] = (22, 22, 232)
    img[100:112, 117:129] = (22, 22, 232)

    # 2) 相邻异类目标紧贴 (crimson_uav 紧邻 amber_uav)
    img[220:234, 200:214] = (22, 22, 232)
    img[220:234, 214:228] = (20, 120, 245)

    # 3) 同类双叶瓣粘连目标（中间存在 2px 窄颈，触发距离变换拆分）
    cv2.circle(img, (340, 150), 9, (20, 120, 245), -1)
    cv2.circle(img, (358, 150), 9, (20, 120, 245), -1)
    img[148:152, 346:352] = (20, 120, 245)

    dets = det.detect_targets(img)
    crimson = [d for d in dets if d.class_id == 'crimson_uav']
    amber = [d for d in dets if d.class_id == 'amber_uav']
    assert len(crimson) == 3  # 2 个间距 5px 的 + 1 个紧邻 amber 的
    assert len(amber) == 3    # 1 个紧邻 crimson 的 + 2 个拆分出来的


def test_motion_and_thermal_cues_boost_shadowed_slate_uav():
    det = MultiSignatureDetector(min_area=24.0)
    prev = np.zeros((360, 480, 3), dtype=np.uint8)
    prev[:, :] = (62, 56, 54)  # dark asphalt

    curr = prev.copy()
    # shadowed electric-violet slate_uav (BGR=(72, 48, 64)) over asphalt (62, 56, 54): 静态对比度低 (ring_de < 18)
    curr[150:166, 200:216] = (72, 48, 64)
    prev[150:166, 175:191] = (72, 48, 64)  # 上一帧处于左侧

    # 无线索时静态暗沥青背景及低对比阴影 violet 均不误报
    static_dets = det.detect_targets(curr)
    assert static_dets == []

    # 提供热红外线索或帧间运动线索时，阴影 slate_uav 应被检出且 cue_score > 0
    thermal = np.full((360, 480), 60, dtype=np.uint8)
    thermal[150:166, 200:216] = 225

    dets_thermal = det.detect_targets(curr, thermal_image=thermal)
    assert len(dets_thermal) == 1
    assert dets_thermal[0].class_id == 'slate_uav'
    assert dets_thermal[0].cue_score > 0.5
    assert dets_thermal[0].conf > 0.6

    # 结合强运动/ROI cue_mask
    cue_mask = np.zeros((360, 480), dtype=np.uint8)
    cue_mask[148:168, 198:218] = 255
    dets_motion = det.detect_targets(curr, prev_bgr=prev, cue_mask=cue_mask)
    assert len(dets_motion) == 1
    assert dets_motion[0].class_id == 'slate_uav'
    assert dets_motion[0].cue_score > 0.3


def test_invalid_inputs_and_backward_compatibility():
    det_legacy = ColorBlobDetector(min_area=24.0, enable_glare_suppression=True)
    assert det_legacy.profiles[0].name == 'crimson_uav'
    for bad in (None, np.empty((0, 0, 3), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.uint8)):
        assert det_legacy.detect_targets(bad) == []
        assert det_legacy.detect(bad) is None
        assert det_legacy.detect_all(bad) == []
        assert det_legacy.detect_with_meta(bad) is None

    # 灰度 / BGRA / float32 [0,1] 输入兼容
    gray = np.zeros((100, 100), dtype=np.uint8)
    assert det_legacy.detect_targets(gray) == []
    bgra = np.zeros((100, 100, 4), dtype=np.uint8)
    bgra[40:50, 40:50] = (15, 20, 240, 255)
    assert det_legacy.detect(bgra) is not None
    flt = bgra[:, :, :3].astype(np.float32) / 255.0
    assert det_legacy.detect_with_meta(flt) is not None

    # YoloDetector 在无 ultralytics 环境下优雅降级
    yolo = YoloDetector()
    if not yolo.available:
        assert yolo.detect_targets(bgra[:, :, :3]) == []
        assert yolo.detect(bgra[:, :, :3]) is None


def test_multi_signature_cpu_throughput_budget():
    cv2.setNumThreads(1)
    det = MultiSignatureDetector(profiles=DEFAULT_TARGET_PROFILES, min_area=24.0)
    frame = _make_cluttered_scene(480, 360)
    frame[100:114, 150:164] = (22, 22, 232)
    frame[240:254, 280:294] = (20, 120, 245)
    frame[120:134, 340:354] = (255, 46, 199)

    # 预热
    for _ in range(5):
        det.detect_targets(frame)

    n_frames = 150
    t0 = time.perf_counter()
    for _ in range(n_frames):
        res = det.detect_targets(frame)
        assert len(res) == 3
    elapsed = time.perf_counter() - t0
    mean_ms = (elapsed / n_frames) * 1000.0
    assert elapsed < 1.5, f'150 frames took {elapsed:.3f}s ({mean_ms:.2f} ms/frame)'


def test_harsh_clutter_concrete_shadow_and_ego_motion_zero_fps():
    det = MultiSignatureDetector(min_area=24.0, enable_glare_suppression=True)
    rng = np.random.default_rng(7)
    base = _make_cluttered_scene(640, 480)
    # 注入噪声、混凝土阴影、bunker_upper、apron_pad_strip 及 canyon_rooftop_trim 干扰
    noise = rng.integers(-8, 9, size=base.shape, dtype=np.int16)
    scene = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    scene[190:270, 220:320] = (48, 46, 45)    # concrete shadow
    scene[20:70, 200:260] = (122, 97, 76)     # bunker_upper
    scene[410:460, 240:340] = (112, 97, 87)   # apron_pad_strip
    scene[20:50, 340:420] = (245, 184, 15)    # canyon_rooftop_trim

    assert det.detect_targets(scene) == []

    # 2 px 相机全局自运动 (ego-motion) 下零误报
    shifted = np.roll(scene, shift=2, axis=1)
    prev_gray = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    assert det.detect_targets(shifted, prev_gray=prev_gray) == []
    assert det.detect_targets(shifted, prev_bgr=scene) == []


def test_touching_8x8_and_14x14_diagonal_and_1px_gap_splits():
    det = MultiSignatureDetector(min_area=24.0, enable_glare_suppression=True)
    img = np.full((360, 480, 3), (55, 120, 70), dtype=np.uint8)

    # 1) 1 px 间距同色目标不得被 close_kernel 粘连
    img[60:70, 60:70] = (22, 22, 232)
    img[60:70, 71:81] = (22, 22, 232)

    # 2) 8x8 仅 1px 颈部粘连目标（area ~ 130 < 160）必须拆分为 2 个
    img[150:158, 150:158] = (255, 46, 199)
    img[150:158, 159:167] = (255, 46, 199)
    img[153:155, 158:159] = (255, 46, 199)

    # 3) 14x14 对角粘连目标（aspect == 1.0）必须拆分为 2 个
    img[220:234, 280:294] = (20, 120, 245)
    img[232:246, 292:306] = (20, 120, 245)

    dets = det.detect_targets(img)
    assert len([d for d in dets if d.class_id == 'crimson_uav']) == 2
    assert len([d for d in dets if d.class_id == 'slate_uav']) == 2
    assert len([d for d in dets if d.class_id == 'amber_uav']) == 2


def test_min_compactness_rejects_hollow_ring():
    prof = SignatureProfile(
        name='crimson_ring_check',
        hsv_ranges=(((0, 110, 55), (12, 255, 255)),),
        min_ring_delta_e=0.0,
        min_area=24.0,
        min_extent=0.20,
        max_aspect_ratio=4.0,
        min_compactness=0.50,
        close_kernel_size=1,
    )
    det = MultiSignatureDetector(profiles=(prof,), min_area=24.0)
    img = np.full((120, 120, 3), (55, 120, 70), dtype=np.uint8)
    # 20x20 镂空方环（中心 10x10 挖空，extent=0.75, aspect=1.0）应被 min_compactness=0.50 拒收
    img[40:60, 40:60] = (22, 22, 232)
    img[45:55, 45:55] = (55, 120, 70)
    assert det.detect_targets(img) == []


def test_profiles_from_json_malformed_inputs():
    bad_payloads = [
        '[{"name": "x", "hsv_ranges": "bad"}]',
        '[{"name": "x", "hsv_ranges": [[[0, 10]]]}]',
        '[{"name": "", "hsv_ranges": [[[0, 0, 0], [10, 10, 10]]]}]',
        '[{"name": "x", "hsv_ranges": [[[0, 0, 0], [10, 10, 10]]], "lab_lower": [1, 2]}]',
        '[{"name": "x", "hsv_ranges": [[[0, 0, 0], [10, 10, 10]]], "min_area": "nan_str"}]',
        '[{"name": "x", "hsv_ranges": [[[0, 0, 0], [10, 10, 10]]], "min_area": "NaN"}]',
        '[{"name": "x", "hsv_ranges": [[[true, 0, 0], [10, 10, 10]]]}]',
    ]
    for payload in bad_payloads:
        assert profiles_from_json(payload) is None

    mixed = (
        '[{"name": "bad", "hsv_ranges": "bad"}, '
        '{"name": "valid_uav", "hsv_ranges": [[[0, 100, 50], [10, 255, 255]]]}]'
    )
    parsed = profiles_from_json(mixed)
    assert parsed is not None and len(parsed) == 1 and parsed[0].name == 'valid_uav'
