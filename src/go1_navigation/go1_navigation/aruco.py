"""Распознавание ArUco-меток DICT_4X4_50 (ТЗ T5, п. 7.4). Без ROS: только numpy и OpenCV.

Правила, заданные ТЗ:
  * публикуются только ТЕКУЩИЕ обнаружения, а не накопленный список;
  * подбор идентификаторов запрещён — ID публикуется, только если метка реально
    распознана на изображении. Поэтому здесь нет никаких списков «всех возможных» ID.

Защита от ложных ID: пониженная доля коррекции ошибок декодера, геометрическая проверка
четырёхугольника и подтверждение по времени (ID должен встретиться в нескольких кадрах
одной камеры). Публикация зависит только от того, что видит камера.
"""
import json
from collections import namedtuple

import cv2
import numpy as np

Detection = namedtuple('Detection', 'id corners')

# DICT_4X4_50: допустимые идентификаторы 0..49.
MAX_ID = 49


def _make_backend():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    if hasattr(aruco, 'DetectorParameters_create'):      # OpenCV <= 4.6 (Ubuntu 22.04)
        params = aruco.DetectorParameters_create()
    else:                                                # OpenCV >= 4.7
        params = aruco.DetectorParameters()
    settings = dict(
        adaptiveThreshWinSizeMin=3, adaptiveThreshWinSizeMax=43, adaptiveThreshWinSizeStep=10,
        minMarkerPerimeterRate=0.02, maxMarkerPerimeterRate=4.0,
        polygonalApproxAccuracyRate=0.04,
        errorCorrectionRate=0.3,                         # по умолчанию 0.6: больше ложных ID
        cornerRefinementMethod=aruco.CORNER_REFINE_SUBPIX)
    for key, value in settings.items():
        if hasattr(params, key):
            setattr(params, key, value)
    if hasattr(aruco, 'ArucoDetector'):
        detector = aruco.ArucoDetector(dictionary, params)
        return lambda gray: detector.detectMarkers(gray)
    return lambda gray: aruco.detectMarkers(gray, dictionary, parameters=params)


def quad_ok(corners, min_side_px=10.0, max_side_ratio=4.0):
    """Геометрическая проверка: выпуклый четырёхугольник разумной формы и размера."""
    c = np.asarray(corners, dtype=np.float32).reshape(4, 2)
    if not np.all(np.isfinite(c)) or not cv2.isContourConvex(c.reshape(4, 1, 2)):
        return False
    sides = np.linalg.norm(c - np.roll(c, -1, axis=0), axis=1)
    if sides.min() < min_side_px:
        return False
    return float(sides.max() / sides.min()) <= max_side_ratio


class ArucoDetector:
    def __init__(self):
        self._detect = _make_backend()

    def detect(self, gray):
        """gray: uint8 (H, W). Возвращает список Detection, прошедших проверки."""
        corners, ids, _ = self._detect(gray)
        out = []
        if ids is None:
            return out
        for c, i in zip(corners, ids.reshape(-1)):
            i = int(i)
            if 0 <= i <= MAX_ID and quad_ok(c):
                out.append(Detection(i, np.asarray(c, dtype=float).reshape(4, 2)))
        return out


class TemporalConfirmer:
    """ID считается подтверждённым, если замечен не менее `confirm` раз за `window` секунд."""

    def __init__(self, confirm=2, window=0.7):
        self.confirm = confirm
        self.window = window
        self.hits = {}

    def update(self, t, ids):
        for i in ids:
            self.hits.setdefault(i, []).append(t)
        for i in list(self.hits):
            self.hits[i] = [x for x in self.hits[i] if t - x <= self.window]
            if not self.hits[i]:
                del self.hits[i]
        return {i for i in set(ids) if len(self.hits.get(i, ())) >= self.confirm}


class DetectionHub:
    """Объединяет подтверждённые ID со всех камер: «текущими» считаются виденные не раньше `hold` секунд назад."""

    def __init__(self, hold=0.3):
        self.hold = hold
        self.last = {}

    def update(self, camera, t, ids):
        for i in ids:
            self.last[(camera, i)] = t

    def current(self, t):
        for key in [k for k, v in self.last.items() if t - v > self.hold]:
            del self.last[key]
        return sorted({i for (_, i) in self.last})


def format_message(ids):
    """Содержимое поля data топика /aruco/detections: {"ids": [3, 7, 12]} или {"ids": []}."""
    return json.dumps({'ids': [int(i) for i in sorted(set(ids))]})


def image_to_gray(msg):
    """sensor_msgs/Image -> uint8 (H, W). Поддержаны rgb8, bgr8, rgba8, bgra8, mono8; иначе None."""
    h, w = int(msg.height), int(msg.width)
    enc = str(msg.encoding).lower()
    channels = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4, 'mono8': 1}.get(enc)
    if channels is None or h <= 0 or w <= 0:
        return None
    step = int(msg.step) if int(msg.step) else w * channels
    buf = np.frombuffer(bytes(msg.data) if not isinstance(msg.data, (bytes, bytearray, np.ndarray))
                        else msg.data, dtype=np.uint8)
    if buf.size < step * h or step < w * channels:
        return None
    img = buf[:step * h].reshape(h, step)[:, :w * channels]
    if channels == 1:
        return np.ascontiguousarray(img)
    img = img.reshape(h, w, channels)
    code = {'rgb8': cv2.COLOR_RGB2GRAY, 'bgr8': cv2.COLOR_BGR2GRAY,
            'rgba8': cv2.COLOR_RGBA2GRAY, 'bgra8': cv2.COLOR_BGRA2GRAY}[enc]
    return cv2.cvtColor(np.ascontiguousarray(img), code)
