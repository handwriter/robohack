"""Рендер ArUco-метки симулятора в виртуальную камеру (для тестов распознавания).

Метка как в сценах T5: пластина 0,25 м, чёрная область 0,1875 м (6 модулей) + белое поле в
1 модуль; такая картинка побайтно совпадает с текстурами aruco_N.png симулятора.
Камеры: фронтальная 640x480 (hfov 1.5184) и боковые 464x400 (hfov 2.094), как в ТЗ п. 8.4.
"""
import math

import cv2
import numpy as np

CAMERAS = {'front': (640, 480, 1.5184), 'side': (464, 400, 2.094)}


def marker_texture(marker_id, module_px=64):
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    gen = aruco.generateImageMarker if hasattr(aruco, 'generateImageMarker') else aruco.drawMarker
    return np.pad(gen(dictionary, marker_id, 6 * module_px, borderBits=1), module_px,
                  constant_values=255)


def render(marker_id, dist, yaw_deg=0.0, camera='front', noise=2.0, background=150, seed=0,
           offset=(0.0, 0.0)):
    """Метка на расстоянии `dist` по оптической оси, повёрнутая вокруг вертикали на yaw_deg."""
    w, h, hfov = CAMERAS[camera]
    f = (w / 2) / math.tan(hfov / 2)
    tex = marker_texture(marker_id)
    s = 0.125
    pts = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], dtype=float)
    a = math.radians(yaw_deg)
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    P = pts @ R.T + np.array([offset[0], offset[1], dist])
    uv = np.stack([f * P[:, 0] / P[:, 2] + w / 2, f * P[:, 1] / P[:, 2] + h / 2],
                  axis=1).astype(np.float32)
    n = tex.shape[0]
    src = np.array([[0, 0], [n, 0], [n, n], [0, n]], dtype=np.float32)
    H = cv2.getPerspectiveTransform(src, uv)
    rng = np.random.default_rng(seed)
    img = np.full((h, w), background, dtype=np.float32) + rng.normal(0, noise, (h, w))
    warped = cv2.warpPerspective(tex, H, (w, h), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full_like(tex, 255), H, (w, h)) > 0
    return np.clip(np.where(mask, warped, img), 0, 255).astype(np.uint8)


def to_rgb8_message(gray, msg_cls):
    """Gray -> sensor_msgs/Image (заглушка) в кодировке rgb8, как публикует Gazebo."""
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    m = msg_cls()
    m.height, m.width = gray.shape
    m.encoding = 'rgb8'
    m.step = gray.shape[1] * 3
    m.data = rgb.tobytes()
    return m
