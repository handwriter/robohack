"""Кватернионы, повороты и фильтр ориентации (чистый numpy, без ROS).

Кватернион везде в порядке ROS: (x, y, z, w).
"""
import math

import numpy as np


def wrap_angle(a):
    """Приводит угол к диапазону [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def quat_is_valid(q):
    """Конечные значения и норма, не сильно отличающаяся от единицы."""
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        return False
    n = float(np.linalg.norm(q))
    return 0.5 < n < 1.5


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    """Произведение кватернионов a * b."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_from_yaw(yaw):
    return np.array([0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)])


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def yaw_of_R(R):
    return math.atan2(R[1, 0], R[0, 0])


def rot2(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def quat_exp(rotvec):
    """Кватернион поворота на вектор rotvec (рад), ось-угол."""
    th = float(np.linalg.norm(rotvec))
    if th < 1e-12:
        return np.array([0.5 * rotvec[0], 0.5 * rotvec[1], 0.5 * rotvec[2], 1.0])
    axis = np.asarray(rotvec) / th
    s = math.sin(0.5 * th)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * th)])


class AttitudeFilter:
    """Запасной оценщик ориентации по гироскопу и акселерометру.

    Используется, только если IMU не отдаёт корректный кватернион. Крен и тангаж
    подтягиваются к направлению силы тяжести, рыскание — чистая интеграция гироскопа
    (плюс поправка yaw_offset от лидара).
    """

    def __init__(self, kp=1.0):
        self.kp = kp
        self.q = np.array([0.0, 0.0, 0.0, 1.0])
        self.initialized = False
        self.yaw_offset = 0.0

    def _init_from_accel(self, accel):
        a = np.asarray(accel, dtype=float)
        n = np.linalg.norm(a)
        if n < 1e-6:
            return
        ax, ay, az = a / n
        roll = math.atan2(ay, az)
        pitch = math.atan2(-ax, math.hypot(ay, az))
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        self.q = np.array([sr * cp, cr * sp, -sr * sp, cr * cp])
        self.q = quat_normalize(self.q)
        self.initialized = True

    def update(self, gyro, accel, dt):
        """Возвращает матрицу поворота «тело -> мир» (с учётом yaw_offset)."""
        if not self.initialized and accel is not None:
            self._init_from_accel(accel)
        w = np.asarray(gyro, dtype=float).copy()
        if accel is not None and self.initialized:
            a = np.asarray(accel, dtype=float)
            n = float(np.linalg.norm(a))
            if 7.0 < n < 12.5:
                R = quat_to_R(self.q)
                g_est = R[2, :]               # направление «вверх» в осях тела
                err = np.cross(a / n, g_est)  # ошибка направления гравитации
                w = w + self.kp * err
        if dt > 0.0:
            self.q = quat_normalize(quat_mul(self.q, quat_exp(w * dt)))
        return Rz(self.yaw_offset) @ quat_to_R(self.q)

    def quaternion(self):
        return quat_normalize(quat_mul(quat_from_yaw(self.yaw_offset), self.q))
