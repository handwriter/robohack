"""Кинематическая одометрия по ногам и детектор неподвижности (чистый numpy).

Идея: опорная стопа неподвижна в мире, поэтому скорость корпуса следует из скорости
стопы относительно корпуса:  v_trunk = -(p_dot + omega x p)  (в осях trunk).
Опорные стопы выбираются по высоте над самой нижней стопой в выровненной по гравитации
системе координат, затем отбрасываются стопы, не согласованные с самой низкой.
"""
from collections import namedtuple

import numpy as np

from . import kinematics as K

LegResult = namedtuple('LegResult', 'v_world height n_stance')


class LegOdometry:
    def __init__(self, stance_height_tol=0.02, consensus_tol=0.12):
        self.stance_height_tol = stance_height_tol
        self.consensus_tol = consensus_tol

    def update(self, q, qd, R_level, R_odom, omega):
        """q, qd: (4, 3); R_level: trunk -> выровненная система (без рыскания);
        R_odom: trunk -> odom; omega: угловая скорость в осях trunk.

        Возвращает LegResult: скорость корпуса в odom (3,), высота корпуса над опорной
        поверхностью (по центрам стоп) и число опорных стоп.
        """
        p = K.foot_positions(q)
        pd = K.foot_velocities(q, qd)
        v_body = -(pd + np.cross(omega, p))
        v_world = (R_odom @ v_body.T).T
        h = (R_level @ p.T).T[:, 2]
        dh = h - h.min()
        tol = self.stance_height_tol
        cand = dh < tol
        anchor = int(np.argmin(dh))
        diff = np.linalg.norm(v_world[:, :2] - v_world[anchor, :2], axis=1)
        keep = cand & (diff < self.consensus_tol)
        keep[anchor] = True
        w = np.clip(1.0 - dh / tol, 0.05, 1.0) ** 2
        w = np.where(keep, w, 0.0)
        ws = w.sum()
        v = (w[:, None] * v_world).sum(axis=0) / ws
        height = float((w * (-h)).sum() / ws)
        return LegResult(v, height, int(keep.sum()))


class StationaryDetector:
    """Определяет, что робот стоит: суставы и корпус практически не движутся.

    Вход в состояние «стоит» — после непрерывного спокойного интервала `hold`;
    выход — сразу, как только скорость суставов или корпуса превысила порог выхода.
    """

    def __init__(self, qd_enter=0.08, qd_exit=0.2, gyro_enter=0.05, gyro_exit=0.12,
                 hold=0.5):
        self.qd_enter = qd_enter
        self.qd_exit = qd_exit
        self.gyro_enter = gyro_enter
        self.gyro_exit = gyro_exit
        self.hold = hold
        self.state = False
        self._calm_since = None

    def update(self, t, qd_rms, gyro_norm):
        if self.state:
            if qd_rms > self.qd_exit or gyro_norm > self.gyro_exit:
                self.state = False
                self._calm_since = None
        else:
            if qd_rms < self.qd_enter and gyro_norm < self.gyro_enter:
                if self._calm_since is None:
                    self._calm_since = t
                if t - self._calm_since >= self.hold:
                    self.state = True
            else:
                self._calm_since = None
        return self.state
