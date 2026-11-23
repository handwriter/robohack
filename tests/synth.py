"""Синтетический стенд для проверки одометрии без Gazebo.

Содержит: комнату с препятствиями и 2D-лидар, генератор рысистой походки (обратная
кинематика ноги), сценарий T4 (стоим, квадрат, стоим) и прогон оценщика.
Это модель идеализированная: опорные стопы неподвижны (плюс настраиваемое
проскальзывание), датчики шумят слабо. Она проверяет алгоритм и проводку данных,
но не заменяет прогон в настоящем Gazebo.
"""
import math
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'go1_odometry'))

from go1_odometry import kinematics as K  # noqa: E402
from go1_odometry.estimator import OdometryEstimator, Params  # noqa: E402
from go1_odometry.geometry import quat_to_R, yaw_of_R  # noqa: E402

DT = 0.005  # 200 Гц, как IMU в симуляторе
T_GAIT = 0.5


# --------------------------------------------------------------------------- комната
class Room:
    def __init__(self, boxes=None, bounds=(-3.4, -3.4, 8.4, 8.4)):
        x0, y0, x1, y1 = bounds
        segs = [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]
        for cx, cy, sx, sy in (boxes if boxes is not None else DEFAULT_BOXES):
            a, b = (cx - sx / 2, cy - sy / 2), (cx + sx / 2, cy + sy / 2)
            segs += [((a[0], a[1]), (b[0], a[1])), ((b[0], a[1]), (b[0], b[1])),
                     ((b[0], b[1]), (a[0], b[1])), ((a[0], b[1]), (a[0], a[1]))]
        s = np.array(segs, dtype=float)
        self.A = s[:, 0, :]
        self.E = s[:, 1, :] - s[:, 0, :]

    def raycast(self, origin, dirs):
        """origin (2,), dirs (n, 2) единичные. Расстояние до ближайшего отрезка."""
        o = np.asarray(origin)[None, None, :]
        d = dirs[:, None, :]
        e = self.E[None, :, :]
        a = self.A[None, :, :]
        den = d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0]
        diff = a - o
        with np.errstate(divide='ignore', invalid='ignore'):
            s = (diff[..., 0] * e[..., 1] - diff[..., 1] * e[..., 0]) / den
            u = (diff[..., 0] * d[..., 1] - diff[..., 1] * d[..., 0]) / den
        hit = (np.abs(den) > 1e-12) & (s > 1e-6) & (u >= 0.0) & (u <= 1.0)
        s = np.where(hit, s, np.inf)
        return s.min(axis=1)


# Препятствия (центр x, центр y, размер x, размер y) — по мотивам odom_validation.world
DEFAULT_BOXES = [(-2.55, -0.3, 0.55, 1.3), (-2.55, 4.35, 0.65, 1.0), (7.5, 1.4, 0.65, 1.25),
                 (3.0, 3.0, 0.5, 0.5), (1.5, 5.5, 0.4, 0.9), (5.5, -1.5, 0.8, 0.4),
                 (-0.5, 6.5, 0.6, 0.6), (4.8, 6.2, 0.5, 1.1)]


# ------------------------------------------------------------------------ траектория
class Scenario:
    """Сценарий T4: неподвижность -> квадрат -> остановка -> неподвижность."""

    def __init__(self, yaw_start=0.0, side=3.0, v=0.2, w=0.3, t_still0=5.0, t_trans=2.0,
                 t_still1=15.0, yaw_bias=0.06, slip=0.05, bob=True, seed=0,
                 omit_still_end=False):
        self.rng = np.random.default_rng(seed)
        segs = [(t_still0, 0.0, 0.0, False)]
        for _ in range(4):
            segs.append((side / v, v, 0.0, True))
            segs.append((math.pi / 2 / w, 0.0, w, True))
        segs.append((t_trans, 0.0, 0.0, False))
        segs.append((t_still1, 0.0, 0.0, False))
        n = int(round(sum(s[0] for s in segs) / DT)) + 1
        x = np.zeros(n)
        y = np.zeros(n)
        yaw = np.zeros(n)
        yaw[0] = yaw_start
        moving = np.zeros(n, dtype=bool)
        i = 0
        self.t_ms = t_still0
        self.t_me_cmd = t_still0 + sum(s[0] for s in segs[1:-2])
        for dur, vv, ww, mv in segs:
            m = int(round(dur / DT))
            for _ in range(m):
                if i + 1 >= n:
                    break
                bias = yaw_bias if mv else 0.0
                yaw[i + 1] = yaw[i] + (ww + bias) * DT
                x[i + 1] = x[i] + vv * math.cos(yaw[i]) * DT
                y[i + 1] = y[i] + vv * math.sin(yaw[i]) * DT
                moving[i] = mv
                i += 1
        n_used = i + 1
        self.n = n_used
        self.t = np.arange(n_used) * DT
        self.x, self.y, self.yaw = x[:n_used], y[:n_used], yaw[:n_used]
        # походка идёт целое число периодов
        cycles = math.ceil((self.t_me_cmd - self.t_ms) / T_GAIT - 1e-9)
        self.t_me = self.t_ms + cycles * T_GAIT
        self.slip = slip
        # плавная амплитуда качки корпуса при движении
        t = self.t
        amp = np.clip((t - self.t_ms) / 0.5, 0, 1) * np.clip((self.t_me - t) / 0.5 + 1.0, 0, 1)
        amp = np.where((t >= self.t_ms) & (t <= self.t_me + 0.5), amp, 0.0)
        ph = 2 * math.pi * (t - self.t_ms) / T_GAIT
        self.roll = 0.03 * np.sin(ph + 0.3) * amp if bob else np.zeros(self.n)
        self.pitch = 0.02 * np.sin(2 * ph) * amp if bob else np.zeros(self.n)
        self.z = 0.30 + (0.006 * np.sin(2 * ph) * amp if bob else 0.0)
        self.R_wt = np.array([Rotation.from_euler('ZYX', [yaw[k], self.pitch[k], self.roll[k]]
                                                  ).as_matrix() for k in range(self.n)])
        self._build_feet()

    # положение номинальной точки стопы в мире (xy) в момент времени tt
    def _nominal(self, leg, tt):
        k = min(max(int(round(tt / DT)), 0), self.n - 1)
        off = np.array([K.HIP_POS[leg][0], K.HIP_POS[leg][1] + K.SIDE[leg] * K.HIP_OFFSET])
        c, s = math.cos(self.yaw[k]), math.sin(self.yaw[k])
        return np.array([self.x[k] + c * off[0] - s * off[1], self.y[k] + s * off[0] + c * off[1]])

    def _body_xy(self, tt):
        k = min(max(int(round(tt / DT)), 0), self.n - 1)
        return np.array([self.x[k], self.y[k]])

    def _stance_pos(self, leg, t_s, tt, p0=None):
        p = p0 if p0 is not None else self._nominal(leg, t_s + T_GAIT / 4)
        return p + self.slip * (self._body_xy(tt) - self._body_xy(t_s))

    def _build_feet(self):
        self.feet = np.zeros((4, self.n, 3))
        offs = {0: 0.0, 1: 0.5, 2: 0.5, 3: 0.0}  # FR, RL — одна пара; FL, RR — другая
        for leg in range(4):
            o = offs[leg]
            p_stand = self._nominal(leg, self.t_ms)
            for i, tt in enumerate(self.t):
                te = min(tt, self.t_me)
                if te < self.t_ms:
                    xy, z = p_stand, 0.02
                else:
                    u = (te - self.t_ms) / T_GAIT + o
                    k = math.floor(u + 1e-12)
                    phi = u - k
                    t_s = self.t_ms + (k - o) * T_GAIT   # начало «периода k»
                    if phi < 0.5 - 1e-12:               # опора
                        first = (o == 0.0 and k == 0)
                        xy = self._stance_pos(leg, t_s, te, p_stand if first else None)
                        z = 0.02
                    else:                               # перенос
                        first = (o == 0.0 and k == 0)
                        lo = self._stance_pos(leg, t_s, t_s + T_GAIT / 2, p_stand if first else None)
                        td = self._nominal(leg, t_s + T_GAIT + T_GAIT / 4)
                        s = (te - (t_s + T_GAIT / 2)) / (T_GAIT / 2)
                        s = min(max(s, 0.0), 1.0)
                        xy = lo + (td - lo) * (0.5 - 0.5 * math.cos(math.pi * s))
                        z = 0.02 + 0.04 * math.sin(math.pi * s)
                if tt > self.t_me:   # после остановки — там, где стояли
                    xy, z = self.feet[leg, i - 1, :2], self.feet[leg, i - 1, 2]
                self.feet[leg, i] = (xy[0], xy[1], z)

    # ------------------------------------------------------------- датчики
    def joint_angles(self):
        """Углы суставов (n, 4, 3) обратной кинематикой."""
        body = np.stack([self.x, self.y, self.z], axis=1)
        q = np.zeros((self.n, 4, 3))
        for leg in range(4):
            pw = self.feet[leg] - body
            pt = np.einsum('nji,nj->ni', self.R_wt, pw)          # R^T (p - pos)
            q[:, leg, :] = ik_leg(leg, pt - K.HIP_POS[leg])
        return q

    def imu(self):
        quats = Rotation.from_matrix(self.R_wt).as_quat()
        R = self.R_wt
        gyro = np.zeros((self.n, 3))
        for k in range(1, self.n - 1):
            gyro[k] = Rotation.from_matrix(R[k - 1].T @ R[k + 1]).as_rotvec() / (2 * DT)
        gyro[0], gyro[-1] = gyro[1], gyro[-2]
        return quats, gyro

    def scan(self, k, room, rng, noise=0.01, n_beams=720, r_max=20.0):
        off = np.array([0.22, 0.0, 0.11])
        R = self.R_wt[k]
        origin3 = np.array([self.x[k], self.y[k], self.z[k]]) + R @ off
        ang = -math.pi + (2 * math.pi / (n_beams - 1)) * np.arange(n_beams)
        d3 = (R @ np.stack([np.cos(ang), np.sin(ang), np.zeros(n_beams)])).T
        h = d3[:, :2]
        hn = np.linalg.norm(h, axis=1)
        s = room.raycast(origin3[:2], h / hn[:, None])
        rng_m = s / hn + rng.normal(0, noise, n_beams)
        rng_m = np.where(rng_m > r_max, np.inf, rng_m)
        return rng_m, -math.pi, 2 * math.pi / (n_beams - 1)


def ik_leg(leg, q):
    """Обратная кинематика: q (n, 3) — цель стопы относительно оси hip -> (n, 3) углы."""
    x, yt, zt = q[:, 0], q[:, 1], q[:, 2]
    y0 = K.SIDE[leg] * K.HIP_OFFSET
    z_pl = -np.sqrt(np.maximum(yt ** 2 + zt ** 2 - y0 ** 2, 1e-9))
    th = np.arctan2(zt, yt) - np.arctan2(z_pl, y0)
    d2 = x ** 2 + z_pl ** 2
    cos_c = np.clip((d2 - K.L1 ** 2 - K.L2 ** 2) / (2 * K.L1 * K.L2), -1, 1)
    tc = -np.arccos(cos_c)
    A = K.L1 + K.L2 * np.cos(tc)
    B = K.L2 * np.sin(tc)
    tt = np.arctan2(-x, -z_pl) - np.arctan2(B, A)
    return np.stack([th, tt, tc], axis=1)


# ---------------------------------------------------------------------------- прогон
class RunResult:
    pass


def run(sc, params=None, use_scans=True, give_accel=False, imu_orientation=True,
        joint_velocity=True, use_joints=True, joint_vel_noise=0.01, room=None, seed=1, t0=100.0, scan_noise=0.01,
        out_every=4, scan_every=10, imu_override=None):
    """Прогоняет оценщик по сценарию. Возвращает RunResult с рядами для метрик."""
    est = OdometryEstimator(params or Params())
    room = room or Room()
    rng = np.random.default_rng(seed)
    q = sc.joint_angles()
    qd = np.gradient(q, DT, axis=0)
    quats, gyro = sc.imu()
    names = list(K.JOINT_NAMES)
    t_out, e_xy, e_yaw, g_xy, g_yaw, e_z, g_z, e_stat = [], [], [], [], [], [], [], []
    cpu_imu = cpu_scan = 0.0
    n_scans = 0
    for k in range(sc.n):
        t = t0 + sc.t[k]
        vel = (qd[k].reshape(-1) + rng.normal(0, joint_vel_noise, 12)).tolist() if joint_velocity else None
        if use_joints:
            est.on_joint_state(t, names, q[k].reshape(-1).tolist(), vel)
        quat = quats[k] if imu_orientation else np.zeros(4)
        if imu_override is not None:
            quat = imu_override(k, quat)
        accel = None
        if give_accel:
            accel = sc.R_wt[k].T @ np.array([0.0, 0.0, 9.81])
        c0 = time.perf_counter()
        est.on_imu(t, quat, gyro[k], accel)
        cpu_imu += time.perf_counter() - c0
        if use_scans and k % scan_every == 0:
            r, amin, ainc = sc.scan(k, room, rng, noise=scan_noise)
            c0 = time.perf_counter()
            est.on_scan(t, r, amin, ainc)
            cpu_scan += time.perf_counter() - c0
            n_scans += 1
        if k % out_every == 0 and est.out is not None:
            o = est.out
            R = quat_to_R(o.quaternion)
            t_out.append(t)
            e_xy.append(o.position[:2].copy())
            e_z.append(o.position[2])
            e_yaw.append(yaw_of_R(R))
            e_stat.append(o.stationary)
            g_xy.append([sc.x[k], sc.y[k]])
            g_z.append(sc.z[k])
            g_yaw.append(sc.yaw[k])
    res = RunResult()
    res.est = est
    res.t = np.array(t_out)
    res.est_xy = np.array(e_xy)
    res.est_yaw = np.array(e_yaw)
    res.est_z = np.array(e_z)
    res.gt_xy = np.array(g_xy)
    res.gt_yaw = np.array(g_yaw)
    res.gt_z = np.array(g_z)
    res.stationary = np.array(e_stat)
    res.cpu_imu_ms = 1000 * cpu_imu / sc.n
    res.cpu_scan_ms = 1000 * cpu_scan / max(1, n_scans)
    return res
