"""Оценщик одометрии Go1: ориентация IMU + кинематика ног + коррекция лидаром.

Модуль не зависит от ROS: узел (node.py) только передаёт сюда данные датчиков.

Состав:
  * ориентация берётся из кватерниона IMU (мировое рыскание приводится к нулю в момент
    старта); при некорректном кватернионе включается запасной фильтр гироскоп+акселерометр;
  * положение интегрируется по скорости корпуса, выведенной из опорных стоп;
  * сканы лидара сопоставляются с локальной картой и поправляют положение;
  * в состоянии «робот стоит» положение замораживается (нулевая скорость, поправок нет);
  * высота — по длине опорных ног.
"""
import math
from collections import deque, namedtuple
from dataclasses import dataclass

import numpy as np

from . import kinematics as K
from .geometry import (AttitudeFilter, Rz, quat_from_yaw, quat_is_valid, quat_mul,
                       quat_normalize, quat_to_R, rot2, wrap_angle, yaw_of_R)
from .leg_odometry import LegOdometry, StationaryDetector
from .scan_matcher import LocalMap, estimate_normals, icp, scan_to_points

Output = namedtuple('Output', 't position quaternion v_body omega_body stationary')


@dataclass
class Params:
    # --- ноги
    leg_scale: float = 1.0
    stance_height_tol: float = 0.02
    consensus_tol: float = 0.12
    joint_timeout: float = 0.1
    # --- неподвижность
    qd_enter: float = 0.08
    qd_exit: float = 0.2
    gyro_enter: float = 0.05
    gyro_exit: float = 0.12
    stationary_hold: float = 0.5
    # --- общее
    max_dt: float = 0.1
    z_tau: float = 0.3
    vel_tau: float = 0.1
    history_size: int = 600
    # --- лидар
    use_lidar: bool = True
    lidar_offset: tuple = (0.22, 0.0, 0.11)
    lidar_r_min: float = 0.25
    lidar_r_max: float = 10.0
    lidar_stride: int = 2
    lidar_gain: float = 0.8
    lidar_yaw_gain: float = 0.5
    max_correction: float = 0.25
    min_inlier_ratio: float = 0.5
    max_rms: float = 0.04
    keyframe_dist: float = 0.25
    keyframe_yaw: float = 0.17
    map_keyframes: int = 8
    reinit_after_rejects: int = 40


class OdometryEstimator:
    def __init__(self, params=None):
        self.p = params or Params()
        p = self.p
        self.leg = LegOdometry(p.stance_height_tol, p.consensus_tol)
        self.stat = StationaryDetector(p.qd_enter, p.qd_exit, p.gyro_enter, p.gyro_exit,
                                       p.stationary_hold)
        self.att = AttitudeFilter()
        self.map = LocalMap(max_keyframes=p.map_keyframes)
        self.lidar_offset = np.asarray(p.lidar_offset, dtype=float)
        self.reset()

    def reset(self):
        self.yaw0 = None
        self.t_prev = None
        self.pos = np.zeros(3)
        self.v_f = np.zeros(3)
        self.z_ready = False
        self.hist = deque(maxlen=self.p.history_size)
        self.out = None
        self._joint_q = None
        self._joint_qd = None
        self._joint_t = None
        self._fd_q = None
        self._fd_t = None
        self._last_quat = None
        self._frozen_cnt = 0
        self._last_yaw_world = None
        self._imu_ok = True
        self._kf = None
        self._reject_run = 0
        self.map = LocalMap(max_keyframes=self.p.map_keyframes)
        self.stats = dict(imu_steps=0, scans=0, scan_accepted=0, scan_rejected=0,
                          fallback_steps=0, stance_min=4)

    # ------------------------------------------------------------------ суставы
    def on_joint_state(self, t, names, position, velocity=None):
        q = K.joints_to_array(names, position)
        if q is None or not np.all(np.isfinite(q)):
            return False
        qd = None
        if velocity is not None and len(velocity) >= len(position):
            qd = K.joints_to_array(names, velocity)
            if qd is not None and not np.all(np.isfinite(qd)):
                qd = None
        if qd is None:
            # скорости не пришли — берём конечные разности по положениям
            if self._fd_t is None:
                qd = np.zeros((4, 3))
                self._fd_q, self._fd_t = q, t
            elif t - self._fd_t >= 0.004:
                qd = (q - self._fd_q) / (t - self._fd_t)
                self._fd_q, self._fd_t = q, t
            else:
                qd = self._joint_qd
        self._joint_q, self._joint_qd, self._joint_t = q, qd, t
        return True

    # --------------------------------------------------------------------- IMU
    def _imu_quat_frozen(self, quat, gyro_norm):
        """Кватернион не меняется, хотя гироскоп показывает вращение — IMU не даёт ориентацию."""
        if self._last_quat is not None and np.array_equal(quat, self._last_quat) \
                and gyro_norm > 0.1:
            self._frozen_cnt += 1
        else:
            self._frozen_cnt = 0
        self._last_quat = np.array(quat, dtype=float)
        return self._frozen_cnt >= 100

    def _orientation(self, quat, gyro, accel, dt):
        gyro_norm = float(np.linalg.norm(gyro))
        ok = quat is not None and quat_is_valid(quat) and \
            not self._imu_quat_frozen(np.asarray(quat, dtype=float), gyro_norm)
        if ok:
            qn = quat_normalize(quat)
            self._imu_ok = True
            return qn, quat_to_R(qn)
        # запасной вариант: гироскоп + акселерометр
        if self._imu_ok and self._last_yaw_world is not None:
            self.att.yaw_offset = self._last_yaw_world
        self._imu_ok = False
        self.stats['fallback_steps'] += 1
        R = self.att.update(gyro, accel, dt)
        return self.att.quaternion(), R

    def on_imu(self, t, quat, gyro, accel=None):
        """Один шаг оценщика по сообщению IMU. Возвращает True, если шаг выполнен."""
        if not math.isfinite(t):
            return False
        if self.t_prev is not None and t <= self.t_prev:
            return False
        gyro = np.asarray(gyro, dtype=float)
        if gyro.shape != (3,) or not np.all(np.isfinite(gyro)):
            gyro = np.zeros(3)
        if accel is not None:
            accel = np.asarray(accel, dtype=float)
            if accel.shape != (3,) or not np.all(np.isfinite(accel)):
                accel = None
        dt = 0.0 if self.t_prev is None else min(t - self.t_prev, self.p.max_dt)

        # --- ориентация
        qw, R_world = self._orientation(quat, gyro, accel, dt)
        yaw = yaw_of_R(R_world)
        self._last_yaw_world = yaw
        if self.yaw0 is None:
            self.yaw0 = yaw
        yaw_rel = wrap_angle(yaw - self.yaw0)
        R_level = Rz(-yaw) @ R_world           # только крен и тангаж
        R_odom = Rz(yaw_rel) @ R_level
        q_odom = quat_normalize(quat_mul(quat_from_yaw(-self.yaw0), qw))

        # --- ноги
        gyro_norm = float(np.linalg.norm(gyro))
        v_leg = np.zeros(3)
        z_meas = None
        qd_rms = None
        if self._joint_q is not None and t - self._joint_t <= self.p.joint_timeout:
            res = self.leg.update(self._joint_q, self._joint_qd, R_level, R_odom, gyro)
            v_leg = res.v_world * self.p.leg_scale
            z_meas = res.height + K.FOOT_RADIUS
            qd_rms = float(np.sqrt(np.mean(self._joint_qd ** 2)))
            self.stats['stance_min'] = min(self.stats['stance_min'], res.n_stance)
        # --- неподвижность
        if qd_rms is None:
            stationary = self.stat.update(t, 1e9, gyro_norm)   # нет данных ног — не «стоит»
        else:
            stationary = self.stat.update(t, qd_rms, gyro_norm)
        v = np.zeros(3) if stationary else v_leg

        # --- интегрирование
        self.pos[0] += v[0] * dt
        self.pos[1] += v[1] * dt
        if z_meas is not None and np.isfinite(z_meas) and 0.1 < z_meas < 0.6:
            if not self.z_ready:
                self.pos[2] = z_meas
                self.z_ready = True
            else:
                a = 1.0 - math.exp(-dt / self.p.z_tau) if dt > 0 else 0.0
                self.pos[2] += a * (z_meas - self.pos[2])
        elif not self.z_ready:
            self.pos[2] = 0.3
        # сглаженная скорость для twist
        if stationary:
            self.v_f[:] = 0.0
        elif dt > 0:
            b = 1.0 - math.exp(-dt / self.p.vel_tau)
            self.v_f += b * (np.array([v[0], v[1], 0.0]) - self.v_f)

        self.hist.append([t, self.pos[0], self.pos[1], yaw_rel, R_level])
        self.t_prev = t
        self.stats['imu_steps'] += 1
        self.out = Output(t, self.pos.copy(), q_odom, R_odom.T @ self.v_f, gyro.copy(),
                          stationary)
        return True

    # ------------------------------------------------------------------- лидар
    def _state_at(self, t):
        best = None
        best_d = 1e9
        for e in reversed(self.hist):
            d = abs(e[0] - t)
            if d < best_d:
                best, best_d = e, d
            elif e[0] < t:
                break
        return best

    def _apply_correction(self, t_scan, delta):
        self.pos[0] += delta[0]
        self.pos[1] += delta[1]
        for e in reversed(self.hist):
            if e[0] < t_scan:
                break
            e[1] += delta[0]
            e[2] += delta[1]
        if self.out is not None:
            self.out = self.out._replace(position=self.pos.copy())

    def _add_keyframe(self, pts, nrm, ok, xy_pose, yaw):
        if self.map.add(pts[ok], nrm[ok]):
            self._kf = (float(xy_pose[0]), float(xy_pose[1]), float(yaw))
            return True
        return False

    def on_scan(self, t, ranges, angle_min, angle_inc):
        """Обрабатывает скан лидара. Возвращает короткую строку-статус (для диагностики)."""
        p = self.p
        if not p.use_lidar or not self.hist:
            return 'off'
        st = self._state_at(t)
        if st is None:
            return 'no_state'
        ts, x, y, yaw, R_level = st
        xy, valid, origin = scan_to_points(ranges, angle_min, angle_inc, R_level,
                                           self.lidar_offset, p.lidar_r_min, p.lidar_r_max)
        if int(valid.sum()) < 60:
            return 'few_points'
        nrm, nok = estimate_normals(xy, valid, origin)
        Rm = rot2(yaw)
        base = np.array([x, y])
        pts = xy @ Rm.T + base
        nrm_o = nrm @ Rm.T
        self.stats['scans'] += 1

        if self.map.empty():
            self._add_keyframe(pts, nrm_o, nok, base, yaw)
            return 'init'
        if self.stat.state:
            return 'stationary'

        sel = np.nonzero(valid)[0][::max(1, p.lidar_stride)]
        res = icp(pts[sel], self.map, pivot=base, estimate_yaw=not self._imu_ok)
        accepted = False
        if res is not None and res.inlier_ratio >= p.min_inlier_ratio and res.rms <= p.max_rms \
                and float(np.linalg.norm(res.translation)) <= p.max_correction:
            accepted = True
        if not accepted:
            self.stats['scan_rejected'] += 1
            self._reject_run += 1
            if self._reject_run >= p.reinit_after_rejects:
                # карта устарела или потеряна — начинаем заново с текущей позы
                self.map = LocalMap(max_keyframes=p.map_keyframes)
                self._add_keyframe(pts, nrm_o, nok, base, yaw)
                self._reject_run = 0
                return 'reinit'
            return 'rejected'
        self._reject_run = 0
        self.stats['scan_accepted'] += 1
        delta = p.lidar_gain * res.translation
        self._apply_correction(ts, delta)
        if not self._imu_ok:
            self.att.yaw_offset += p.lidar_yaw_gain * res.yaw
        base_c = base + delta
        kx, ky, kyaw = self._kf
        if math.hypot(base_c[0] - kx, base_c[1] - ky) >= p.keyframe_dist or \
                abs(wrap_angle(yaw - kyaw)) >= p.keyframe_yaw:
            self._add_keyframe(pts + delta, nrm_o, nok, base_c, yaw)
        return 'ok'
