"""Сопоставление сканов 2D-лидара с локальной картой (point-to-line ICP).

Рыскание робота известно из IMU, поэтому по умолчанию ищется только сдвиг (2 степени
свободы) — это намного устойчивее полного ICP. Слабо определённые направления
(например, вдоль гладкой стены) удерживаются слабым приором на предсказанную позу.
Требуются numpy и scipy (cKDTree).
"""
from collections import deque, namedtuple

import numpy as np
from scipy.spatial import cKDTree

from .geometry import rot2

IcpResult = namedtuple('IcpResult', 'translation yaw inlier_ratio rms info_min n_used')


def scan_to_points(ranges, angle_min, angle_inc, R_level, lidar_offset, r_min, r_max):
    """Лучи лидара -> горизонтальные 2D-точки в выровненной (без рыскания) системе trunk.

    Лучи сначала переносятся в trunk (смещение лидара), затем поворачиваются на крен и
    тангаж корпуса (R_level), после чего берутся (x, y): для вертикальных стен это точная
    горизонтальная проекция. Возвращает (xy (N, 2), valid (N,), origin (2,)).
    """
    ranges = np.asarray(ranges, dtype=float)
    n = ranges.shape[0]
    ang = angle_min + angle_inc * np.arange(n)
    valid = np.isfinite(ranges) & (ranges >= r_min) & (ranges <= r_max)
    r = np.where(valid, ranges, 0.0)
    pts = np.stack([r * np.cos(ang), r * np.sin(ang), np.zeros(n)], axis=1) + lidar_offset
    xy = (pts @ R_level.T)[:, :2]
    origin = (R_level @ np.asarray(lidar_offset, dtype=float))[:2]
    return xy, valid, origin


def estimate_normals(xy, valid, origin, k=2, max_gap=0.3, min_incidence=0.15):
    """Нормали по соседним лучам. Возвращает (normals (N, 2), ok (N,))."""
    a = np.roll(xy, k, axis=0)
    b = np.roll(xy, -k, axis=0)
    ok = valid & np.roll(valid, k) & np.roll(valid, -k)
    t = b - a
    tn = np.linalg.norm(t, axis=1)
    d1 = np.linalg.norm(xy - a, axis=1)
    d2 = np.linalg.norm(b - xy, axis=1)
    ok &= (d1 < max_gap) & (d2 < max_gap) & (tn > 1e-6)
    tn = np.where(tn > 1e-6, tn, 1.0)
    nrm = np.stack([-t[:, 1], t[:, 0]], axis=1) / tn[:, None]
    ray = xy - origin
    rn = np.linalg.norm(ray, axis=1)
    rn = np.where(rn > 1e-6, rn, 1.0)
    inc = np.abs(np.einsum('ij,ij->i', nrm, ray / rn[:, None]))
    ok &= inc > min_incidence
    return nrm, ok


class LocalMap:
    """Объединение нескольких последних ключевых сканов в системе odom."""

    def __init__(self, max_keyframes=8, max_points=360):
        self._frames = deque(maxlen=max_keyframes)
        self.max_points = max_points
        self.pts = None
        self.nrm = None
        self.tree = None

    def empty(self):
        return self.tree is None

    def add(self, pts, nrm):
        if pts.shape[0] > self.max_points:
            sel = np.linspace(0, pts.shape[0] - 1, self.max_points).astype(int)
            pts, nrm = pts[sel], nrm[sel]
        if pts.shape[0] < 10:
            return False
        self._frames.append((pts.copy(), nrm.copy()))
        self.pts = np.concatenate([f[0] for f in self._frames])
        self.nrm = np.concatenate([f[1] for f in self._frames])
        self.tree = cKDTree(self.pts)
        return True


def icp(cur, lmap, pivot, estimate_yaw=False, iters=12, huber=0.03, prior_w=2.0,
        max_dist0=0.5, min_pts=30):
    """Совмещает текущий скан `cur` (N, 2, в odom при предсказанной позе) с картой.

    Возвращает IcpResult с поправкой (translation (2,), yaw) или None при неудаче.
    pivot — положение робота (центр поворота поправки по рысканию).
    """
    tree, mp, mn = lmap.tree, lmap.pts, lmap.nrm
    P = cur.copy()
    t_tot = np.zeros(2)
    psi_tot = 0.0
    for it in range(iters):
        max_d = max(0.12, max_dist0 * 0.65 ** it)
        d, idx = tree.query(P, k=1, distance_upper_bound=max_d)
        m = np.isfinite(d)
        if int(m.sum()) < min_pts:
            return None
        q = mp[idx[m]]
        n = mn[idx[m]]
        p = P[m]
        r = np.einsum('ij,ij->i', n, p - q)
        ar = np.abs(r)
        w = np.where(ar < huber, 1.0, huber / np.maximum(ar, 1e-9))
        if estimate_yaw:
            rel = p - pivot
            jr = np.stack([-rel[:, 1], rel[:, 0]], axis=1)
            a = np.column_stack([n, np.einsum('ij,ij->i', n, jr)])
        else:
            a = n
        A = (a * w[:, None]).T @ a
        b = (a * (w * r)[:, None]).sum(axis=0)
        A[:2, :2] += prior_w * np.eye(2)
        b[:2] += prior_w * t_tot
        A += 1e-9 * np.eye(A.shape[0])
        try:
            delta = -np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        dt = delta[:2]
        psi = float(delta[2]) if estimate_yaw else 0.0
        if estimate_yaw:
            P = pivot + (P - pivot) @ rot2(psi).T + dt
            t_tot = rot2(psi) @ t_tot + dt
            psi_tot += psi
        else:
            P = P + dt
            t_tot = t_tot + dt
        if np.linalg.norm(dt) < 1e-4 and abs(psi) < 1e-5:
            break
    # итоговые показатели качества
    d, idx = tree.query(P, k=1, distance_upper_bound=0.3)
    m = np.isfinite(d)
    n_used = int(m.sum())
    if n_used < min_pts:
        return None
    n = mn[idx[m]]
    r = np.einsum('ij,ij->i', n, P[m] - mp[idx[m]])
    good = np.abs(r) < 0.05
    ratio = float(good.sum()) / float(cur.shape[0])
    rms = float(np.sqrt(np.mean(r[good] ** 2))) if good.any() else 1.0
    info = (n[good].T @ n[good]) if good.any() else np.zeros((2, 2))
    info_min = float(np.linalg.eigvalsh(info)[0])
    return IcpResult(t_tot, psi_tot, ratio, rms, info_min, n_used)
