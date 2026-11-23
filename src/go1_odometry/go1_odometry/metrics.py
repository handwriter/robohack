"""Локальный аналог оценки T4: RPE по участкам ~1 м, дрейф за 10 с, частота публикации.

Реализация по разделам 6.2, 6.4 и 6.5 ТЗ. Это не код Judge, а приближение для проверки:
Judge вправе делить путь и совмещать начальные позы иначе.

CSV для командной строки: t, est_x, est_y, est_yaw, gt_x, gt_y, gt_yaw
  python3 -m go1_odometry.metrics run.csv
"""
import math
import sys

import numpy as np

from .geometry import rot2, wrap_angle


def compute_rpe(gt_xy, est_xy, gt_yaw0, est_yaw0, seg_len=1.0):
    """Средняя относительная ошибка перемещения на участках фактического пути ~seg_len.

    Оценённая траектория совмещается с истинной по начальной позе (поворот на
    gt_yaw0 - est_yaw0). Возвращает (средняя_ошибка, число_участков, список_ошибок).
    """
    gt_xy = np.asarray(gt_xy, dtype=float)
    est_xy = np.asarray(est_xy, dtype=float)
    R = rot2(gt_yaw0 - est_yaw0)
    step = np.linalg.norm(np.diff(gt_xy, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(step)])
    n = len(gt_xy)
    errs = []
    i = 0
    while True:
        j = int(np.searchsorted(cum, cum[i] + seg_len))
        if j >= n:
            break
        length = cum[j] - cum[i]
        d_gt = gt_xy[j] - gt_xy[i]
        d_est = R @ (est_xy[j] - est_xy[i])
        errs.append(float(np.linalg.norm(d_gt - d_est) / length))
        i = j
    if not errs:
        return float('nan'), 0, errs
    return float(np.mean(errs)), len(errs), errs


def compute_drift(t, est_xy, est_yaw, window=10.0):
    """Изменение оценённых положения (м) и рыскания (градусы) за последние `window` секунд."""
    t = np.asarray(t, dtype=float)
    i0 = int(np.searchsorted(t, t[-1] - window))
    dpos = float(np.linalg.norm(np.asarray(est_xy)[-1] - np.asarray(est_xy)[i0]))
    dyaw = abs(wrap_angle(float(est_yaw[-1]) - float(est_yaw[i0])))
    return dpos, math.degrees(dyaw)


def compute_rate(t, window=5.0):
    """(средняя частота, минимальная частота в окне `window`, максимальный интервал) по меткам t."""
    t = np.asarray(t, dtype=float)
    if len(t) < 2:
        return 0.0, 0.0, float('inf')
    gaps = np.diff(t)
    mean_rate = (len(t) - 1) / (t[-1] - t[0])
    min_rate = mean_rate
    if t[-1] - t[0] >= window:
        starts = np.arange(t[0], t[-1] - window + 1e-9, 0.5)
        counts = [np.searchsorted(t, s + window) - np.searchsorted(t, s) for s in starts]
        min_rate = float(min(counts)) / window
    return float(mean_rate), float(min_rate), float(gaps.max())


def score(rpe, drift_pos, drift_yaw_deg, min_rate):
    """Баллы T4 по таблице 6.5 (часть критериев — по локальным оценкам)."""
    pts = 0
    if min_rate >= 20.0:
        pts += 3
    if rpe <= 0.50:
        pts += 5
    if rpe <= 0.20:
        pts += 6
    if drift_pos <= 0.10 and drift_yaw_deg <= 5.0:
        pts += 6
    return pts


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    d = np.loadtxt(argv[1], delimiter=',', skiprows=1)
    t, ex, ey, eyaw, gx, gy, gyaw = d.T
    rpe, n, _ = compute_rpe(np.c_[gx, gy], np.c_[ex, ey], gyaw[0], eyaw[0])
    dpos, dyaw = compute_drift(t, np.c_[ex, ey], eyaw)
    mean_rate, min_rate, max_gap = compute_rate(t)
    print('Частота /odom: средняя %.1f Гц, минимум за 5 с %.1f Гц, макс. интервал %.0f мс'
          % (mean_rate, min_rate, 1000 * max_gap))
    print('RPE: %.1f %% по %d участкам' % (100 * rpe, n))
    print('Дрейф за последние 10 с: %.3f м, %.2f град' % (dpos, dyaw))
    print('Баллы (локальная оценка, из 20): %d' % score(rpe, dpos, dyaw, min_rate))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
