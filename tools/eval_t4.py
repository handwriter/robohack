#!/usr/bin/env python3
"""ЛОКАЛЬНЫЙ ИНСТРУМЕНТ: сравнивает /odom участника с публичным /odom/ground_truth.

Пишет CSV (t, est_x, est_y, est_yaw, gt_x, gt_y, gt_yaw) и по Ctrl+C печатает RPE по участкам
~1 м, дрейф за последние 10 с, частоту /odom и локальную оценку баллов (приближение
критериев ТЗ T4 п. 6.4-6.5; официальная оценка закрыта). /odom/ground_truth есть только в
публичной симуляции, решение его не использует.

  python3 tools/eval_t4.py --csv run.csv        # затем: python3 -m go1_odometry.metrics run.csv
"""
import argparse
import bisect
import math
import os
import sys

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'go1_odometry'))
from go1_odometry import metrics  # noqa: E402


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp(msg):
    return msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec


class Eval(Node):
    def __init__(self):
        super().__init__('t4_eval', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        qos = QoSProfile(depth=500, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.gt_t, self.gt = [], []
        self.rows = []
        self.create_subscription(Odometry, '/odom/ground_truth', self.on_gt, qos)
        self.create_subscription(Odometry, '/odom', self.on_est, qos)

    def on_gt(self, m):
        p = m.pose.pose
        self.gt_t.append(stamp(m))
        self.gt.append((p.position.x, p.position.y, yaw_of(p.orientation)))

    def on_est(self, m):
        t = stamp(m)
        i = bisect.bisect_right(self.gt_t, t) - 1
        if i < 0:
            return
        p = m.pose.pose
        g = self.gt[i]
        self.rows.append((t, p.position.x, p.position.y, yaw_of(p.orientation), g[0], g[1], g[2]))


def report(rows, csv_path):
    d = np.array(rows)
    np.savetxt(csv_path, d, delimiter=',', header='t,est_x,est_y,est_yaw,gt_x,gt_y,gt_yaw',
               comments='')
    t, ex, ey, eyaw, gx, gy, gyaw = d.T
    rpe, n, _ = metrics.compute_rpe(np.c_[gx, gy], np.c_[ex, ey], gyaw[0], eyaw[0])
    dpos, dyaw = metrics.compute_drift(t, np.c_[ex, ey], eyaw)
    mean_rate, min_rate, max_gap = metrics.compute_rate(t)
    print('\nСообщений /odom: %d, файл: %s' % (len(d), csv_path))
    print('Частота: средняя %.1f Гц, минимум за 5 с %.1f Гц, макс. интервал %.0f мс'
          % (mean_rate, min_rate, 1000 * max_gap))
    print('RPE: %.1f %% по %d участкам' % (100 * rpe, n))
    print('Дрейф за последние 10 с: %.3f м, %.2f град' % (dpos, dyaw))
    print('Локальная оценка баллов: %d из 20' % metrics.score(rpe, dpos, dyaw, min_rate))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='t4_run.csv')
    args = ap.parse_args()
    rclpy.init()
    node = Eval()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if len(node.rows) > 20:
            report(node.rows, args.csv)
        else:
            print('Слишком мало данных: проверьте, что /odom и /odom/ground_truth публикуются')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
