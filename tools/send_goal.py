#!/usr/bin/env python3
"""ЛОКАЛЬНЫЙ ИНСТРУМЕНТ (роль Judge, T5): начальная поза + цель Action /navigate_to_pose.

Публикует /initialpose (0, 0, yaw 0), ждёт Action-сервер и отправляет цель в системе map.
Если доступен публичный /odom/ground_truth, по завершении печатает фактическую ошибку цели
по критерию ТЗ (<= 0,5 м и <= 30 град). Решение участника этот скрипт не использует; в
Docker-образ он не входит.

  python3 tools/send_goal.py                   # цель трассы 1 м: (22.7, 0, 0)
  python3 tools/send_goal.py --x 22.7 --y 0.35 # цель трассы 2 м
"""
import argparse
import math
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Client(Node):
    def __init__(self):
        super().__init__('t5_goal_client', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.gt = None
        self.create_subscription(
            Odometry, '/odom/ground_truth', self.on_gt,
            QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                       reliability=ReliabilityPolicy.BEST_EFFORT))
        self.init_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.client = ActionClient(self, NavigateToPose, '/navigate_to_pose')

    def on_gt(self, msg):
        p = msg.pose.pose
        self.gt = (p.position.x, p.position.y, yaw_of(p.orientation))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x', type=float, default=22.7)
    ap.add_argument('--y', type=float, default=0.0)
    ap.add_argument('--yaw', type=float, default=0.0)
    ap.add_argument('--timeout', type=float, default=600.0, help='секунды реального времени')
    args = ap.parse_args()
    rclpy.init()
    node = Client()
    try:
        print('Ждём Action-сервер /navigate_to_pose ...')
        while not node.client.wait_for_server(timeout_sec=1.0):
            rclpy.spin_once(node, timeout_sec=0.1)
        init = PoseWithCovarianceStamped()
        init.header.frame_id = 'map'
        init.pose.pose.orientation.w = 1.0
        node.init_pub.publish(init)
        rclpy.spin_once(node, timeout_sec=0.5)
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x, goal.pose.pose.position.y = args.x, args.y
        goal.pose.pose.orientation.z = math.sin(args.yaw / 2)
        goal.pose.pose.orientation.w = math.cos(args.yaw / 2)
        goal.pose.header.stamp = node.get_clock().now().to_msg()
        fut = node.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, fut)
        handle = fut.result()
        if not handle.accepted:
            print('Цель отклонена')
            return
        print('Цель принята: map (%.2f, %.2f, yaw %.2f)' % (args.x, args.y, args.yaw))
        res = handle.get_result_async()
        t0 = time.monotonic()
        while rclpy.ok() and not res.done() and time.monotonic() - t0 < args.timeout:
            rclpy.spin_once(node, timeout_sec=0.2)
        print('Статус Nav2:', res.result().status if res.done() else 'таймаут')
        if node.gt is not None:
            ex, ey = args.x - node.gt[0], args.y - node.gt[1]
            err, dyaw = math.hypot(ex, ey), abs(wrap(args.yaw - node.gt[2]))
            ok = err <= 0.5 and dyaw <= math.radians(30)
            print('Фактическая ошибка по ground truth: %.2f м, %.1f град -> цель %s'
                  % (err, math.degrees(dyaw), 'ДОСТИГНУТА' if ok else 'не достигнута'))
            print('Внимание: у Judge своя система координат; здесь ground truth считается в map.')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
