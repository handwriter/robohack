#!/usr/bin/env python3
"""ЛОКАЛЬНЫЙ ИНСТРУМЕНТ (роль Judge): ведёт робота по квадрату, как в ТЗ T4, п. 6.3.

Неподвижная фаза -> 4 x (прямая 0,2 м/с + поворот 0,3 рад/с на 90 град) -> остановка ->
финальная неподвижная фаза. Решение участника /cmd_vel не публикует; этот скрипт нужен
только для локальной проверки и в Docker-образ не входит.

  python3 tools/drive_square.py [--side 3.0] [--still0 5] [--still1 15]
Запускать в контейнере симулятора (или там, где виден ROS-домен) после /go1_ready.
"""
import argparse
import math

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool


class Driver(Node):
    def __init__(self, a):
        super().__init__('t4_square_driver', parameter_overrides=[
            rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        v, w = 0.2, 0.3
        self.plan = [(a.still0, 0.0, 0.0)]
        for _ in range(4):
            self.plan.append((a.side / v, v, 0.0))
            self.plan.append((math.pi / 2 / w, 0.0, w))
        self.plan += [(2.0, 0.0, 0.0), (a.still1, 0.0, 0.0)]
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/go1_ready', self.on_ready, latched)
        self.t_start = None
        self.timer = self.create_timer(0.05, self.tick)

    def on_ready(self, msg):
        if msg.data and self.t_start is None:
            self.t_start = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().info('/go1_ready получен, сценарий начат')

    def tick(self):
        if self.t_start is None:
            return
        t = self.get_clock().now().nanoseconds * 1e-9 - self.t_start
        for dur, v, w in self.plan:
            if t < dur:
                msg = Twist()
                msg.linear.x, msg.angular.z = v, w
                self.pub.publish(msg)
                return
            t -= dur
        self.pub.publish(Twist())
        self.get_logger().info('сценарий завершён')
        self.timer.cancel()
        raise SystemExit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--side', type=float, default=3.0, help='длина стороны квадрата, м')
    ap.add_argument('--still0', type=float, default=5.0)
    ap.add_argument('--still1', type=float, default=15.0)
    args = ap.parse_args()
    rclpy.init()
    node = Driver(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
