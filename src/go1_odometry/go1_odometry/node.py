#!/usr/bin/env python3
"""ROS 2-узел одометрии Go1 (задание T4).

Подписки (только публичные датчики): /imu/data, /joint_states, /scan, /go1_ready.
Публикации: /odom (nav_msgs/Odometry, frame_id=odom, child_frame_id=trunk) и TF odom -> trunk.
Узел НЕ публикует /cmd_vel и НЕ использует /odom/ground_truth (в официальных
испытаниях его нет). Метки времени берутся из сообщений датчиков (симуляционное время).
"""
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy)
from sensor_msgs.msg import Imu, JointState, LaserScan
from std_msgs.msg import Bool
from tf2_ros import TransformBroadcaster

from .estimator import OdometryEstimator, Params

BEST_EFFORT = QoSProfile(depth=500, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
BEST_EFFORT_SCAN = QoSProfile(depth=20, history=HistoryPolicy.KEEP_LAST,
                              reliability=ReliabilityPolicy.BEST_EFFORT)
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)
OUT_QOS = QoSProfile(depth=100, history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.RELIABLE)


def stamp_seconds(stamp):
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


class OdometryNode(Node):
    def __init__(self):
        super().__init__('go1_odometry')
        d = Params()
        self.declare_parameter('use_lidar', d.use_lidar)
        self.declare_parameter('publish_rate', 100.0)
        self.declare_parameter('reset_on_ready', True)
        self.declare_parameter('leg_scale', d.leg_scale)
        self.declare_parameter('lidar_gain', d.lidar_gain)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'trunk')
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        params = Params(use_lidar=bool(g('use_lidar')), leg_scale=float(g('leg_scale')),
                        lidar_gain=float(g('lidar_gain')))
        self.est = OdometryEstimator(params)
        self.odom_frame = str(g('odom_frame'))
        self.base_frame = str(g('base_frame'))
        self.period = 1.0 / max(float(g('publish_rate')), 25.0)   # не реже 25 Гц
        self.reset_on_ready = bool(g('reset_on_ready'))
        self._ready_seen = False
        self._last_pub_t = None
        self._last_good = None
        self._bad_count = 0
        self._published = 0
        self._log_t = None

        self.odom_pub = self.create_publisher(Odometry, '/odom', OUT_QOS)
        self.tf_pub = TransformBroadcaster(self)
        self.create_subscription(Imu, '/imu/data', self.on_imu, BEST_EFFORT)
        self.create_subscription(JointState, '/joint_states', self.on_joints, BEST_EFFORT)
        if params.use_lidar:
            self.create_subscription(LaserScan, '/scan', self.on_scan, BEST_EFFORT_SCAN)
        self.create_subscription(Bool, '/go1_ready', self.on_ready, LATCHED)
        self.get_logger().info('go1_odometry запущен (лидар: %s); ждём /imu/data и /joint_states'
                               % ('вкл' if params.use_lidar else 'выкл'))

    # ------------------------------------------------------------------ входы
    def _time(self, msg):
        """Время сообщения датчика (сек) по header.stamp; при нулевой метке — часы узла."""
        t = stamp_seconds(msg.header.stamp)
        if t <= 0.0:
            t = self.get_clock().now().nanoseconds * 1e-9
        return t

    def on_ready(self, msg):
        """Симулятор закончил подъём робота: начинаем оценку заново с неподвижной позы."""
        if msg.data and not self._ready_seen:
            self._ready_seen = True
            if self.reset_on_ready:
                self.est.reset()
                self._last_pub_t = None
                self._last_good = None
                self.get_logger().info('/go1_ready: оценка сброшена, начало координат odom здесь')

    def on_joints(self, msg):
        self.est.on_joint_state(self._time(msg), list(msg.name), list(msg.position),
                                list(msg.velocity) if len(msg.velocity) else None)

    def on_scan(self, msg):
        try:
            self.est.on_scan(self._time(msg), np.asarray(msg.ranges, dtype=float),
                             float(msg.angle_min), float(msg.angle_increment))
        except Exception as exc:  # лидар не должен ронять одометрию
            self.get_logger().warn('ошибка обработки скана: %r' % (exc,), throttle_duration_sec=5.0)

    def on_imu(self, msg):
        o = msg.orientation
        quat = None
        if not (len(msg.orientation_covariance) and msg.orientation_covariance[0] < 0.0):
            quat = [o.x, o.y, o.z, o.w]
        gyro = [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        acc = [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z]
        t = self._time(msg)
        try:
            ok = self.est.on_imu(t, quat, gyro, acc)
        except Exception as exc:
            self.get_logger().error('ошибка оценщика: %r' % (exc,), throttle_duration_sec=5.0)
            return
        if not ok or self.est.out is None:
            return
        if self._last_pub_t is not None and t - self._last_pub_t < self.period - 0.002:
            return
        self._publish(msg.header.stamp, t)

    # ---------------------------------------------------------------- выходы
    @staticmethod
    def _valid(out):
        p, q = out.position, out.quaternion
        return bool(np.all(np.isfinite(p)) and np.all(np.isfinite(q))
                    and np.all(np.isfinite(out.v_body)) and np.all(np.isfinite(out.omega_body))
                    and abs(float(np.linalg.norm(q)) - 1.0) < 1e-3)

    def _publish(self, stamp, t):
        out = self.est.out
        if not self._valid(out):
            self._bad_count += 1
            self.get_logger().error('некорректная оценка (NaN/Inf/кватернион), сообщение пропущено',
                                    throttle_duration_sec=2.0)
            if self._last_good is not None:      # возвращаем последнее корректное состояние
                self.est.pos[:] = self._last_good
                self.est.v_f[:] = 0.0
            return
        self._last_good = out.position.copy()
        self._last_pub_t = t
        p, q, v, w = out.position, out.quaternion, out.v_body, out.omega_body

        od = Odometry()
        od.header.stamp = stamp
        od.header.frame_id = self.odom_frame
        od.child_frame_id = self.base_frame
        od.pose.pose.position.x = float(p[0])
        od.pose.pose.position.y = float(p[1])
        od.pose.pose.position.z = float(p[2])
        od.pose.pose.orientation.x = float(q[0])
        od.pose.pose.orientation.y = float(q[1])
        od.pose.pose.orientation.z = float(q[2])
        od.pose.pose.orientation.w = float(q[3])
        od.twist.twist.linear.x = float(v[0])
        od.twist.twist.linear.y = float(v[1])
        od.twist.twist.linear.z = float(v[2])
        od.twist.twist.angular.x = float(w[0])
        od.twist.twist.angular.y = float(w[1])
        od.twist.twist.angular.z = float(w[2])
        pc = [0.0] * 36
        tc = [0.0] * 36
        for i, var in enumerate((0.01, 0.01, 0.01, 0.01, 0.01, 0.02)):
            pc[7 * i] = var
        for i, var in enumerate((0.01, 0.01, 0.01, 0.01, 0.01, 0.01)):
            tc[7 * i] = var
        od.pose.covariance = pc
        od.twist.covariance = tc
        self.odom_pub.publish(od)

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self.odom_frame
        tf.child_frame_id = self.base_frame
        tf.transform.translation.x = float(p[0])
        tf.transform.translation.y = float(p[1])
        tf.transform.translation.z = float(p[2])
        tf.transform.rotation.x = float(q[0])
        tf.transform.rotation.y = float(q[1])
        tf.transform.rotation.z = float(q[2])
        tf.transform.rotation.w = float(q[3])
        self.tf_pub.sendTransform(tf)

        self._published += 1
        if self._log_t is None:
            self._log_t = t
            self.get_logger().info('первое сообщение /odom опубликовано')
        elif t - self._log_t >= 10.0:
            self._log_t = t
            s = self.est.stats
            self.get_logger().info(
                'odom=(%.2f, %.2f) стоит=%s сканы принято/отклонено=%d/%d опорных_мин=%d'
                % (p[0], p[1], out.stationary, s['scan_accepted'], s['scan_rejected'],
                   s['stance_min']))


def main(args=None):
    rclpy.init(args=args)
    node = OdometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
