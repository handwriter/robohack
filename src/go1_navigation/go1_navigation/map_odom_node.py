#!/usr/bin/env python3
"""Публикует TF map -> odom (ТЗ T5, п. 7.2): цепочка map -> odom -> trunk.

Одометрия T4 уже опирается на сканы лидара, поэтому отдельный SLAM не нужен: по умолчанию
map -> odom — тождественное преобразование (старт робота принят за (0, 0, yaw=0)). Если Judge
присылает начальную позу в /initialpose (PoseWithCovarianceStamped, frame map), преобразование
пересчитывается: T_map_odom = T_map_trunk * T_odom_trunk^-1 по последней позе /odom.
"""
import math
import sys

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import TransformBroadcaster

from .frames import map_to_odom, yaw_from_quat

BEST_EFFORT = QoSProfile(depth=50, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
RELIABLE = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                      reliability=ReliabilityPolicy.RELIABLE)


class MapOdomNode(Node):
    def __init__(self):
        super().__init__('go1_map_odom')
        self.declare_parameter('publish_rate', 20.0)
        self.declare_parameter('future_dating', 0.1)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.map_frame = str(g('map_frame'))
        self.odom_frame = str(g('odom_frame'))
        self.future = float(g('future_dating'))
        self.T = (0.0, 0.0, 0.0)           # map -> odom
        self.odom_pose = None              # последняя поза trunk в odom
        self.pending_init = None
        self.tf_pub = TransformBroadcaster(self)
        self.create_subscription(Odometry, '/odom', self.on_odom, BEST_EFFORT)
        self.create_subscription(PoseWithCovarianceStamped, '/initialpose', self.on_init, RELIABLE)
        self.create_timer(1.0 / max(float(g('publish_rate')), 1.0), self.on_timer)
        self.get_logger().info('go1_map_odom: map -> odom = тождественное, пока нет /initialpose')

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        pose = (float(p.x), float(p.y), yaw_from_quat(q.x, q.y, q.z, q.w))
        if all(math.isfinite(v) for v in pose):
            self.odom_pose = pose
            if self.pending_init is not None:
                self._apply_init(self.pending_init)
                self.pending_init = None

    def on_init(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        init = (float(p.x), float(p.y), yaw_from_quat(q.x, q.y, q.z, q.w))
        if not all(math.isfinite(v) for v in init):
            return
        if msg.header.frame_id not in ('', self.map_frame):
            self.get_logger().warn('/initialpose в системе %r, ожидается %r — пропущено'
                                   % (msg.header.frame_id, self.map_frame))
            return
        if self.odom_pose is None:
            self.pending_init = init
        else:
            self._apply_init(init)

    def _apply_init(self, init):
        self.T = map_to_odom(init, self.odom_pose)
        self.get_logger().info('/initialpose принята: map -> odom = (%.2f, %.2f, %.1f град)'
                               % (self.T[0], self.T[1], math.degrees(self.T[2])))

    def on_timer(self):
        now = self.get_clock().now().nanoseconds
        if now <= 0:
            return
        t = now + int(self.future * 1e9)    # метка чуть вперёд: tf2 не экстраполирует в будущее
        tf = TransformStamped()
        tf.header.stamp.sec = int(t // 1_000_000_000)
        tf.header.stamp.nanosec = int(t % 1_000_000_000)
        tf.header.frame_id = self.map_frame
        tf.child_frame_id = self.odom_frame
        tf.transform.translation.x = float(self.T[0])
        tf.transform.translation.y = float(self.T[1])
        tf.transform.rotation.z = math.sin(0.5 * self.T[2])
        tf.transform.rotation.w = math.cos(0.5 * self.T[2])
        self.tf_pub.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = MapOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
