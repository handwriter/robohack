#!/usr/bin/env python3
"""ROS 2-узел распознавания ArUco (T5): пять RGB-камер -> /aruco/detections (std_msgs/String).

Публикуется JSON {"ids": [...]} с ТЕКУЩИМИ обнаружениями (виденными в последние ~0,3 с),
пустой список {"ids": []}, если меток нет. Накопленный список не ведётся и перебор
идентификаторов не выполняется (ТЗ T5, п. 7.4). Глубина не используется.
"""
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from .aruco import ArucoDetector, DetectionHub, TemporalConfirmer, format_message, image_to_gray

CAMERAS = ('/camera', '/camera_chin', '/camera_left', '/camera_right', '/camera_rearDown')

# Как у публичных топиков камер (ТЗ, п. 8.4): RELIABLE, VOLATILE, KEEP_LAST.
CAMERA_QOS = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                        reliability=ReliabilityPolicy.RELIABLE,
                        durability=DurabilityPolicy.VOLATILE)
OUT_QOS = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.RELIABLE)


class ArucoNode(Node):
    def __init__(self):
        super().__init__('go1_aruco')
        self.declare_parameter('confirm_frames', 2)
        self.declare_parameter('hold_time', 0.3)
        self.declare_parameter('publish_rate', 10.0)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.detector = ArucoDetector()
        self.hub = DetectionHub(hold=float(g('hold_time')))
        self.confirmers = {c: TemporalConfirmer(confirm=int(g('confirm_frames'))) for c in CAMERAS}
        self.pub = self.create_publisher(String, '/aruco/detections', OUT_QOS)
        for cam in CAMERAS:
            self.create_subscription(Image, cam + '/image_raw',
                                     lambda msg, c=cam: self.on_image(c, msg), CAMERA_QOS)
        self.create_timer(1.0 / max(float(g('publish_rate')), 1.0), self.on_timer)
        self.get_logger().info('go1_aruco запущен: DICT_4X4_50, камеры: %s' % ', '.join(CAMERAS))

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_image(self, cam, msg):
        gray = image_to_gray(msg)
        if gray is None:
            self.get_logger().warn('%s: неподдерживаемый формат изображения %r'
                                   % (cam, msg.encoding), throttle_duration_sec=10.0)
            return
        try:
            found = {d.id for d in self.detector.detect(gray)}
        except Exception as exc:    # распознавание не должно ронять узел
            self.get_logger().warn('%s: ошибка распознавания: %r' % (cam, exc),
                                   throttle_duration_sec=10.0)
            return
        t = self._now()
        confirmed = self.confirmers[cam].update(t, found)
        if confirmed:
            self.hub.update(cam, t, confirmed)

    def on_timer(self):
        t = self._now()
        if t <= 0.0:
            return
        self.pub.publish(String(data=format_message(self.hub.current(t))))


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    sys.exit(main())
