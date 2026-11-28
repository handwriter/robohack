"""Проводка aruco_node и map_odom_node на заглушках ROS (без установленного ROS)."""
import json
import math
import sys

import pytest

import marker_scene
import test_node as stubs


@pytest.fixture(scope='module')
def modules():
    saved = dict(sys.modules)
    stubs.install_stubs()
    for name in ('go1_navigation.aruco_node', 'go1_navigation.map_odom_node'):
        sys.modules.pop(name, None)
    import go1_navigation.aruco_node as aruco_node
    import go1_navigation.map_odom_node as map_odom_node
    yield aruco_node, map_odom_node
    for name in list(sys.modules):
        if name not in saved and name.split('.')[0] in (
                'rclpy', 'geometry_msgs', 'nav_msgs', 'sensor_msgs', 'std_msgs', 'tf2_ros'):
            sys.modules.pop(name)
    sys.modules.pop('go1_navigation.aruco_node', None)
    sys.modules.pop('go1_navigation.map_odom_node', None)


def image_msg(marker_id, camera, t_ns):
    from sensor_msgs.msg import Image
    gray = marker_scene.render(marker_id, 1.0, 0, camera) if marker_id is not None else \
        marker_scene.render(0, 50.0, 0, camera)           # метка слишком далеко — кадр пустой
    m = marker_scene.to_rgb8_message(gray, Image)
    m.header.stamp.sec, m.header.stamp.nanosec = divmod(t_ns, 1_000_000_000)
    return m


def published_ids(node):
    return [json.loads(m.data)['ids'] for m in node.pubs['/aruco/detections'].sent]


# ------------------------------------------------------------------ ArUco
def test_aruco_topics_and_qos(modules):
    aruco_node, _ = modules
    node = aruco_node.ArucoNode()
    expected = {c + '/image_raw' for c in ('/camera', '/camera_chin', '/camera_left',
                                          '/camera_right', '/camera_rearDown')}
    assert set(node.subs) == expected                       # глубина не используется
    assert set(node.pubs) == {'/aruco/detections'}
    for s in node.subs.values():
        assert (s.qos.reliability, s.qos.durability, s.qos.depth) == ('RELIABLE', 'VOLATILE', 1)
    assert node.pubs['/aruco/detections'].msg_type.__name__ == 'M'


def test_aruco_publishes_only_current_confirmed_ids(modules):
    aruco_node, _ = modules
    node = aruco_node.ArucoNode()
    tick = node.timers[0].cb
    sub = node.subs['/camera_left/image_raw'].cb
    t = 100_000_000_000

    def at(dt_ms):
        node._clock_ns = t + dt_ms * 1_000_000
        return node._clock_ns

    at(0)
    tick()
    assert published_ids(node)[-1] == []                    # пустой список, если меток нет
    sub(image_msg(7, 'side', at(100)))                      # один кадр — ещё не подтверждено
    tick()
    assert published_ids(node)[-1] == []
    sub(image_msg(7, 'side', at(200)))                      # второй кадр — подтверждено
    tick()
    assert published_ids(node)[-1] == [7]
    at(300)
    tick()
    assert published_ids(node)[-1] == [7]                   # ещё «текущее»
    at(900)
    tick()
    assert published_ids(node)[-1] == []                    # метка пропала — исчезла из сообщения


def test_aruco_no_id_without_marker_in_view(modules):
    aruco_node, _ = modules
    node = aruco_node.ArucoNode()
    tick = node.timers[0].cb
    for cam in node.subs:
        for k in range(6):
            node._clock_ns = 50_000_000_000 + k * 100_000_000
            node.subs[cam].cb(image_msg(None, 'front', node._clock_ns))
            tick()
    assert all(ids == [] for ids in published_ids(node))    # никакого перебора идентификаторов


def test_aruco_bad_image_is_ignored(modules):
    aruco_node, _ = modules
    node = aruco_node.ArucoNode()
    from sensor_msgs.msg import Image
    m = Image()
    m.height, m.width, m.encoding, m.step, m.data = 4, 4, '32FC1', 16, b'\x00' * 64
    node.subs['/camera/image_raw'].cb(m)                    # не должно бросать исключений
    node._clock_ns = 1_000_000_000
    node.timers[0].cb()
    assert published_ids(node) == [[]]


# ------------------------------------------------------------------ map -> odom
def odom_msg(x, y, yaw):
    from nav_msgs.msg import Odometry
    m = Odometry()
    m.pose.pose.position.x, m.pose.pose.position.y = x, y
    m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return m


def init_msg(x, y, yaw, frame='map'):
    from geometry_msgs.msg import PoseWithCovarianceStamped
    m = PoseWithCovarianceStamped()
    m.header.frame_id = frame
    m.pose.pose.position.x, m.pose.pose.position.y = x, y
    m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return m


def tf_yaw(tf):
    return 2.0 * math.atan2(tf.transform.rotation.z, tf.transform.rotation.w)


def test_map_odom_identity_by_default_with_future_stamp(modules):
    _, map_odom = modules
    node = map_odom.MapOdomNode()
    node.timers[0].cb()
    assert node.tf_pub.sent == []                           # часы ещё не идут — не публикуем
    node._clock_ns = 10_000_000_000
    node.timers[0].cb()
    tf = node.tf_pub.sent[-1]
    assert (tf.header.frame_id, tf.child_frame_id) == ('map', 'odom')
    assert (tf.transform.translation.x, tf.transform.translation.y, tf_yaw(tf)) == (0.0, 0.0, 0.0)
    assert tf.header.stamp.sec == 10 and tf.header.stamp.nanosec == 100_000_000   # now + 0.1 с


def test_map_odom_applies_initialpose(modules):
    _, map_odom = modules
    node = map_odom.MapOdomNode()
    node._clock_ns = 5_000_000_000
    node.subs['/odom'].cb(odom_msg(0.4, 0.0, 0.0))
    node.subs['/initialpose'].cb(init_msg(2.0, 1.0, math.pi / 2))
    node.timers[0].cb()
    tf = node.tf_pub.sent[-1]
    # проверка: T_map_odom * T_odom_trunk == заданная поза в map
    c, s = math.cos(tf_yaw(tf)), math.sin(tf_yaw(tf))
    x = tf.transform.translation.x + c * 0.4
    y = tf.transform.translation.y + s * 0.4
    assert abs(x - 2.0) < 1e-6 and abs(y - 1.0) < 1e-6 and abs(tf_yaw(tf) - math.pi / 2) < 1e-6


def test_map_odom_initialpose_before_first_odom_is_kept(modules):
    _, map_odom = modules
    node = map_odom.MapOdomNode()
    node._clock_ns = 5_000_000_000
    node.subs['/initialpose'].cb(init_msg(1.0, 0.0, 0.0))
    node.timers[0].cb()
    assert node.tf_pub.sent[-1].transform.translation.x == 0.0      # пока одометрии нет — тождественно
    node.subs['/odom'].cb(odom_msg(0.0, 0.0, 0.0))
    node.timers[0].cb()
    assert abs(node.tf_pub.sent[-1].transform.translation.x - 1.0) < 1e-9


def test_map_odom_ignores_bad_initialpose(modules):
    _, map_odom = modules
    node = map_odom.MapOdomNode()
    node._clock_ns = 5_000_000_000
    node.subs['/odom'].cb(odom_msg(0.0, 0.0, 0.0))
    node.subs['/initialpose'].cb(init_msg(float('nan'), 0.0, 0.0))
    node.subs['/initialpose'].cb(init_msg(3.0, 0.0, 0.0, frame='odom'))
    node.timers[0].cb()
    assert node.tf_pub.sent[-1].transform.translation.x == 0.0
