"""Проводка ROS-узла (node.py) на заглушках rclpy/сообщений — без установленного ROS.

Проверяет то, что можно проверить без Gazebo: имена топиков и типов, QoS, frame_id,
метки времени, частоту, отсутствие NaN/Inf, согласованность /odom и TF, отсутствие
публикаций в /cmd_vel и подписок на /odom/ground_truth. Настоящий прогон в симуляторе
этим тестом НЕ заменяется.
"""
import sys
import types

import numpy as np
import pytest

import synth
from go1_odometry import kinematics as K


# ------------------------------------------------------------------ заглушки ROS
class Ns(types.SimpleNamespace):
    pass


def make_msg_class(fields):
    class M:
        def __init__(self, **kw):
            for k, v in fields().items():
                setattr(self, k, v)
            for k, v in kw.items():
                setattr(self, k, v)
    return M


def vec3():
    return Ns(x=0.0, y=0.0, z=0.0)


def quat():
    return Ns(x=0.0, y=0.0, z=0.0, w=1.0)


def header():
    return Ns(stamp=Ns(sec=0, nanosec=0), frame_id='')


class StubNode:
    registry = []

    def __init__(self, name):
        self.params, self.pubs, self.subs = {}, {}, {}
        self._clock_ns = 0
        self.logs = []
        StubNode.registry.append(self)

    def declare_parameter(self, name, value):
        self.params[name] = value

    def get_parameter(self, name):
        return Ns(value=self.params[name])

    def create_publisher(self, msg_type, topic, qos):
        pub = Ns(msg_type=msg_type, topic=topic, qos=qos, sent=[])
        pub.publish = pub.sent.append
        self.pubs[topic] = pub
        return pub

    def create_subscription(self, msg_type, topic, cb, qos):
        self.subs[topic] = Ns(msg_type=msg_type, cb=cb, qos=qos)

    def get_logger(self):
        node = self
        return Ns(info=lambda m, **k: node.logs.append(('info', m)),
                  warn=lambda m, **k: node.logs.append(('warn', m)),
                  error=lambda m, **k: node.logs.append(('error', m)))

    def get_clock(self):
        node = self
        return Ns(now=lambda: Ns(nanoseconds=node._clock_ns))

    def destroy_node(self):
        pass


class StubTfBroadcaster:
    def __init__(self, node):
        self.sent = []

    def sendTransform(self, tf):
        self.sent.append(tf)


class QoSProfile:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def install_stubs():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    enum = lambda *names: Ns(**{n: n for n in names})  # noqa: E731
    mod('rclpy', init=lambda args=None: None, spin=lambda n: None, try_shutdown=lambda: None)
    mod('rclpy.node', Node=StubNode)
    mod('rclpy.qos', QoSProfile=QoSProfile,
        DurabilityPolicy=enum('TRANSIENT_LOCAL', 'VOLATILE'),
        HistoryPolicy=enum('KEEP_LAST'), ReliabilityPolicy=enum('RELIABLE', 'BEST_EFFORT'))
    mod('geometry_msgs')
    mod('geometry_msgs.msg', TransformStamped=make_msg_class(
        lambda: dict(header=header(), child_frame_id='',
                     transform=Ns(translation=vec3(), rotation=quat()))))
    mod('nav_msgs')
    mod('nav_msgs.msg', Odometry=make_msg_class(
        lambda: dict(header=header(), child_frame_id='',
                     pose=Ns(pose=Ns(position=vec3(), orientation=quat()), covariance=[0.0] * 36),
                     twist=Ns(twist=Ns(linear=vec3(), angular=vec3()), covariance=[0.0] * 36))))
    mod('sensor_msgs')
    mod('sensor_msgs.msg',
        Imu=make_msg_class(lambda: dict(header=header(), orientation=quat(),
                                        orientation_covariance=[0.0] * 9,
                                        angular_velocity=vec3(), linear_acceleration=vec3())),
        JointState=make_msg_class(lambda: dict(header=header(), name=[], position=[], velocity=[])),
        LaserScan=make_msg_class(lambda: dict(header=header(), angle_min=0.0, angle_increment=0.0,
                                              ranges=[])))
    mod('std_msgs')
    mod('std_msgs.msg', Bool=make_msg_class(lambda: dict(data=False)))
    mod('tf2_ros', TransformBroadcaster=StubTfBroadcaster)


@pytest.fixture(scope='module')
def node_module():
    saved = {k: v for k, v in sys.modules.items()}
    install_stubs()
    sys.modules.pop('go1_odometry.node', None)
    import go1_odometry.node as node_mod
    yield node_mod
    sys.modules.pop('go1_odometry.node', None)
    for k in list(sys.modules):
        if k not in saved and k.split('.')[0] in ('rclpy', 'geometry_msgs', 'nav_msgs',
                                                  'sensor_msgs', 'std_msgs', 'tf2_ros'):
            sys.modules.pop(k)


def stamp_of(t):
    sec = int(t)
    return Ns(sec=sec, nanosec=int(round((t - sec) * 1e9)))


def feed(node_mod, sc, t0=50.0, with_ready=True):
    """Подаёт в узел поток датчиков сценария через его реальные колбэки."""
    from sensor_msgs.msg import Imu, JointState, LaserScan
    from std_msgs.msg import Bool
    node = node_mod.OdometryNode()
    if with_ready:
        node.subs['/go1_ready'].cb(Bool(data=True))
    room = synth.Room()
    rng = np.random.default_rng(1)
    q = sc.joint_angles()
    qd = np.gradient(q, synth.DT, axis=0)
    quats, gyro = sc.imu()
    for k in range(sc.n):
        t = t0 + sc.t[k]
        js = JointState()
        js.header.stamp = stamp_of(t)
        js.name = list(K.JOINT_NAMES)
        js.position = q[k].reshape(-1).tolist()
        js.velocity = qd[k].reshape(-1).tolist()
        node.subs['/joint_states'].cb(js)
        if k % 10 == 0:
            r, amin, ainc = sc.scan(k, room, rng)
            ls = LaserScan()
            ls.header.stamp = stamp_of(t)
            ls.ranges = [float(x) for x in r]
            ls.angle_min, ls.angle_increment = amin, ainc
            node.subs['/scan'].cb(ls)
        imu = Imu()
        imu.header.stamp = stamp_of(t)
        imu.orientation = Ns(x=quats[k][0], y=quats[k][1], z=quats[k][2], w=quats[k][3])
        imu.angular_velocity = Ns(x=gyro[k][0], y=gyro[k][1], z=gyro[k][2])
        imu.linear_acceleration = Ns(x=0.0, y=0.0, z=9.81)
        node.subs['/imu/data'].cb(imu)
    return node


@pytest.fixture(scope='module')
def run(node_module):
    sc = synth.Scenario(yaw_start=1.0, side=1.5, t_still0=3.0, t_still1=12.0)
    node = feed(node_module, sc)
    return sc, node


def test_topics_types_and_no_forbidden_interfaces(run):
    _, node = run
    assert set(node.subs) == {'/imu/data', '/joint_states', '/scan', '/go1_ready'}
    assert set(node.pubs) == {'/odom'}                       # нет /cmd_vel и прочего
    assert 'cmd_vel' not in ' '.join(node.pubs) and 'ground_truth' not in ' '.join(node.subs)
    assert node.pubs['/odom'].qos.reliability == 'RELIABLE'  # совместимо с любым подписчиком
    for t in ('/imu/data', '/joint_states', '/scan'):
        assert node.subs[t].qos.reliability == 'BEST_EFFORT'   # совместимо с любым издателем


def test_odom_fields_rate_and_finiteness(run):
    sc, node = run
    msgs = node.pubs['/odom'].sent
    assert len(msgs) > 100
    t = np.array([m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec for m in msgs])
    assert np.all(np.diff(t) > 0)                            # метки возрастают
    gaps = np.diff(t)
    assert gaps.max() < 0.05                                 # частота не ниже 20 Гц везде
    assert (len(t) - 1) / (t[-1] - t[0]) >= 20.0
    for m in msgs:
        assert m.header.frame_id == 'odom' and m.child_frame_id == 'trunk'
        p, o = m.pose.pose.position, m.pose.pose.orientation
        vals = [p.x, p.y, p.z, o.x, o.y, o.z, o.w, m.twist.twist.linear.x, m.twist.twist.linear.y,
                m.twist.twist.angular.z]
        assert np.all(np.isfinite(vals))
        assert abs(np.linalg.norm([o.x, o.y, o.z, o.w]) - 1.0) < 1e-6
        assert np.all(np.isfinite(m.pose.covariance)) and np.all(np.isfinite(m.twist.covariance))


def test_tf_matches_odom(run):
    _, node = run
    odoms = node.pubs['/odom'].sent
    tfs = node.tf_pub.sent
    assert len(tfs) == len(odoms)
    for m, tf in zip(odoms[::50], tfs[::50]):
        assert (tf.header.stamp.sec, tf.header.stamp.nanosec) == \
            (m.header.stamp.sec, m.header.stamp.nanosec)
        assert tf.header.frame_id == 'odom' and tf.child_frame_id == 'trunk'
        a, b = tf.transform.translation, m.pose.pose.position
        assert (a.x, a.y, a.z) == (b.x, b.y, b.z)
        a, b = tf.transform.rotation, m.pose.pose.orientation
        assert (a.x, a.y, a.z, a.w) == (b.x, b.y, b.z, b.w)


def test_odom_accuracy_through_node(run):
    from go1_odometry import metrics
    from go1_odometry.geometry import quat_to_R, yaw_of_R
    sc, node = run
    msgs = node.pubs['/odom'].sent
    t = np.array([m.header.stamp.sec + 1e-9 * m.header.stamp.nanosec for m in msgs]) - 50.0
    est_xy = np.array([[m.pose.pose.position.x, m.pose.pose.position.y] for m in msgs])
    est_yaw = np.array([yaw_of_R(quat_to_R([m.pose.pose.orientation.x, m.pose.pose.orientation.y,
                                            m.pose.pose.orientation.z, m.pose.pose.orientation.w]))
                        for m in msgs])
    idx = np.clip(np.round(t / synth.DT).astype(int), 0, sc.n - 1)
    gt_xy = np.c_[sc.x[idx], sc.y[idx]]
    rpe, n, _ = metrics.compute_rpe(gt_xy, est_xy, sc.yaw[idx][0], est_yaw[0])
    dpos, dyaw = metrics.compute_drift(t, est_xy, est_yaw)
    assert n >= 3 and rpe <= 0.20
    assert dpos <= 0.10 and dyaw <= 5.0


def test_ready_resets_origin_once(node_module):
    from std_msgs.msg import Bool
    node = node_module.OdometryNode()
    node.est.pos[:] = [5.0, 5.0, 0.3]
    node.subs['/go1_ready'].cb(Bool(data=True))
    assert np.allclose(node.est.pos, 0.0)
    node.est.pos[:] = [1.0, 2.0, 0.3]
    node.subs['/go1_ready'].cb(Bool(data=True))              # повтор не сбрасывает
    assert np.allclose(node.est.pos, [1.0, 2.0, 0.3])


def test_invalid_estimate_is_never_published(node_module):
    from sensor_msgs.msg import Imu
    node = node_module.OdometryNode()
    node.est.on_imu(1.0, [0, 0, 0, 1], [0, 0, 0])
    node.est.pos[0] = float('nan')                          # принудительно портим состояние
    imu = Imu()
    imu.header.stamp = stamp_of(1.01)
    node.subs['/imu/data'].cb(imu)
    assert node.pubs['/odom'].sent == [] and node.tf_pub.sent == []
