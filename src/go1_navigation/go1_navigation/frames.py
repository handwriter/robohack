"""Плоские преобразования SE(2) для цепочки TF map -> odom -> trunk (без ROS)."""
import math


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def compose(a, b):
    """a * b для поз (x, y, yaw): сначала b в системе a."""
    c, s = math.cos(a[2]), math.sin(a[2])
    return (a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2]))


def inverse(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return (-(c * a[0] + s * a[1]), -(-s * a[0] + c * a[1]), wrap(-a[2]))


def map_to_odom(pose_in_map, pose_in_odom):
    """map -> odom по позе робота, заданной в map (инициализация Judge), и текущей позе в odom.

    T_map_odom = T_map_trunk * T_odom_trunk^-1.
    """
    return compose(pose_in_map, inverse(pose_in_odom))
