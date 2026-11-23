"""Прямая кинематика ног Unitree Go1 (числа из go1_description/urdf, чистый numpy).

Порядок ног как в ТЗ: FR, FL, RR, RL; для каждой ноги суставы hip, thigh, calf.
Система координат: trunk (x вперёд, y влево, z вверх). Стопа — центр сферы радиуса 0.02.

Цепочка одной ноги по URDF:
  trunk --(±LEG_X, ±LEG_Y, 0)--> hip (ось X) --(0, ±HIP_OFFSET, 0)--> thigh (ось Y)
  --(0, 0, -L1)--> calf (ось Y) --(0, 0, -L2)--> foot
"""
import numpy as np

LEGS = ('FR', 'FL', 'RR', 'RL')
JOINT_TYPES = ('hip', 'thigh', 'calf')
JOINT_NAMES = tuple('%s_%s_joint' % (leg, j) for leg in LEGS for j in JOINT_TYPES)

LEG_X = 0.1881
LEG_Y = 0.04675
HIP_OFFSET = 0.08
L1 = 0.213
L2 = 0.213
FOOT_RADIUS = 0.02

# Положение оси hip каждой ноги в trunk и знак бокового смещения (mirror в URDF).
HIP_POS = np.array([
    [LEG_X, -LEG_Y, 0.0],    # FR
    [LEG_X, LEG_Y, 0.0],     # FL
    [-LEG_X, -LEG_Y, 0.0],   # RR
    [-LEG_X, LEG_Y, 0.0],    # RL
])
SIDE = np.array([-1.0, 1.0, -1.0, 1.0])


def foot_positions(q):
    """q: массив (4, 3) углов [hip, thigh, calf]. Возвращает (4, 3) — стопы в trunk."""
    q = np.asarray(q, dtype=float)
    th, tt, tc = q[:, 0], q[:, 1], q[:, 2]
    y0 = SIDE * HIP_OFFSET
    x = -L1 * np.sin(tt) - L2 * np.sin(tt + tc)
    z = -L1 * np.cos(tt) - L2 * np.cos(tt + tc)
    ch, sh = np.cos(th), np.sin(th)
    y_out = y0 * ch - z * sh
    z_out = y0 * sh + z * ch
    return HIP_POS + np.stack([x, y_out, z_out], axis=1)


def foot_velocities(q, qd):
    """Скорость стоп в осях trunk (4, 3) по углам q и скоростям суставов qd, оба (4, 3)."""
    q = np.asarray(q, dtype=float)
    qd = np.asarray(qd, dtype=float)
    th, tt, tc = q[:, 0], q[:, 1], q[:, 2]
    y0 = SIDE * HIP_OFFSET
    z = -L1 * np.cos(tt) - L2 * np.cos(tt + tc)
    # производные плоской части по thigh и calf
    dx_t = -L1 * np.cos(tt) - L2 * np.cos(tt + tc)
    dx_c = -L2 * np.cos(tt + tc)
    dz_t = L1 * np.sin(tt) + L2 * np.sin(tt + tc)
    dz_c = L2 * np.sin(tt + tc)
    ch, sh = np.cos(th), np.sin(th)
    # d/d hip
    dy_h = -y0 * sh - z * ch
    dz_h = y0 * ch - z * sh
    vx = dx_t * qd[:, 1] + dx_c * qd[:, 2]
    vy = dy_h * qd[:, 0] - sh * (dz_t * qd[:, 1] + dz_c * qd[:, 2])
    vz = dz_h * qd[:, 0] + ch * (dz_t * qd[:, 1] + dz_c * qd[:, 2])
    return np.stack([vx, vy, vz], axis=1)


def joints_to_array(names, values):
    """Раскладывает значения JointState по порядку (FR, FL, RR, RL) x (hip, thigh, calf).

    Возвращает массив (4, 3) или None, если какого-то сустава нет в сообщении.
    """
    index = {n: i for i, n in enumerate(names)}
    try:
        idx = [index[n] for n in JOINT_NAMES]
    except KeyError:
        return None
    vals = np.asarray(values, dtype=float)
    if vals.shape[0] <= max(idx):
        return None
    return vals[idx].reshape(4, 3)
