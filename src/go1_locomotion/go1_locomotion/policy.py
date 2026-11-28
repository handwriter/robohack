"""Structured locomotion policy, evaluated with only LowState and Twist.

The learned weights control a periodic foot trajectory and IMU feedback. IK
decodes the feet into 12 motor targets in Unitree order FR, FL, RR, RL.
"""

import json
from dataclasses import dataclass

import numpy as np


HOME = np.tile([0.0, 0.9, -1.8], 4)
LENGTH = 0.213
SIDE = np.array([-1., 1., -1., 1.])
FRONT = np.array([1., 1., -1., -1.])
HIP_Y = 0.04675 * SIDE
HIP_X = 0.1881 * FRONT
OUT = 0.08 * SIDE
PHASE_OFFSETS = np.array([0., 0.5, 0.5, 0.])
MAX_DT = 0.04
CONTROL_PERIOD = 0.006

# name, prior, minimum, maximum. The zero weight vector means the hand-set
# prior; the optimizer explores the continuous bounded parameter space.
WEIGHT_SPEC = (
    ('frequency_hz', 2.0, 1.45, 2.9),
    ('stance_fraction', 0.63, 0.52, 0.79),
    ('leg_extension_m', 0.295, 0.248, 0.343),
    ('fore_aft_bias_m', 0.005, -0.07, 0.08),
    ('rear_front_bias_m', 0.0, -0.045, 0.045),
    ('step_gain', 1.1, 0.55, 2.15),
    ('swing_height_m', 0.075, 0.037, 0.135),
    ('stance_depth_m', 0.012, 0.0, 0.038),
    ('stance_width_m', 0.021, 0.0, 0.065),
    ('roll_feedback', 0.21, 0.0, 0.65),
    ('pitch_feedback', 0.16, 0.0, 0.65),
    ('roll_rate_feedback', 0.016, 0.0, 0.065),
    ('pitch_rate_feedback', 0.014, 0.0, 0.065),
    ('kp', 54.0, 29.0, 92.0),
    ('kd', 1.45, 0.65, 3.2),
    ('swing_velocity_ff', 0.1, 0.0, 0.75),
    ('yaw_feedback', 2.0, 0.0, 4.5),
    ('yaw_rate_feedback', 0.25, 0.0, 0.7),
)
_SPEC = np.array([spec[1:] for spec in WEIGHT_SPEC])
N_WEIGHTS = len(WEIGHT_SPEC)


def decode(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (N_WEIGHTS,) or not np.isfinite(weights).all() or np.max(np.abs(weights)) > 1.0:
        raise ValueError('invalid T3 policy weight vector')
    prior, low, high = _SPEC.T
    return prior + np.where(weights < 0, prior - low, high - prior) * weights


def load_weights(path):
    with open(path, encoding='utf-8') as file:
        checkpoint = json.load(file)
    if checkpoint.get('format') != 'go1-structured-policy-v1':
        raise ValueError('unknown T3 checkpoint format')
    if checkpoint.get('parameters') != [spec[0] for spec in WEIGHT_SPEC]:
        raise ValueError('T3 checkpoint parameter order mismatch')
    weights = np.asarray(checkpoint['weights'], dtype=float)
    decode(weights)
    return weights


def orientation(quat):
    """Return roll, pitch and yaw from the Unitree quaternion (w, x, y, z)."""
    w, x, y, z = quat / np.linalg.norm(quat)
    return np.array((
        np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
        np.arcsin(np.clip(2 * (w*y - z*x), -1.0, 1.0)),
        np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)),
    ))


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def inverse_kinematics(x, y, z):
    """Foot centres relative to their hip axes, x forward and z up."""
    sagittal_z = -np.sqrt(np.maximum(y*y + z*z - OUT*OUT, 0.025**2))
    hip = np.arctan2(z, y) - np.arctan2(sagittal_z, OUT)
    c = (x*x + sagittal_z*sagittal_z) / (2 * LENGTH**2) - 1
    calf = -np.arccos(np.clip(c, -0.98, 0.65))
    thigh = np.arctan2(-x, -sagittal_z) - calf / 2
    return np.column_stack((hip, thigh, calf)).reshape(12)


@dataclass(frozen=True)
class MotorTargets:
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    kp: float
    kd: float


class WalkingPolicy:
    def __init__(self, weights):
        self.params = decode(weights)
        self.reset()

    def reset(self):
        self.phase = 0.
        self.ramp = 0.
        self.command = np.zeros(3)
        self.yaw_target = None
        self.last_q = None
        self.last_dq = np.zeros(12)

    def step(self, dt, q, dq, quat, gyro, command):
        dt = float(np.clip(dt, 0.001, MAX_DT))
        frequency, duty, height, x0, x_bias, stride, lift, depth, width, kr, kpitch, dr, dp, kp, kd, ff, ky, dy = self.params
        roll, pitch, yaw = orientation(quat)
        command = np.clip(command, [-0.6, -0.3, -0.8], [0.6, 0.3, 0.8])
        if self.last_q is None:
            self.last_q = np.clip(q, HOME - 0.6, HOME + 0.6)
        active = np.max(np.abs(command)) > 0.025
        self.command += min(dt / 0.15, 1.) * (command - self.command)
        self.ramp = np.clip(self.ramp + dt * (2.5 if active else -2.5), 0., 1.)
        if self.yaw_target is None or self.ramp == 0.:
            self.yaw_target = yaw
        else:
            self.yaw_target = wrap_angle(self.yaw_target + self.command[2] * dt)
        if self.ramp:
            self.phase = (self.phase + dt * frequency) % 1.
        else:
            self.phase = 0.

        yaw_rate = np.clip(ky * wrap_angle(self.yaw_target - yaw) + self.command[2] - dy * gyro[2], -1., 1.)
        vx = self.command[0] - yaw_rate * (HIP_Y + OUT)
        vy = self.command[1] + yaw_rate * HIP_X
        phase = (self.phase + PHASE_OFFSETS) % 1.
        stance = phase < duty
        u = np.where(stance, phase / duty, (phase - duty) / (1 - duty))
        stroke = np.where(stance, 0.5 - u, -0.5 + u*u*(3 - 2*u)) * duty * stride / frequency
        vertical = np.where(stance, -depth * np.sin(np.pi * u), lift * np.sin(np.pi * u)**2)
        x = x0 + FRONT * x_bias + self.ramp * vx * stroke
        y = OUT + SIDE * width + self.ramp * vy * stroke
        # Extend the legs on the low side to generate a restoring moment.
        roll_term = np.clip(kr * roll + dr * gyro[0], -0.09, 0.09)
        pitch_term = np.clip(kpitch * pitch + dp * gyro[1], -0.09, 0.09)
        z = -height + self.ramp * vertical + SIDE * roll_term - FRONT * pitch_term
        target = inverse_kinematics(x, y, z)
        # Smooth startup and post-loss recovery without large target jumps.
        target = self.last_q + np.clip(target - self.last_q, -11 * dt, 11 * dt)
        target_dq = np.clip((target - self.last_q) / dt, -12., 12.)
        self.last_dq += min(dt / 0.022, 1.) * (target_dq - self.last_dq)
        self.last_q = target
        # Reduce velocity feedforward when a joint is already moving faster
        # than the target in the same direction (e.g. after touchdown).
        drive = self.last_dq - np.clip(dq * np.sign(self.last_dq), 0., np.abs(self.last_dq)) * np.sign(self.last_dq)
        return MotorTargets(target, ff * drive, np.zeros(12), float(kp), float(kd))
