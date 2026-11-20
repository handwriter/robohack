import math

import numpy as np

N_JOINTS = 12

Q_MIN = np.array([-0.863, -0.686, -2.818] * 4)
Q_MAX = np.array([0.863, 4.501, -0.888] * 4)
KP_MAX = 1000.0
KD_MAX = 100.0
TAU_MAX = 100.0
DQ_MAX = 20.0
MAX_STEP = math.radians(5.0)
EPS = 1e-6

HOME_Q = np.array([0.0, 0.9, -1.8] * 4)


def f32(x):
    return np.asarray(x, dtype=np.float32).astype(float)


def clip_q(q):
    return f32(np.clip(q, Q_MIN + EPS, Q_MAX - EPS))


class SafetyFilter:
    def __init__(self, checker, q0=HOME_Q, hold_kp=40.0, hold_kd=2.0, catch_up_step=math.radians(0.5)):
        self.checker = checker
        self.hold_kp = hold_kp
        self.hold_kd = hold_kd
        self.catch_up_step = min(catch_up_step, MAX_STEP)

        q0 = clip_q(q0)
        if not checker.is_free(q0):
            raise ValueError('initial pose is in self-collision')
        self.q = q0
        self.goal = q0.copy()
        self.dq = np.zeros(N_JOINTS)
        self.tau = np.zeros(N_JOINTS)
        self.kp = np.full(N_JOINTS, float(hold_kp))
        self.kd = np.full(N_JOINTS, float(hold_kd))
        self.holding = True
        self.accepted = 0
        self.rejected = 0

    def command(self, q, dq, tau, kp, kd):
        values = np.array([q, dq, tau, kp, kd], dtype=float)
        if not np.all(np.isfinite(values)):
            self.rejected += 1
            return False

        q, dq, tau, kp, kd = values[:, :N_JOINTS]
        self.goal = clip_q(q)
        self.dq = f32(np.clip(dq, -DQ_MAX, DQ_MAX))
        self.tau = f32(np.clip(tau, -TAU_MAX, TAU_MAX))
        self.kp = f32(np.clip(kp, 0.0, KP_MAX))
        self.kd = f32(np.clip(kd, 0.0, KD_MAX))
        self.holding = False
        self.accepted += 1

        if np.max(np.abs(self.goal - self.q)) <= MAX_STEP:
            self._move_to(self.goal)
        else:
            self._catch_up()
        return True

    def tick(self):
        if not self.holding:
            self._catch_up()

    def hold(self):
        if self.holding:
            return
        self.holding = True
        self.goal = self.q.copy()
        self.dq = np.zeros(N_JOINTS)
        self.tau = np.zeros(N_JOINTS)
        weak = self.kp < 1.0
        self.kp[weak] = self.hold_kp
        self.kd[weak] = self.hold_kd

    def output(self):
        return self.q, self.dq, self.tau, self.kp, self.kd

    def _catch_up(self):
        delta = self.goal - self.q
        if not np.any(delta):
            return
        step = np.clip(delta, -self.catch_up_step, self.catch_up_step)
        self._move_to(clip_q(self.q + step))

    def _move_to(self, target):
        if np.max(np.abs(target - self.q)) > MAX_STEP:
            return
        frac = self.checker.safe_fraction(self.q, target)
        if frac >= 1.0:
            self.q = target.copy()
            return
        # дальше идти нельзя, останавливаемся на последней безопасной точке
        if frac > 0.0:
            q = clip_q(self.q + (target - self.q) * frac)
            if self.checker.is_free(q) and np.max(np.abs(q - self.q)) <= MAX_STEP:
                self.q = q
        self.goal = self.q.copy()
