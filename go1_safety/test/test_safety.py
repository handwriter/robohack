import csv
import math
import os

import mujoco
import numpy as np
import pytest

from go1_safety.collision import CollisionChecker
from go1_safety.safety import HOME_Q, MAX_STEP, Q_MAX, Q_MIN, SafetyFilter

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'unitree_go1')
ZERO = [0.0] * 20


class Judge:
    """Независимая проверка: mj_geomDistance по 655 парам, шаг 0.1 градуса."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(os.path.join(MODEL_DIR, 'scene.xml'))
        self.d = mujoco.MjData(self.m)
        with open(os.path.join(MODEL_DIR, 't2_collision_pairs.csv')) as f:
            self.pairs = [(int(r['geom1_id']), int(r['geom2_id'])) for r in csv.DictReader(f)
                          if r['diagnostic_policy'] == 'checked']
        assert len(self.pairs) == 655
        self.d.qpos[2] = 10.0
        self.d.qpos[3] = 1.0

    def depth(self, q):
        self.d.qpos[7:19] = q
        mujoco.mj_kinematics(self.m, self.d)
        worst = 0.0
        for a, b in self.pairs:
            dist = mujoco.mj_geomDistance(self.m, self.d, a, b, 0.01, None)
            worst = max(worst, -dist)
        return worst

    def transition_ok(self, q0, q1):
        n = max(1, math.ceil(np.max(np.abs(q1 - q0)) / math.radians(0.1)))
        return all(self.depth(q0 + (q1 - q0) * i / n) <= 0.001 for i in range(n + 1))


@pytest.fixture(scope='module')
def checker():
    return CollisionChecker(MODEL_DIR)


@pytest.fixture(scope='module')
def judge():
    return Judge()


def cmd(q, kp=40.0, kd=2.0, dq=0.0, tau=0.0):
    q = list(q) + [0.0] * 8
    return (q, [dq] * 12 + [0.0] * 8, [tau] * 12 + [0.0] * 8, [kp] * 12 + [0.0] * 8, [kd] * 12 + [0.0] * 8)


def check_output(f, prev):
    q, dq, tau, kp, kd = f.output()
    for v in (q, dq, tau, kp, kd):
        assert np.all(np.isfinite(v))
    assert np.all(q >= Q_MIN) and np.all(q <= Q_MAX)
    assert np.all(np.abs(dq) <= 20) and np.all(np.abs(tau) <= 100)
    assert np.all((kp >= 0) & (kp <= 1000)) and np.all((kd >= 0) & (kd <= 100))
    q32 = q.astype(np.float32).astype(float)
    assert np.array_equal(q, q32)
    if prev is not None:
        assert np.max(np.abs(q - prev)) <= MAX_STEP
    return q.copy()


def test_home_is_free(checker, judge):
    assert checker.is_free(HOME_Q)
    assert judge.depth(HOME_Q) <= 0.001


def test_checker_matches_judge(checker, judge):
    rng = np.random.default_rng(1)
    for _ in range(300):
        q = rng.uniform(Q_MIN, Q_MAX)
        if checker.is_free(q):
            assert judge.depth(q) <= 0.001


def test_safe_commands_pass(checker):
    f = SafetyFilter(checker)
    q = HOME_Q.copy()
    for i in range(100):
        q = q.copy()
        q[1::3] += math.radians(0.3) * math.sin(i / 10)
        q32 = q.astype(np.float32).astype(float)
        assert f.command(*cmd(q32, kp=60.0, kd=3.0, dq=0.5, tau=1.0))
        out, dq, tau, kp, kd = f.output()
        assert np.array_equal(out, q32)
        assert np.allclose(kp, 60.0) and np.allclose(dq, 0.5) and np.allclose(tau, 1.0)


def test_nan_inf_rejected(checker):
    f = SafetyFilter(checker)
    f.command(*cmd(HOME_Q))
    before = [x.copy() for x in f.output()]
    for bad in (math.nan, math.inf, -math.inf):
        for field in range(5):
            c = [list(x) for x in cmd(HOME_Q + 0.01)]
            c[field][7] = bad
            assert not f.command(*c)
        c = [list(x) for x in cmd(HOME_Q + 0.01)]
        c[2][15] = bad
        assert not f.command(*c)
    after = f.output()
    for a, b in zip(before, after):
        assert np.array_equal(a, b)


def test_limits_and_params_clipped(checker):
    f = SafetyFilter(checker)
    q = HOME_Q.copy()
    q[0] = 0.86
    f.command(*cmd(q, kp=5000.0, kd=-3.0, dq=100.0, tau=-500.0))
    q, dq, tau, kp, kd = f.output()
    assert np.all(kp == 1000.0) and np.all(kd == 0.0)
    assert np.all(dq == 20.0) and np.all(tau == -100.0)
    for _ in range(50):
        bad = HOME_Q.copy()
        bad[0] = 3.0
        bad[2] = 1.0
        f.command(*cmd(bad))
        check_output(f, None)


def test_hold(checker):
    f = SafetyFilter(checker)
    q = HOME_Q.copy()
    q[1] += 0.05
    f.command(*cmd(q, kp=0.0, kd=0.0, dq=1.0, tau=2.0))
    f.hold()
    out, dq, tau, kp, kd = f.output()
    assert np.array_equal(out, q.astype(np.float32).astype(float))
    assert np.all(dq == 0) and np.all(tau == 0)
    assert np.all(kp > 0) and np.all(kd > 0)


def test_random_stream(checker, judge):
    rng = np.random.default_rng(7)
    f = SafetyFilter(checker)
    prev = check_output(f, None)
    goal = HOME_Q.copy()
    for i in range(1500):
        r = rng.random()
        if r < 0.05:
            goal = rng.uniform(Q_MIN - 0.5, Q_MAX + 0.5)
        elif r < 0.10:
            c = [list(x) for x in cmd(goal)]
            c[rng.integers(5)][rng.integers(20)] = rng.choice([math.nan, math.inf, -math.inf])
            assert not f.command(*c)
        else:
            goal = goal + rng.normal(0, math.radians(2), 12)
        f.command(*cmd(goal))
        q = check_output(f, prev)
        assert judge.transition_ok(prev, q)
        for _ in range(rng.integers(0, 4)):
            f.tick()
            q2 = check_output(f, q)
            assert judge.transition_ok(q, q2)
            q = q2
        prev = q


def test_collision_target_blocked(checker, judge):
    f = SafetyFilter(checker)
    rng = np.random.default_rng(3)
    bad = None
    while bad is None:
        q = rng.uniform(Q_MIN, Q_MAX)
        if not checker.is_free(q):
            bad = q
    prev = check_output(f, None)
    for _ in range(2000):
        f.command(*cmd(bad))
        q = check_output(f, prev)
        assert judge.transition_ok(prev, q)
        prev = q
    assert checker.is_free(prev)
