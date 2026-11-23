"""Проверки алгоритма на синтетическом стенде (см. synth.py) — без ROS и Gazebo."""
import math

import numpy as np
import pytest

import synth
from go1_odometry import kinematics as K
from go1_odometry import metrics
from go1_odometry.estimator import Params
from go1_odometry.leg_odometry import StationaryDetector


def evaluate(r):
    rpe, n, _ = metrics.compute_rpe(r.gt_xy, r.est_xy, r.gt_yaw[0], r.est_yaw[0])
    dpos, dyaw = metrics.compute_drift(r.t, r.est_xy, r.est_yaw)
    mean_rate, min_rate, max_gap = metrics.compute_rate(r.t)
    return dict(rpe=rpe, n=n, dpos=dpos, dyaw=dyaw, min_rate=min_rate, max_gap=max_gap)


@pytest.fixture(scope='module')
def scenario():
    # укороченный квадрат 2 м: достаточно для нескольких участков по 1 м
    return synth.Scenario(yaw_start=0.4, side=2.0, t_still0=3.0, t_still1=12.0)


def test_kinematics_roundtrip_and_jacobian():
    sc = synth.Scenario(side=1.0, t_still0=1.0, t_still1=1.0)
    q = sc.joint_angles()
    body = np.stack([sc.x, sc.y, sc.z], axis=1)
    for k in range(0, sc.n, 97):
        for leg in range(4):
            pt = sc.R_wt[k].T @ (sc.feet[leg, k] - body[k])
            p = K.foot_positions(np.tile(q[k][leg], (4, 1)))[leg]
            assert np.allclose(p, pt, atol=1e-9)
    rng = np.random.default_rng(3)
    qq = np.tile([0.0, 0.8, -1.6], (4, 1)) + rng.normal(0, 0.2, (4, 3))
    qd = rng.normal(0, 1, (4, 3))
    eps = 1e-6
    num = (K.foot_positions(qq + eps * qd) - K.foot_positions(qq - eps * qd)) / (2 * eps)
    assert np.allclose(num, K.foot_velocities(qq, qd), atol=1e-6)


def test_full_pipeline_meets_t4_criteria(scenario):
    r = synth.run(scenario)
    m = evaluate(r)
    assert m['min_rate'] >= 20.0
    assert m['rpe'] <= 0.20
    assert m['dpos'] <= 0.10 and m['dyaw'] <= 5.0
    assert np.all(np.isfinite(r.est_xy)) and np.all(np.isfinite(r.est_yaw))
    assert np.abs(r.est_z - r.gt_z).max() < 0.05


def test_legs_only_is_usable_without_lidar(scenario):
    m = evaluate(synth.run(scenario, use_scans=False))
    assert m['rpe'] <= 0.20 and m['dpos'] <= 0.10


def test_lidar_only_without_joint_states(scenario):
    m = evaluate(synth.run(scenario, use_joints=False))
    assert m['rpe'] <= 0.20 and m['dpos'] <= 0.10


def test_joint_velocities_missing_uses_finite_differences(scenario):
    m = evaluate(synth.run(scenario, joint_velocity=False))
    assert m['rpe'] <= 0.20


def test_invalid_imu_quaternion_falls_back_to_gyro_accel(scenario):
    r = synth.run(scenario, imu_orientation=False, give_accel=True)
    m = evaluate(r)
    assert r.est.stats['fallback_steps'] > 0
    assert m['rpe'] <= 0.20 and m['dyaw'] <= 5.0


@pytest.mark.parametrize('yaw0', [0.0, -2.5, 3.0])
def test_arbitrary_initial_heading(yaw0):
    sc = synth.Scenario(yaw_start=yaw0, side=2.0, t_still0=3.0, t_still1=12.0)
    m = evaluate(synth.run(sc))
    assert m['rpe'] <= 0.20 and m['dpos'] <= 0.10


def test_leg_slip_is_compensated_by_lidar():
    sc = synth.Scenario(side=2.0, t_still0=3.0, t_still1=12.0, slip=0.15)
    assert evaluate(synth.run(sc))['rpe'] <= 0.20


def test_stationary_detector_hysteresis():
    d = StationaryDetector(hold=0.5)
    assert not d.update(0.0, 0.01, 0.01)       # ещё не прошло hold
    assert not d.update(0.3, 0.01, 0.01)
    assert d.update(0.6, 0.01, 0.01)           # прошло hold — стоит
    assert d.update(0.7, 0.15, 0.01)           # между порогами входа и выхода — остаётся
    assert not d.update(0.8, 0.5, 0.01)        # движение — сразу выходит


def test_estimator_ignores_bad_inputs():
    from go1_odometry.estimator import OdometryEstimator
    e = OdometryEstimator(Params())
    assert not e.on_imu(float('nan'), [0, 0, 0, 1], [0, 0, 0])
    assert e.on_imu(1.0, [0, 0, 0, 1], [0, 0, 0])
    assert not e.on_imu(0.9, [0, 0, 0, 1], [0, 0, 0])          # время назад
    assert e.on_imu(1.01, [float('nan')] * 4, [float('inf'), 0, 0], [0, 0, 9.81])
    o = e.out
    assert np.all(np.isfinite(o.position)) and np.all(np.isfinite(o.quaternion))
    assert abs(np.linalg.norm(o.quaternion) - 1.0) < 1e-9


def test_metrics_formulas():
    # прямая 4 м, оценка с масштабом 1.2 -> ошибка 20 % на каждом участке
    s = np.linspace(0, 4, 401)
    gt = np.c_[s, 0 * s]
    rpe, n, _ = metrics.compute_rpe(gt, 1.2 * gt, 0.0, 0.0)
    assert n >= 3 and abs(rpe - 0.2) < 0.01
    # начальные позы разные: оценка повёрнута на 90 градусов — после совмещения ошибка 0
    est = np.c_[0 * s, s]
    rpe, _, _ = metrics.compute_rpe(gt, est, 0.0, math.pi / 2)
    assert rpe < 1e-9
    assert metrics.score(0.15, 0.05, 2.0, 50.0) == 20
    assert metrics.score(0.4, 0.05, 2.0, 50.0) == 14
    assert metrics.score(0.8, 0.5, 9.0, 10.0) == 0
