"""Контракт /entrypoint.sh из ТЗ T1 (раздел 3.3): ping за 10 с, один JSON в stdout."""
import json
import os
import subprocess
import time

ENTRYPOINT = os.path.join(os.path.dirname(__file__), '..', 'entrypoint.sh')


def run(*args):
    t0 = time.monotonic()
    p = subprocess.run(['bash', ENTRYPOINT, *args], capture_output=True, text=True, timeout=10)
    return p, time.monotonic() - t0


def test_ping_contract():
    p, dt = run('ping')
    assert p.returncode == 0 and dt < 10.0
    assert p.stdout.strip() == p.stdout.strip('\n').strip()        # нет посторонних строк
    data = json.loads(p.stdout)                                    # ровно один JSON-объект
    assert data['protocol_version'] == '1.0'
    assert isinstance(data['supported_tasks'], list) and data['supported_tasks'] == ['T2', 'T4', 'T5']
    assert all(t in ('T2', 'T3', 'T4', 'T5') for t in data['supported_tasks'])
    assert isinstance(data['self_trained_rl'], bool)
    assert data['preferred_speed_mps'] is None                     # T3 не реализовано
    assert len(p.stdout.strip().splitlines()) == 1


def test_unsupported_tasks_do_not_pollute_stdout():
    for task in ('T3',):
        p, _ = run(task)
        assert p.returncode != 0 and p.stdout == ''
    p, _ = run('bogus')
    assert p.returncode != 0 and p.stdout == ''
