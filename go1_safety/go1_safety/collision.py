import csv
import math
import os

import mujoco
import numpy as np


class CollisionChecker:
    def __init__(self, model_dir, max_depth=0.0002, step=math.radians(0.05)):
        self.model = mujoco.MjModel.from_xml_path(os.path.join(model_dir, 'scene.xml'))
        self.data = mujoco.MjData(self.model)
        self.max_depth = max_depth
        self.step = step

        n = self.model.ngeom
        self.pairs = np.zeros((n, n), dtype=bool)
        with open(os.path.join(model_dir, 't2_collision_pairs.csv')) as f:
            for row in csv.DictReader(f):
                if row['diagnostic_policy'] == 'checked':
                    a, b = int(row['geom1_id']), int(row['geom2_id'])
                    self.pairs[a, b] = self.pairs[b, a] = True

        self.qadr = []
        for leg in ('FR', 'FL', 'RR', 'RL'):
            for joint in ('hip', 'thigh', 'calf'):
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f'{leg}_{joint}_joint')
                self.qadr.append(self.model.jnt_qposadr[jid])

        # корпус поднят над полом, чтобы не было контактов с полом
        self.data.qpos[:] = 0.0
        self.data.qpos[2] = 10.0
        self.data.qpos[3] = 1.0

    def min_distance(self, q):
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)
        ncon = self.data.ncon
        if ncon == 0:
            return math.inf
        geom = self.data.contact.geom[:ncon]
        dist = self.data.contact.dist[:ncon]
        mask = self.pairs[geom[:, 0], geom[:, 1]]
        if not mask.any():
            return math.inf
        return float(dist[mask].min())

    def is_free(self, q):
        return self.min_distance(q) >= -self.max_depth

    def safe_fraction(self, q0, q1):
        q0 = np.asarray(q0, dtype=float)
        dq = np.asarray(q1, dtype=float) - q0
        n = max(1, int(math.ceil(np.max(np.abs(dq)) / self.step)))
        for i in range(1, n + 1):
            if not self.is_free(q0 + dq * (i / n)):
                return (i - 1) / n
        return 1.0
