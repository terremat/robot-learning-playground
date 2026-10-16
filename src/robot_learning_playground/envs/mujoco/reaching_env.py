
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = ROOT / "assets/franka_emika_panda/scene.xml"


class ReachingEnv:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)

        self.tcp_id = self.model.site("tcp").id
        self.home_id = self.model.key("home").id

        self.control_dt = 0.01
        self.physics_dt = self.model.opt.timestep
        self.n_substeps = round(self.control_dt / self.physics_dt)

        assert np.isclose(
            self.n_substeps * self.physics_dt,
            self.control_dt,
        )

        self.max_steps = 500
        self.success_threshold = 0.01
        self.max_cartesian_speed = 0.15
        self.max_joint_speed = 0.8
        self.damping = 0.05

        self.rng = np.random.default_rng()
        self.target = np.zeros(3)
        self.q_ref = np.zeros(7)
        self.step_count = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetDataKeyframe(
            self.model, self.data, self.home_id
        )
        mujoco.mj_forward(self.model, self.data)

        self.q_ref = self.data.qpos[:7].copy()
        self.step_count = 0

        # Random target near initial TCP position
        x_home = self.data.site_xpos[self.tcp_id].copy()

        offset = self.rng.uniform(
            low=[-0.08, -0.10, -0.05],
            high=[0.08, 0.10, 0.08],
        )

        self.target = x_home + offset

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)

        assert action.shape == (3,)

        # Cartesian velocity command
        speed = np.linalg.norm(action)

        if speed > self.max_cartesian_speed:
            action = action * self.max_cartesian_speed / speed

        mujoco.mj_forward(self.model, self.data)

        # Compute translational Jacobian
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        mujoco.mj_jacSite(
            self.model,
            self.data,
            jacp,
            jacr,
            self.tcp_id,
        )

        J = jacp[:, :7]

        # Damped least-squares inverse
        A = J @ J.T + self.damping**2 * np.eye(3)
        qdot = J.T @ np.linalg.solve(A, action)

        qdot = np.clip(
            qdot,
            -self.max_joint_speed,
            self.max_joint_speed,
        )

        # Persistent joint reference
        self.q_ref += qdot * self.control_dt

        self.q_ref = np.clip(
            self.q_ref,
            self.model.actuator_ctrlrange[:7, 0],
            self.model.actuator_ctrlrange[:7, 1],
        )

        self.data.ctrl[:7] = self.q_ref

        # Physics integration
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        # Refresh derived kinematics for observation
        mujoco.mj_forward(self.model, self.data)

        self.step_count += 1

        obs = self._get_obs()
        info = self._get_info()

        terminated = info["is_success"]
        truncated = self.step_count >= self.max_steps

        reward = -info["distance"]

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos[:7],
            self.data.qvel[:7],
            self.data.site_xpos[self.tcp_id],
            self.target,
        ]).astype(np.float32)

    def _get_info(self):
        distance = np.linalg.norm(
            self.target - self.data.site_xpos[self.tcp_id]
        )

        return {
            "distance": float(distance),
            "is_success": bool(
                distance < self.success_threshold
            ),
        }
