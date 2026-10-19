from pathlib import Path

import mujoco
import numpy as np

from robot_learning_playground.envs.mujoco.cartesian_controller import (
    CartesianController,
)

ROOT = Path(__file__).resolve().parents[4]
SCENE_PATH = ROOT / "assets/pick_place/scene.xml"

class PickPlaceEnv:
    def __init__(self, scene_path: Path = SCENE_PATH, control_steps: int = 5):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)

        self.control_steps = control_steps
        self.control_dt = self.model.opt.timestep * control_steps

        self.controller = CartesianController(
            self.model,
            self.data,
            prefix="panda",
            tcp_name="tcp",
        )

        self.body_names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)]
        self.object_names = [f for f in self.body_names if f.startswith("cube_")]
        self.target_names = [f for f in self.body_names if f.startswith("target_")]
        print("Object names:", self.object_names)
        print("Target names:", self.target_names)

        self.gripper_id = self.model.actuator("panda_actuator8").id
        self.tcp_id = self.model.site("panda_tcp").id

        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0) # Home position
        mujoco.mj_forward(self.model, self.data)

        self.controller.reset()

        return self.get_observation()

    def get_observation(self):
        return {
            "time": float(self.data.time),
            "tcp_position": self.data.site_xpos[self.tcp_id].copy(),
            "tcp_rotation": self.data.site_xmat[self.tcp_id].reshape(3, 3).copy(),
            "joint_positions": self.data.qpos[self.controller.qpos_ids].copy(),
            "object_positions": {name: self.data.body(name).xpos.copy() for name in self.object_names},
            "target_positions": {name: self.data.body(name).xpos.copy() for name in self.target_names},
        }

    def step(self, action):
        error = self.controller.control_step(
            target_position=action.position,
            target_orientation=action.orientation,
            control_dt=self.control_dt,
        )

        self.data.ctrl[self.gripper_id] = action.gripper

        for _ in range(self.control_steps):
            mujoco.mj_step(self.model, self.data)

        observation = self.get_observation()

        info = {
            "cartesian_error": error.copy(),
        }

        return observation, info