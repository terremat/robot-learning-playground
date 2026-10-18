import numpy as np
import mujoco

from robot_learning_playground.envs.mujoco.utils import get_qpos_ids, get_dof_ids

class CartesianController:
    def __init__(self, model, data, prefix="",tcp_name="tcp"):
        self.model = model
        self.data = data
        self.gain = 2.0
        self.damping = 0.05
        self.max_cartesian_speed = 0.15 # m/s
        self.max_joint_speed = 0.8  # rad/s
        self.tcp_name = tcp_name
        self.prefix = f"{prefix}_" if prefix else ""

        self.tcp_id = self.model.site(f"{self.prefix}{self.tcp_name}").id

        self.joint_names = [f"{self.prefix}joint{i}" for i in range(1, 8)]
        self.actuator_ids = [self.model.actuator(f"{self.prefix}actuator{i}").id for i in range(1, 8)]
        self.qpos_ids = get_qpos_ids(self.model, self.joint_names)
        self.dof_ids = get_dof_ids(self.model, self.joint_names)

        # Initialize Jacobian matrices (allocate once to avoid repeated allocation)
        self.jacobian_pos = np.zeros((3, self.model.nv))
        self.jacobian_rot = np.zeros((3, self.model.nv))

        # Initialize actuators command to current configuration
        self.qpos_cmd = self.data.qpos[self.qpos_ids].copy()


    def reset(self):
        self.qpos_cmd = self.data.qpos[self.qpos_ids].copy()
        self.data.ctrl[self.actuator_ids] = self.qpos_cmd


    def compute_velocities(self, desired_cartesian_velocity):

        # Limit the maximum Cartesian speed
        speed = np.linalg.norm(desired_cartesian_velocity)
        if speed > self.max_cartesian_speed:  
            desired_cartesian_velocity *= self.max_cartesian_speed / speed

        # Compute Jacobian for all model degrees of freedom (nv)
        mujoco.mj_jacSite(self.model, self.data, self.jacobian_pos, self.jacobian_rot, self.tcp_id)

        # Use only the 7 robot arm joints
        jacobian = self.jacobian_pos[:, self.dof_ids]  

        # Compute joint velocities using damped least squares
        lambda_identity = self.damping**2 * np.eye(3)
        dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + lambda_identity, desired_cartesian_velocity)

        dq = np.clip(
            dq,
            -self.max_joint_speed,
            self.max_joint_speed,
        )

        return dq


    def control_step(self, target_position, control_dt=0.01):

        # Get current TCP position
        current_tcp_pos = self.data.site_xpos[self.tcp_id]

        # Compute position error
        position_error = target_position - current_tcp_pos

        # Compute desired Cartesian velocity using proportional control
        desired_cartesian_velocity = self.gain * position_error

        # Compute joint velocities using the Jacobian
        dq = self.compute_velocities(desired_cartesian_velocity)

        # Intgrate joint velocities to get desired joint positions
        self.qpos_cmd += dq * control_dt
        
        self.qpos_cmd = np.clip(
            self.qpos_cmd,
            self.model.actuator_ctrlrange[self.actuator_ids, 0],
            self.model.actuator_ctrlrange[self.actuator_ids, 1],
        )

        # Apply computed joint positions to the simulation
        self.data.ctrl[self.actuator_ids] = self.qpos_cmd
        
        return np.linalg.norm(position_error)