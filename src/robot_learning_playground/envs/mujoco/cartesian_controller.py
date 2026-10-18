import numpy as np
import mujoco

from robot_learning_playground.envs.mujoco.utils import get_qpos_ids, get_dof_ids

class CartesianController:
    def __init__(self, model, data, prefix="",tcp_name="tcp"):
        self.model = model
        self.data = data
        self.gain = 2.0
        self.damping = 0.05
        self.max_linear_speed  = 0.15 # m/s
        self.max_angular_speed = 0.5  # rad/s
        self.max_joint_speed   = 0.8  # rad/s
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


    def compute_position_error(self, target_position):
        current_tcp_pos = self.data.site_xpos[self.tcp_id]
        return target_position - current_tcp_pos


    def compute_orientation_error(self, target_orientation):
        current_tcp_rot = self.data.site_xmat[self.tcp_id].reshape(3, 3)
        R_error = target_orientation @ current_tcp_rot.T

        quat_error = np.zeros(4)
        mujoco.mju_mat2Quat(quat_error, R_error.flatten())

        orientation_error = np.zeros(3)
        mujoco.mju_quat2Vel(orientation_error, quat_error, 1.0)

        return orientation_error


    def compute_limit_velocity(self, velocity, max_speed):
        speed = np.linalg.norm(velocity)
        if speed > max_speed:
            return velocity * (max_speed / speed)
        return velocity


    def compute_joint_velocities(self, desired_twist):
        
        # Compute Jacobian for all model degrees of freedom (nv)
        mujoco.mj_jacSite(self.model, self.data, self.jacobian_pos, self.jacobian_rot, self.tcp_id)

        # Extract jacobians for only the robot arm joints
        if desired_twist.shape[0] == 6:
            jacobian = np.vstack([
                self.jacobian_pos[:, self.dof_ids],
                self.jacobian_rot[:, self.dof_ids]
            ])
        else:
            jacobian = self.jacobian_pos[:, self.dof_ids]

        # Compute joint velocities using damped least squares
        lambda_identity = self.damping**2 * np.eye(desired_twist.shape[0])
        dq = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + lambda_identity, desired_twist)

        dq = np.clip(
            dq,
            -self.max_joint_speed,
            self.max_joint_speed,
        )
        return dq


    def control_step(self, target_position, target_orientation=None, control_dt=0.01):

        # Compute desired linear velocity using proportional control on linear cartesian error
        position_error = self.compute_position_error(target_position)
        desired_linear_velocity = self.gain * position_error
        desired_linear_velocity = self.compute_limit_velocity(desired_linear_velocity, self.max_linear_speed)

        desired_twist = desired_linear_velocity
        error = np.concatenate([position_error,  np.full(3, np.nan)])

        if target_orientation is not None:
            rotation_error = self.compute_orientation_error(target_orientation)
            desired_angular_velocity = self.gain * rotation_error
            desired_angular_velocity = self.compute_limit_velocity(desired_angular_velocity, self.max_angular_speed)

            # Combine linear and angular velocities into a single twist
            desired_twist = np.concatenate([desired_linear_velocity, desired_angular_velocity])
            error = np.concatenate([position_error, rotation_error])

        # Compute joint velocities using the Jacobian
        dq = self.compute_joint_velocities(desired_twist)

        # Intgrate joint velocities to get desired joint positions
        self.qpos_cmd += dq * control_dt
        
        self.qpos_cmd = np.clip(
            self.qpos_cmd,
            self.model.actuator_ctrlrange[self.actuator_ids, 0],
            self.model.actuator_ctrlrange[self.actuator_ids, 1],
        )

        # Apply computed joint positions to the simulation
        self.data.ctrl[self.actuator_ids] = self.qpos_cmd
        
        return error