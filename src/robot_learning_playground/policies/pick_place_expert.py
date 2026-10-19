from dataclasses import dataclass

import numpy as np

from robot_learning_playground.common.actions import CartesianAction
from robot_learning_playground.common.tasks import PickPlaceTask

@dataclass
class Step:
    name: str
    position: np.ndarray
    gripper: float
    duration: float = 0.0


def compute_position_error(target_position, current_position):
    return np.linalg.norm(target_position - current_position)

def compute_rotation_error(target_rotation, current_rotation):
    rotation_error_matrix = target_rotation @ current_rotation.T
    cos_angle = np.clip(
        (np.trace(rotation_error_matrix) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    return np.arccos(cos_angle)


class PickPlaceExpert:
    def __init__(
        self,
        pregrasp_height: float = 0.10,
        lift_height: float = 0.10,
        position_tolerance: float = 0.005,
        rotation_tolerance: float = 0.05,
        gripper_open: float = 255.0,
        gripper_closed: float = 0.0,
        gripper_duration: float = 1.0,
    ):
        self.pregrasp_height = pregrasp_height
        self.lift_height = lift_height

        self.position_tolerance = position_tolerance
        self.rotation_tolerance = rotation_tolerance

        self.gripper_open = gripper_open
        self.gripper_closed = gripper_closed
        self.gripper_duration = gripper_duration

        self.steps = []
        self.step_index = 0
        self.step_elapsed = 0.0
        self.target_rotation = None
        self.phase_start_time = 0.0

    def reset(self, observation, task):
        object_position = observation["object_positions"][task.object_name].copy()
        target_position = observation["target_positions"][task.target_name].copy()
        self.target_rotation = observation["tcp_rotation"].copy()
        self.phase_start_time = observation["time"]

        offset = np.array([0.0, 0.0, self.pregrasp_height])
        lift_offset = np.array([0.0, 0.0, self.lift_height])

        pregrasp_position = object_position + offset
        grasp_position = object_position.copy()
        lift_position = grasp_position + lift_offset

        # Keep the same height as the grasp position.
        place_position = np.array([
            target_position[0],
            target_position[1],
            grasp_position[2],
        ])

        target_above = place_position + lift_offset

        # Define the pick-and-place steps
        self.steps = [
            Step("PREGRASP",        pregrasp_position, self.gripper_open),
            Step("DESCEND",         grasp_position,    self.gripper_open),
            Step("CLOSE",           grasp_position,    self.gripper_closed, duration=0.0),
            Step("LIFT",            lift_position,     self.gripper_closed),
            Step("MOVE_TO_TARGET",  target_above,      self.gripper_closed),
            Step("PLACE",           place_position,    self.gripper_closed),
            Step("OPEN",            place_position,    self.gripper_open, duration=0.0),
            Step("RETREAT",         target_above,      self.gripper_open),
        ]
        self.step_index = 0
        self.step_elapsed = 0.0

    @property
    def done(self):
        return self.step_index >= len(self.steps)

    @property
    def phase(self):
        if self.done:
            return "DONE"
        return self.steps[self.step_index].name

    def predict(self, observation) -> CartesianAction | None:
        if self.done:
            return None

        step = self.steps[self.step_index]

        # Check conditions for moving to the next step
        if step.duration > 0:
            elapsed = observation["time"] - self.phase_start_time
            completed = elapsed >= step.duration

        else:
            position_error = compute_position_error(step.position, observation["tcp_position"])
            rotation_error = compute_rotation_error(self.target_rotation, observation["tcp_rotation"])

            completed = (
                position_error < self.position_tolerance
                and rotation_error < self.rotation_tolerance
            )

        if completed:
            print(f"{step.name} completed")

            self.step_index += 1
            self.phase_start_time = observation["time"]

            if self.done:
                return None

            step = self.steps[self.step_index]

        return CartesianAction(
            position=step.position.copy(),
            orientation=self.target_rotation.copy(),
            gripper=step.gripper,
        )