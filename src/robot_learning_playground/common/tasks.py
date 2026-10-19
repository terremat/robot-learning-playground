from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PickPlaceTask:
    object_name: str
    target_name: str
    success_tolerance: float = 0.03

    def is_success(self, observation):
        object_pos = observation["object_positions"][self.object_name]
        target_pos = observation["target_positions"][self.target_name]

        distance_xy = np.linalg.norm(
            object_pos[:2] - target_pos[:2]
        )

        return bool(distance_xy < self.success_tolerance)