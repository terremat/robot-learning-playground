from dataclasses import dataclass
import numpy as np


@dataclass
class CartesianAction:
    position: np.ndarray
    orientation: np.ndarray
    gripper: float