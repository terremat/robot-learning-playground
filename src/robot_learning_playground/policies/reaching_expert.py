
import numpy as np


class ReachingExpert:
    def __init__(self, kp=2.0, max_speed=0.15):
        self.kp = kp
        self.max_speed = max_speed

    def predict(self, obs):
        tcp = obs[14:17]
        target = obs[17:20]

        velocity = self.kp * (target - tcp)

        speed = np.linalg.norm(velocity)

        if speed > self.max_speed:
            velocity *= self.max_speed / speed

        return velocity.astype(np.float32)
