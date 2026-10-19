import time
import argparse
import numpy as np

import mujoco.viewer

from robot_learning_playground.common.actions import CartesianAction
from robot_learning_playground.common.tasks import PickPlaceTask
from robot_learning_playground.envs.mujoco.pick_place_env import PickPlaceEnv
from robot_learning_playground.policies.pick_place_expert import PickPlaceExpert

def parse_args():
    parser = argparse.ArgumentParser(description="Run the pick and place expert policy in the Mujoco environment.")
    parser.add_argument("--object", type=str, default="cube_red", help="Name of the object to pick.")
    parser.add_argument("--target", type=str, default="target_blue", help="Name of the target location to place the object.")
    return parser.parse_args()

def main():
    args = parse_args()

    env = PickPlaceEnv()
    expert = PickPlaceExpert()

    task = PickPlaceTask(object_name=args.object, target_name=args.target)
    print(f"Task: Pick {task.object_name} and place it on {task.target_name}")

    observation = env.reset()
    expert.reset(observation, task)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            
            action = expert.predict(observation)

            if action is None:
                print("Expert episode completed.")
                break

            observation, info = env.step(action)

            error = info["cartesian_error"]

            viewer.sync()
            time.sleep(env.control_dt)

    cube_position = observation["object_positions"][task.object_name]
    target_position = observation["target_positions"][task.target_name]

    placement_error_xy = (
        (cube_position[:2] - target_position[:2]) ** 2
    ).sum() ** 0.5

    print("Final cube position:", cube_position)
    print("Placement error XY:", placement_error_xy)


if __name__ == "__main__":
    main()