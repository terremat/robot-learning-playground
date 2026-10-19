from pathlib import Path
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from robot_learning_playground.common.actions import CartesianAction
from robot_learning_playground.common.tasks import PickPlaceTask
from robot_learning_playground.envs.mujoco.pick_place_env import PickPlaceEnv
from robot_learning_playground.policies.pick_place_expert import PickPlaceExpert

def parse_args():
    parser = argparse.ArgumentParser(description="Run the pick and place expert policy in the Mujoco environment.")
    parser.add_argument("--object", type=str, default="cube_red", help="Name of the object to pick.")
    parser.add_argument("--target", type=str, default="target_blue", help="Name of the target location to place the object.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for environment reset.")
    parser.add_argument("--randomize", action="store_true", help="Randomize the environment at each reset.")
    return parser.parse_args()


def contact_normal_forces(model, data, cube_name):
    forces = {
        "left": 0.0,
        "right": 0.0,
        "table": 0.0,
    }

    cube_id = model.body(cube_name).id
    wrench = np.zeros(6)

    for i in range(data.ncon):
        contact = data.contact[i]

        body1 = model.geom_bodyid[contact.geom1]
        body2 = model.geom_bodyid[contact.geom2]

        if cube_id not in (body1, body2):
            continue

        other_body = body2 if body1 == cube_id else body1
        name = model.body(other_body).name

        if name == "panda_left_finger":
            key = "left"
        elif name == "panda_right_finger":
            key = "right"
        elif name == "table":
            key = "table"
        else:
            continue

        mujoco.mj_contactForce(model, data, i, wrench)
        forces[key] += wrench[0]

    return forces

def main():
    args = parse_args()

    env = PickPlaceEnv()
    expert = PickPlaceExpert()

    task = PickPlaceTask(object_name=args.object, target_name=args.target)
    print(f"Task: Pick {task.object_name} and place it on {task.target_name}")

    observation = env.reset(task=task, seed=args.seed, randomize=args.randomize)
    expert.reset(observation, task)

    print("Number of sensors:", env.model.nsensor)

    for i in range(env.model.nsensor):
        print(env.model.sensor(i).name)
        
    print("Force:", env.data.sensor("wrist_force").data)
    print("Torque:", env.data.sensor("wrist_torque").data)

    renderer = mujoco.Renderer(
        env.model,
        height=480,
        width=640,
    )

    frames = []
    next_capture_time = env.data.time
    camera_dt = 0.1

    step_count = 0
    contact_log = []
    ft_log = []

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            
            action = expert.predict(observation)

            if action is None:
                print("Expert episode completed.")
                break

            observation, reward, terminated, truncated, info = env.step(action)

            # Log contact forces and wrist force/torque
            forces = contact_normal_forces(
                env.model,
                env.data,
                task.object_name,
            )

            contact_log.append({
                "time": env.data.time,
                "phase": expert.phase,
                **forces,
            })

            force = env.data.sensor("wrist_force").data.copy()
            torque = env.data.sensor("wrist_torque").data.copy()

            ft_log.append({
                "time": env.data.time,
                "phase": expert.phase,
                "fx": force[0],
                "fy": force[1],
                "fz": force[2],
                "tx": torque[0],
                "ty": torque[1],
                "tz": torque[2],
            })
                    
            # Capture images at a fixed interval
            if env.data.time + 1e-9 >= next_capture_time:
                renderer.update_scene(env.data, camera="overview")
                rgb = renderer.render().copy()

                # Depth
                # renderer.enable_depth_rendering()
                # renderer.update_scene(env.data, camera="overview")
                # depth = renderer.render().copy()
                # renderer.disable_depth_rendering()

                # Segmentation
                # renderer.enable_segmentation_rendering()
                # renderer.update_scene(env.data, camera="overview")
                # segmentation = renderer.render().copy()
                # renderer.disable_segmentation_rendering()

                frames.append({
                    "time": float(env.data.time),
                    "rgb": rgb,
                    "tcp_position": observation["tcp_position"].copy(),
                })

                next_capture_time += camera_dt

            error = info["cartesian_error"]
            step_count += 1

            viewer.sync()
            time.sleep(env.control_dt)


    renderer.close()
    print(f"Captured {len(frames)} frames")
    print("First frame time:", frames[0]["time"])
    print("Last frame time:", frames[-1]["time"])

    cube_position = observation["object_positions"][task.object_name]
    target_position = observation["target_positions"][task.target_name]

    placement_error_xy = (
        (cube_position[:2] - target_position[:2]) ** 2
    ).sum() ** 0.5

    print("Final cube position:", cube_position)
    print("Placement error XY:", placement_error_xy)


    df = pd.DataFrame(contact_log)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df["time"], df["left"], label="Left finger")
    ax.plot(df["time"], df["right"], label="Right finger")
    ax.plot(df["time"], df["table"], label="Table")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Normal contact force [N]")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig("outputs/contact_forces.png", dpi=150)
    plt.close(fig)

    df_ft = pd.DataFrame(ft_log)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(df_ft["time"], df_ft["fx"], label="Fx")
    ax.plot(df_ft["time"], df_ft["fy"], label="Fy")
    ax.plot(df_ft["time"], df_ft["fz"], label="Fz")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Wrist force [N]")
    ax.set_title("Wrist F/T sensor")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig("outputs/wrist_forces.png", dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    main()