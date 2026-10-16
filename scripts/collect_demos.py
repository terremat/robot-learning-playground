"""
This script collects demonstrations from the ReachingEnv using the ReachingExpert policy.
"""

import argparse
import time
import numpy as np

from pathlib import Path
from contextlib import nullcontext

import mujoco.viewer

from robot_learning_playground.envs.mujoco.reaching_env import ReachingEnv
from robot_learning_playground.policies.reaching_expert import ReachingExpert

def draw_target(viewer, target):
    with viewer.lock():
        scene = viewer.user_scn
        scene.ngeom = 0

        mujoco.mjv_initGeom(
            scene.geoms[0],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.015, 0, 0],
            pos=target,
            mat=np.eye(3).flatten(),
            rgba=[0.2, 1.0, 0.2, 0.8],
        )
        scene.ngeom = 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    env = ReachingEnv()
    expert = ReachingExpert()

    observations = []
    actions = []
    episode_ids = []

    n_episodes = args.episodes
    successes = 0

    viewer_context = (
        mujoco.viewer.launch_passive(env.model, env.data)
        if args.render
        else nullcontext()
    )

    with viewer_context as viewer:
        for episode in range(n_episodes):
            obs, info = env.reset(seed=1000 + episode)

            if viewer is not None:
                draw_target(viewer, env.target)
                viewer.sync()
                time.sleep(0.5)  # Pausa per vedere il nuovo target

            for step in range(env.max_steps):
                if viewer is not None and not viewer.is_running():
                    print("Viewer closed. Stopping collection.")
                    return
        
                action = expert.predict(obs)

                # Save (observation_t, action_t)
                observations.append(obs.copy())
                actions.append(action.copy())
                episode_ids.append(episode)

                obs, reward, terminated, truncated, info = env.step(action)

                if viewer is not None:
                    draw_target(viewer, env.target)
                    viewer.sync()
                    time.sleep(env.control_dt)

                if terminated or truncated:
                    break

            successes += int(info["is_success"])

            if (episode + 1) % 10 == 0:
                print(
                    f"Episodes: {episode + 1:03d}/{n_episodes:03d} | "
                    f"Success rate: {successes / (episode + 1):.1%}"
                )

    observations = np.stack(observations).astype(np.float32)
    actions = np.stack(actions).astype(np.float32)
    episode_ids = np.asarray(episode_ids, dtype=np.int32)

    output = Path("data/reaching_expert.npz")
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        observations=observations,
        actions=actions,
        episode_ids=episode_ids,
    )

    print(f"\nSaved to: {output}")
    print("Observations:", observations.shape)
    print("Actions:", actions.shape)
    print("Episode IDs:", episode_ids.shape)
    print("Success rate:", f"{successes / n_episodes:.1%}")

if __name__ == "__main__":
    main()
