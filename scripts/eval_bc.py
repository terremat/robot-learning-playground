import os
import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch

import matplotlib.pyplot as plt

from robot_learning_playground.envs.mujoco.reaching_env import ReachingEnv
from robot_learning_playground.policies.reaching_mlp import ReachingMLP
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
            rgba=[1, 0.2, 0.2, 0.8],
        )
        scene.ngeom = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--target-scale", type=float, default=1.0)
    parser.add_argument("--policy", choices=["expert", "bc"], default="bc")
    parser.add_argument("--checkpoint", default="checkpoints/reaching_bc.pt")
    parser.add_argument("--plot-actions", action="store_true")
    args = parser.parse_args()

    env = ReachingEnv(target_scale=args.target_scale)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    policy = ReachingMLP(
        obs_dim=checkpoint["obs_dim"],
        action_dim=checkpoint["action_dim"],
    )
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()

    expert = ReachingExpert()

    successes = 0
    distances = []
    steps_to_finish = []

    viewer = (
        mujoco.viewer.launch_passive(env.model, env.data)
        if args.render
        else None
    )

    try:
        for episode in range(args.episodes):
            # Different seeds from training demonstrations
            obs, info = env.reset(seed=10_000 + episode)
            actions = []
            
            if viewer is not None:
                if not viewer.is_running():
                    break
                draw_target(viewer, env.target)
                viewer.sync()
                time.sleep(0.5)

            for step in range(env.max_steps):

                # Policy inference
                if args.policy == "expert":
                    action = expert.predict(obs)
                else:
                    with torch.inference_mode():
                        obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                        action = policy(obs_tensor).squeeze(0).numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                actions.append(action)

                if viewer is not None:
                    if not viewer.is_running():
                        break
                    draw_target(viewer, env.target)
                    viewer.sync()
                    time.sleep(env.control_dt)

                if terminated or truncated:
                    break

            successes += int(info["is_success"])
            distances.append(info["distance"])
            steps_to_finish.append(step + 1)

            print(
                f"Episode {episode:03d} | "
                f"Steps: {step + 1:3d} | "
                f"Distance: {info['distance']:.4f} m | "
                f"Success: {info['is_success']}"
            )

            os.makedirs(f"outputs/{args.policy}_{args.target_scale}", exist_ok=True)
            if not args.render and args.plot_actions:
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.plot(actions)
                plt.title("Actions over time")
                plt.xlabel("Step")
                plt.ylabel("Action value")
                plt.subplot(1, 2, 2)
                plt.hist(distances, bins=20, color="blue", alpha=0.7)
                plt.title("Final distances histogram")
                plt.xlabel("Distance (m)")
                plt.ylabel("Frequency")
                plt.tight_layout()
                plt.savefig(f"outputs/{args.policy}_{args.target_scale}/eval_episode_{episode:03d}.png")

    finally:
        if viewer is not None:
            viewer.close()

    n = len(distances)

    print("\n===== BC EVALUATION =====")
    print(f"Episodes evaluated: {n}")
    if n:
        print(f"Success rate: {successes / n:.1%}")
        print(f"Mean final distance: {np.mean(distances):.4f} m")
        print(f"Mean episode steps: {np.mean(steps_to_finish):.1f}")

    with open(f"outputs/{args.policy}_{args.target_scale}/eval_summary.txt", "w") as f:
        f.write("===== BC EVALUATION =====\n")
        f.write(f"Episodes evaluated: {n}\n")
        if n:
            f.write(f"Success rate: {successes / n:.1%}\n")
            f.write(f"Mean final distance: {np.mean(distances):.4f} m\n")
            f.write(f"Mean episode steps: {np.mean(steps_to_finish):.1f}\n")


if __name__ == "__main__":
    main()
