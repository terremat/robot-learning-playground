
import argparse
from pathlib import Path

import numpy as np
import torch

from robot_learning_playground.envs.mujoco.reaching_env import ReachingEnv
from robot_learning_playground.policies.reaching_expert import ReachingExpert
from robot_learning_playground.policies.reaching_mlp import ReachingMLP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--target-scale", type=float, default=2.0)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/reaching_bc.pt",
    )
    parser.add_argument(
        "--output",
        default="data/reaching_dagger.npz",
    )
    args = parser.parse_args()

    env = ReachingEnv(target_scale=args.target_scale)
    expert = ReachingExpert()

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

    observations = []
    expert_actions = []
    episode_ids = []

    successes = 0

    for episode in range(args.episodes):
        obs, info = env.reset(seed=20_000 + episode)

        for step in range(env.max_steps):
            # Action chosen by the learned policy
            with torch.inference_mode():
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                bc_action = policy(obs_tensor).squeeze(0).numpy()

            # Expert label for the same observation
            expert_action = expert.predict(obs)

            # Store the expert label, NOT the BC action
            observations.append(obs.copy())
            expert_actions.append(expert_action.copy())
            episode_ids.append(episode)

            # The BC policy controls the robot
            obs, reward, terminated, truncated, info = env.step(bc_action)

            if terminated or truncated:
                break

        successes += int(info["is_success"])

        if (episode + 1) % 10 == 0:
            print(
                f"Episodes: {episode + 1}/{args.episodes} | "
                f"BC success: {successes / (episode + 1):.1%}"
            )

    observations = np.stack(observations).astype(np.float32)
    expert_actions = np.stack(expert_actions).astype(np.float32)
    episode_ids = np.asarray(episode_ids, dtype=np.int32)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output,
        observations=observations,
        actions=expert_actions,
        episode_ids=episode_ids,
    )

    print(f"\nSaved to: {output}")
    print("Observations:", observations.shape)
    print("Expert actions:", expert_actions.shape)
    print("Episodes:", args.episodes)
    print("BC success during collection:", f"{successes / args.episodes:.1%}")


if __name__ == "__main__":
    main()
