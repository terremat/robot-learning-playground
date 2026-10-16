
from robot_learning_playground.envs.mujoco.reaching_env import ReachingEnv
from robot_learning_playground.policies.reaching_expert import ReachingExpert


def main():
    env = ReachingEnv()
    expert = ReachingExpert()

    successes = 0
    n_episodes = 20

    for episode in range(n_episodes):
        obs, info = env.reset(seed=episode)

        for step in range(env.max_steps):
            action = expert.predict(obs)

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        successes += int(info["is_success"])

        print(
            f"Episode {episode:02d} | "
            f"Steps: {step + 1:3d} | "
            f"Distance: {info['distance']:.4f} m | "
            f"Success: {info['is_success']}"
        )

    print(f"\nSuccess rate: {successes / n_episodes:.1%}")


if __name__ == "__main__":
    main()
