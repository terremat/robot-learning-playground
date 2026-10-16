"""
This script converts the reaching expert dataset from a NumPy format to a LeRobot dataset format.

The output dataset can be inspect using the LeRobot dataset viewer:
    robot-learning-playground % lerobot-dataset-viz \
    --repo-id local/reaching_expert \
    --root data/lerobot/reaching_expert \
    --mode local \
    --episode-index 0
"""

from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


INPUT_PATH = Path("data/reaching_expert.npz")
OUTPUT_PATH = Path("data/lerobot/reaching_expert")

REPO_ID = "local/reaching_expert"
FPS = 100


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_PATH} already exists. "
            "Choose a new output directory or remove the old one."
        )

    with np.load(INPUT_PATH) as data:
        observations = data["observations"]
        actions = data["actions"]
        episode_ids = data["episode_ids"]

    assert observations.ndim == 2 and observations.shape[1] == 20
    assert actions.shape == (len(observations), 3)
    assert episode_ids.shape == (len(observations),)
    assert np.all(episode_ids[1:] >= episode_ids[:-1])

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (20,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["vx", "vy", "vz"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=OUTPUT_PATH,
        fps=FPS,
        robot_type="franka_panda",
        features=features,
        use_videos=False,
    )

    episode_count = 0

    for i in range(len(observations)):
        # When the episode changes, save the previous one.
        if i > 0 and episode_ids[i] != episode_ids[i - 1]:
            dataset.save_episode()
            episode_count += 1

        dataset.add_frame({
            "observation.state": observations[i],
            "action": actions[i],
            "task": "Reach the target position with the robot TCP",
        })

    # Save the final episode.
    if len(observations) > 0:
        dataset.save_episode()
        episode_count += 1

    dataset.finalize()

    print(f"Dataset saved to: {OUTPUT_PATH}")
    print(f"Episodes: {episode_count}")
    print(f"Frames: {len(observations)}")


if __name__ == "__main__":
    main()