
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from robot_learning_playground.policies.reaching_mlp import ReachingMLP


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    dataset = LeRobotDataset(
        repo_id="local/reaching_expert",
        root="data/lerobot/reaching_expert",
    )
    print("Dataset size:", len(dataset))

    # Split by episode, not by individual frame.
    episode_ids = np.asarray(
        dataset.hf_dataset["episode_index"]
    )

    unique_episodes = np.unique(episode_ids)
    rng = np.random.default_rng(42)
    rng.shuffle(unique_episodes)

    n_train = int(0.8 * len(unique_episodes))
    train_episodes = unique_episodes[:n_train]
    val_episodes = unique_episodes[n_train:]

    train_indices = np.where(
        np.isin(episode_ids, train_episodes)
    )[0]

    val_indices = np.where(
        np.isin(episode_ids, val_episodes)
    )[0]

    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_size=256,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        batch_size=256,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model = ReachingMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 100

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_count = 0

        for batch in train_loader:
            obs = batch["observation.state"].float().to(device)
            action = batch["action"].float().to(device)

            pred = model(obs)
            loss = loss_fn(pred, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(obs)
            train_count += len(obs)

        model.eval()
        val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                obs = batch["observation.state"].float().to(device)
                action = batch["action"].float().to(device)

                pred = model(obs)
                loss = loss_fn(pred, action)

                val_loss += loss.item() * len(obs)
                val_count += len(obs)

        train_loss /= train_count
        val_loss /= val_count

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:03d} | "
                f"Train MSE: {train_loss:.6f} | "
                f"Val MSE: {val_loss:.6f}"
            )

    output = Path("checkpoints/reaching_bc.pt")
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": 20,
            "action_dim": 3,
        },
        output,
    )

    print(f"\nModel saved to: {output}")


if __name__ == "__main__":
    main()
