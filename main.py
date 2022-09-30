from pathlib import Path

import matplotlib.pyplot as plt
import torch
from enlighten import Manager
from torch import Tensor
from torch.utils.data import DataLoader

from datasets import PixelRegressionDataset
from models import Regressor0, Regressor1, Regressor2


grid_size = 32
num_epochs = 80
batch_size = 32
device = torch.device("cuda")


def experiment(train_mask: Tensor, test_mask: Tensor, manager: Manager, save_path: Path) -> None:
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    axes = fig.subplot_mosaic(
        """
        ABC
        DDD
        """
    )
    tests = [
        (Regressor0, axes["A"], "No CoordConv"),
        (Regressor1, axes["B"], "CoordConv at input"),
        (Regressor2, axes["C"], "CoordConv at each layer"),
    ]

    experiment_pbar = manager.counter(total=len(tests), desc="Experiment progress", unit="models", leave=False)
    for model_ctr, ax, description in tests:
        trainset = PixelRegressionDataset(size=grid_size, mask=train_mask)
        testset = PixelRegressionDataset(size=grid_size, mask=test_mask)

        train_data_loader = DataLoader(
            dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True
        )
        test_data_loader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=2)

        model = model_ctr().to(device)
        optimizer = torch.optim.RAdam(params=model.parameters(), lr=0.001)

        avg_losses = []
        train_pbar = manager.counter(total=num_epochs, desc="Training progress", unit="epochs", leave=False)
        for _ in range(num_epochs):
            losses = []
            epoch_pbar = manager.counter(total=len(train_data_loader), desc="Epoch progress", unit="iters", leave=False)
            for img, target in train_data_loader:
                img, target = img.to(device), target.to(device)

                pred = model(img)

                loss = (target - pred).pow(2).sum(dim=1).mean(dim=0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                epoch_pbar.update()
            epoch_pbar.close()

            avg_loss = sum(losses) / len(losses)
            avg_losses.append(avg_loss)
            train_pbar.update()
        train_pbar.close()

        train_targets = []
        test_targets = []
        preds = []
        model.eval()
        for img, target in train_data_loader:
            img, target = img.to(device), target.to(device)
            with torch.no_grad():
                pred = model(img)
            train_targets.append(target)
            preds.append(pred)
        for img, target in test_data_loader:
            img, target = img.to(device), target.to(device)
            with torch.no_grad():
                pred = model(img)
            test_targets.append(target)
            preds.append(pred)
        train_targets_np = torch.cat(train_targets, dim=0).cpu().numpy()
        test_targets_np = torch.cat(test_targets, dim=0).cpu().numpy()
        preds_np = torch.cat(preds, dim=0).cpu().numpy()

        ax.set_title(description)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.scatter(x=train_targets_np[:, 0], y=train_targets_np[:, 1], color="green", s=25.0, label="Train set")
        ax.scatter(x=test_targets_np[:, 0], y=test_targets_np[:, 1], color="blue", s=25.0, label="Test set")
        ax.scatter(x=preds_np[:, 0], y=preds_np[:, 1], color="red", s=12.5, label="Predictions")
        ax.legend()
        axes["D"].plot(avg_losses, label=description)

        experiment_pbar.update()
    experiment_pbar.close()

    axes["D"].grid(True)
    axes["D"].legend()
    axes["D"].set_xlabel("Epoch")
    axes["D"].set_ylabel("L2 loss")
    fig.suptitle("Pixel coordinate regression")

    plt.savefig(save_path)
    plt.close(fig=fig)


def main() -> None:
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    manager = Manager()

    overfit_mask = torch.ones((grid_size, grid_size), dtype=torch.bool)
    uniform_split_mask = torch.rand((grid_size, grid_size)) < 0.75
    quadrant_split = torch.ones((grid_size, grid_size), dtype=torch.bool)
    quadrant_split[grid_size // 2 :, grid_size // 2 :] = False

    train_test_masks: list[tuple[Tensor, Tensor, str]] = [
        (overfit_mask, overfit_mask, "Overfit grid"),
        (uniform_split_mask, ~uniform_split_mask, "Uniform split"),
        (quadrant_split, ~quadrant_split, "Quadrant split"),
    ]

    test_pbar = manager.counter(total=len(train_test_masks), desc="Test progress", unit="experiments", leave=False)
    for train_mask, test_mask, experiment_name in train_test_masks:
        save_name = experiment_name.lower().replace(" ", "_")
        experiment(train_mask=train_mask, test_mask=test_mask, manager=manager, save_path=out_dir / f"{save_name}.png")
        test_pbar.update()
    test_pbar.close()

    manager.stop()


if __name__ == "__main__":
    main()
