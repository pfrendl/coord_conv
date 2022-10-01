from pathlib import Path

import matplotlib.pyplot as plt
import torch
from enlighten import Manager
from torch import Tensor
from torch.utils.data import DataLoader

from datasets import PixelRegressionDataset
from models import Regressor0, Regressor1, Regressor2, Regressor3


grid_size = 32
num_epochs = 1000
batch_size = 32
device = torch.device("cuda")
out_dir = Path("outputs")


def loss_fn(pred: Tensor, target: Tensor) -> Tensor:
    return (target - pred).pow(2).sum(dim=1).mean(dim=0)


def experiment(train_mask: Tensor, test_mask: Tensor, manager: Manager, experiment_name: str) -> None:
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    axes = fig.subplot_mosaic(
        """
        ABE
        CDF
        """
    )
    tests = [
        (Regressor0, axes["A"], "No CoordConv"),
        (Regressor1, axes["B"], "CoordConv at input"),
        (Regressor2, axes["C"], "CoordConv at each layer"),
        (Regressor3, axes["D"], "Attention-based model"),
    ]

    experiment_pbar = manager.counter(total=len(tests), desc="Experiment progress", unit="models", leave=False)
    for model_ctr, ax, description in tests:
        train_set = PixelRegressionDataset(size=grid_size, mask=train_mask)
        test_set = PixelRegressionDataset(size=grid_size, mask=test_mask)

        train_data_loader = DataLoader(
            dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True
        )
        test_data_loader = DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True
        )

        model = model_ctr().to(device)
        optimizer = torch.optim.RAdam(params=model.parameters(), lr=0.001)

        avg_train_losses = []
        avg_test_losses = []
        train_pbar = manager.counter(total=num_epochs, desc="Training progress", unit="epochs", leave=False)
        for _ in range(num_epochs):

            model.train()
            train_losses = []
            train_epoch_pbar = manager.counter(
                total=len(train_data_loader), desc="Train epoch progress", unit="iters", leave=False
            )
            for img, target in train_data_loader:
                img, target = img.to(device), target.to(device)

                pred = model(img)

                loss = loss_fn(pred=pred, target=target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                train_epoch_pbar.update()
            train_epoch_pbar.close()
            avg_train_losses.append(sum(train_losses) / len(train_losses))

            model.eval()
            test_losses = []
            test_epoch_pbar = manager.counter(
                total=len(test_data_loader), desc="Test epoch progress", unit="iters", leave=False
            )
            with torch.no_grad():
                for img, target in test_data_loader:
                    img, target = img.to(device), target.to(device)
                    pred = model(img)
                    loss = loss_fn(pred=pred, target=target)
                    test_losses.append(loss.item())
                    test_epoch_pbar.update()
            test_epoch_pbar.close()
            avg_test_losses.append(sum(test_losses) / len(test_losses))

            train_pbar.update()
        train_pbar.close()

        train_targets = []
        test_targets = []
        train_preds = []
        test_preds = []
        model.eval()
        with torch.no_grad():
            calc_train_outputs_pbar = manager.counter(
                total=len(train_data_loader) + len(test_data_loader),
                desc="Calculate train outputs",
                unit="iters",
                leave=False,
            )
            for img, target in train_data_loader:
                img, target = img.to(device), target.to(device)
                pred = model(img)
                train_targets.append(target)
                train_preds.append(pred)
                calc_train_outputs_pbar.update()
            calc_train_outputs_pbar.close()

            calc_test_outputs_pbar = manager.counter(
                total=len(train_data_loader) + len(test_data_loader),
                desc="Calculate test outputs",
                unit="iters",
                leave=False,
            )
            for img, target in test_data_loader:
                img, target = img.to(device), target.to(device)
                pred = model(img)
                test_targets.append(target)
                test_preds.append(pred)
                calc_test_outputs_pbar.update()
            calc_test_outputs_pbar.close()

        train_targets_np = torch.cat(train_targets, dim=0).cpu().numpy()
        test_targets_np = torch.cat(test_targets, dim=0).cpu().numpy()
        train_preds_np = torch.cat(train_preds, dim=0).cpu().numpy()
        test_preds_np = torch.cat(test_preds, dim=0).cpu().numpy()

        ax.set_title(description)
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.scatter(x=train_targets_np[:, 0], y=train_targets_np[:, 1], color="green", s=25.0, label="Train set targets")
        ax.scatter(x=test_targets_np[:, 0], y=test_targets_np[:, 1], color="blue", s=25.0, label="Test set targets")
        ax.scatter(
            x=train_preds_np[:, 0], y=train_preds_np[:, 1], color="purple", s=12.5, label="Train set predictions"
        )
        ax.scatter(x=test_preds_np[:, 0], y=test_preds_np[:, 1], color="red", s=12.5, label="Test set predictions")
        ax.legend(loc="upper right")
        axes["E"].plot(avg_train_losses, label=description)
        axes["F"].plot(avg_test_losses, label=description)

        experiment_pbar.update()
    experiment_pbar.close()

    axes["E"].set_title("Train loss")
    axes["E"].grid(True)
    axes["E"].legend()
    axes["E"].set_xlabel("Epoch")
    axes["E"].set_ylabel("L2 loss")

    axes["F"].set_title("Test loss")
    axes["F"].grid(True)
    axes["F"].legend()
    axes["F"].set_xlabel("Epoch")
    axes["F"].set_ylabel("L2 loss")

    fig.suptitle(f"Pixel coordinate regression - {experiment_name}")

    save_name = experiment_name.lower().replace(" ", "_")
    plt.savefig(out_dir / f"{save_name}.png")
    plt.close(fig=fig)


def main() -> None:
    out_dir.mkdir(exist_ok=True)
    manager = Manager()

    overfit_mask = torch.ones((grid_size, grid_size), dtype=torch.bool)
    uniform_split_mask = torch.rand((grid_size, grid_size)) < 0.75
    quadrant_split = torch.ones((grid_size, grid_size), dtype=torch.bool)
    quadrant_split[grid_size // 2 :, : grid_size // 2] = False

    train_test_masks: list[tuple[Tensor, Tensor, str]] = [
        (overfit_mask, overfit_mask, "Overfit grid"),
        (uniform_split_mask, ~uniform_split_mask, "Uniform split"),
        (quadrant_split, ~quadrant_split, "Quadrant split"),
    ]

    test_pbar = manager.counter(total=len(train_test_masks), desc="Test progress", unit="experiments", leave=False)
    for train_mask, test_mask, experiment_name in train_test_masks:
        experiment(
            train_mask=train_mask,
            test_mask=test_mask,
            manager=manager,
            experiment_name=experiment_name,
        )
        test_pbar.update()
    test_pbar.close()

    manager.stop()


if __name__ == "__main__":
    main()
