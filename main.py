from pathlib import Path

import enlighten
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from datasets import PixelRegressionDataset
from models import Regressor


def main() -> None:
    num_epochs = 80
    batch_size = 32
    device = torch.device("cuda")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    manager = enlighten.Manager()

    test_bar = manager.counter(total=2, desc="Test progress", unit="models", leave=False)
    for coord_conv, ax in zip([False, True], axes):
        dataset = PixelRegressionDataset(size=32)
        assert len(dataset) % batch_size == 0

        train_data_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True
        )
        test_data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        model = Regressor(coord_conv=coord_conv).to(device)
        optimizer = torch.optim.RAdam(params=model.parameters(), lr=0.001)

        avg_losses = []
        train_bar = manager.counter(total=num_epochs, desc="Training progress", unit="epochs", leave=False)
        for _ in range(num_epochs):
            losses = []
            epoch_bar = manager.counter(total=len(train_data_loader), desc="Epoch progress", unit="iters", leave=False)
            for img, target in train_data_loader:
                img, target = img.to(device), target.to(device)

                pred = model(img)

                loss = (target - pred).pow(2).sum(dim=1).mean(dim=0)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                epoch_bar.update()
            epoch_bar.close()

            avg_loss = sum(losses) / len(losses)
            avg_losses.append(avg_loss)
            train_bar.update()
        train_bar.close()

        targets = []
        preds = []
        model.eval()
        for img, target in test_data_loader:
            img, target = img.to(device), target.to(device)
            with torch.no_grad():
                pred = model(img)
            targets.append(target)
            preds.append(pred)
        targets_np = torch.cat(targets, dim=0).cpu().numpy()
        preds_np = torch.cat(preds, dim=0).cpu().numpy()

        ax.set_title(f"coord_conv = {coord_conv}")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.scatter(x=targets_np[:, 0], y=targets_np[:, 1], color="green", s=25.0)
        ax.scatter(x=preds_np[:, 0], y=preds_np[:, 1], color="red", s=12.5)
        axes[-1].plot(avg_losses, label=f"coord_conv = {coord_conv}")

        test_bar.update()
    test_bar.close()

    axes[-1].grid(True)
    axes[-1].legend()
    axes[-1].set_xlabel("Epoch")
    axes[-1].set_ylabel("L2 loss")
    fig.subplots_adjust(wspace=0.4)
    fig.suptitle("Pixel coordinate regression")

    plt.savefig(out_dir / f"test.png")
    plt.close(fig=fig)

    manager.stop()


if __name__ == "__main__":
    main()
