import argparse
import itertools
import os

import flox
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from flox.data import federated_split
from flox.nn import FloxModule
from topo import balanced_flock
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from consts import *


class SmallConvModel(FloxModule):
    def __init__(self, lr: float = DEFAULT_LR, device: str | None = None):
        super().__init__()
        self.lr = lr
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10)
        self.last_accuracy = None

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        inputs = inputs.to(DEVICE)
        targets = targets.to(DEVICE)
        preds = self(inputs)
        loss = F.cross_entropy(preds, targets)

        self.last_accuracy = self.accuracy(preds, targets)

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=self.lr)


def main(args: argparse.Namespace):
    suite = {
        "trials": list(range(args.trials)),
        "labels_alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "samples_alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "num_workers": [5, 10, 15],
    }

    data = FashionMNIST(
        root=os.environ["TORCH_DATASETS"],
        download=False,
        train=True,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
            ]
        ),
    )
    histories = []

    for s in itertools.product(*suite.values()):
        trials, labels_alpha, samples_alpha, num_workers = s
        flock = balanced_flock(20, (2, 4))
        fed_data = federated_split(
            data,
            flock,
            num_classes=10,
            labels_alpha=labels_alpha,
            samples_alpha=samples_alpha,
        )
        print(f"{flock.number_of_workers=}")

        _, his = flox.federated_fit(
            flock,
            SmallConvModel(),
            fed_data,
            num_global_rounds=1,
            strategy="fedsgd",
            launcher_cfg={"max_workers": 10},
        )
        histories.append(his)
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", "-t", type=int, default=1)
    parser.add_argument("--rounds", "-r", type=int, default=10)
    parser.add_argument("--epochs", "-e", type=int, default=3)
    main(parser.parse_args())
