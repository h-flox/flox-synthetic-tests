import argparse
import logging
import pickle
import flox
import pandas as pd
import torch
import torchmetrics

from datetime import datetime
from pathlib import Path
from torch import nn
from torch.nn import functional as F

from flox import Flock
from flox.data import federated_split
from flox.nn import FloxModule
from flox.strategies import load_strategy
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from balanced_topo import balanced_tree_with_fixed_leaves


logging.basicConfig(
    format="(%(levelname)s  - %(asctime)s) ❯ %(message)s", level=logging.INFO
)


class SmallConvModel(FloxModule):
    def __init__(self, lr: float = 0.01, device: str | None = None):
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
        preds = self.forward(inputs)
        loss = F.cross_entropy(preds, targets)
        self.last_accuracy = self.accuracy(preds, targets)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=self.lr)


def single_test(parsed_args: argparse.Namespace, height: int) -> pd.DataFrame:
    # STEP 1: Load the data to be trained on.
    data = FashionMNIST(
        root=parsed_args.root,
        download=False,
        train=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        ),
    )
    logging.info("FashionMNIST data is loaded.")

    # STEP 2: Stage the federated learning process by setting up the topo and partitioning the data.
    tree = balanced_tree_with_fixed_leaves(parsed_args.workers, height)
    flock = Flock(tree)
    logging.info("Flock topology initialized.")
    fed_data = federated_split(
        data, flock, 10, parsed_args.samples_alpha, parsed_args.labels_alpha
    )
    logging.info("FashionMNIST data partitioning across the flock topology completed.")

    # STEP 3: Run the FL process and return the results.
    results = []
    for strategy_name in ["fedavg", "fedprox", "fedsgd"]:
        logging.info(f"Beginning FL process for {strategy_name=}.")
        strategy = load_strategy(strategy_name, participation=parsed_args.participation)
        torch.manual_seed(0)
        _, res = flox.federated_fit(
            flock,
            SmallConvModel(),
            fed_data,
            num_global_rounds=parsed_args.rounds,
            # strategy="fedavg",
            strategy=strategy,
            kind="sync-v2",
            debug_mode=False,
            launcher_kind="process",
            launcher_cfg={"max_workers": parsed_args.exec_workers},
        )
        res["strategy"] = strategy_name
        results.append(res)
        logging.info(f"Finished FL process for {strategy_name=}.")

    return pd.concat(results)


def main(parsed_args: argparse.Namespace):
    timestamp = str(datetime.now()).split(".")[0].replace(" ", "__")
    outdir = Path(parsed_args.outdir)
    outdir = outdir / timestamp
    if not outdir.exists():
        outdir.mkdir(parents=True)

    results = []
    heights = (1, 2, 4, 8)
    for i, h in enumerate(heights):
        logging.info(f"STARTING test {i+1} out of {len(heights)}.")
        res = single_test(parsed_args, h)
        res["height"] = h
        res["workers"] = parsed_args.workers
        results.append(res)
        out_path = outdir / f"height={h}__workers={parsed_args.workers}.feather"

        res.to_feather(out_path)
        with open(outdir / f"{timestamp}__config.pkl", "wb") as f:
            pickle.dump(parsed_args, f)

        logging.info(f"FINISHED test {i+1} out of {len(heights)}.")

    return results


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--root",
        required=True,
        type=str,
        help="The root directory of where the FashionMNIST data are stored.",
    )
    args.add_argument(
        "--outdir",
        "-o",
        type=str,
        required=True,
        help="The directory where the results are stored.",
    )
    args.add_argument("--model", "-m", type=int, default=1)
    args.add_argument("--exec_workers", "-e", type=int, default=8)
    args.add_argument("--labels_alpha", "-l", type=float, default=1.0)
    args.add_argument("--samples_alpha", "-s", type=float, default=3.0)
    args.add_argument("--participation", "-p", type=float, default=1 / 32)  # 1/32
    args.add_argument("--rounds", "-r", type=int, default=200)
    args.add_argument(
        "--workers", "-w", type=int, default=256
    )  # nathaniel-hudson: Don't recommend changing.

    main(args.parse_args())
