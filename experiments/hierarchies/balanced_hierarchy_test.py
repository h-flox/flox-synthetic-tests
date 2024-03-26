import argparse
import logging
from datetime import datetime
from pathlib import Path

import flox
import pandas as pd
from balanced_topo import balanced_tree_with_fixed_leaves
from flox import Flock
from flox.data import federated_split
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from models import get_model

logging.basicConfig(
    format="(%(levelname)s  - %(asctime)s) ❯ %(message)s", level=logging.INFO
)


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
    logging.info("Beginning FL process.")
    model = get_model(parsed_args.model)
    _, results = flox.federated_fit(
        flock,
        model,
        fed_data,
        num_global_rounds=parsed_args.rounds,
        strategy="fedavg",
        kind="sync-v2",
        debug_mode=False,
        launcher_kind="process",
        launcher_cfg={"max_workers": parsed_args.exec_workers},
    )
    logging.info("Finished FL process.")

    return results


def main(parsed_args: argparse.Namespace):
    outdir = Path(parsed_args.outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)

    results = []
    heights = (1, 2, 4, 8)
    timestamp = str(datetime.now()).split(".")[0].replace(" ", "__")
    for i, h in enumerate(heights):
        logging.info(f"STARTING test {i+1} out of {len(heights)}.")
        res = single_test(parsed_args, h)
        results.append(res)
        out_path = outdir / f"{timestamp}__height={h}__workers={args.workers}.feather"
        res.to_feather(out_path)
        logging.info(f"FINISHED test {i+1} out of {len(heights)}.")

    return results


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--root",
        "-r",
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
    args.add_argument("--model", "-m", type=int, default=0)
    args.add_argument("--exec_workers", "-e", type=int, default=10)
    args.add_argument("--labels_alpha", "-l", type=float, default=100.0)
    args.add_argument("--samples_alpha", "-s", type=float, default=100.0)
    args.add_argument("--participation", "-p", type=float, default=1.0)
    args.add_argument("--rounds", "-r", type=int, default=200)
    args.add_argument(
        "--workers", "-w", type=int, default=256
    )  # nathaniel-hudson: Don't recommend changing.

    main(args.parse_args())
