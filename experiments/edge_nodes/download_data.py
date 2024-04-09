import argparse

from torchvision import transforms
from torchvision.datasets import FashionMNIST

if __name__ == "__main__":
    args = argparse = argparse.ArgumentParser()
    args.add_argument("--root", "-r", required=True, help="")
    parsed_args = args.parse_args()

    FashionMNIST(
        root=parsed_args.root,
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        ),
    )
