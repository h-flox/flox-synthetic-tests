import logging
import os
from time import perf_counter

import globus_compute_sdk as funcx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from flox import federated_fit
from flox.data import federated_split
from flox.flock import Flock, FlockNode
from flox.nn import FloxModule
from proxystore.connectors.endpoint import EndpointConnector
from proxystore.proxy import Proxy
from proxystore.store import Store
from torchvision import transforms
from torchvision.datasets import FashionMNIST

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


def matmul_test(flock: Flock, node: FlockNode):
    def work(data: np.ndarray | Proxy[np.ndarray], pstore: Store):
        import numpy as np

        result = np.matmul(data, data)
        return pstore.proxy(result)

    connector = EndpointConnector([n.proxystore_endpoint for n in flock.nodes()])
    store = Store(name="default", connector=connector)
    arr = np.random.randn(1000, 1000)

    with funcx.Executor(node.globus_compute_endpoint) as gce:
        start_time = perf_counter()
        fut = gce.submit(work, store.proxy(arr), store)
        print(fut.result())
        end_time = perf_counter()
        print(f"Computation took {end_time - start_time} seconds.")


def main():
    # flock = Flock.from_yaml("hawfinch_topo.yaml")
    flock = Flock.from_yaml("hawfinch_topo_small.yaml")

    # TODO: Convert this into a logical data class that just has the subset and the logic to load
    #       the data on the device.
    data = FashionMNIST(
        root=os.environ["TORCH_DATASETS"],
        train=True,
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.5, 0.5)]
        ),
    )
    fed_data = federated_split(data, flock, 10, 3.0, 1.0)
    logging.info("Starting federated fitting.")
    module, history = federated_fit(
        flock,
        None,  # SmallConvModel(),
        None,  # fed_data,
        num_global_rounds=1,
        strategy="fedsgd",
        kind="sync-v2",
        launcher_kind="globus-compute",
        debug_mode=True,
        logging=True,
    )
    print("Finished learning!")


if __name__ == "__main__":
    main()
