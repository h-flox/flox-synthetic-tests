import typing as t

import torch
from torch import nn
from torch.nn import functional as F

if t.TYPE_CHECKING:
    from flox.nn import FloxModule


class Net(FloxModule):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.last_accuracy = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        return loss


class ResNet(FloxModule):
    def __init__(self, kind: int | str, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.classifier_head = None

        if isinstance(kind, str):
            kind = int(kind)
        assert isinstance(kind, int)

        match kind:
            case 18:
                from torchvision.models import resnet18

                self._model = resnet18(weights=None)
            case 34:
                from torchvision.models import resnet34

                self._model = resnet34(weights=None)
            case 50:
                from torchvision.models import resnet50

                self._model = resnet50(weights=None)
            case 101:
                from torchvision.models import resnet101

                self._model = resnet101(weights=None)
            case 152:
                from torchvision.models import resnet152

                self._model = resnet152(weights=None)
            case _:
                options = (18, 34, 50, 101, 152)
                raise ValueError(
                    f"Illegal value for `ResNet` kind. Must be one of {options}."
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._model(x)
        x = torch.flatten(x)
        if self.classifier_head is None:
            in_features = x.shape[0]
            out_features = self.num_classes
            self.classifier_head = nn.Linear(in_features, out_features)
        x = self.classifier_head(x)
        return x

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        return loss


def get_model(name: int | str) -> FloxModule:
    match name:
        case 0:
            return Net()
        case 18 | "18" | "resnet18":
            return ResNet(18)
        case 34 | "34" | "resnet34":
            return ResNet(34)
        case 50 | "50" | "resnet50":
            return ResNet(50)
        case 101 | "101" | "resnet101":
            return ResNet(101)
        case 152 | "152" | "resnet152":
            return ResNet(152)
