from __future__ import annotations

import typing as t
import warnings

import pandas as pd
from torch import nn
from torch.nn import functional as F
from torchvision.models import (
    squeezenet1_0,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

if t.TYPE_CHECKING:
    import torch

MODEL_CODES = [0, 1, 3, 18, 34, 50, 101, 152]


class TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SmallNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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


def get_model(model_code: int) -> nn.Module:
    match model_code:
        case 0:
            return TinyNet()
        case 1:
            return SmallNet()
        case 3:
            return squeezenet1_0(weights=None)
        case 18:
            return resnet18(weights=None)
        case 34:
            return resnet34(weights=None)
        case 50:
            return resnet50(weights=None)
        case 101:
            return resnet101(weights=None)
        case 152:
            return resnet152(weights=None)


def get_model_size(model: torch.nn.Module) -> int:
    """
    This is used because the size reported by `sys.getsizeof(...)` is
    incorrect for PyTorch modules.

    Reference:
        https://discuss.pytorch.org/t/finding-model-size/130275/2

    Args:
        model (torch.nn.Module): The PyTorch model.

    Returns:
        Size of the PyTorch module in bytes.
    """
    param_size = sum(
        param.nelement() * param.element_size() for param in model.parameters()
    )
    buffer_size = sum(
        buffer.nelement() * buffer.element_size() for buffer in model.buffers()
    )
    byte = param_size + buffer_size
    return byte


def count_parameters(model: nn.Module) -> int:
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    records: list[dict[str, float]] = []
    for code in MODEL_CODES:
        print(f"❯ Loading model '{code}'... ", end="")
        # model = AutoModelForImageClassification.from_pretrained(model_name)
        model = get_model(code)

        byte = get_model_size(model)
        kilo = byte / 1024
        mega = kilo / 1024
        giga = mega / 1024

        preface = f"model size for {model.__class__.__name__} is"
        if giga > 1:
            print(f"{preface} {giga:0.3f} GB")
        elif mega > 1:
            print(f"{preface} {mega:0.3f} MB")
        elif kilo > 1:
            print(f"{preface} {kilo:0.3f} KB")
        else:
            print(f"{preface} {byte:0.3f} bytes")

        records.append(
            {
                "model_name": model.__class__.__name__,
                # "num_layers": int(model_name.split("-")[-1]),
                "model_size_gb": giga,
                "model_size_mb": mega,
                "model_size_kb": kilo,
                "model_size_bytes": byte,
                "num_params": count_parameters(model),
            }
        )

    data = pd.DataFrame.from_records(records)
    data.to_csv("resnet-sizes.csv")
