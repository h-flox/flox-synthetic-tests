import numpy as np
import pandas as pd

from datetime import timedelta


def flower_results():
    data = {
        "framework": ["flower" for _ in range(7)],
        "model": ["simple" for _ in range(7)],
        "workers": [2, 4, 8, 16, 32, 64, 128],
        "flower_aggr_time": [
            0.42440681578591466,
            0.6442288761027157,
            0.6689941002987325,
            2.0164664718322456,
            2.7738771508447826,
            5.8041068678721786,
            8.649411479011178
        ],
        "time": [
            timedelta(minutes=0, seconds=9.662),
            timedelta(minutes=0, seconds=15.283),
            timedelta(minutes=0, seconds=27.484),
            timedelta(minutes=0, seconds=53.935),
            timedelta(minutes=1, seconds=48.778),
            timedelta(minutes=3, seconds=42.512),
            timedelta(minutes=7, seconds=32.002),
        ],
    }
    data["time"] = [t.total_seconds() for t in data["time"]]
    return pd.DataFrame.from_dict(data)


def flox_results():
    data_simple = {
        "framework": ["flox" for _ in range(7)],
        "model": ["simple" for _ in range(7)],
        "workers": [2, 4, 8, 16, 32, 64, 128],
        "time": [
            timedelta(seconds=32),
            timedelta(seconds=31),
            timedelta(seconds=35),
            timedelta(seconds=32),
            timedelta(seconds=36),
            timedelta(seconds=55),
            timedelta(seconds=67),
        ],
    }
    data_simple["time"] = [t.total_seconds() for t in data_simple["time"]]
    df_simple = pd.DataFrame.from_dict(data_simple)

    data_resnet18 = {
        "framework": ["flox" for _ in range(3)],
        "model": ["resnet18" for _ in range(3)],
        "workers": [2, 32, 128],
        "time": [
            timedelta(seconds=37),
            timedelta(seconds=67),
            timedelta(seconds=228),
        ],
    }
    data_resnet18["time"] = [t.total_seconds() for t in data_resnet18["time"]]
    df_resnet18 = pd.DataFrame.from_dict(data_resnet18)

    return pd.concat([df_simple, df_resnet18])


if __name__ == "__main__":
    df = pd.concat([
        flower_results(),
        flox_results(),
    ])
    df.to_feather("kyle_results.feather")
