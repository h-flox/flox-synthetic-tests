from dataclasses import dataclass, field

import torch
from flox import Flock
from flox.jobs.local_training import pure_debug_train_job
from flox.runtime.transfer import ProxyStoreTransfer
from flox.strategies.strategy import DefaultWorkerStrategy, DefaultTrainerStrategy
from globus_compute_sdk import Client
from globus_compute_sdk.serialize import (
    DillDataBase64,
    CombinedCode,
    ComputeSerializer,
    DillCodeSource,
)


class NodeKind:
    def __init__(self):
        pass

    def to_str(self) -> str:
        return "worker"


@dataclass
class Node:
    idx: int = field(default=0)
    kind: NodeKind = field(default_factory=lambda: NodeKind())


def basic_test(serializer):
    def greet(foo, bar="bar"):
        return f"{foo} {bar}!"

    fn, args, kwargs = serializer.check_strategies(greet, "foo", bar="bar")
    print(fn(*args, **kwargs))
    print(greet("foo", bar="bar"))


def local_train_debug_job_test(serializer):
    # connector = EndpointConnector([n.proxystore_endpoint for n in flock.nodes()])
    # store = Store(name="default", connector=connector)
    # transfer = BaseTransfer()
    flock = Flock.from_yaml("hawfinch_topo_small.yaml")
    transfer = ProxyStoreTransfer(flock)
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 1),
    )

    sanity_check_out = pure_debug_train_job(
        node=Node(),
        parent=Node(),
        global_model=model,
        module_state_dict=model.state_dict(),
        dataset=None,
        transfer=transfer,
        worker_strategy=DefaultWorkerStrategy(),
        trainer_strategy=DefaultTrainerStrategy(),
    )

    fn, args, kwargs = serializer.check_strategies(
        pure_debug_train_job,
        node=Node(),
        parent=Node(),
        global_model=model,
        module_state_dict=transfer.proxy(model.state_dict()),
        dataset=None,
        transfer=transfer,
        worker_strategy=DefaultWorkerStrategy(),
        trainer_strategy=DefaultTrainerStrategy(),
    )
    test_run = fn(*args, **kwargs)

    print(test_run)


if __name__ == "__main__":
    # gcx = Executor("db06abc0-68e4-4b9b-a397-cfc50212f5fc", client=gcc)

    gcc = Client(
        code_serialization_strategy=DillCodeSource(),
        data_serialization_strategy=DillDataBase64(),
    )
    serializer = ComputeSerializer(
        strategy_code=CombinedCode(), strategy_data=DillDataBase64()
    )

    basic_test(serializer)
    local_train_debug_job_test(serializer)
