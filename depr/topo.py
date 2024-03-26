import networkx as nx
import random
import typing as t
from flox.flock import Flock, NodeKind


def choose_parents(tree: nx.DiGraph, children, parents):
    children_without_parents = [child for child in children]

    for parent in parents:
        child = random.choice(children_without_parents)
        children_without_parents.remove(child)
        tree.add_edge(parent, child)

    for child in children_without_parents:
        parent = random.choice(parents)
        tree.add_edge(parent, child)


def balanced_flock(
        workers: int, aggr_shape: t.Collection[int] | None = None, return_nx: bool = False
) -> Flock | nx.DiGraph:
    client_idx = "client"
    flock = nx.DiGraph()
    flock.add_node(
        client_idx,
        kind=NodeKind.LEADER,
        proxystore_endpoint=None,
        globus_compute_endpoint=None,
    )
    worker_nodes = []
    for i in range(workers):
        idx = f"w{i + 1}"
        flock.add_node(
            idx,
            kind=NodeKind.WORKER,
            proxystore_endpoint=None,
            globus_compute_endpoint=None,
        )
        worker_nodes.append(idx)

    if aggr_shape is None:
        for worker in worker_nodes:
            flock.add_edge(client_idx, worker)
        return flock

    # Validate the values of the `aggr_shape` argument.
    for i in range(len(aggr_shape) - 1):
        v0, v1 = aggr_shape[i], aggr_shape[i + 1]
        if v0 > v1:
            raise ValueError(
                "Argument `aggr_shape` must have ascending values "
                "(i.e., no value can be larger than the preceding value)."
            )
        if not 0 < v0 <= workers or not 0 < v1 <= workers:
            raise ValueError(
                f"Values in `aggr_shape` must be in range (0, `{workers=}`]."
            )

    aggr_idx = 1
    last_aggrs = [client_idx]
    for num_aggrs in aggr_shape:
        if not 0 < num_aggrs <= workers:
            raise ValueError("")

        curr_aggrs = []
        for aggr in range(num_aggrs):
            idx = f"a{aggr_idx}"
            flock.add_node(
                idx,
                kind=NodeKind.AGGREGATOR,
                proxystore_endpoint=None,
                globus_compute_endpoint=None,
            )
            curr_aggrs.append(idx)
            aggr_idx += 1

        choose_parents(flock, curr_aggrs, last_aggrs)
        last_aggrs = curr_aggrs

    choose_parents(flock, worker_nodes, last_aggrs)

    if return_nx:
        return flock
    return Flock(flock)
