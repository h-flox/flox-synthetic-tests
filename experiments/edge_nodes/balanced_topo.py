import math
import typing as t

import networkx as nx


def num_leaves(graph: nx.DiGraph) -> int:
    if not nx.is_tree(graph):
        raise ValueError("Graph is not a tree.")

    count = 0
    for node in graph.nodes():
        if len(list(graph.successors(node))) == 0:
            count += 1

    return count


def balanced_tree_with_fixed_leaves(
    leaves: int,
    height: int,
    create_using=nx.DiGraph,
    rounding: t.Literal["round", "floor", "ceil"] = "round",
) -> nx.DiGraph:
    r"""
    Creates a balanced tree with a (roughly) fixed number of leaves.

    By default, `networkx` provides the `balanced_tree` function which generates a balanced
    tree using the branching factor and the height of the tree. This function wraps that function
    and computes the branching factor using $b = \lfloor l^{1/h} \rfloor where $l$ is the number
    of leaves and $h$ is the height.

    Notes:
        Because the calculation for $b$ (described above) is not always going to result in an
        integer, this function will use the floor of $l^{1/h}$. Unless you are wise about your
        parameters for `leaves` and `height`, you will have more leaves than originally specified.
        So, be mindful of this.

    Args:
        leaves (int): Approximate number of leaves in the resulting tree (see note).
        height (int): Height of the tree.
        create_using: Is used to specify the type of network constructed.
            Defaults to `nx.DiGraph`.
        rounding (t.Literal): How to round the branching factor.

    Returns:
        The constructed, balanced tree.
    """
    if leaves < 1:
        raise ValueError("Value for arg `leaves` must be at least 1.")

    branching_factor = leaves ** (1 / height)
    match rounding:
        case "round":
            r = round(branching_factor)
        case "floor":
            r = math.floor(branching_factor)
        case "ceil":
            r = math.ceil(branching_factor)
        case _:
            raise ValueError(f"Illegal value for arg `rounding`.")

    return nx.balanced_tree(r, height, create_using)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    leaves = 10000
    x, y = [], []
    for h in range(1, 10 + 1):
        tree = balanced_tree_with_fixed_leaves(1000, 4)
        bf = leaves ** (1 / h)
        x.append(h)
        y.append(bf)

    plt.plot(x, y)
    plt.xlabel("Height")
    plt.ylabel("Branching Factor")
    plt.yscale("log")
    plt.show()
