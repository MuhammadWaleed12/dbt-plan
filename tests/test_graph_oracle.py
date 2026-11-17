"""Independent reachability checks for batched downstream traversal."""

import random

import pytest

from dbt_plan.manifest import find_downstream_batch


def transitive_closure(
    roots: list[str],
    child_map: dict[str, list[str]],
    *,
    models_only: bool,
) -> dict[str, list[str]]:
    """Compute reachability with a matrix, independently of production BFS."""
    nodes = sorted(
        set(roots)
        | set(child_map)
        | {child for children in child_map.values() for child in children}
    )
    positions = {node: index for index, node in enumerate(nodes)}
    reachable = [[False] * len(nodes) for _ in nodes]

    for parent, children in child_map.items():
        for child in children:
            reachable[positions[parent]][positions[child]] = True

    for intermediate in range(len(nodes)):
        for source in range(len(nodes)):
            if not reachable[source][intermediate]:
                continue
            for target in range(len(nodes)):
                reachable[source][target] = (
                    reachable[source][target] or (reachable[intermediate][target])
                )

    return {
        root: [
            node
            for node in nodes
            if node != root
            and reachable[positions[root]][positions[node]]
            and (not models_only or node.startswith("model."))
        ]
        for root in roots
    }


def assert_matches_oracle(
    roots: list[str],
    child_map: dict[str, list[str]],
    results: dict[str, list[str]],
    *,
    models_only: bool,
    context: str,
) -> None:
    expected = transitive_closure(roots, child_map, models_only=models_only)
    assert results == expected, context
    for root, downstream in results.items():
        assert root not in downstream, context
        assert downstream == sorted(set(downstream)), context


def generated_graph(seed: int, node_count: int = 20) -> dict[str, list[str]]:
    """Build a bounded graph with fixed edge cases plus seeded extra edges."""
    if node_count < 12:
        raise ValueError("node_count must be at least 12")

    rng = random.Random(seed)
    models = [f"model.bookstore.m{index}" for index in range(node_count - 4)]
    non_models = [
        "test.bookstore.not_null_m0",
        "test.bookstore.unique_m4",
        "exposure.bookstore.dashboard",
        "source.bookstore.orders",
    ]
    nodes = models + non_models
    graph = {node: [] for node in nodes}

    fixed_edges = [
        (models[0], models[1]),
        (models[0], models[1]),  # duplicate edge
        (models[1], models[2]),
        (models[2], models[0]),  # cycle
        (models[3], models[3]),  # self-loop
        (models[4], non_models[0]),
        (non_models[0], models[5]),
        (models[5], non_models[2]),
        (non_models[2], models[6]),
        (models[0], models[7]),
        (models[4], models[7]),  # shared subtree
    ]
    for parent, child in fixed_edges:
        graph[parent].append(child)

    # Keep the last model disconnected while varying the rest of the graph.
    connected_nodes = nodes[:-5] + non_models
    for parent in connected_nodes:
        for child in connected_nodes:
            if rng.random() < 0.08:
                graph[parent].append(child)
                if rng.random() < 0.15:
                    graph[parent].append(child)

    return graph


def permute_graph(child_map: dict[str, list[str]], seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    items = list(child_map.items())
    rng.shuffle(items)
    permuted: dict[str, list[str]] = {}
    for node, children in items:
        shuffled_children = list(children)
        rng.shuffle(shuffled_children)
        permuted[node] = shuffled_children
    return permuted


def test_transitive_closure_hand_calculated_controls() -> None:
    child_map = {
        "model.bookstore.a": ["model.bookstore.b", "model.bookstore.c"],
        "model.bookstore.b": ["model.bookstore.d"],
        "model.bookstore.c": ["model.bookstore.d"],
        "model.bookstore.d": ["model.bookstore.a", "test.bookstore.unique_d"],
        "test.bookstore.unique_d": ["exposure.bookstore.dashboard"],
        "exposure.bookstore.dashboard": [],
    }
    roots = ["model.bookstore.a", "model.bookstore.b"]

    assert transitive_closure(roots, child_map, models_only=True) == {
        "model.bookstore.a": [
            "model.bookstore.b",
            "model.bookstore.c",
            "model.bookstore.d",
        ],
        "model.bookstore.b": [
            "model.bookstore.a",
            "model.bookstore.c",
            "model.bookstore.d",
        ],
    }
    assert transitive_closure(roots[:1], child_map, models_only=False) == {
        "model.bookstore.a": [
            "exposure.bookstore.dashboard",
            "model.bookstore.b",
            "model.bookstore.c",
            "model.bookstore.d",
            "test.bookstore.unique_d",
        ]
    }


def test_oracle_rejects_a_dropped_reachable_result() -> None:
    child_map = {
        "model.bookstore.orders": ["model.bookstore.order_items"],
        "model.bookstore.order_items": ["model.bookstore.revenue"],
        "model.bookstore.revenue": [],
    }
    roots = ["model.bookstore.orders"]
    incomplete = {"model.bookstore.orders": ["model.bookstore.order_items"]}

    with pytest.raises(AssertionError):
        assert_matches_oracle(
            roots,
            child_map,
            incomplete,
            models_only=True,
            context="negative control",
        )


@pytest.mark.parametrize("models_only", [True, False])
def test_batch_matches_independent_oracle_across_seeded_graphs(
    models_only: bool,
) -> None:
    for seed in range(50):
        child_map = generated_graph(seed)
        roots = list(child_map)
        random.Random(seed + 1_000).shuffle(roots)
        permuted = permute_graph(child_map, seed + 2_000)
        edges = [(parent, child) for parent, children in child_map.items() for child in children]
        context = f"seed={seed}, models_only={models_only}, edges={edges!r}"

        results = find_downstream_batch(roots, permuted, models_only=models_only)

        assert_matches_oracle(
            roots,
            child_map,
            results,
            models_only=models_only,
            context=context,
        )
