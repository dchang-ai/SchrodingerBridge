"""Structural restrictions concern the affine conditioner, not the full score."""
import json
from collections import Counter
from itertools import combinations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dsb.local_made import LocalMADE
from dsb.made import MADE


@pytest.mark.parametrize("dim,hidden,depth,constraints", [
    (3, 9, 1, {}),
    (3, 17, 3, {"max_degree": 3}),
    (2, 24, 2, {"radius": 2.5}),
    (2, 31, 3, {"planar": True}),
    (2, 64, 2, {"max_degree": 5, "radius": 3.0, "planar": True}),
])
def test_graph_constraints_reachability_and_conditional_jacobian(dim, hidden, depth, constraints):
    net = LocalMADE(dim, hidden, depth, components=2, key=jax.random.PRNGKey(7), **constraints)
    graph, metadata = net.graph(), net.metadata()
    json.dumps({"graph": graph, "metadata": metadata})
    positions, edges = graph["positions"], graph["edges"]
    degree, incoming, outgoing = Counter(), Counter(), Counter()
    lengths = []
    support = {(layer, source, destination) for layer, affine in enumerate(net.layers)
               for destination, row in enumerate(affine.indices) for source in row if source >= 0}
    assert {tuple(edge) for edge in edges} == support
    for layer, source, destination in edges:
        degree[layer, source] += 1
        degree[layer + 1, destination] += 1
        outgoing[layer, source] += 1
        incoming[layer + 1, destination] += 1
        lengths.append(np.linalg.norm(np.array(positions[layer][source]) - positions[layer + 1][destination]))
    assert metadata["active_weights"] == len(edges)
    assert metadata["conditioner_max_degree"] == max(degree.values(), default=0)
    assert metadata["conditioner_max_edge_length"] == pytest.approx(max(lengths, default=0))
    if "max_degree" in constraints:
        bound = constraints["max_degree"]
        assert max(degree.values(), default=0) <= bound
        assert all(count <= bound // 2 for (layer, _), count in incoming.items() if layer <= depth)
        assert all(count <= (bound + 1) // 2 for (layer, _), count in outgoing.items() if layer > 0)
    if "radius" in constraints:
        assert all(length <= constraints["radius"] for length in lengths)
    noncrossing = all(len({tuple(p) for p in layer}) == len(layer) for layer in positions)
    for layer in range(depth + 1):
        band = [(source, destination) for band_index, source, destination in edges if band_index == layer]
        for (s1, d1), (s2, d2) in combinations(band, 2):
            dx_source = positions[layer][s1][0] - positions[layer][s2][0]
            dx_destination = positions[layer + 1][d1][0] - positions[layer + 1][d2][0]
            noncrossing &= dx_source * dx_destination >= 0
    assert metadata["conditioner_planar_drawing"] == noncrossing
    if constraints.get("planar"):
        assert noncrossing
    assert all(y == layer for layer, nodes in enumerate(positions) for _, y in nodes)
    assert positions[0][0][0] == 0
    assert positions[0][dim - 1][0] == hidden - 1
    for nodes in positions[1:-1]:
        np.testing.assert_array_equal(sorted(x for x, _ in nodes), np.arange(hidden))
    if dim == 2:
        spatial_order = np.argsort(np.array(positions[1])[:, 0])
        spatial_ranks = np.asarray(net.layers[0].output_ranks)[spatial_order]
        np.testing.assert_array_equal(spatial_ranks[:2 * (hidden // 2)], np.tile([1, 0], hidden // 2))
    if "radius" in constraints or constraints.get("planar"):
        np.testing.assert_array_equal(np.array(positions[-1])[:6, 0], np.arange(6))
        np.testing.assert_array_equal(np.array(positions[-1])[-6:, 0], np.arange(hidden - 6, hidden))

    reached = [{i} if i < dim else set() for i in range(dim + 16)]
    for layer in range(depth + 1):
        following = [set() for _ in positions[layer + 1]]
        for band, source, destination in edges:
            if band == layer:
                following[destination].update(reached[source])
        reached = following
    parents = np.zeros((dim, dim), dtype=bool)
    for parameter, ancestors in enumerate(reached):
        for ancestor in ancestors:
            parents[parameter // 6, ancestor] = True
    np.testing.assert_array_equal(metadata["conditional_parents"], parents)
    assert not np.any(np.triu(parents))
    x = jnp.linspace(-0.5, 0.7, dim)
    for jacobian in jax.jacfwd(lambda z: net.conditional_parameters(3, z))(x):
        assert np.all(np.asarray(jacobian).transpose(0, 2, 1)[~parents] == 0)


def test_unconstrained_matches_dense_and_pruning_is_key_independent():
    key = jax.random.PRNGKey(5)
    dense = MADE(3, 11, 2, components=2, key=key)
    sparse = LocalMADE(3, 11, 2, components=2, key=key)
    x = jnp.array([0.2, -0.4, 0.7])
    for actual, expected in zip(sparse.conditional_parameters(4, x), dense.conditional_parameters(4, x)):
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(sparse(4, x), dense(4, x), atol=1e-6, rtol=1e-5)
    assert sparse.metadata()["active_weights"] == sum(int(layer.mask().sum()) for layer in dense.layers)
    a = LocalMADE(2, 19, 2, max_degree=4, key=key)
    b = LocalMADE(2, 19, 2, max_degree=4, key=jax.random.PRNGKey(92))
    assert a.graph() == b.graph()
    assert a.metadata()["conditioner_max_edge_length"] > 3


@pytest.mark.parametrize("hidden,depth", [(3, 1), (17, 3), (64, 2)])
def test_degree_one_has_factorized_score_and_no_time_feature_path(hidden, depth):
    net = LocalMADE(2, hidden, depth, components=2, max_degree=1, key=jax.random.PRNGKey(2))
    net = eqx.tree_at(lambda m: m.layers[-1].bias, net, jnp.linspace(-0.4, 0.6, 12))
    assert not np.any(net.metadata()["conditional_parents"])
    assert not any(net.metadata()["conditional_time_reachable"])
    x = jnp.array([0.25, -0.7])
    jacobian = np.asarray(jax.jacfwd(lambda z: net(5, z))(x))
    assert jacobian[0, 1] == jacobian[1, 0] == 0
    assert np.all(np.abs(np.diag(jacobian)) > 1e-7)
    np.testing.assert_allclose(net(5, x)[0], net(5, x.at[1].set(4.0))[0], atol=0, rtol=0)
    np.testing.assert_allclose(net(5, x)[1], net(5, x.at[0].set(-3.0))[1], atol=0, rtol=0)


def test_padding_cannot_affect_values_or_train_as_edges():
    net = LocalMADE(2, 13, 2, components=2, max_degree=3, key=jax.random.PRNGKey(8))
    assert all(jnp.issubdtype(leaf.dtype, jnp.floating) for leaf in jax.tree_util.tree_leaves(net))
    x = jnp.array([-0.2, 0.4])
    grads = eqx.filter_grad(lambda m: m.log_prob(2, x))(net)
    padded = 0
    for index, (layer, gradient) in enumerate(zip(net.layers, grads.layers)):
        padding = np.asarray(layer.indices) < 0
        padded += int(padding.sum())
        assert np.all(np.asarray(layer.weight)[padding] == 0)
        assert np.all(np.asarray(gradient.weight)[padding] == 0)
        changed = eqx.tree_at(lambda m: m.layers[index].weight, net,
                             jnp.where(padding, 123.0, layer.weight))
        np.testing.assert_allclose(changed.log_prob(2, x), net.log_prob(2, x), atol=0, rtol=0)
    assert padded > 0


def test_geometric_width_and_vertical_distance_boundaries():
    for constraints in ({"radius": 2.0}, {"planar": True}):
        with pytest.raises(ValueError, match="disjoint output clusters"):
            LocalMADE(2, 11, components=2, key=jax.random.PRNGKey(0), **constraints)
    empty = LocalMADE(2, 12, components=2, radius=0.99, key=jax.random.PRNGKey(0))
    assert empty.graph()["edges"] == []
    assert empty.metadata()["conditioner_max_edge_length"] == 0
