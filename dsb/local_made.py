"""Sparse affine-conditioner restrictions of the normalized MADE density.

The graph starts at data coordinates and the 16 *already computed* sinusoidal
features. Time-embedding generation, scalar biases, mixture-head arithmetic,
the time gain, and reverse differentiation are outside this graph. In
particular, its triangular dependencies are not the full score dependencies.
"""
from __future__ import annotations

import math
from itertools import zip_longest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .made import MADE


class _SparseLinear(eqx.Module):
    weight: jax.Array
    bias: jax.Array
    indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    input_ranks: tuple[int, ...] = eqx.field(static=True)
    output_ranks: tuple[int, ...] = eqx.field(static=True)
    strict: bool = eqx.field(static=True)

    def __init__(self, dense, edges):
        self.input_ranks = dense.input_ranks
        self.output_ranks = dense.output_ranks
        self.strict = dense.strict
        rows = [[] for _ in dense.output_ranks]
        for source, destination in edges:
            rows[destination].append(source)
        width = max(1, max(map(len, rows)))
        self.indices = tuple(tuple(sorted(row)) + (-1,) * (width - len(row)) for row in rows)
        indices = jnp.asarray(self.indices)
        active = indices >= 0
        original_fan_in = dense.mask().sum(axis=1)
        fan_in = jnp.maximum(active.sum(axis=1), 1)
        selected = jnp.take_along_axis(dense.weight, jnp.maximum(indices, 0), axis=1)
        self.weight = jnp.where(active, selected * jnp.sqrt(original_fan_in / fan_in)[:, None], 0.0)
        self.bias = dense.bias

    def __call__(self, x):
        indices = jnp.asarray(self.indices)
        weights = jnp.where(indices >= 0, self.weight, 0.0)
        return jnp.sum(weights * x[jnp.maximum(indices, 0)], axis=1) + self.bias


def _select_edges(layer, source_x, destination_x, incoming, outgoing, radius, planar, seed):
    destinations, sources = np.nonzero(np.asarray(layer.mask()))
    distances = np.hypot(source_x[sources] - destination_x[destinations], 1.0)
    eligible = np.ones(len(sources), dtype=bool) if radius is None else distances <= radius
    candidates = list(zip(sources[eligible].tolist(), destinations[eligible].tolist()))
    if planar:
        candidates.sort(key=lambda edge: (source_x[edge[0]], edge[0],
                                         abs(source_x[edge[0]] - destination_x[edge[1]]), edge[1]))
    elif radius is not None:
        candidates.sort(key=lambda edge: (abs(source_x[edge[0]] - destination_x[edge[1]]), edge))
    elif incoming is not None or outgoing is not None:
        np.random.default_rng(seed).shuffle(candidates)

    used_in = np.zeros(len(destination_x), dtype=int)
    used_out = np.zeros(len(source_x), dtype=int)
    accepted = []
    previous_source = None
    previous_max = current_max = -math.inf
    for source, destination in candidates:
        if planar and source != previous_source:
            previous_max = max(previous_max, current_max)
            previous_source = source
        if (incoming is not None and used_in[destination] >= incoming
                or outgoing is not None and used_out[source] >= outgoing
                or planar and destination_x[destination] < previous_max):
            continue
        accepted.append((source, destination))
        used_in[destination] += 1
        used_out[source] += 1
        current_max = max(current_max, destination_x[destination])
    return accepted


class LocalMADE(MADE):
    """MADE with bounded total degree, Euclidean locality and/or planar bands.

    Hidden nodes reserve floor(degree/2) incoming and ceil(degree/2) outgoing
    edges. Endpoints may use the full degree. Pruning is deterministic and
    independent of the parameter key; it is a greedy construction, not a
    maximum-edge planar subgraph or a guarantee of surviving communication.
    Pruning may remove time-feature paths as well as data paths; the learned
    per-step gain is still time dependent. The mixture head adds fixed work per
    coordinate/component, and reverse differentiation adds propagation along
    the reversed graph; neither inherits the conditioner-only graph bounds.
    """

    max_degree: int | None = eqx.field(static=True)
    radius: float | None = eqx.field(static=True)
    planar: bool = eqx.field(static=True)
    positions: tuple[tuple[tuple[float, float], ...], ...] = eqx.field(static=True)

    def __init__(self, dim, hidden=128, depth=2, components=8, mode="score",
                 num_steps=20, output_scale=0.01, *, key, max_degree=None,
                 radius=None, planar=False):
        if max_degree is not None and (not isinstance(max_degree, int)
                                       or isinstance(max_degree, bool) or max_degree < 1):
            raise ValueError("max_degree must be a positive integer or None")
        if radius is not None and (not math.isfinite(radius) or radius < 0):
            raise ValueError("radius must be finite and nonnegative or None")
        if not isinstance(planar, bool):
            raise ValueError("planar must be a boolean")
        super().__init__(dim, hidden, depth, components, mode, num_steps, output_scale, key=key)
        self.max_degree, self.radius, self.planar = max_degree, radius, planar
        cluster = 3 * components
        if (radius is not None or planar) and hidden < dim * cluster:
            raise ValueError("geometric modes require hidden >= dim * 3 * components for disjoint output clusters")

        data_x = np.linspace(0.0, hidden - 1.0, dim)
        input_x = np.concatenate((data_x, np.linspace(0.0, hidden - 1.0, 18)[1:-1]))
        groups = [[i for i, rank in enumerate(self.layers[0].output_ranks) if rank == r]
                  for r in sorted(set(self.layers[0].output_ranks), reverse=True)]
        interleaved = [i for row in zip_longest(*groups) for i in row if i is not None]
        hidden_x = np.empty(hidden)
        hidden_x[interleaved] = np.arange(hidden)
        cluster_starts = np.linspace(0.0, hidden - cluster, dim)
        output_x = np.concatenate([start + np.arange(cluster) for start in cluster_starts])
        layer_x = [input_x] + [(hidden_x + layer % 2) % hidden for layer in range(depth)] + [output_x]
        if planar and any(len(set(xs)) != len(xs) for xs in layer_x):
            raise ValueError("planar drawing requires distinct node positions within each layer")
        self.positions = tuple(tuple((float(x), float(y)) for x in xs) for y, xs in enumerate(layer_x))
        layers = []
        for index, dense in enumerate(self.layers):
            incoming = None if max_degree is None else (max_degree if index == depth else max_degree // 2)
            outgoing = None if max_degree is None else (max_degree if index == 0 else (max_degree + 1) // 2)
            edges = _select_edges(dense, layer_x[index], layer_x[index + 1], incoming, outgoing,
                                  radius, planar, seed=index)
            layers.append(_SparseLinear(dense, edges))
        self.layers = tuple(layers)

    def graph(self):
        """Return structural affine edges; padded weight slots are never edges."""
        return {
            "positions": [[list(position) for position in layer] for layer in self.positions],
            "edges": [[index, source, destination] for index, layer in enumerate(self.layers)
                      for destination, sources in enumerate(layer.indices) for source in sources if source >= 0],
        }

    def metadata(self):
        degrees = [np.zeros(len(layer), dtype=int) for layer in self.positions]
        reach = np.eye(self.dim + 16, dtype=bool)
        longest = 0.0
        noncrossing = all(len(set(layer)) == len(layer) for layer in self.positions)
        for index, layer in enumerate(self.layers):
            next_reach = np.zeros((len(layer.indices), self.dim + 16), dtype=bool)
            destinations = {}
            for destination, sources in enumerate(layer.indices):
                for source in sources:
                    if source < 0:
                        continue
                    degrees[index][source] += 1
                    degrees[index + 1][destination] += 1
                    next_reach[destination] |= reach[source]
                    sx = self.positions[index][source][0]
                    dx = self.positions[index + 1][destination][0]
                    longest = max(longest, math.hypot(sx - dx, 1.0))
                    destinations.setdefault(source, []).append(dx)
            previous_max = -math.inf
            for source in sorted(destinations, key=lambda s: self.positions[index][s][0]):
                xs = destinations[source]
                noncrossing = noncrossing and min(xs) >= previous_max
                previous_max = max(previous_max, max(xs))
            reach = next_reach
        conditional_reach = reach.reshape(self.dim, 3 * self.components, self.dim + 16).any(axis=1)
        return {
            **super().metadata(),
            "architecture": "local_made",
            "max_degree": self.max_degree,
            "radius": self.radius,
            "planar": self.planar,
            "conditioner_max_degree": max(int(d.max()) for d in degrees),
            "conditioner_max_edge_length": longest,
            "conditioner_planar_drawing": bool(noncrossing),
            "active_weights": sum(int(degree.sum()) for degree in degrees) // 2,
            "conditional_parents": conditional_reach[:, :self.dim].tolist(),
            "conditional_time_reachable": conditional_reach[:, self.dim:].any(axis=1).tolist(),
            "graph_boundary": "scalar affine conditioner, including precomputed time features; excludes biases, time embedding generation, mixture head, time gain and reverse differentiation",
            "edge_semantics": "trainable structural support, excluding permanently masked padding",
            "spatial_rank_assignment": "round_robin_rank_groups_with_alternating_layer_shift",
        }
