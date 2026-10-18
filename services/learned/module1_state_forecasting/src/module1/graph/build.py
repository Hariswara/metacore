"""The microgrid graph: one node per asset, edges only where a conductor actually runs.

Each island is its own electrical system, so the graph is a disjoint union of per-island stars --
every asset connects to its island bus. No inter-island edge is drawn: there is no cable between
islands, and inventing one would let message passing carry information along a path that does not
exist. Shared weather across islands already reaches every node through its own features.

The node set mirrors ``module1.assemble.ISLAND_ASSETS`` so the forecaster and the contract
producer can never disagree about which nodes exist.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from module1.assemble import ISLAND_ASSETS

ISLAND_ORDER: tuple[str, ...] = ("Eluvaitivu", "Analaitivu", "Nainativu", "Delft-Neduntivu")


@dataclass(frozen=True)
class MicrogridGraph:
    node_ids: tuple[str, ...]        # "Island/asset"
    node_island: tuple[str, ...]
    node_asset: tuple[str, ...]
    adjacency: np.ndarray            # (N, N) symmetric, 0/1, no self loops

    @property
    def n(self) -> int:
        return len(self.node_ids)

    def normalized(self) -> np.ndarray:
        """GCN propagation matrix  D^-1/2 (A + I) D^-1/2  (Kipf and Welling, 2017)."""
        a = self.adjacency + np.eye(self.n)
        d = a.sum(axis=1)
        inv = 1.0 / np.sqrt(d)
        return a * inv[:, None] * inv[None, :]


def build_graph(islands: tuple[str, ...] = ISLAND_ORDER) -> MicrogridGraph:
    ids, isl, asset = [], [], []
    for island in islands:
        for a in ISLAND_ASSETS[island]:
            ids.append(f"{island}/{a}")
            isl.append(island)
            asset.append(a)
    n = len(ids)
    adj = np.zeros((n, n))
    for island in islands:
        members = [i for i in range(n) if isl[i] == island]
        bus = next(i for i in members if asset[i] == "bus")
        for i in members:
            if i != bus:
                adj[bus, i] = adj[i, bus] = 1.0
    return MicrogridGraph(tuple(ids), tuple(isl), tuple(asset), adj)


def check_invariants(g: MicrogridGraph) -> None:
    """Property checks the proposal commits to (NFR8).

    Symmetric, loop-free, each island connected, no edge between islands.
    """
    a = g.adjacency
    if not np.array_equal(a, a.T):
        raise ValueError("adjacency is not symmetric")
    if np.any(np.diag(a) != 0):
        raise ValueError("adjacency has self loops")
    for island in set(g.node_island):
        members = [i for i in range(g.n) if g.node_island[i] == island]
        seen, stack = {members[0]}, [members[0]]
        while stack:
            u = stack.pop()
            for v in members:
                if a[u, v] and v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(seen) != len(members):
            raise ValueError(f"{island} is not connected")
    cross = [(i, j) for i in range(g.n) for j in range(g.n)
             if a[i, j] and g.node_island[i] != g.node_island[j]]
    if cross:
        raise ValueError(f"inter-island edges present: {cross[:3]}")
