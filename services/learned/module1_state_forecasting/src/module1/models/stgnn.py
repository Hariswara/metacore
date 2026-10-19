"""The learned head of Module 1 and the models it is measured against.

Everything upstream (fusion, graph construction, the target) is a fixed rule; this file holds the
only learned part. Graph convolution is written out directly rather than imported from PyTorch
Geometric: on a graph of ten nodes the dense form ``A_hat @ H @ W`` is exact, fast, and short
enough to explain line by line.

* ``STGNN``      per-node GRU over the history window, then two GCN layers over the asset graph,
                 then a linear head to t+1 .. t+H. The 64-d post-GCN vector is the node embedding
                 the M1 -> M2 contract reserves (``EMBEDDING_DIM``).
* ``GraphFree``  identical, with message passing removed -- isolates what the graph contributes.
* ``TimeFree``   identical GCN, but sees only the latest hour -- isolates the temporal encoder.
* ``LSTM1D``     univariate per-node LSTM on the vulnerability history alone: the conventional
                 one-dimensional forecaster, with no fused features and no graph. With
                 ``residual=True`` it gets the same residual head as ST-GNN v0.1, so the comparison
                 isolates architecture rather than output parameterisation.

Two variants, both kept so results stay reproducible:

* **v0** -- two GCN layers; the head predicts the absolute score through a sigmoid.
* **v0.1** -- one GCN layer (two smeared the bus and its single asset together on a star graph);
  the head predicts the *change* from the last observed score, clipped to [0, 1]. Both changes
  were declared before v0.1 was trained, in response to v0's t+1 error and spatial hit rate.
"""

from __future__ import annotations

import torch
from metacore_contracts.state_schema import EMBEDDING_DIM  # the M1 -> M2 pin, not a local choice
from torch import nn

__all__ = ["EMBEDDING_DIM", "GCNLayer", "LSTM1D", "STGNN", "persistence"]


class GCNLayer(nn.Module):
    """H' = A_hat H W  (Kipf and Welling, ICLR 2017)."""

    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)

    def forward(self, h: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        return a_hat @ self.lin(h)


class STGNN(nn.Module):
    def __init__(self, n_features: int, horizon: int, a_hat: torch.Tensor,
                 hidden: int = EMBEDDING_DIM, use_graph: bool = True,
                 use_time: bool = True, gcn_layers: int = 2, residual: bool = False,
                 history_col: int = -1) -> None:
        super().__init__()
        self.use_graph, self.use_time = use_graph, use_time
        self.gcn_layers, self.residual, self.history_col = gcn_layers, residual, history_col
        self.register_buffer("a_hat", a_hat if use_graph else torch.eye(a_hat.shape[0]))
        if use_time:
            self.encoder = nn.GRU(n_features, hidden, batch_first=True)
        else:
            self.encoder = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU())
        self.gcn1 = GCNLayer(hidden, hidden)
        self.gcn2 = GCNLayer(hidden, hidden) if gcn_layers >= 2 else None
        self.act = nn.ReLU()
        self.head = nn.Linear(hidden, horizon)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, N, F) -> per-node embedding (B, N, hidden)."""
        b, seq_len, n, f = x.shape
        if self.use_time:
            seq = x.permute(0, 2, 1, 3).reshape(b * n, seq_len, f)
            _, h_last = self.encoder(seq)
            h = h_last[-1].reshape(b, n, -1)
        else:
            h = self.encoder(x[:, -1])
        h = h + self.act(self.gcn1(h, self.a_hat))
        if self.gcn2 is not None:
            h = h + self.act(self.gcn2(h, self.a_hat))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, N, F) -> (B, N, H), in [0, 1]."""
        out = self.head(self.embed(x))
        if self.residual:
            last = x[:, -1, :, self.history_col].unsqueeze(-1)
            return torch.clamp(last + out, 0.0, 1.0)
        return torch.sigmoid(out)


class LSTM1D(nn.Module):
    """Conventional baseline: each node's own vulnerability history, nothing else."""

    def __init__(self, horizon: int, hidden: int = 32, history_col: int = -1,
                 residual: bool = False) -> None:
        super().__init__()
        self.col = history_col
        self.residual = residual
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, seq_len, n, _ = x.shape
        seq = x[..., self.col].permute(0, 2, 1).reshape(b * n, seq_len, 1)
        _, (h, _) = self.lstm(seq)
        out = self.head(h[-1]).reshape(b, n, -1)
        if self.residual:
            last = x[:, -1, :, self.col].unsqueeze(-1)
            return torch.clamp(last + out, 0.0, 1.0)
        return torch.sigmoid(out)


def persistence(x: torch.Tensor, horizon: int, history_col: int = -1) -> torch.Tensor:
    """The floor: tomorrow looks like now. (B, L, N, F) -> (B, N, H)."""
    last = x[:, -1, :, history_col]
    return last.unsqueeze(-1).expand(-1, -1, horizon)
