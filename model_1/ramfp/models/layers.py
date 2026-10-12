"""
Graph attention over an 11-node formation graph.

Implemented in plain PyTorch. The graph is tiny and every graph in a batch has
exactly 11 nodes, so a dense [B, 11, 11] adjacency mask is both simpler and
faster than the scatter kernels a sparse library would use — and it removes
torch-geometric from the critical path of the install.

`DenseGAT` follows Velickovic et al. (2018): additive attention with a LeakyReLU
score, masked softmax over neighbours, multi-head concatenation.

If the world-model teammate needs PyG for the 22-node bipartite graph, set
encoder.use_pyg: true and `PyGGATWrapper` provides the same interface, so the
two halves of the project can disagree about dependencies without either of
them having to change.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseGATLayer(nn.Module):
    """One multi-head graph attention layer over a dense adjacency mask."""

    def __init__(self, d_in: int, d_out: int, heads: int = 4,
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        if concat and d_out % heads != 0:
            raise ValueError(f"d_out ({d_out}) must be divisible by heads ({heads})")
        self.heads = heads
        self.concat = concat
        self.d_head = d_out // heads if concat else d_out

        self.proj = nn.Linear(d_in, heads * self.d_head, bias=False)
        # additive attention: separate source and target scoring vectors
        self.att_src = nn.Parameter(torch.empty(1, heads, self.d_head))
        self.att_dst = nn.Parameter(torch.empty(1, heads, self.d_head))
        self.bias = nn.Parameter(torch.zeros(d_out if concat else self.d_head))
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x   : [B, N, d_in]
        adj : [B, N, N] bool — adj[b, i, j] True means node i may attend to j.
        out : [B, N, d_out]
        """
        B, N, _ = x.shape
        h = self.proj(x).view(B, N, self.heads, self.d_head)         # [B,N,H,dh]

        # additive scores: alpha[b,h,i,j] = a_src . h_i + a_dst . h_j
        s_src = (h * self.att_src).sum(-1)                            # [B,N,H]
        s_dst = (h * self.att_dst).sum(-1)                            # [B,N,H]
        scores = s_src.permute(0, 2, 1).unsqueeze(-1) + s_dst.permute(0, 2, 1).unsqueeze(-2)
        scores = F.leaky_relu(scores, negative_slope=0.2)             # [B,H,N,N]

        mask = adj.unsqueeze(1)                                       # [B,1,N,N]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = self.dropout(torch.softmax(scores, dim=-1))

        h_t = h.permute(0, 2, 1, 3)                                   # [B,H,N,dh]
        out = torch.matmul(attn, h_t)                                 # [B,H,N,dh]
        out = out.permute(0, 2, 1, 3)                                 # [B,N,H,dh]
        out = out.reshape(B, N, -1) if self.concat else out.mean(dim=2)
        return out + self.bias


class DenseGAT(nn.Module):
    """Stack of DenseGATLayers with residual connections and layer norm."""

    def __init__(self, d_in: int, d_hidden: int, n_layers: int = 2,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_hidden) if d_in != d_hidden else nn.Identity()
        self.layers = nn.ModuleList(
            DenseGATLayer(d_hidden, d_hidden, heads=heads, dropout=dropout)
            for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d_hidden) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            h = norm(h + self.dropout(F.elu(layer(h, adj))))
        return h


class PyGGATWrapper(nn.Module):
    """
    torch-geometric path, same interface as DenseGAT. Only imported when
    encoder.use_pyg is true, so the package installs without PyG present.
    """

    def __init__(self, d_in: int, d_hidden: int, n_layers: int = 2,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        try:
            from torch_geometric.nn import GATv2Conv  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "encoder.use_pyg is true but torch-geometric is not installed. "
                "Either `pip install torch-geometric` matching your torch/CUDA "
                "build, or set encoder.use_pyg: false to use DenseGAT."
            ) from exc
        from torch_geometric.nn import GATv2Conv

        self.input_proj = nn.Linear(d_in, d_hidden) if d_in != d_hidden else nn.Identity()
        self.convs = nn.ModuleList(
            GATv2Conv(d_hidden, d_hidden // heads, heads=heads, dropout=dropout)
            for _ in range(n_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d_hidden) for _ in range(n_layers))
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _dense_to_edge_index(adj: torch.Tensor) -> torch.Tensor:
        """[B, N, N] bool -> a single batched edge_index with node offsets."""
        B, N, _ = adj.shape
        b, i, j = adj.nonzero(as_tuple=True)
        return torch.stack([i + b * N, j + b * N], dim=0)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        edge_index = self._dense_to_edge_index(adj)
        h = self.input_proj(x).reshape(B * N, -1)
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + self.dropout(F.elu(conv(h, edge_index))))
        return h.view(B, N, -1)


def build_gat(use_pyg: bool, **kwargs) -> nn.Module:
    return PyGGATWrapper(**kwargs) if use_pyg else DenseGAT(**kwargs)
