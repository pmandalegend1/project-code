#!/usr/bin/env python3
"""
test_batched_forward.py -- the batched forward must be the SAME model.

BipartiteFormationGNN.forward handled exactly one 22-node graph: it sliced
x[:11] and x[11:] directly, so a batch could not be passed. `forward_batch`
was a Python loop over `forward`, so a batch of 32 ran 32 forward passes and
held 32 autograd graphs alive before one backward. That cost roughly 6x the
epoch time and most of the GPU memory.

Making forward batch-aware is a rewrite of the hot path, so it has to be shown
that the rewrite did not change what the model computes. That is what this
file is for: it is not a smoke test, it is the evidence that the speed-up is
free.

    python test_batched_forward.py
"""

from __future__ import annotations

import json

import pandas as pd
import torch
from torch_geometric.data import Batch

from models.world_model.bipartite_gnn import BipartiteFormationGNN
from models.world_model.graph_builder import build_bipartite_graph

DATA = "data/processed"


def make_graphs(n=24):
    vocab = {int(c): i for i, c in
             enumerate(json.load(open(f"{DATA}/formation_vocab.json"))["vocab"])}
    rms = pd.read_parquet(f"{DATA}/role_matchup_stats.parquet")
    sty = pd.read_parquet(f"{DATA}/team_style_clusters.parquet")
    codes = sorted(vocab)
    graphs = []
    for i in range(n):
        ctx = {
            "minute": 20 + i, "scoreline_home": i % 3, "scoreline_away": (i + 1) % 3,
            "ppda": 6.0 + (i % 9), "home_team_id": int(sty.team_id.iloc[i % len(sty)]),
            "away_team_id": int(sty.team_id.iloc[(i + 3) % len(sty)]),
        }
        graphs.append(build_bipartite_graph(codes[i % len(codes)],
                                            codes[(i * 7) % len(codes)],
                                            ctx, rms, sty, vocab))
    return graphs, vocab


def test_batched_forward_matches_loop():
    torch.manual_seed(0)
    graphs, _ = make_graphs(24)
    model = BipartiteFormationGNN().eval()

    with torch.no_grad():
        loop = torch.stack([model(g) for g in graphs])
        batched = model(Batch.from_data_list(graphs))

    assert loop.shape == batched.shape == (len(graphs), 4)
    max_diff = (loop - batched).abs().max().item()
    assert torch.allclose(loop, batched, atol=1e-5), f"outputs diverged by {max_diff:.3e}"
    print(f"  batched == per-graph loop (max diff {max_diff:.1e})")


def test_single_graph_still_returns_1d():
    """The adapter and the counter-resistance builder pass one graph at a time."""
    torch.manual_seed(0)
    graphs, _ = make_graphs(2)
    model = BipartiteFormationGNN().eval()
    with torch.no_grad():
        out = model(graphs[0])
    assert out.shape == (4,), out.shape
    print("  single graph still returns shape (4,)")


def test_gradients_flow_through_the_batched_path():
    torch.manual_seed(0)
    graphs, _ = make_graphs(16)
    model = BipartiteFormationGNN().train()
    model(Batch.from_data_list(graphs)).sum().backward()
    params = list(model.parameters())
    dead = [i for i, p in enumerate(params)
            if p.grad is None or p.grad.abs().sum().item() == 0.0]
    assert not dead, f"{len(dead)} of {len(params)} parameters got no gradient: {dead}"
    print(f"  all {len(params)} parameters receive gradient")


def test_batch_order_is_preserved():
    """Row i of the batched output must belong to graph i, not some other graph."""
    torch.manual_seed(0)
    graphs, _ = make_graphs(12)
    model = BipartiteFormationGNN().eval()
    with torch.no_grad():
        batched = model(Batch.from_data_list(graphs))
        for i in (0, 5, 11):
            one = model(graphs[i])
            assert torch.allclose(one, batched[i], atol=1e-5), f"row {i} is the wrong graph"
    print("  batched rows line up with their graphs")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(fns)} tests\n")
    for fn in fns:
        fn()
    print("\nall tests passed")
