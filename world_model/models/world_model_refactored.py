#!/usr/bin/env python3
"""
world_model_refactored.py (v2) -- RAMFP formation world model with structural fixes.

Input per window (all produced by dense_data.py)
    h_roles, a_roles : [B, 11] long   canonical role id of each position (0..10)
    h_adj,   a_adj   : [B, 11, 11]    intra-team adjacency (with self loops)
    ctx              : [B, C]         game state; enters ONLY at the FiLM readout

Pieces
    RoleReadout   pools positions into the 11 canonical roles (mean over positions
                  that share a role), per-role learned-query attention, plus the
                  flattened per-role block and role counts.
    FiLMReadout   context dropout (p=0.3 train) + FiLM on LayerNorm(g) + a
                  formation-only skip path.
    SwapSym       exact xg antisymmetry and possession sum = 100.
    BipartiteLayer  masked intra-team attention (formation adjacency) and masked
                  cross-team attention (role matchup edges, with wr / xg_rate
                  as attention bias).

Target order: [xg_delta, possession_pct, ppda, shots_on_target]. SwapSym outputs
RAW units; `zscore()` maps them to the z-scored space the loss uses.

Self-test:  python models/world_model_refactored.py
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_ROLES = 11
N_POS = 11
IDX_XG, IDX_POSS, IDX_PPDA, IDX_SOT = 0, 1, 2, 3
NEG = -1e4
ROLE_MIRROR = torch.tensor([0, 5, 4, 3, 2, 1, 6, 7, 8, 9, 10], dtype=torch.long)


def mirror_roles(roles: torch.Tensor) -> torch.Tensor:
    """Lateral mirror augmentation: relabel RB<->LB and RCB<->LCB on BOTH teams.
    Graph structure (adjacency, matchup edges between positions) is unchanged,
    which is exactly what a pitch reflection does. Targets are unchanged too."""
    return ROLE_MIRROR.to(roles.device)[roles]


# --------------------------------------------------------------------------- #
# C. Role-indexed readout
# --------------------------------------------------------------------------- #
class RoleReadout(nn.Module):
    """Pool positions -> 11 canonical roles -> [attention-pooled || flattened || counts].

    w_r = softmax_r( <R_r, q_r> ) over roles present in the formation;
    R_r is the mean embedding of the positions holding role r (a 442 has two ST
    positions, a 433 three CM). Absent roles are zeroed and masked, so every
    formation, including unseen ones, lands in the same 11-slot layout.
    """

    def __init__(self, node_dim: int, n_roles: int = N_ROLES):
        super().__init__()
        self.n_roles, self.node_dim = n_roles, node_dim
        self.queries = nn.Parameter(torch.randn(n_roles, node_dim) * node_dim ** -0.5)
        self.scale = node_dim ** -0.5
        self.out_dim = node_dim + n_roles * node_dim + n_roles

    def forward(self, H: torch.Tensor, roles: torch.Tensor, return_weights: bool = False):
        onehot = F.one_hot(roles, self.n_roles).to(H.dtype)             # [B, P, R]
        count = onehot.sum(1)                                           # [B, R]
        present = count > 0
        R = torch.einsum("bpr,bpd->brd", onehot, H) / count.clamp(min=1).unsqueeze(-1)
        scores = (R * self.queries).sum(-1) * self.scale                # [B, R]
        scores = scores.masked_fill(~present, float("-inf"))
        w = torch.softmax(scores, dim=-1)
        pooled = (w.unsqueeze(-1) * R).sum(1)                           # [B, D]
        out = torch.cat([pooled, R.flatten(1), count / N_POS], dim=-1)
        return (out, w) if return_weights else out


# --------------------------------------------------------------------------- #
# B. Context dropout + FiLM readout
# --------------------------------------------------------------------------- #
class FiLMReadout(nn.Module):
    """h_out = f_form(g) + Head( LN(g) * (1 + gamma(z)) + beta(z) ),  z = enc(ctx)."""

    def __init__(self, g_dim: int, ctx_dim: int, out_dim: int = 4, hidden: int = 256,
                 ctx_hidden: int = 64, ctx_dropout: float = 0.3, dropout: float = 0.1):
        super().__init__()
        self.ctx_dropout = ctx_dropout
        self.f_form = nn.Sequential(nn.Linear(g_dim, hidden), nn.GELU(),
                                    nn.Dropout(dropout), nn.Linear(hidden, out_dim))
        self.norm = nn.LayerNorm(g_dim)
        self.ctx_enc = nn.Sequential(nn.Linear(ctx_dim, ctx_hidden), nn.GELU())
        self.gamma = nn.Linear(ctx_hidden, g_dim)
        self.beta = nn.Linear(ctx_hidden, g_dim)
        for m in (self.gamma, self.beta):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        self.head = nn.Sequential(nn.Linear(g_dim, hidden), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(hidden, out_dim))

    def forward(self, g: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        if self.training and self.ctx_dropout > 0:
            keep = (torch.rand(ctx.shape[0], 1, device=ctx.device) >= self.ctx_dropout).to(ctx.dtype)
            ctx = ctx * keep
        z = self.ctx_enc(ctx)
        return self.f_form(g) + self.head(self.norm(g) * (1.0 + self.gamma(z)) + self.beta(z))


# --------------------------------------------------------------------------- #
# Backbone
# --------------------------------------------------------------------------- #
class BiasedMHA(nn.Module):
    """Multi-head attention with an additive logit bias [B, H or 1, Nq, Nk]."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        assert dim % heads == 0
        self.h, self.dk = heads, dim // heads
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
        self.drop = nn.Dropout(dropout)

    def forward(self, xq, xkv, bias):
        B, Nq, _ = xq.shape
        Nk = xkv.shape[1]
        q = self.q(xq).view(B, Nq, self.h, self.dk).transpose(1, 2)
        k = self.k(xkv).view(B, Nk, self.h, self.dk).transpose(1, 2)
        v = self.v(xkv).view(B, Nk, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk) + bias
        att = self.drop(torch.softmax(att, dim=-1))
        return self.o((att @ v).transpose(1, 2).reshape(B, Nq, -1))


class BipartiteLayer(nn.Module):
    """Masked intra-team attention -> masked cross-team attention -> FFN.
    Weights are shared by both teams, so the layer has no home/away asymmetry."""

    def __init__(self, dim: int, heads: int, dropout: float, edge_feat_dim: int = 6):
        super().__init__()
        self.intra = BiasedMHA(dim, heads, dropout)
        self.cross = BiasedMHA(dim, heads, dropout)
        self.edge_bias = nn.Linear(edge_feat_dim, heads)
        self.n1, self.n2, self.n3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x, opp, intra_bias, cross_mask_bias, cross_feats):
        x = self.n1(x + self.intra(x, x, intra_bias))
        eb = self.edge_bias(cross_feats).permute(0, 3, 1, 2)             # [B, H, 11, 11]
        x = self.n2(x + self.cross(x, opp, cross_mask_bias + eb))
        return self.n3(x + self.ffn(x))


class WorldModelCore(nn.Module):
    """f(home, away, ctx) -> [B, 4] on the z-score scale (not yet symmetrised)."""

    def __init__(self, edge_stats: np.ndarray, ctx_dim: int, node_dim: int = 128,
                 layers: int = 3, heads: int = 4, out_dim: int = 4, hidden: int = 256,
                 ctx_dropout: float = 0.3, dropout: float = 0.1):
        super().__init__()
        self.register_buffer("S", torch.as_tensor(edge_stats, dtype=torch.float32))   # [11,11,3]
        self.role_emb = nn.Embedding(N_ROLES, node_dim)
        self.layers = nn.ModuleList(BipartiteLayer(node_dim, heads, dropout) for _ in range(layers))
        self.role_readout = RoleReadout(node_dim)
        self.readout = FiLMReadout(2 * self.role_readout.out_dim, ctx_dim, out_dim, hidden,
                                   ctx_dropout=ctx_dropout, dropout=dropout)

    def _cross(self, rx, ry):
        """Cross-team structure from the viewpoint of team x. feats[b,i,j] =
        [stats(x_i attacks y_j) (3), stats(y_j attacks x_i) (3)]."""
        s1 = self.S[rx.unsqueeze(2), ry.unsqueeze(1)]                    # [B,11,11,3]
        s2 = self.S[ry.unsqueeze(1), rx.unsqueeze(2)]
        feats = torch.cat([s1, s2], dim=-1)
        present = torch.maximum(s1[..., 0], s2[..., 0]) > 0              # [B,11,11]
        has_any = present.any(-1, keepdim=True)                          # rows with no edge attend uniformly
        allow = present | ~has_any
        return (~allow).to(feats.dtype).unsqueeze(1) * NEG, feats

    @staticmethod
    def _intra(adj):
        return ((adj <= 0).to(adj.dtype) * NEG).unsqueeze(1)

    def forward(self, h_roles, h_adj, a_roles, a_adj, ctx):
        h, a = self.role_emb(h_roles), self.role_emb(a_roles)
        ib_h, ib_a = self._intra(h_adj), self._intra(a_adj)
        cm_h, cf_h = self._cross(h_roles, a_roles)
        cm_a, cf_a = self._cross(a_roles, h_roles)
        for layer in self.layers:
            h, a = layer(h, a, ib_h, cm_h, cf_h), layer(a, h, ib_a, cm_a, cf_a)
        g = torch.cat([self.role_readout(h, h_roles), self.role_readout(a, a_roles)], dim=-1)
        return self.readout(g, ctx)


# --------------------------------------------------------------------------- #
# A. Swap-antisymmetric wrapper
# --------------------------------------------------------------------------- #
def make_ctx_swapper(perm: Sequence[int], sign: Sequence[float]) -> Callable[[torch.Tensor], torch.Tensor]:
    perm_t = torch.as_tensor(list(perm), dtype=torch.long)
    sign_t = torch.as_tensor(list(sign), dtype=torch.float32)

    def fn(ctx):
        return ctx[:, perm_t.to(ctx.device)] * sign_t.to(device=ctx.device, dtype=ctx.dtype)

    return fn


class SwapSym(nn.Module):
    """xg = std*0.5*(o(A,B)-o(B,A)); possession = c + std*0.5*(p(A,B)-p(B,A)),
    c = poss_center (0.5 for possession stored as a fraction, 50 for percent),
    so poss(A,B) + poss(B,A) = 2c.
    ppda / shots_on_target are home-team quantities, taken from o(A,B) only.
    Output is RAW units; use zscore() for the loss."""

    def __init__(self, core, ctx_swap, target_mean: Sequence[float], target_std: Sequence[float],
                 poss_center: float = 0.5):
        super().__init__()
        self.core, self.ctx_swap, self.poss_center = core, ctx_swap, float(poss_center)
        self.register_buffer("mean", torch.as_tensor(list(target_mean), dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(list(target_std), dtype=torch.float32))

    def forward(self, h_roles, h_adj, a_roles, a_adj, ctx):
        B = h_roles.shape[0]
        o = self.core(torch.cat([h_roles, a_roles]), torch.cat([h_adj, a_adj]),
                      torch.cat([a_roles, h_roles]), torch.cat([a_adj, h_adj]),
                      torch.cat([ctx, self.ctx_swap(ctx)]))
        o_ab, o_ba = o[:B], o[B:]
        anti = 0.5 * (o_ab - o_ba)
        return torch.stack([
            anti[:, IDX_XG] * self.std[IDX_XG],
            self.poss_center + anti[:, IDX_POSS] * self.std[IDX_POSS],
            o_ab[:, IDX_PPDA] * self.std[IDX_PPDA] + self.mean[IDX_PPDA],
            o_ab[:, IDX_SOT] * self.std[IDX_SOT] + self.mean[IDX_SOT],
        ], dim=-1)

    def zscore(self, raw):
        return (raw - self.mean) / self.std


def build_world_model(edge_stats, ctx_dim, ctx_swap, target_mean, target_std, poss_center=0.5, **kw) -> SwapSym:
    return SwapSym(WorldModelCore(edge_stats, ctx_dim, **kw), ctx_swap, target_mean, target_std, poss_center)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _rand_batch(B, gen):
    roles = torch.randint(0, N_ROLES, (B, N_POS), generator=gen)
    a = (torch.rand(B, N_POS, N_POS, generator=gen) > 0.7).float()
    a = ((a + a.transpose(1, 2)) > 0).float()
    return roles, torch.clamp(a + torch.eye(N_POS), max=1.0)


def _run_checks():
    try:
        from dense_data import ctx_swap_perm, ctx_dim
    except ImportError:
        from models.dense_data import ctx_swap_perm, ctx_dim
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    B, C = 6, ctx_dim(2)
    perm, sign = ctx_swap_perm(2)
    swapper = make_ctx_swapper(perm, sign)
    S = np.zeros((11, 11, 3), np.float32)
    for a, d in [(0, 10), (1, 9), (1, 4), (6, 8), (7, 7), (9, 1), (9, 5), (10, 3), (10, 2), (8, 6)]:
        S[a, d] = (1.0, 0.01 * a, 0.05)
    mean, std = [0.03, 0.51, 13.4, 0.9], [0.32, 0.157, 11.2, 0.95]
    model = build_world_model(S, C, swapper, mean, std, node_dim=32, layers=2, heads=4, hidden=64)

    hr, ha = _rand_batch(B, gen)
    ar, aa = _rand_batch(B, gen)
    ctx = torch.zeros(B, C)
    ctx[torch.arange(B), torch.randint(0, 5, (B,))] = 1
    ctx[torch.arange(B), 5 + torch.randint(0, 6, (B,))] = 1
    ctx[torch.arange(B), 11 + torch.randint(0, 2, (B,))] = 1
    ctx[torch.arange(B), 13 + torch.randint(0, 2, (B,))] = 1

    model.eval()
    y = model(hr, ha, ar, aa, ctx)
    assert y.shape == (B, 4) and torch.isfinite(y).all()
    print(f"[ok] shape {tuple(y.shape)}, finite")

    # RoleReadout: weights sum to 1; absent roles get zero weight; layout fixed
    ro = RoleReadout(32)
    out, w = ro(torch.randn(B, N_POS, 32), hr, return_weights=True)
    assert out.shape == (B, ro.out_dim) and torch.allclose(w.sum(-1), torch.ones(B), atol=1e-5)
    absent = F.one_hot(hr, N_ROLES).sum(1) == 0
    assert (w[absent] == 0).all()
    print(f"[ok] RoleReadout out {tuple(out.shape)}; absent roles carry zero weight")

    # swap exactness
    ys = model(ar, aa, hr, ha, swapper(ctx))
    xg_res = (y[:, IDX_XG] + ys[:, IDX_XG]).abs().max().item()
    poss_res = (y[:, IDX_POSS] + ys[:, IDX_POSS] - 1.0).abs().max().item()
    assert xg_res < 1e-5 and poss_res < 1e-4, (xg_res, poss_res)
    print(f"[ok] swap: xg residual {xg_res:.1e}, possession sum (should be 1) deviation {poss_res:.1e}")

    # identical formations + swap-invariant context -> xg 0, possession 50
    c_sym = (ctx + swapper(ctx)) / 2
    yi = model(hr, ha, hr, ha, c_sym)
    assert yi[:, IDX_XG].abs().max() < 1e-5 and (yi[:, IDX_POSS] - 0.5).abs().max() < 1e-4
    print("[ok] mirror matchup, symmetric context -> xg=0, possession=0.5")

    # position-permutation invariance (the model sees a graph, not an ordering)
    p = torch.randperm(N_POS, generator=gen)
    y_p = model(hr[:, p], ha[:, p][:, :, p], ar, aa, ctx)
    assert torch.allclose(y, y_p, atol=1e-4), (y - y_p).abs().max()
    print("[ok] shuffling position order of a team leaves outputs unchanged")

    # eval deterministic, train stochastic (context dropout)
    model.train()
    a1, a2 = model(hr, ha, ar, aa, ctx), model(hr, ha, ar, aa, ctx)
    model.eval()
    assert torch.allclose(model(hr, ha, ar, aa, ctx), model(hr, ha, ar, aa, ctx))
    assert not torch.allclose(a1, a2)
    print("[ok] eval deterministic; train stochastic")

    # mirror augmentation
    assert (mirror_roles(mirror_roles(hr)) == hr).all()
    ym = model(mirror_roles(hr), ha, mirror_roles(ar), aa, ctx)
    assert ym.shape == y.shape and torch.isfinite(ym).all()
    print("[ok] mirror_roles is an involution and the model accepts mirrored inputs")

    # gradients reach every parameter
    model.train()
    model.core.readout.ctx_dropout = 0.0
    loss = F.mse_loss(model.zscore(model(hr, ha, ar, aa, ctx)), torch.randn(B, 4))
    model.zero_grad()
    loss.backward()
    dead = [n for n, q in model.named_parameters() if q.grad is None or q.grad.abs().sum() == 0]
    assert not dead, f"no gradient: {dead}"
    print(f"[ok] gradients reach all {sum(1 for _ in model.parameters())} parameter tensors")
    print(f"[info] params: {sum(q.numel() for q in model.parameters()):,}")
    print("\nall checks passed")


if __name__ == "__main__":
    _run_checks()
