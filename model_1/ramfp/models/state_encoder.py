"""
The state encoder: match context -> z_t in R^256.

Two branches, as in section 5.3 of the draft.

  1. EventTransformer — the last N=50 events, tokenised on five fields, read by
     a CAUSAL transformer. Causality is a deliberate choice: the pretraining
     task is next-event-type prediction, and using a causal mask means the same
     weights serve pretraining and downstream encoding with no train/finetune
     mismatch. The summary vector is the hidden state at the final real token,
     which is exactly the position the pretraining objective optimises.

  2. FormationEncoder — the current shape as an 11-node graph. Node features
     are a role embedding (shared across formations via the band x lateral
     vocabulary in vocab.py, which is contribution N5), plus PPDA bucket,
     scoreline and inferred game phase broadcast to every node.

Concatenate, project, done. Nothing here knows about opponents or planning.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..config import EncoderConfig
from ..vocab import (
    EVENT_TYPE_VOCAB, N_MINUTE_BUCKETS, N_OUTCOMES, N_PHASES, N_PPDA_BUCKETS,
    N_ROLES, N_SCORE_BUCKETS, N_ZONES,
)
from .layers import build_gat

PAD_ID = 0  # reserved padding index; real vocabularies are offset by one


class EventTokenEmbedding(nn.Module):
    """Five categorical fields -> one d_model token vector."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        d = cfg.d_model
        self.type_emb = nn.Embedding(EVENT_TYPE_VOCAB + 1, d // 2, padding_idx=PAD_ID)
        self.zone_emb = nn.Embedding(N_ZONES + 1, d // 4, padding_idx=PAD_ID)
        self.team_emb = nn.Embedding(3, d // 8, padding_idx=PAD_ID)
        self.minute_emb = nn.Embedding(N_MINUTE_BUCKETS + 1, d // 8, padding_idx=PAD_ID)
        self.outcome_emb = nn.Embedding(N_OUTCOMES + 1, d // 8, padding_idx=PAD_ID)
        fused = d // 2 + d // 4 + d // 8 * 3
        self.fuse = nn.Linear(fused, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [
            self.type_emb(batch["event_type"]),
            self.zone_emb(batch["event_zone"]),
            self.team_emb(batch["event_team"]),
            self.minute_emb(batch["event_minute"]),
            self.outcome_emb(batch["event_outcome"]),
        ]
        return self.norm(self.fuse(torch.cat(parts, dim=-1)))


class EventTransformer(nn.Module):
    """Causal transformer over the tokenised event sequence."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.tokens = EventTokenEmbedding(cfg)
        self.pos = nn.Parameter(torch.zeros(1, cfg.seq_len, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.dim_ff,
            dropout=cfg.dropout, activation="gelu", batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns per-position hidden states [B, T, d_model]."""
        x = self.tokens(batch)
        T = x.size(1)
        x = x + self.pos[:, :T]
        causal = torch.triu(
            torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        pad = batch["event_type"].eq(PAD_ID)
        # a fully padded row would produce NaNs in softmax; keep position 0 alive
        pad = pad & ~(pad.all(dim=1, keepdim=True) &
                      (torch.arange(T, device=x.device) == 0))
        h = self.encoder(x, mask=causal, src_key_padding_mask=pad)
        return self.norm(h)

    @staticmethod
    def pool_last(h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Hidden state at the last real token of each sequence."""
        idx = (lengths - 1).clamp(min=0).view(-1, 1, 1).expand(-1, 1, h.size(-1))
        return h.gather(1, idx).squeeze(1)


class FormationEncoder(nn.Module):
    """GAT over the 11-node formation graph."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.role_emb = nn.Embedding(N_ROLES, cfg.gat_dim // 2)
        self.ppda_emb = nn.Embedding(N_PPDA_BUCKETS, cfg.gat_dim // 6)
        self.score_emb = nn.Embedding(N_SCORE_BUCKETS, cfg.gat_dim // 6)
        self.phase_emb = nn.Embedding(N_PHASES, cfg.gat_dim // 6)
        d_node = cfg.gat_dim // 2 + 3 * (cfg.gat_dim // 6)
        self.gat = build_gat(
            cfg.use_pyg, d_in=d_node, d_hidden=cfg.gat_dim,
            n_layers=cfg.gat_layers, heads=cfg.gat_heads, dropout=cfg.dropout)
        self.norm = nn.LayerNorm(cfg.gat_dim)

    def forward(self, roles: torch.Tensor, adj: torch.Tensor,
                ppda_b: torch.Tensor, score_b: torch.Tensor,
                phase: torch.Tensor) -> torch.Tensor:
        """
        roles  : [B, 11] role ids
        adj    : [B, 11, 11] bool
        others : [B] context ids, broadcast to every node
        returns: [B, gat_dim] mean-pooled graph embedding
        """
        n = roles.size(1)
        ctx = torch.cat([
            self.ppda_emb(ppda_b), self.score_emb(score_b), self.phase_emb(phase)
        ], dim=-1).unsqueeze(1).expand(-1, n, -1)
        x = torch.cat([self.role_emb(roles), ctx], dim=-1)
        return self.norm(self.gat(x, adj).mean(dim=1))


class StateEncoder(nn.Module):
    """z_t = Linear([z_trans ; z_gnn]) in R^{d_context}."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.events = EventTransformer(cfg)
        self.formation = FormationEncoder(cfg)
        self.fuse = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.gat_dim, cfg.d_context),
            nn.GELU(),
            nn.LayerNorm(cfg.d_context),
        )
        # pretraining head: next event type at every position
        self.lm_head = nn.Linear(cfg.d_model, EVENT_TYPE_VOCAB + 1)

    # -- pretraining -------------------------------------------------------
    def pretrain_logits(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """[B, T, vocab] next-event-type logits (position t predicts t+1)."""
        return self.lm_head(self.events(batch))

    # -- downstream --------------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        h = self.events(batch)
        z_trans = EventTransformer.pool_last(h, batch["length"])
        z_gnn = self.formation(
            batch["roles"], batch["adj"], batch["ppda_bucket"],
            batch["score_bucket"], batch["phase"])
        return self.fuse(torch.cat([z_trans, z_gnn], dim=-1))

    # -- checkpointing -----------------------------------------------------
    def save(self, path: str) -> None:
        torch.save({"cfg": vars(self.cfg), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "StateEncoder":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(EncoderConfig(**blob["cfg"]))
        model.load_state_dict(blob["state_dict"])
        return model
