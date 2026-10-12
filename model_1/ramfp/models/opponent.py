"""
Style-conditioned opponent co-behavior model (section 5.5).

    q_tau   = [emb(f_src) ; emb(f_tgt) ; emb(intent)]        the proposed change
    KV      = [z_t ; e_phi]                                   context and style
    h_opp   = CrossAttention(Q=q_tau, K=V=KV)
    P(f_opp | tau, s_t, phi) = softmax(W_o h_opp)

Two things the draft leaves implicit and this implementation makes explicit:

1. `style_conditioned=False` removes e_phi from the key-value input entirely
   rather than zeroing it, so the ablation is a true architectural ablation
   (E1 baseline 4) rather than a masked input the model could still exploit.

2. The KL term is KL(p || uniform) = log|V| - H(p), so minimising it maximises
   predictive entropy. It is a calibration regulariser, not a distillation
   loss — with ~1,500 training examples over 15 classes the model will
   otherwise become confidently wrong, which wrecks the ECE column of Table 3
   and, downstream, the conformal stopping rule in the planner.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import EncoderConfig, OpponentConfig
from ..vocab import N_FORMATIONS, N_INTENTS, N_STYLES, UNK_FORMATION_ID
from .state_encoder import StateEncoder

N_FORMATION_SLOTS = N_FORMATIONS + 1        # + UNK
N_OUTPUT_CLASSES = N_FORMATIONS             # UNK is never a prediction target


class OpponentCoBehaviorModel(nn.Module):
    def __init__(self, enc_cfg: EncoderConfig, opp_cfg: OpponentConfig,
                 encoder: Optional[StateEncoder] = None,
                 style_conditioned: bool = True):
        super().__init__()
        self.enc_cfg, self.opp_cfg = enc_cfg, opp_cfg
        self.style_conditioned = style_conditioned
        self.encoder = encoder if encoder is not None else StateEncoder(enc_cfg)

        d = opp_cfg.d_model
        self.f_emb = nn.Embedding(N_FORMATION_SLOTS, d // 2)
        self.intent_emb = nn.Embedding(N_INTENTS, d // 4)
        self.query_proj = nn.Sequential(
            nn.Linear(d // 2 * 2 + d // 4, d), nn.GELU(), nn.LayerNorm(d))

        self.style_emb = nn.Embedding(N_STYLES, opp_cfg.d_style)
        self.style_proj = nn.Linear(opp_cfg.d_style, d)
        self.ctx_proj = nn.Linear(d, d)

        self.attn = nn.MultiheadAttention(
            embed_dim=d, num_heads=opp_cfg.n_heads,
            dropout=opp_cfg.dropout, batch_first=True)
        self.norm_q = nn.LayerNorm(d)
        self.norm_o = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, d * 2), nn.GELU(), nn.Dropout(opp_cfg.dropout), nn.Linear(d * 2, d))
        self.out = nn.Linear(d, N_OUTPUT_CLASSES)

        # Second head: does the opponent change shape at all?
        #
        # On 1200 matches the opponent's formation moved in 39 of 614
        # decisions (6.4%), spread over 10 target formations -- far too few
        # to learn a 15-way choice, and a one-line rule that always predicts
        # "no change" already scores 93.6%.  What IS learnable, and what the
        # planner actually consumes, is the propensity to respond: 125 of
        # 614 (20.4%) positives for "the opponent intervened".
        #
        # So the trunk feeds two heads.  `out` keeps answering "which
        # formation", trained only on the decisions where a change really
        # happened, so it never collapses into the persistence rule.
        # `respond_out` answers "will they change at all".  predict_proba
        # recombines them into one distribution over formations.
        self.respond_out = nn.Linear(d, 1)

        if opp_cfg.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    # ----------------------------------------------------------------- core
    def build_query(self, f_src: torch.Tensor, f_tgt: torch.Tensor,
                    intent: torch.Tensor) -> torch.Tensor:
        q = torch.cat([self.f_emb(f_src), self.f_emb(f_tgt), self.intent_emb(intent)], dim=-1)
        return self.norm_q(self.query_proj(q))

    def head(self, z_t: torch.Tensor, f_src: torch.Tensor, f_tgt: torch.Tensor,
             intent: torch.Tensor, style: torch.Tensor,
             return_respond: bool = False):
        """Logits from an already-encoded state. Used by the inference loop to
        score many candidate transitions against one cached z_t.

        Returns formation logits, or (formation logits, respond logit) when
        return_respond is True.  The default keeps the single-tensor contract
        every existing caller relies on.
        """
        q = self.build_query(f_src, f_tgt, intent).unsqueeze(1)          # [B,1,d]
        kv = [self.ctx_proj(z_t).unsqueeze(1)]                            # [B,1,d]
        if self.style_conditioned:
            kv.append(self.style_proj(self.style_emb(style)).unsqueeze(1))
        kv_tensor = torch.cat(kv, dim=1)                                  # [B,1|2,d]

        h, _ = self.attn(q, kv_tensor, kv_tensor, need_weights=False)
        h = self.norm_o(q + h).squeeze(1)
        h = h + self.ffn(h)
        logits = self.out(h)
        if return_respond:
            return logits, self.respond_out(h).squeeze(-1)
        return logits

    def forward(self, batch: Dict[str, torch.Tensor],
                return_respond: bool = False):
        z_t = self.encoder(batch)
        return self.head(z_t, batch["f_src"], batch["f_tgt"], batch["intent"],
                         batch["style"], return_respond=return_respond)

    # ----------------------------------------------------------------- loss
    def loss(self, logits: torch.Tensor, target: torch.Tensor,
             respond_logit: Optional[torch.Tensor] = None,
             respond_target: Optional[torch.Tensor] = None,
             changed: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Formation CE (+ KL calibration) and, optionally, the respond BCE.

        `changed` is a boolean mask marking the decisions where the
        opponent's formation actually moved.  When it is given, the
        formation head is trained on those rows only: including the
        unchanged ones would teach it to copy the opponent's current
        formation, which is the persistence baseline, and the head would
        then add nothing the planner does not already know.

        Passing neither respond_logit nor changed reproduces the original
        single-head behaviour exactly.
        """
        device = logits.device
        if changed is None:
            sel = torch.ones(len(logits), dtype=torch.bool, device=device)
        else:
            sel = changed.to(device=device, dtype=torch.bool)

        if int(sel.sum()) > 0:
            ce = F.cross_entropy(
                logits[sel], target[sel],
                label_smoothing=self.opp_cfg.label_smoothing)
            logp = F.log_softmax(logits[sel], dim=-1)
            p = logp.exp()
            # KL(p || uniform) = sum p (log p + log |V|) = log|V| - H(p)
            kl = (p * logp).sum(-1).mean() + torch.log(
                torch.tensor(float(N_OUTPUT_CLASSES), device=device))
        else:
            # A batch can legitimately contain no positives at 6% base rate.
            ce = logits.sum() * 0.0
            kl = logits.sum() * 0.0

        total = ce + self.opp_cfg.kl_lambda * kl
        out = {"ce": ce.detach(), "kl": kl.detach(),
               "n_changed": torch.tensor(float(int(sel.sum())))}

        if respond_logit is not None and respond_target is not None:
            y = respond_target.to(device=device, dtype=logits.dtype)
            pos_weight = None
            if self.opp_cfg.respond_pos_weight > 0:
                pos_weight = torch.tensor(
                    self.opp_cfg.respond_pos_weight, device=device)
            else:
                # Balance automatically from the batch: at a ~11% base rate
                # an unweighted BCE converges happily on "never responds".
                #
                # Only when the batch actually holds both classes.  The
                # obvious n_neg / n_pos becomes exactly 0 in an all-positive
                # batch, and a pos_weight of 0 silently deletes the whole
                # BCE term -- the loss goes to zero and training looks like
                # it converged. Caught by test_two_heads.
                n_pos = y.sum()
                n_neg = len(y) - n_pos
                if n_pos > 0 and n_neg > 0:
                    pos_weight = (n_neg / n_pos).detach()
            bce = F.binary_cross_entropy_with_logits(
                respond_logit, y, pos_weight=pos_weight)
            total = total + self.opp_cfg.respond_weight * bce
            out["bce"] = bce.detach()

        out["loss"] = total
        return out

    # -------------------------------------------------------------- helpers
    @torch.no_grad()
    def predict_proba(self, batch: Dict[str, torch.Tensor],
                      mixture: bool = False) -> torch.Tensor:
        """Distribution over the opponent's next formation.

        mixture=False keeps the plain softmax of the formation head, which
        is the "given they change, to what?" distribution.

        mixture=True returns what the planner should actually consume:

            P(f) = (1 - p_respond) * 1[f = current] + p_respond * softmax(f)

        i.e. the formation head's answer weighted by the probability that a
        change happens at all, with the remaining mass parked on the shape
        the opponent is already in.  This is the calibrated object the
        conformal terminator reads, and it needs batch["opp_formation_id"].
        """
        if not mixture:
            return torch.softmax(self(batch), dim=-1)
        logits, respond_logit = self(batch, return_respond=True)
        p_change = torch.sigmoid(respond_logit).unsqueeze(-1)
        p_form = torch.softmax(logits, dim=-1)
        stay = torch.zeros_like(p_form)
        stay.scatter_(1, batch["opp_formation_id"].view(-1, 1).long(), 1.0)
        return (1.0 - p_change) * stay + p_change * p_form

    def save(self, path: str) -> None:
        torch.save({
            "enc_cfg": vars(self.enc_cfg),
            "opp_cfg": vars(self.opp_cfg),
            "style_conditioned": self.style_conditioned,
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "OpponentCoBehaviorModel":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(EncoderConfig(**blob["enc_cfg"]), OpponentConfig(**blob["opp_cfg"]),
                    style_conditioned=blob["style_conditioned"])
        model.load_state_dict(blob["state_dict"])
        # map_location decides where the CHECKPOINT tensors are read to;
        # load_state_dict then copies their values into this module's own
        # parameters, which were constructed on the CPU. So without the move
        # below, load(..., map_location="cuda") handed back a CPU model while
        # the caller believed it had a CUDA one, and the mismatch surfaced far
        # away as "Expected all tensors to be on the same device" inside an
        # embedding lookup. Honour the argument its name promises.
        return model.to(map_location)
