#!/usr/bin/env python3
"""
world_model_adapter.py -- the trained bipartite GNN behind the opponent
model's WorldModel protocol, in place of StubWorldModel.

Why this matters more than it looks
-----------------------------------
ramfp/inference.py ships a StubWorldModel whose own docstring says: "Its
outputs are a deterministic function of formation geometry -- plausible
enough to exercise the plumbing, and useless as a result. Never report
numbers that passed through this." Every critic downstream reads the world
model, so while the stub is in place the revision loop's rankings carry no
evidence at all. This adapter is what turns that plumbing into a result.

Contract
--------
    predict(f_ours, f_theirs, state) -> {"xg_delta", "possession", "ppda", "shots"}

Orientation, stated because getting it wrong is silent
------------------------------------------------------
The world model was trained on HOME-relative targets:

    xg_delta        home xG minus away xG over the window
    possession_pct  home share of passes, 0..1
    ppda            passes the HOME side made per AWAY defensive action,
                    i.e. how hard the AWAY team pressed
    shots_on_target shots on target by BOTH sides in the window

f_ours is placed at home and f_theirs away, so xg_delta and possession come
out in our favour as the protocol expects. `ppda` does NOT: it is the
opponent pressing us, not us pressing them. It is returned under the
protocol's name because that is the quantity the model was trained to
predict, and repeated under an unambiguous key. A critic that wants our own
pressing intensity is asking for something this model was never fitted to.

`shots` is a two-sided count, not a differential. Also repeated under an
unambiguous key.
"""

from __future__ import annotations

import json
import os
from typing import Dict

import pandas as pd
import torch

from models.world_model.bipartite_gnn import BipartiteFormationGNN
from models.world_model.graph_builder import build_bipartite_graph

# Checkpoint and data paths resolve against this file, so the planner can
# import the adapter from one directory and plan from another. graph_builder
# anchors its own JSON paths the same way.
_HERE = os.path.dirname(os.path.abspath(__file__))

TARGETS = ["xg_delta", "possession_pct", "ppda", "shots_on_target"]

# Synthetic ids so the graph builder's style lookup can be fed directly from
# the GameState instead of from a team table that inference does not have.
_OURS_ID, _THEIRS_ID = -1, -2


class TrainedWorldModel:
    """Step 10, for real. is_stub is False, which is the whole point."""

    is_stub = False

    def __init__(self, checkpoint_path="checkpoints/world_model_best.pt",
                 data_dir="data/processed", device=None, symmetrise=True):
        # same reason as the module-level path pinning above
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(_HERE, checkpoint_path)
        if not os.path.isabs(data_dir):
            data_dir = os.path.join(_HERE, data_dir)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"{checkpoint_path} not found. Train it first:\n"
                f"  python train_world_model.py --config configs/world_model.yaml"
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt["cfg"]
        self.norm = ckpt["norm_stats"]
        self.val_mse = ckpt.get("val_mse")
        self.val_pearson = ckpt.get("val_pearson")

        self.model = BipartiteFormationGNN(
            node_feat_dim=cfg["node_feat_dim"], node_dim=cfg["node_dim"],
            gat_heads=cfg["gnn_heads"], gnn_layers=cfg["gnn_layers"],
            output_dim=cfg["output_dim"], edge_feat_dim=cfg["edge_feat_dim"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

        # match the feature set the checkpoint was trained with, or every
        # prediction is made from a node feature the model never saw in training
        from models.world_model import graph_builder as _gb
        # Checkpoints written before this flag existed were all trained WITH the
        # ppda node feature, so absent means True here. Defaulting to False
        # would silently feed such a model a feature set it never saw.
        self.use_ppda_feature = bool(cfg.get("use_ppda_feature", True))
        _gb.USE_PPDA_FEATURE = self.use_ppda_feature
        # same for the edge set: checkpoints before bidirectional edges
        # existed were trained home -> away only
        self.bidirectional_edges = bool(cfg.get("bidirectional_edges", False))
        _gb.BIDIRECTIONAL_INTER = self.bidirectional_edges
        # Exact swap consistency at inference. The planner
        # does not know the venue, so each prediction is the antisymmetric
        # average of "ours in slot 0" and "theirs in slot 0":
        #   xg_delta   = (f(A,B) - f(B,A)) / 2            -> flips sign on swap
        #   possession = (f(A,B) + 1 - f(B,A)) / 2        -> shares sum to 100
        #   shots      = (f(A,B) + f(B,A)) / 2            -> symmetric, as defined
        #   ppda       = f(A,B)  (the opponent pressing us; its mirror is a
        #                         different quantity the model does not output)
        # Both identities then hold to rounding error by construction, for
        # any checkpoint. symmetrise=False reproduces the raw model.
        self.symmetrise = bool(symmetrise)
        if self.use_ppda_feature:
            print("NOTE: this checkpoint was trained with ppda as a node feature. "
                  "ppda is also target #3 and is measured over the same window, so "
                  "its reported correlation is inflated by leakage.")

        with open(os.path.join(data_dir, "formation_vocab.json")) as f:
            vocab = json.load(f)["vocab"]
        self.vocab = {int(c): i for i, c in enumerate(vocab)}
        # Use the role-matchup table this checkpoint was trained with. A
        # cross-fitted checkpoint records a table computed without its excluded
        # matches; older checkpoints record the default path, so they are
        # unaffected.
        rms_path = cfg.get("role_matchup_stats_path") or os.path.join(data_dir, "role_matchup_stats.parquet")
        if not os.path.isabs(rms_path):
            rms_path = os.path.join(_HERE, rms_path)
        if not os.path.exists(rms_path):
            rms_path = os.path.join(data_dir, "role_matchup_stats.parquet")
        self.rms = pd.read_parquet(rms_path)
        self.rms_path = rms_path
        self.excluded_matches_file = cfg.get("excluded_matches_file")

    def _style_frame(self, ours_style, theirs_style):
        return pd.DataFrame({
            "team_id": [_OURS_ID, _THEIRS_ID],
            "style_cluster": [int(ours_style), int(theirs_style)],
        })

    def _raw(self, f_slot0, f_slot1, goal_diff_slot0, style0, style1, minute, ppda):
        """One forward pass, slot 0 flagged as the home side; denormalised."""
        ctx = {
            "minute": float(minute),
            "scoreline_home": max(int(goal_diff_slot0), 0),
            "scoreline_away": max(-int(goal_diff_slot0), 0),
            "ppda": ppda,
            "home_team_id": _OURS_ID,
            "away_team_id": _THEIRS_ID,
        }
        graph = build_bipartite_graph(
            int(f_slot0), int(f_slot1), ctx, self.rms,
            self._style_frame(style0, style1), self.vocab,
        ).to(self.device)
        out = self.model(graph).reshape(-1)
        return {
            t: float(out[i]) * self.norm[t]["std"] + self.norm[t]["mean"]
            for i, t in enumerate(TARGETS)
        }

    @torch.no_grad()
    def predict(self, f_ours: int, f_theirs: int, state) -> Dict[str, float]:
        ours_style = int(getattr(state, "our_style", 0) or 0)
        theirs_style = int(getattr(state, "opp_style", 0) or 0)
        gd = int(getattr(state, "goal_diff", 0))
        minute = float(getattr(state, "minute", 45))
        ppda = getattr(state, "opp_ppda", None)

        vals = self._raw(f_ours, f_theirs, gd, ours_style, theirs_style, minute, ppda)
        if self.symmetrise:
            rev = self._raw(f_theirs, f_ours, -gd, theirs_style, ours_style, minute, ppda)
            vals = {
                "xg_delta": 0.5 * (vals["xg_delta"] - rev["xg_delta"]),
                "possession_pct": 0.5 * (vals["possession_pct"] + 1.0 - rev["possession_pct"]),
                "ppda": vals["ppda"],
                "shots_on_target": 0.5 * (vals["shots_on_target"] + rev["shots_on_target"]),
            }

        return {
            "xg_delta": vals["xg_delta"],
            "possession": vals["possession_pct"] * 100.0,
            "ppda": vals["ppda"],
            "shots": vals["shots_on_target"],
            # unambiguous aliases; see the orientation note in the module docstring
            "opponent_pressing_ppda": vals["ppda"],
            "shots_on_target_both_sides": vals["shots_on_target"],
        }

    def consistency_check(self, formations=None, state=None, verbose=True):
        """Does the model contradict itself when the sides are swapped?

        Two of the four targets are antisymmetric by definition: swapping
        which side is home must flip xg_delta's sign and make the two
        possession shares sum to 100. Nothing in the architecture enforces
        either -- home and away are separate node blocks with a one-directional
        inter-team edge set -- so the model can, and by default will, violate
        both. This needs no labels and no held-out data, which makes it the
        cheapest honest statement available about whether the world model's
        outputs mean what the critics downstream read them as meaning.

        Report the two numbers alongside any result that passes through this
        model. A large possession residual means the ranking a critic produces
        depends on an arbitrary choice of which formation was called "home".
        """
        from dataclasses import dataclass

        @dataclass
        class _Default:
            opp_style: int = 0
            goal_diff: int = 0
            minute: int = 45
            opp_ppda: float = 12.0
            our_style: int = 0

        state = state or _Default()
        formations = formations or sorted(self.vocab)[:12]

        # The swapped call must swap the WHOLE state, not only the
        # formations: the other side sees the opposite goal difference and
        # the styles exchanged.
        from dataclasses import replace as _replace
        try:
            swapped = _replace(state, goal_diff=-int(getattr(state, "goal_diff", 0)),
                               opp_style=int(getattr(state, "our_style", 0) or 0),
                               our_style=int(getattr(state, "opp_style", 0) or 0))
        except TypeError:
            swapped = state

        xg_residuals, poss_residuals = [], []
        for i, a in enumerate(formations):
            for b in formations[i + 1:]:
                ab = self.predict(a, b, state)
                ba = self.predict(b, a, swapped)
                xg_residuals.append(abs(ab["xg_delta"] + ba["xg_delta"]))
                poss_residuals.append(abs(ab["possession"] + ba["possession"] - 100.0))

        n = len(xg_residuals)
        out = {
            "pairs_tested": n,
            "xg_delta_antisymmetry_mean_abs_residual": sum(xg_residuals) / max(n, 1),
            "xg_delta_antisymmetry_max_abs_residual": max(xg_residuals, default=0.0),
            "possession_sum_mean_abs_residual_pct": sum(poss_residuals) / max(n, 1),
            "possession_sum_max_abs_residual_pct": max(poss_residuals, default=0.0),
        }
        if verbose:
            print(f"swap consistency over {n} formation pairs")
            print(f"  |xg_delta(A,B) + xg_delta(B,A)|      mean {out['xg_delta_antisymmetry_mean_abs_residual']:.4f}  "
                  f"max {out['xg_delta_antisymmetry_max_abs_residual']:.4f}   (0 if consistent)")
            print(f"  |possession(A,B) + possession(B,A) - 100|  mean {out['possession_sum_mean_abs_residual_pct']:.2f}pp  "
                  f"max {out['possession_sum_max_abs_residual_pct']:.2f}pp  (0 if consistent)")
        return out

    def provenance(self) -> Dict[str, object]:
        """What a results table has to state alongside any number from here."""
        return {
            "is_stub": False,
            "val_mse_total": self.val_mse,
            "val_pearson_per_target": self.val_pearson,
            "targets": TARGETS,
            "note": "ppda is the opponent pressing us; shots is a two-sided count",
            "symmetrised": self.symmetrise,
            "bidirectional_edges": self.bidirectional_edges,
        }


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class _S:
        opp_style: int = 1
        goal_diff: int = -1
        minute: int = 70
        opp_ppda: float = 9.0

    wm = TrainedWorldModel()
    print("provenance:", json.dumps(wm.provenance(), indent=2, default=str))
    print()
    wm.consistency_check()
    print()
    for ours, theirs in [(442, 433), (433, 442), (4231, 352), (41212, 3511), (3412, 4231)]:
        p = wm.predict(ours, theirs, _S())
        print(f"  {ours} vs {theirs}:  xg_delta={p['xg_delta']:+.3f}  "
              f"possession={p['possession']:.1f}%  ppda={p['ppda']:.1f}  shots={p['shots']:.2f}")
