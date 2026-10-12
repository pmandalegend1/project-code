"""
End-to-end demo of the inference loop.

    python scripts/demo_inference.py --config configs/smoke.yaml \
        --opponent models/opponent.pt

Fits the conformal terminator on a held-out calibration split, then plans at a
single decision point and prints the ranked recommendations. The world model is
stubbed until step 10 lands, and every affected line says so.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ramfp.config import Config
from ramfp.data.datasets import TripleDataset, collate
from ramfp.data.synthetic import generate_triples
from ramfp.eval.metrics import conformal_coverage, conformal_set_sizes
from ramfp.inference import ConformalTerminator, RAMFPPlanner, demo_state
from ramfp.models.opponent import OpponentCoBehaviorModel
from torch.utils.data import DataLoader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--opponent", default="models/opponent.pt")
    ap.add_argument("--n-calibration", type=int, default=400)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    device = cfg.resolved_device()
    model = OpponentCoBehaviorModel.load(args.opponent, map_location=device).to(device).eval()

    # --- calibration for the conformal stopping rule ----------------------
    cal_df, _ = generate_triples(args.n_calibration, seed=12345,
                                 seq_len=model.enc_cfg.seq_len)
    dl = DataLoader(TripleDataset(cal_df, model.enc_cfg.seq_len, knn=model.enc_cfg.knn),
                    batch_size=64, collate_fn=collate)
    probs, ys = [], []
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            probs.append(torch.softmax(model(b), -1).cpu().numpy())
            ys.append(b["target"].cpu().numpy())
    probs, ys = np.concatenate(probs), np.concatenate(ys)

    term = ConformalTerminator(cfg.inference.conformal_alpha,
                               cfg.inference.conformal_threshold).fit(probs, ys)
    sizes = conformal_set_sizes(probs, term.qhat)
    print(f"conformal calibration on {len(ys)} held-out triples")
    print(f"  alpha={cfg.inference.conformal_alpha}  qhat={term.qhat:.4f}")
    print(f"  empirical coverage {conformal_coverage(probs, ys, term.qhat):.3f} "
          f"(target {1 - cfg.inference.conformal_alpha:.2f})")
    print(f"  mean prediction-set size {sizes.mean():.2f}  median {np.median(sizes):.0f}  "
          f"(theta_conf={cfg.inference.conformal_threshold})")
    suggested = int(np.median(sizes))
    if cfg.inference.conformal_threshold < suggested:
        print(f"  WARNING: theta_conf ({cfg.inference.conformal_threshold}) is below the "
              f"median set size ({suggested}), so most candidates terminate at round 0\n"
              f"           and the revision loop never runs. With |V|=15 classes and a "
              f"Bayes ceiling near 0.32,\n"
              f"           a 90% conformal set is naturally large. Calibrate theta_conf "
              f"against this distribution\n"
              f"           — e.g. the median ({suggested}) or the 75th percentile "
              f"({int(np.percentile(sizes, 75))}) — rather than picking a round number.\n"
              f"           Report the chosen value and the set-size distribution in the paper.")
    print()

    # --- plan --------------------------------------------------------------
    planner = RAMFPPlanner(model, cfg=cfg.inference, terminator=term, device=device)
    state = demo_state()
    print(f"decision point: minute {state.minute}, {state.f_src} vs "
          f"{state.opp_formation}, goal difference {state.goal_diff:+d}, "
          f"opponent style {state.opp_style} (PPDA {state.opp_ppda})\n")

    recs = planner.plan(state, top_n=3)
    print(f"{len(recs)} candidates survived the tactical grammar; top 5:\n")
    for i, r in enumerate(recs[:5], 1):
        print(f"{i}. {r.explain()}\n")


if __name__ == "__main__":
    main()
