"""
planner_diagnostics.py -- critic diagnostics computed from the planner's real
outputs, kept in one testable place.

Why this exists (team-lead audit, "dead critic diagnostic")
-----------------------------------------------------------
eval_planner.py used to compute critic agreement and "effective weights"
inline, and both could quietly stop measuring what they claimed:

* Agreement was a Spearman rho over the raw candidate list, which repeats a
  formation once per intent, so the same target was counted several times.
  Decisions where a critic was constant (for example with the stub world
  model) were skipped without a word, so the table could come out empty or
  rest on a handful of decisions, and nothing said so.
* "Effective weight" was the spread of RAW scores pooled across all
  decisions. The planner z-scores critics within each decision before
  weighting them, and ranks only within a decision, so the pooled raw spread
  was not the influence a critic actually has on a ranking.

Here, agreement is computed per distinct target formation, using the same
"best score per target" rule that turns each critic into a ranking. Every
decision is counted as used or skipped with a reason, and a critic that never
varies is reported as dead. The effective weight is measured on the scores as
the planner aggregates them.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Iterable, List, Sequence

import numpy as np
from scipy.stats import spearmanr


def per_target(recs: Sequence, key: str = "critic_scores") -> Dict[int, Dict[str, float]]:
    """{f_tgt: {critic: best score over that target's candidates}}."""
    out: Dict[int, Dict[str, float]] = {}
    for r in recs:
        cs = getattr(r, key)
        d = out.setdefault(int(r.f_tgt), {})
        for k, v in cs.items():
            d[k] = max(d.get(k, -np.inf), float(v))
    return out


def critic_agreement(recs: Sequence, min_targets: int = 4):
    """Spearman rho between every pair of critics over one decision's targets.

    Returns (rhos, status). rhos maps (a, b) -> rho for the pairs that could be
    measured. status is "ok", "too_few_targets", or "constant:<critic>,..."
    when some critic gives every target the same score (that critic carries
    no ranking information in this decision).
    """
    tab = per_target(recs)
    if len(tab) < min_targets:
        return {}, "too_few_targets"
    names = sorted(next(iter(tab.values())).keys())
    cols = {c: np.array([tab[t][c] for t in sorted(tab)]) for c in names}
    constant = [c for c in names if np.ptp(cols[c]) == 0]
    rhos = {}
    for a, b in combinations(names, 2):
        if a in constant or b in constant:
            continue
        rho = spearmanr(cols[a], cols[b]).statistic
        if np.isfinite(rho):
            rhos[(a, b)] = float(rho)
    status = "ok" if not constant else "constant:" + ",".join(constant)
    return rhos, status


class AgreementTracker:
    """Accumulates critic_agreement over decisions and says how many counted."""

    def __init__(self):
        self.rhos = defaultdict(list)
        self.status = Counter()
        self.const_count = Counter()
        self.n = 0

    def add(self, recs: Sequence) -> None:
        self.n += 1
        rhos, status = critic_agreement(recs)
        self.status[status.split(":")[0]] += 1
        if status.startswith("constant:"):
            for c in status.split(":", 1)[1].split(","):
                self.const_count[c] += 1
        for k, v in rhos.items():
            self.rhos[k].append(v)

    def summary(self) -> Dict:
        return {
            "decisions": self.n,
            "status": dict(self.status),
            "constant_in_decisions": dict(self.const_count),
            "dead_critics": [c for c, k in self.const_count.items() if k == self.n and self.n],
            "pairs": {f"{a}|{b}": {"mean_rho": float(np.mean(v)), "decisions": len(v)}
                      for (a, b), v in sorted(self.rhos.items())},
        }


def effective_weights_as_used(recs: Sequence, normalise, weights: Sequence[float]) -> Dict[str, float]:
    """Share of within-decision score variation each critic contributes, measured
    on the normalised scores the planner actually aggregates.

    `normalise` is the planner's own `_normalise_critics`; `weights` its
    configured critic weights, in critic order.
    """
    scaled = normalise([r.critic_scores for r in recs])
    if not scaled:
        return {}
    names = list(scaled[0].keys())
    w = (list(weights) + [0.0] * len(names))[:len(names)]
    spread = {n: wi * float(np.std([s[n] for s in scaled])) for n, wi in zip(names, w)}
    tot = sum(spread.values())
    return {n: (v / tot if tot else 0.0) for n, v in spread.items()}
