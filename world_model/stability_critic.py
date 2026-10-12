"""
stability_critic.py -- the stability critic the planner uses.

Project audit, "stability critic capacity": the learned MLP lost to a
logistic regression on structural distance alone, and the planner was running
an unlearned placeholder (-distance/100) whose sign assumption scored below
chance. The audit's fix is to prefer the simpler baseline. This critic is that
baseline: P(concede within 600 s) = sigmoid(a + b * delta_struct), with a and b
fitted by train_stability.py (written to results/stability_logistic.json).
The score is -P(concede), so higher is safer, as for the other critics.
"""
from __future__ import annotations

import json
import math
import os

from critics import structural_distance


class LogisticStabilityCritic:
    name = "stability"
    is_stub = False

    def __init__(self, path):
        with open(path) as f:
            p = json.load(f)
        self.a, self.b = float(p["intercept"]), float(p["coef"])
        self.path = path
        self.provenance = p
        # critics.structural_distance reads canonical_roles.json from a path
        # relative to THIS directory, lazily, on first use; the planner calls
        # it from another directory. Load (and cache) it here instead.
        import critics as _critics
        here = os.getcwd()
        try:
            os.chdir(os.path.dirname(os.path.abspath(__file__)))
            _critics._get_struct_features()
        finally:
            os.chdir(here)

    def concede_probability(self, f_src, f_tgt) -> float:
        z = self.a + self.b * float(structural_distance(int(f_src), int(f_tgt)))
        return 1.0 / (1.0 + math.exp(-z))

    def score(self, f_src, f_tgt, f_opp, state, wm_out):
        return -self.concede_probability(f_src, f_tgt)


def load(wm_dir, filename="results/stability_logistic.json"):
    path = filename if os.path.isabs(filename) else os.path.join(wm_dir, filename)
    return LogisticStabilityCritic(path) if os.path.exists(path) else None
