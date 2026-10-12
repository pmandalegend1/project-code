"""
Bounded Revision Loop
=====================
Implements Algorithm 1 (RAMFP Inference), lines 9–22.
Paper reference: Section 5.8.
Pipeline reference: Section F.

File path: models/integration/revision_loop.py

FIX — Note 1: θ_conf broken, loop never runs.
    Root cause: pipeline spec suggests θ_conf = 5 as a hardcoded constant.
    At α = 0.10 on |V| = 15 classes with a moderately calibrated opponent
    model, conformal set sizes at round m = 0 routinely exceed 5, causing
    the loop to terminate before even one revision executes. This is a
    silent failure — the pipeline runs without error but revision_rounds
    is always 0, making B4 and B5 in the ablation table identical to B3.
    Fix: θ_conf is calibrated at runtime via calibrate_theta_conf() as the
    80th-percentile conformal set size on held-out calibration data. This
    ensures at most 20% of calibration examples would trigger early
    termination at round 0, so the loop executes for the majority of
    inference calls.

NOTE — Note 5: ECE directly affects model selection.
    The conformal predictor's validity guarantee (coverage ≥ 1 - α = 90%)
    holds only if the opponent model's softmax probabilities are
    exchangeably drawn from the same distribution as the calibration set.
    A model selected by Top-1 accuracy alone can be overconfident
    (low entropy, high ECE), causing the conformal quantile to be too low
    and prediction sets too small — the set size will rarely exceed θ_conf
    even when the model is genuinely uncertain, so the loop runs all M
    rounds even when it should stop. Conversely, an underconfident model
    gives sets too large and the loop terminates too early.
    REQUIRED: the opponent model checkpoint passed to BoundedRevisionLoop
    MUST be selected by: argmax Top-1 Acc subject to ECE ≤ 0.15.
    This constraint must be enforced in train_opponent_model.py
    (the opponent model module). An assertion is included at initialization below.
    The ECE bound of 0.15 matches the empirical OOD threshold in Section 5.8.

Context updater — pipeline spec §F:
    UpdateContext is implemented as nn.Linear(258, 256) — projecting
    z_t (256) concatenated with [ŷ_poss, ŷ_PPDA] (2 scalars) back to
    256. This linear is initialized randomly and frozen after training
    (or trained as an auxiliary head of the world model; coordinate with
    the opponent model module). It is registered as a module in BoundedRevisionLoop so
    its state_dict is saved alongside the pipeline checkpoint.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CONTEXT_DIM    = 256
MAX_ROUNDS     = 2      # hard cap from paper §5.8
ECE_THRESHOLD  = 0.15   # §5.8 OOD criterion
ACC_DROP_LIMIT = 0.05   # 5 pp drop triggers OOD flag
N_BINS_ECE     = 10
ALPHA          = 0.10   # conformal significance level

# ── Named types ───────────────────────────────────────────────────────────────
@dataclass
class CriticScores:
    xg:         float
    stability:  float
    resistance: float

    @property
    def aggregate(self) -> float:
        return (self.xg + self.stability + self.resistance) / 3.0


@dataclass
class ScoredTransition:
    """
    Full output record per candidate.
    Matches the namedtuple spec in pipeline §G, extended with revision metadata.
    """
    f_src:          int
    f_tgt:          int
    intent:         str
    total_score:    float
    c_xg:           float
    c_stab:         float
    c_res:          float
    f_opp_predicted: int
    opp_dist:       torch.Tensor   # (|V|,) calibrated probabilities
    conf_set_size:  int = 0
    revision_rounds: int = 0


@dataclass
class RevisionLoopOutput:
    ranked:             List[ScoredTransition]
    rounds_run:         int
    termination_reason: str    # "max_rounds" | "conformal" | "ood"


# ── Context updater (pipeline §F) ─────────────────────────────────────────────
class ContextUpdater(nn.Module):
    """
    UpdateContext(z_t, ŷ_τ) → z'_t  ∈ ℝ²⁵⁶

    Projects [z_t ∈ ℝ²⁵⁶ ; ŷ_poss ∈ ℝ ; ŷ_PPDA ∈ ℝ] → ℝ²⁵⁶.
    Pipeline spec §F: "nn.Linear(258, 256), frozen after training".
    Initialize to near-identity by zeroing the extra 2 input dimensions'
    weights; this makes round m=0 and m=1 outputs comparable in magnitude
    without requiring a warm-up period.
    """

    def __init__(self, context_dim: int = CONTEXT_DIM) -> None:
        super().__init__()
        self.proj = nn.Linear(context_dim + 2, context_dim, bias=False)
        # Near-identity init: copy identity block, zero out the extra columns.
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[:context_dim, :context_dim] = torch.eye(context_dim)
        # Frozen by default per pipeline spec. Call .unfreeze() to train jointly.
        self._freeze()

    def _freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        """Call this to enable joint training with the world model."""
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(
        self,
        z_t: torch.Tensor,     # (256,) or (B, 256)
        poss: float,           # predicted possession ∈ [0, 1]
        ppda: float,           # predicted PPDA (normalise to [0, 1] before passing)
    ) -> torch.Tensor:
        scalar = torch.tensor([poss, ppda], dtype=z_t.dtype, device=z_t.device)
        if z_t.dim() == 1:
            return self.proj(torch.cat([z_t, scalar], dim=0))
        # batched
        scalar = scalar.unsqueeze(0).expand(z_t.shape[0], -1)
        return self.proj(torch.cat([z_t, scalar], dim=-1))


# ── Conformal predictor ───────────────────────────────────────────────────────
class ConformalPredictor:
    """
    Split conformal predictor for the opponent co-behavior model.

    Non-conformity score: s(x, y) = 1 − p̂_y(x).
    Prediction set: Ŝ(x) = {y : 1 − p̂_y(x) ≤ q̂}
    where q̂ = ⌈(n+1)(1−α)⌉/n -th empirical quantile of calibration scores.

    Reference: Vovk et al. (2005); Angelopoulos & Bates (2021).
    
    """

    def __init__(self, alpha: float = ALPHA) -> None:
        self.alpha = alpha
        self._q_hat: Optional[float] = None

    def fit(self, cal_probs: torch.Tensor, cal_labels: torch.Tensor) -> None:
        """
        cal_probs:  (N, |V|) softmax probabilities on calibration set.
        cal_labels: (N,)     ground-truth formation indices.
        """
        n = cal_probs.shape[0]
        scores = 1.0 - cal_probs[torch.arange(n), cal_labels]   # (N,)
        level = math.ceil((n + 1) * (1.0 - self.alpha)) / n
        level = min(level, 1.0)
        self._q_hat = float(torch.quantile(scores, level).item())
        logger.info("ConformalPredictor fitted: q̂ = %.4f (α = %.2f, n = %d)",
                    self._q_hat, self.alpha, n)

    @property
    def q_hat(self) -> float:
        if self._q_hat is None:
            raise RuntimeError("ConformalPredictor.fit() must be called before inference.")
        return self._q_hat

    def prediction_set(self, probs: torch.Tensor) -> List[int]:
        """probs: (|V|,) → list of formation indices that cannot be excluded."""
        return [i for i, p in enumerate(probs.tolist()) if (1.0 - p) <= self.q_hat]

    def set_size(self, probs: torch.Tensor) -> int:
        return len(self.prediction_set(probs))


# ── θ_conf calibration ──────────────────────────────
def calibrate_theta_conf(
    conformal: ConformalPredictor,
    cal_probs: torch.Tensor,
    percentile: float = 0.80,
) -> int:
    """
    Calibrates θ_conf as the (percentile)-th quantile of conformal set sizes
    observed at revision round m = 0 on the held-out calibration set.

    This guarantees that at most (1 − percentile) = 20% of calibration
    examples trigger early loop termination at round 0, so the revision
    loop executes for the majority of inference calls.

    Why this fixes the bug:
        With hardcoded θ_conf = 5 and |V| = 15, any opponent model with
        ECE > 0 will produce set sizes frequently above 5, causing the loop
        to terminate at m = 0. The calibrated threshold is data-adaptive and
        automatically accounts for the model's calibration quality.

    Args:
        conformal:  A *fitted* ConformalPredictor (calibration set passed
                    during fitting; a *different* held-out split should be
                    used here to avoid data leakage).
        cal_probs:  (N, |V|) softmax probabilities from the opponent model
                    on a held-out OOD validation set (not the conformal
                    calibration set).
        percentile: Fraction of examples that should NOT trigger early
                    termination at round 0. Default 0.80 (80th percentile).

    Returns:
        θ_conf (int): set-size threshold ≥ 1.
    """
    if not (0.5 < percentile < 1.0):
        raise ValueError(f"percentile must be in (0.5, 1.0), got {percentile}")

    sizes = torch.tensor(
        [conformal.set_size(cal_probs[i]) for i in range(cal_probs.shape[0])],
        dtype=torch.float,
    )
    theta = int(math.ceil(torch.quantile(sizes, percentile).item()))
    theta = max(theta, 1)     # never zero
    theta = min(theta, 14)    # never ≥ |V|; if so, the predictor is meaningless

    logger.info(
        "θ_conf calibrated to %d (%.0f-th pctl of set sizes on OOD val, "
        "mean=%.1f, min=%d, max=%d)",
        theta, percentile * 100,
        sizes.mean().item(), int(sizes.min().item()), int(sizes.max().item()),
    )
    return theta


# ── ECE utility ───────────────────────────────────────────────────────────────
def compute_ece(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = N_BINS_ECE,
) -> float:
    """
    Expected Calibration Error over a mini-batch.

    probs:  (N, |V|) softmax probabilities.
    labels: (N,)     integer ground-truth indices.
    """
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels).float()
    ece = 0.0
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    n = len(labels)
    for i in range(n_bins):
        lo, hi = float(bin_edges[i]), float(bin_edges[i + 1])
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        acc_b  = accuracies[mask].mean().item()
        conf_b = confidences[mask].mean().item()
        ece   += (mask.sum().item() / n) * abs(acc_b - conf_b)
    return ece


def assert_ece_constraint(
    opponent_model_ece: float,
    threshold: float = ECE_THRESHOLD,
) -> None:
    """
    Asserts that the opponent model checkpoint meets the ECE requirement.
    Must be called at pipeline initialization before the revision loop runs.

    flag #5: if this assertion fails, the conformal stopping
    criterion is unreliable. Fix the opponent model checkpoint selection
    in train_opponent_model.py before running E3/E4.
    """
    if opponent_model_ece > threshold:
        raise ValueError(
            f"Opponent model ECE = {opponent_model_ece:.4f} exceeds threshold "
            f"{threshold}. "
            f"The conformal stopping criterion is unreliable with this checkpoint. "
            f"Reselect the checkpoint using: argmax Top-1 Acc s.t. ECE ≤ {threshold}. "
        )


# ── OOD monitor ───────────────────────────────────────────────────────────────
class OODMonitor:
    """
    Tracks opponent model Top-1 accuracy and ECE across revision rounds
    as context shifts further from the training distribution.

    Empirical OOD bound (§5.8): loop is capped when accuracy drops > 5 pp
    OR ECE exceeds 0.15 relative to round m = 0 baseline.
    """

    def __init__(self) -> None:
        self._baseline_acc: Optional[float] = None
        self._baseline_ece: Optional[float] = None
        self.history: List[Dict] = []

    def set_baseline(self, acc: float, ece: float) -> None:
        self._baseline_acc = acc
        self._baseline_ece = ece
        self.history.append({"round": 0, "acc": acc, "ece": ece, "ood": False})

    def check(self, round_idx: int, acc: float, ece: float) -> bool:
        """Returns True if OOD degradation is detected."""
        self.history.append({"round": round_idx, "acc": acc, "ece": ece})
        if self._baseline_acc is None:
            return False
        acc_degraded = (self._baseline_acc - acc) > ACC_DROP_LIMIT
        ece_exceeded = ece > ECE_THRESHOLD
        ood = acc_degraded or ece_exceeded
        self.history[-1]["ood"] = ood
        if ood:
            logger.warning(
                "OOD detected at round %d: acc=%.3f (baseline=%.3f, drop=%.3f), "
                "ece=%.3f (threshold=%.2f)",
                round_idx, acc, self._baseline_acc,
                self._baseline_acc - acc, ece, ECE_THRESHOLD,
            )
        return ood


# ── Main revision loop ────────────────────────────────────────────────────────
class BoundedRevisionLoop(nn.Module):
    """
    Bounded Revision Loop — Algorithm 1, lines 9–22.

    Parameters
    ----------
    opponent_model  : nn.Module
        Interface: forward(z_t, tau_emb, style_emb) → (|V|,) log-probabilities.
        MUST be selected by: argmax Top-1 Acc s.t. ECE ≤ 0.15 (flag #5).
    world_model     : callable
        Interface: (f_tgt: int, f_opp: int, z_t: Tensor) → dict with keys
        'xg', 'possession', 'ppda', 'shots'.
    critics         : callable
        Interface: (f_tgt, f_opp, wm_out, tau) → CriticScores.
    conformal       : ConformalPredictor
        Must be fitted before instantiating this class.
    theta_conf      : int
        Calibrated via calibrate_theta_conf(). Do NOT hardcode.
    formation_vocab : List[int]
        Ordered list of formation codes matching opponent model output indices.
    context_updater : ContextUpdater
        Shared instance (may be frozen per pipeline spec §F).
    opponent_ece    : float
        ECE of the opponent model on the held-out calibration set.
        Used to assert flag #5 constraint at init time.
    max_rounds      : int ≤ 2
    """

    def __init__(
        self,
        opponent_model:  nn.Module,
        world_model:     Callable,
        critics:         Callable,
        conformal:       ConformalPredictor,
        theta_conf:      int,
        formation_vocab: List[int],
        context_updater: Optional[ContextUpdater] = None,
        opponent_ece:    float = 0.0,
        max_rounds:      int = MAX_ROUNDS,
    ) -> None:
        super().__init__()

        assert_ece_constraint(opponent_ece)   # flag #5 guard

        if max_rounds > MAX_ROUNDS:
            warnings.warn(
                f"max_rounds={max_rounds} exceeds paper bound M ≤ {MAX_ROUNDS}. "
                "OOD degradation beyond round 2 is not characterized."
            )

        self.opp_model  = opponent_model
        self.world_model = world_model
        self.critics    = critics
        self.conformal  = conformal
        self.theta_conf = theta_conf
        self.vocab      = formation_vocab
        self.max_rounds = max_rounds
        self.context_updater = context_updater or ContextUpdater()
        self._ood_monitor: Optional[OODMonitor] = None

    # ── helpers ───────────────────────────────────────────────────────────────

    def _query_opponent(
        self,
        tau: ScoredTransition,
        z_t: torch.Tensor,
        style_emb: torch.Tensor,
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Returns (opp_probs (|V|,), f_opp_code, conf_set_size).
        """
        # Build tau embedding vector [f_src_idx, f_tgt_idx, intent_idx]
        tau_emb = torch.tensor(
            [self.vocab.index(tau.f_src) if tau.f_src in self.vocab else 0,
             self.vocab.index(tau.f_tgt) if tau.f_tgt in self.vocab else 0,
             _INTENT_IDX.get(tau.intent, 0)],
            dtype=torch.long, device=z_t.device,
        )
        with torch.no_grad():
            logits    = self.opp_model(z_t, tau_emb, style_emb)    # (|V|,)
        probs     = F.softmax(logits, dim=-1)                       # (|V|,)
        f_opp_idx = int(probs.argmax().item())
        f_opp     = self.vocab[f_opp_idx]
        set_size  = self.conformal.set_size(probs)
        return probs, f_opp, set_size

    def _score_full(
        self,
        tau: ScoredTransition,
        z_t: torch.Tensor,
        style_emb: torch.Tensor,
        round_idx: int,
    ) -> ScoredTransition:
        """Score a candidate end-to-end: opp model → world model → critics."""
        probs, f_opp, set_size = self._query_opponent(tau, z_t, style_emb)
        wm_out  = self.world_model(tau.f_tgt, f_opp, z_t)
        c       = self.critics(tau.f_tgt, f_opp, wm_out, tau)
        return ScoredTransition(
            f_src           = tau.f_src,
            f_tgt           = tau.f_tgt,
            intent          = tau.intent,
            total_score     = c.aggregate,
            c_xg            = c.xg,
            c_stab          = c.stability,
            c_res           = c.resistance,
            f_opp_predicted = f_opp,
            opp_dist        = probs,
            conf_set_size   = set_size,
            revision_rounds = round_idx,
        )

    # ── main interface ────────────────────────────────────────────────────────

    @torch.no_grad()
    def run(
        self,
        initial_candidates: List[ScoredTransition],
        z_t: torch.Tensor,
        style_emb: torch.Tensor,
        ood_monitor: Optional[OODMonitor] = None,
    ) -> RevisionLoopOutput:
        """
        Execute Algorithm 1, lines 9–22.

        Args:
            initial_candidates : top-K ScoredTransitions from the generator,
                                  already scored at round m = 0 by inference.py.
            z_t                : context vector (256,) from state encoder.
            style_emb          : opponent style embedding (32,) from clustering.
            ood_monitor        : optional OODMonitor; if provided, OOD checks
                                  are performed at each round.

        Returns:
            RevisionLoopOutput with all candidates re-ranked.
        """
        if not initial_candidates:
            return RevisionLoopOutput(ranked=[], rounds_run=0,
                                      termination_reason="empty_input")

        candidates = sorted(initial_candidates, key=lambda x: x.total_score,
                            reverse=True)
        termination_reason = "max_rounds"
        last_round = 0

        for m in range(1, self.max_rounds + 1):
            last_round = m
            top3 = candidates[:3]
            revised_map: Dict[Tuple[int, str], ScoredTransition] = {}
            conformal_break = False

            for tau in top3:
                # Step 1: update context with previous world-model predictions
                # The wrapper passed in as `world_model` may hand back possession
                # as 0-100 (the adapter's convention) or as 0-1 (some internal
                # wrappers normalise). Dividing an already-normalised value by
                # 100 would make the possession signal vanish into [0, 0.01].
                # Detect the convention from the value itself: possession
                # shares above 1.5 are on the 0-100 scale.
                wm_prev = self.world_model(tau.f_tgt, tau.f_opp_predicted, z_t)

                poss_raw = float(wm_prev.get("possession", 50.0))
                poss_norm = poss_raw / 100.0 if poss_raw > 1.5 else poss_raw

                ppda_raw = float(wm_prev.get("ppda", 12.0))
                # _ppda_oh in graph_builder buckets at 6, 9, 13; use the same
                # window when normalising for the context updater so both
                # places interpret "low / mid / high pressing" the same way.
                ppda_norm = max(0.0, min(1.0, (ppda_raw - 5.0) / 15.0))

                z_prime = self.context_updater(z_t, poss_norm, ppda_norm)

                # Step 2: re-query opponent model with z'_t
                probs_new, f_opp_new, set_size_new = self._query_opponent(
                    tau, z_prime, style_emb
                )

                # Step 3: conformal termination check (Algorithm 1, line 15)
                if set_size_new > self.theta_conf:
                    logger.info(
                        "Conformal termination at round %d: set_size=%d > θ_conf=%d "
                        "(τ: %d→%d, %s)",
                        m, set_size_new, self.theta_conf,
                        tau.f_src, tau.f_tgt, tau.intent,
                    )
                    conformal_break = True
                    termination_reason = "conformal"
                    break

                # Step 4: optional OOD check
                if ood_monitor is not None:
                    # Approximate per-candidate acc/ece on this single example
                    approx_acc = float(
                        f_opp_new == tau.f_opp_predicted
                    )  # rough; full OOD eval is done externally
                    approx_ece = float(1.0 - probs_new.max().item())
                    if ood_monitor.check(m, approx_acc, approx_ece):
                        termination_reason = "ood"
                        conformal_break = True
                        break

                # Step 5: re-score with updated context and opponent prediction
                wm_new = self.world_model(tau.f_tgt, f_opp_new, z_prime)
                c_new  = self.critics(tau.f_tgt, f_opp_new, wm_new, tau)

                revised_map[(tau.f_tgt, tau.intent)] = ScoredTransition(
                    f_src           = tau.f_src,
                    f_tgt           = tau.f_tgt,
                    intent          = tau.intent,
                    total_score     = c_new.aggregate,
                    c_xg            = c_new.xg,
                    c_stab          = c_new.stability,
                    c_res           = c_new.resistance,
                    f_opp_predicted = f_opp_new,
                    opp_dist        = probs_new,
                    conf_set_size   = set_size_new,
                    revision_rounds = m,
                )

            if conformal_break:
                break

            # Merge revised top-3 back into full candidate list
            candidates = [
                revised_map.get((c.f_tgt, c.intent), c) for c in candidates
            ]
            candidates.sort(key=lambda x: x.total_score, reverse=True)

        return RevisionLoopOutput(
            ranked=candidates,
            rounds_run=last_round,
            termination_reason=termination_reason,
        )


# ── Intent index map ──────────────────────────────────────────────────────────
_INTENT_IDX = {
    "offensive": 0, "defensive": 1, "wide": 2,
    "narrow": 3, "pressing": 4, "conservative": 5,
}


# ── Factory ───────────────────────────────────────────────────────────────────
def build_revision_loop(
    opponent_model:     nn.Module,
    world_model:        Callable,
    critics:            Callable,
    cal_probs_conformal: torch.Tensor,    # (N, |V|) for fitting conformal predictor
    cal_labels_conformal: torch.Tensor,   # (N,)
    cal_probs_theta:    torch.Tensor,     # (N', |V|) for calibrating θ_conf (different split)
    formation_vocab:    List[int],
    context_updater:    Optional[ContextUpdater] = None,
    opponent_ece:       float = 0.0,
    alpha:              float = ALPHA,
    theta_percentile:   float = 0.80,
    max_rounds:         int = MAX_ROUNDS,
) -> Tuple[BoundedRevisionLoop, int]:
    """
    Convenience factory that fits the conformal predictor and calibrates
    θ_conf before constructing the revision loop.

    Args:
        cal_probs_conformal / cal_labels_conformal :
            Held-out split for fitting the conformal predictor(
            conformal_predictor.py should produce this).
        cal_probs_theta :
            A *different* held-out split (OOD val) for calibrating θ_conf.
            Using the same split for both would cause data leakage into θ_conf.

    Returns:
        (loop, theta_conf)
    """
    conformal = ConformalPredictor(alpha=alpha)
    conformal.fit(cal_probs_conformal, cal_labels_conformal)

    theta_conf = calibrate_theta_conf(conformal, cal_probs_theta,
                                      percentile=theta_percentile)

    loop = BoundedRevisionLoop(
        opponent_model  = opponent_model,
        world_model     = world_model,
        critics         = critics,
        conformal       = conformal,
        theta_conf      = theta_conf,
        formation_vocab = formation_vocab,
        context_updater = context_updater,
        opponent_ece    = opponent_ece,
        max_rounds      = max_rounds,
    )
    return loop, theta_conf


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    VOCAB = [4231, 433, 442, 4141, 352, 4321, 4312, 343,
             451, 4411, 532, 541, 3412, 4222, 3421]
    V = len(VOCAB)

    # Synthetic opponent model: returns uniform log-probs
    class _FakeOpp(nn.Module):
        def forward(self, z, tau, style):
            return torch.zeros(V)

    # Synthetic world model and critics
    def _wm(f_tgt, f_opp, z_t):
        return {"xg": 0.1, "possession": 0.55, "ppda": 8.0, "ppda_norm": 0.4, "shots": 1.2}

    def _critics(f_tgt, f_opp, wm_out, tau):
        return CriticScores(xg=wm_out["xg"], stability=0.8, resistance=0.6)

    # Build synthetic calibration data (uniform probabilities → large set sizes)
    N_CAL = 200
    cal_p  = torch.softmax(torch.randn(N_CAL, V), dim=-1)
    cal_l  = torch.randint(0, V, (N_CAL,))
    cal_p2 = torch.softmax(torch.randn(N_CAL, V), dim=-1)

    loop, theta = build_revision_loop(
        opponent_model        = _FakeOpp(),
        world_model           = _wm,
        critics               = _critics,
        cal_probs_conformal   = cal_p,
        cal_labels_conformal  = cal_l,
        cal_probs_theta       = cal_p2,
        formation_vocab       = VOCAB,
        opponent_ece          = 0.10,   # within threshold
        max_rounds            = 2,
    )
    print(f"θ_conf = {theta}")

    # Build dummy initial candidates
    dummy_cands = [
        ScoredTransition(
            f_src=4231, f_tgt=f, intent="offensive",
            total_score=float(i) / 10,
            c_xg=0.1, c_stab=0.8, c_res=0.6,
            f_opp_predicted=442, opp_dist=torch.ones(V) / V,
        )
        for i, f in enumerate(VOCAB[:5])
    ]

    z_t       = torch.zeros(256)
    style_emb = torch.zeros(32)

    out = loop.run(dummy_cands, z_t, style_emb)
    print(f"Rounds run: {out.rounds_run}, Termination: {out.termination_reason}")
    print(f"Top candidate: {out.ranked[0].f_tgt}, score={out.ranked[0].total_score:.3f}")
