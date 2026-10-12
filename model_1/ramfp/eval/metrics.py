"""Metrics for E1: Top-k accuracy, Expected Calibration Error, NLL, macro-F1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np


@dataclass
class E1Metrics:
    top1: float
    top3: float
    ece: float
    nll: float
    macro_f1: float
    mean_entropy: float
    n: int

    def as_row(self) -> Dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (f"Top-1 {self.top1:.4f} | Top-3 {self.top3:.4f} | ECE {self.ece:.4f} "
                f"| NLL {self.nll:.4f} | macroF1 {self.macro_f1:.4f} | n={self.n}")


def top_k_accuracy(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    if k >= probs.shape[1]:
        return 1.0
    topk = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    return float((topk == y[:, None]).any(axis=1).mean())


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """
    Standard confidence-based ECE (Guo et al. 2017): bin predictions by the
    top-1 probability and average |accuracy - confidence| weighted by bin size.
    """
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict[str, np.ndarray]:
    """Per-bin confidence/accuracy/count, for the calibration figure."""
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    c, a, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        c.append(conf[m].mean() if m.sum() else np.nan)
        a.append(correct[m].mean() if m.sum() else np.nan)
        n.append(int(m.sum()))
    return {"edges": edges, "confidence": np.array(c), "accuracy": np.array(a),
            "count": np.array(n)}


def macro_f1(probs: np.ndarray, y: np.ndarray) -> float:
    pred = probs.argmax(axis=1)
    classes = np.arange(probs.shape[1])
    f1s = []
    for c in classes:
        tp = float(((pred == c) & (y == c)).sum())
        fp = float(((pred == c) & (y != c)).sum())
        fn = float(((pred != c) & (y == c)).sum())
        if tp + fn == 0:          # class absent from this split
            continue
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> E1Metrics:
    eps = 1e-12
    p = np.clip(probs, eps, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return E1Metrics(
        top1=top_k_accuracy(p, y, 1),
        top3=top_k_accuracy(p, y, 3),
        ece=expected_calibration_error(p, y, n_bins),
        nll=float(-np.log(p[np.arange(len(y)), y]).mean()),
        macro_f1=macro_f1(p, y),
        mean_entropy=float((-p * np.log(p)).sum(axis=1).mean()),
        n=int(len(y)),
    )


# --------------------------------------------------------------------------
# The rare-event question: does the opponent change shape at all?
# --------------------------------------------------------------------------
# On 2651 StatsBomb matches the opponent's formation moves in 235 of 2138
# decisions -- 11.0%.  "Predict no change" therefore scores 89.0%, so plain
# accuracy cannot separate a useful model from a constant.  These metrics
# are the ones that can: AUROC and average precision are threshold-free,
# and recall on the positives is the quantity the planner actually needs,
# because a response the model misses is a response the loop never plans
# around.
# --------------------------------------------------------------------------
@dataclass
class ResponseMetrics:
    base_rate: float              # share of decisions where the shape moved
    persistence_accuracy: float   # accuracy of the "nothing changes" rule
    accuracy: float
    recall: float                 # of real changes, how many were caught
    precision: float
    f1: float
    balanced_accuracy: float
    auroc: float                  # threshold-free; 0.5 = no better than chance
    average_precision: float      # threshold-free; compare against base_rate
    brier: float
    threshold: float
    n: int
    n_positive: int

    def as_row(self) -> Dict[str, float]:
        return asdict(self)

    def __str__(self) -> str:
        return (f"AUROC {self.auroc:.4f} | AP {self.average_precision:.4f} "
                f"(base {self.base_rate:.4f}) | recall {self.recall:.4f} "
                f"| precision {self.precision:.4f} | acc {self.accuracy:.4f} "
                f"vs persistence {self.persistence_accuracy:.4f} | n={self.n}")


def evaluate_response(p_change: np.ndarray, changed: np.ndarray,
                      threshold: float = 0.5) -> ResponseMetrics:
    """Score the respond head.

    `threshold` should be picked on validation, not here: at an 11% base
    rate 0.5 is a poor operating point and will look like the model predicts
    nothing.  AUROC and average precision do not depend on it, which is why
    they are the numbers to lead with.
    """
    p = np.asarray(p_change, dtype=np.float64).ravel()
    y = np.asarray(changed).astype(int).ravel()
    pred = (p >= threshold).astype(int)

    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    tn = float(((pred == 0) & (y == 0)).sum())

    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    both_classes = 0 < y.sum() < len(y)
    if both_classes:
        from sklearn.metrics import average_precision_score, roc_auc_score
        auroc = float(roc_auc_score(y, p))
        ap = float(average_precision_score(y, p))
    else:
        # A split with one class present has no ROC; report NaN rather than
        # a number that looks like a score.
        auroc = float("nan")
        ap = float("nan")

    return ResponseMetrics(
        base_rate=float(y.mean()),
        persistence_accuracy=float((y == 0).mean()),
        accuracy=float((pred == y).mean()),
        recall=recall, precision=precision, f1=f1,
        balanced_accuracy=float((recall + specificity) / 2.0),
        auroc=auroc, average_precision=ap,
        brier=float(((p - y) ** 2).mean()),
        threshold=float(threshold),
        n=int(len(y)), n_positive=int(y.sum()),
    )


def evaluate_conditional(probs: np.ndarray, y: np.ndarray,
                         changed: np.ndarray, n_bins: int = 10) -> E1Metrics:
    """Formation metrics on the decisions where the shape actually moved.

    This is the only subset on which "which formation will they switch to"
    is a real question.  Scored over every decision instead, the formation
    head is graded mostly on rows whose answer is "the one they are already
    in", and any model that learns to copy that input scores well without
    having learned anything about responses.
    """
    m = np.asarray(changed).astype(bool).ravel()
    if not m.any():
        return E1Metrics(top1=float("nan"), top3=float("nan"), ece=float("nan"),
                         nll=float("nan"), macro_f1=float("nan"),
                         mean_entropy=float("nan"), n=0)
    return evaluate(np.asarray(probs)[m], np.asarray(y)[m], n_bins)


# --------------------------------------------------------------------------
# Split conformal prediction — used by the planner's termination criterion
# --------------------------------------------------------------------------
def conformal_qhat(cal_probs: np.ndarray, cal_y: np.ndarray, alpha: float = 0.10) -> float:
    """
    Fit a split-conformal threshold on held-out calibration data using the
    standard 1 - p_true nonconformity score, with the finite-sample corrected
    quantile level. The calibration set MUST be examples the model never
    trained on, otherwise coverage is meaningless.
    """
    n = len(cal_y)
    scores = 1.0 - cal_probs[np.arange(n), cal_y]
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def conformal_set_sizes(probs: np.ndarray, qhat: float) -> np.ndarray:
    """Number of classes that cannot be excluded at the 1 - alpha level."""
    return (1.0 - probs <= qhat).sum(axis=1)


def conformal_coverage(probs: np.ndarray, y: np.ndarray, qhat: float) -> float:
    return float((1.0 - probs[np.arange(len(y)), y] <= qhat).mean())
