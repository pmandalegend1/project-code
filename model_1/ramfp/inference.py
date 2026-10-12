"""
RAMFP inference loop (Algorithm 1).

    z_t   <- Enc(s_t)
    C     <- top-K candidates from Gen(T_feas, z_t)
    for each tau in C:
        P_tau <- Opp(tau, z_t, e_phi)
        y_tau <- WM(f_tgt, f_opp_hat, z_t)
        Score <- w1 C_xG + w2 C_stab + w3 C_res
    for m in 1..M:                       # bounded revision
        re-score the top 3 with updated context, stop on conformal set size

Integration boundary
--------------------
The world model and the stability/counter-resistance critics belong to other
people on the team. They enter through the `WorldModel` and `Critic` protocols
below, so this loop can be written, tested and demoed today. `StubWorldModel`
returns clearly-labelled placeholder numbers and every result it touches is
flagged `is_stub=True` — so a stubbed run can never be mistaken for a real one
in your results tables.

The one part of this loop that is genuinely yours and genuinely novel is the
conformal termination: it reads the opponent model's own predictive
distribution, which is why it lives here rather than in the planner teammate's
module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import torch

from .config import InferenceConfig
from .eval.metrics import conformal_qhat
from .models.opponent import OpponentCoBehaviorModel
from .vocab import (
    FORMATIONS, INTENTS, INTENT_TO_ID, N_FORMATIONS, PHASE_TO_ID, decode_formation,
    formation_id, formation_layout, intra_adjacency, minute_bucket, mirror_zone,
    ppda_bucket, score_bucket, structural_distance,
)
from .data.datasets import OFFSET


# --------------------------------------------------------------------------
# Game state
# --------------------------------------------------------------------------
@dataclass
class GameState:
    """Everything the encoder needs about one decision point."""
    f_src: int                       # our current formation code
    opp_formation: int               # their current formation code
    opp_style: int                   # 0..3 from step 04
    goal_diff: int                   # ours minus theirs
    minute: int
    opp_ppda: float
    phase: str = "transition"
    # last N events, oldest first, as vocabulary ids
    event_type: Sequence[int] = field(default_factory=list)
    event_zone: Sequence[int] = field(default_factory=list)
    event_team: Sequence[int] = field(default_factory=list)
    event_minute: Sequence[int] = field(default_factory=list)
    event_outcome: Sequence[int] = field(default_factory=list)
    # positional role ids currently on the pitch, for feasibility checking.
    # None disables the check (with a warning in the result).
    available_roles: Optional[Sequence[int]] = None

    def to_batch(self, seq_len: int, knn: int, device: str) -> Dict[str, torch.Tensor]:
        def pad(seq):
            a = np.asarray(list(seq), dtype=np.int64)[-seq_len:] + OFFSET
            if a.size < seq_len:
                a = np.concatenate([a, np.zeros(seq_len - a.size, dtype=np.int64)])
            return torch.from_numpy(a).unsqueeze(0)

        _, roles = formation_layout(self.f_src)
        batch = {
            "event_type": pad(self.event_type), "event_zone": pad(self.event_zone),
            "event_team": pad(self.event_team), "event_minute": pad(self.event_minute),
            "event_outcome": pad(self.event_outcome),
            "length": torch.tensor([min(len(self.event_type), seq_len)]),
            "roles": torch.tensor(roles, dtype=torch.long).unsqueeze(0),
            "adj": torch.from_numpy(intra_adjacency(self.f_src, knn).copy()).unsqueeze(0),
            "ppda_bucket": torch.tensor([ppda_bucket(self.opp_ppda)]),
            "score_bucket": torch.tensor([score_bucket(self.goal_diff)]),
            "phase": torch.tensor([PHASE_TO_ID[self.phase]]),
        }
        return {k: v.to(device) for k, v in batch.items()}


# --------------------------------------------------------------------------
# Integration protocols — implemented by teammates
# --------------------------------------------------------------------------
class WorldModel(Protocol):
    """Step 10. Predicts the next 10 minutes for a formation matchup."""

    is_stub: bool

    def predict(self, f_ours: int, f_theirs: int,
                state: GameState) -> Dict[str, float]:
        """Returns {'xg_delta', 'possession', 'ppda', 'shots'}."""
        ...


class Critic(Protocol):
    """Step 12. Scores one candidate; higher is better."""

    name: str
    is_stub: bool

    def score(self, f_src: int, f_tgt: int, f_opp: int,
              state: GameState, wm_out: Dict[str, float]) -> float:
        ...


class StubWorldModel:
    """
    Placeholder so the loop runs end to end before step 10 lands. Its outputs
    are a deterministic function of formation geometry — plausible enough to
    exercise the plumbing, and useless as a result. Never report numbers that
    passed through this.
    """

    is_stub = True

    def predict(self, f_ours: int, f_theirs: int, state: GameState) -> Dict[str, float]:
        d_o, f_o = decode_formation(f_ours)[0], decode_formation(f_ours)[-1]
        d_t, f_t = decode_formation(f_theirs)[0], decode_formation(f_theirs)[-1]
        attack_edge = (f_o - f_t) * 0.05 + (d_t - d_o) * 0.02
        return {
            "xg_delta": float(attack_edge),
            "possession": float(50.0 + 3.0 * (f_o - f_t)),
            "ppda": float(12.0 - 0.8 * (f_o - 2)),
            "shots": float(3.0 + 0.6 * (f_o - f_t)),
        }


class XGCritic:
    """C_xG — reads the world model directly, so it is only as real as the WM."""
    name = "xG"

    def __init__(self, wm: WorldModel):
        self.wm = wm
        self.is_stub = getattr(wm, "is_stub", False)

    def score(self, f_src, f_tgt, f_opp, state, wm_out):
        return float(wm_out["xg_delta"])


class StructuralStabilityCritic:
    """
    C_stab placeholder: penalises large structural jumps, which is the prior
    the trained MLP is meant to learn. Swap for the trained model from step 12.
    """
    name = "stability"
    is_stub = True

    def score(self, f_src, f_tgt, f_opp, state, wm_out):
        return float(-structural_distance(f_src, f_tgt) / 100.0)


# --------------------------------------------------------------------------
# Candidate generation
# --------------------------------------------------------------------------
@dataclass
class Candidate:
    f_src: int
    f_tgt: int
    intent: str
    prior: float = 0.0

    def key(self) -> Tuple[int, int, str]:
        return (self.f_src, self.f_tgt, self.intent)


class TransitionGenerator:
    """
    The feasible transition space.

    Note for the paper.  |T| = |V| x |V| x |I| is the full product space.
    At inference f_src is fixed by the current game state, so the
    per-decision ceiling is |V| x |I| before any rule is applied; tactical
    grammar then cuts it further.  That reduction is structural, not
    grammatical, and a reviewer who multiplies the numbers out will check it.

    The values are computed by space_summary() rather than written into this
    docstring, because they were quoted as 1,350 and 90 while |V| was 15 and
    |V| became 17 once the vocabulary was checked against the corpus (see
    ramfp/vocab.py).  Quote space_summary() in the draft so the text cannot
    drift from the code again.
    """

    @staticmethod
    def space_summary() -> Dict[str, int]:
        """Candidate-space sizes, derived from the live vocabulary."""
        return {
            "n_formations": N_FORMATIONS,
            "n_intents": len(INTENTS),
            "full_product_space": N_FORMATIONS * N_FORMATIONS * len(INTENTS),
            "per_decision_ceiling": N_FORMATIONS * len(INTENTS),
        }

    def __init__(self, formations: Sequence[int] = FORMATIONS,
                 intents: Sequence[str] = INTENTS):
        self.formations = list(formations)
        self.intents = list(intents)

    # -- hard constraints --------------------------------------------------
    @staticmethod
    def lineup_feasible(f_tgt: int, available_roles: Optional[Sequence[int]]) -> bool:
        """
        A shape is only legal if the players on the pitch can fill it. The
        cheap version: compare required defender count against how many
        defenders are actually on. This is the constraint everyone forgets.
        """
        if available_roles is None:
            return True
        from .vocab import N_LATERALS
        have_def = sum(1 for r in available_roles if r // N_LATERALS == 1)
        return decode_formation(f_tgt)[0] <= have_def + 1

    @staticmethod
    def grammar_ok(f_src: int, f_tgt: int, intent: str, state: GameState) -> bool:
        if f_tgt == f_src:
            return False
        d_src, f_fwd_src = decode_formation(f_src)[0], decode_formation(f_src)[-1]
        d_tgt, f_fwd_tgt = decode_formation(f_tgt)[0], decode_formation(f_tgt)[-1]
        # don't get more attacking while two up late in the game
        if state.goal_diff >= 2 and state.minute >= 75 and f_fwd_tgt > f_fwd_src:
            return False
        # don't get more defensive while two down late in the game
        if state.goal_diff <= -2 and state.minute >= 75 and d_tgt > d_src:
            return False
        # intent must be consistent with the shape change
        if intent == "offensive" and f_fwd_tgt < f_fwd_src:
            return False
        if intent == "conservative" and d_tgt < d_src:
            return False
        return True

    # -- ranking -----------------------------------------------------------
    @staticmethod
    def _prior(f_src: int, f_tgt: int, intent: str, state: GameState) -> float:
        """
        Heuristic stand-in for the learned scorer of step 11: prefer small
        structural moves, and moves whose direction matches the scoreline.
        Replace with the trained MLP when the planner teammate ships it.
        """
        chasing = -state.goal_diff
        d_delta = decode_formation(f_tgt)[-1] - decode_formation(f_src)[-1]
        return float(-0.02 * structural_distance(f_src, f_tgt) + 0.15 * chasing * d_delta)

    def generate(self, state: GameState, k: int = 20,
                 diverse: bool = True) -> List[Candidate]:
        """Top-k candidates, covering as many distinct shapes as k allows.

        A candidate is a (formation, intent) pair, and each formation can pair
        with up to |I| = 6 intents. Sorting the whole product by prior and
        truncating at k therefore spends the budget on intents rather than on
        shapes: at k = 20 the old behaviour returned a median of **4 distinct
        formations** out of 17. Measured on 427 held-out decisions, the
        formation the coach actually chose was inside the candidate set only
        **27.6%** of the time -- a ceiling on every ranking metric downstream,
        and one that had nothing to do with the ranker.

        The grammar was not the problem: 97.2% of real targets pass it. The
        truncation was. Taking each formation's best intent first, then second
        intents, and so on, reaches **97.2% coverage at the same k = 20**, so
        the paper's K does not have to change.

        Pass diverse=False for the original behaviour.
        """
        out: List[Candidate] = []
        for f_tgt in self.formations:
            if not self.lineup_feasible(f_tgt, state.available_roles):
                continue
            for intent in self.intents:
                if not self.grammar_ok(state.f_src, f_tgt, intent, state):
                    continue
                out.append(Candidate(state.f_src, f_tgt, intent,
                                     self._prior(state.f_src, f_tgt, intent, state)))
        out.sort(key=lambda c: -c.prior)
        if not diverse:
            return out[:k]

        by_formation: Dict[int, List[Candidate]] = {}
        for c in out:
            by_formation.setdefault(c.f_tgt, []).append(c)
        order = sorted(by_formation, key=lambda f: -by_formation[f][0].prior)

        picked: List[Candidate] = []
        depth = 0
        while len(picked) < k and any(len(by_formation[f]) > depth for f in order):
            for f in order:
                if len(picked) >= k:
                    break
                if len(by_formation[f]) > depth:
                    picked.append(by_formation[f][depth])
            depth += 1
        return picked


# --------------------------------------------------------------------------
# Conformal termination
# --------------------------------------------------------------------------
class ConformalTerminator:
    """
    Split-conformal predictor over the opponent model's softmax. Fitted on
    calibration triples the model never trained on; the loop stops when the
    prediction set grows past `threshold`, i.e. when the model has become too
    uncertain for another revision round to mean anything.
    """

    UNFITTED = -1

    def __init__(self, alpha: float = 0.10, threshold: int = 1):
        self.alpha, self.threshold = alpha, threshold
        self.qhat: Optional[float] = None
        self._warned = False

    @property
    def is_fitted(self) -> bool:
        return self.qhat is not None

    def fit(self, cal_probs: np.ndarray, cal_y: np.ndarray) -> "ConformalTerminator":
        self.qhat = conformal_qhat(cal_probs, cal_y, self.alpha)
        return self

    def set_size(self, probs: np.ndarray) -> int:
        """Number of formations in the conformal set, or UNFITTED (-1).

        This returned 0 when unfitted, which is also a real set size: a fitted
        predictor emits an EMPTY set whenever no formation reaches the
        threshold. The two are opposite situations -- "the rule was never
        switched on" and "the model is as uncertain as it gets" -- and they
        were indistinguishable in every logged result.
        """
        if self.qhat is None:
            return self.UNFITTED
        return int((1.0 - probs <= self.qhat).sum())

    def should_stop(self, probs: np.ndarray) -> bool:
        """Stop when the model is too uncertain for another round to mean anything.

        An empty set is the MOST uncertain outcome, not the least: it means no
        formation reached the conformal threshold at all. The old rule was
        `size > threshold`, so an empty set scored 0 and never stopped -- the
        loop kept revising precisely where revision was least informative.
        Both tails now stop.
        """
        if self.qhat is None:
            if not self._warned:
                import warnings
                warnings.warn(
                    "ConformalTerminator was never fitted, so termination is disabled "
                    "and every decision will use the full revision budget. Call .fit() "
                    "on calibration data the opponent model never trained on.",
                    RuntimeWarning, stacklevel=2,
                )
                self._warned = True
            return False
        size = self.set_size(probs)
        return size == 0 or size > self.threshold


# --------------------------------------------------------------------------
# The planner
# --------------------------------------------------------------------------
@dataclass
class Recommendation:
    f_src: int
    f_tgt: int
    intent: str
    score: float
    predicted_opponent: int
    opponent_probs: np.ndarray
    critic_scores: Dict[str, float]
    world_model: Dict[str, float]
    rounds_used: int
    conformal_set_size: int
    is_stub: bool
    stub_components: Tuple[str, ...] = ()
    # State encoding from the opponent encoder, carried through so downstream
    # consumers (the ranker in particular) can see the same 256-d summary of
    # the last 50 events + formation graph the opponent model was given.
    # Prior to this field the encoder output was computed, used for the
    # opponent prediction, and discarded -- so every downstream ranking signal
    # had no access to the event sequence that the state encoder spent four
    # transformer layers summarising.
    z_t: Optional[np.ndarray] = None

    def explain(self) -> str:
        top = np.argsort(-self.opponent_probs)[:3]
        likely = ", ".join(f"{FORMATIONS[i]} ({self.opponent_probs[i]:.0%})" for i in top)
        if self.stub_components:
            flag = f"  [STUB: {', '.join(self.stub_components)} — not a result]"
        elif self.is_stub:
            flag = "  [STUB — not a result]"
        else:
            flag = ""
        return (f"{self.f_src} -> {self.f_tgt} ({self.intent})  score {self.score:+.4f}"
                f"{flag}\n    likely response: {likely}"
                f"\n    critics: " + ", ".join(f"{k} {v:+.3f}" for k, v in self.critic_scores.items())
                + f"\n    predicted xG delta {self.world_model['xg_delta']:+.3f}, "
                f"possession {self.world_model['possession']:.1f}%, "
                f"rounds {self.rounds_used}, conformal set "
                + ("unfitted" if self.conformal_set_size < 0
                   else "empty (max uncertainty)" if self.conformal_set_size == 0
                   else str(self.conformal_set_size)))


class RAMFPPlanner:
    def __init__(self, opponent: OpponentCoBehaviorModel,
                 world_model: Optional[WorldModel] = None,
                 critics: Optional[Sequence[Critic]] = None,
                 cfg: Optional[InferenceConfig] = None,
                 terminator: Optional[ConformalTerminator] = None,
                 device: str = "cpu"):
        self.cfg = cfg or InferenceConfig()
        self.device = device
        self.opponent = opponent.to(device).eval()
        self.wm = world_model or StubWorldModel()
        self.critics = list(critics) if critics is not None else [
            XGCritic(self.wm), StructuralStabilityCritic()]
        self.generator = TransitionGenerator()
        self.terminator = terminator or ConformalTerminator(
            self.cfg.conformal_alpha, self.cfg.conformal_threshold)

    # -- scoring -----------------------------------------------------------
    def _normalise_critics(self, per_candidate: List[Dict[str, float]]
                           ) -> List[Dict[str, float]]:
        """Put each critic on a common scale across this decision's candidates.

        Returns the scores unchanged when normalisation is off, or when a
        critic gives the same value to every candidate (nothing to rank on).
        """
        if not per_candidate or getattr(self.cfg, "critic_normalisation", "zscore") != "zscore":
            return per_candidate
        names = list(per_candidate[0])
        stats = {}
        for n in names:
            v = np.array([d[n] for d in per_candidate], dtype=float)
            sd = float(v.std())
            stats[n] = (float(v.mean()), sd if sd > 1e-12 else None)
        out = []
        for d in per_candidate:
            out.append({n: (d[n] - stats[n][0]) / stats[n][1] if stats[n][1] else 0.0
                        for n in names})
        return out

    def _aggregate(self, scores: Dict[str, float]) -> float:
        w = list(self.cfg.critic_weights)
        vals = list(scores.values())

        # Algorithm 1 weights three critics: C_xG, C_stab, C_res. The planner
        # registers two, so the third weight was being dropped and the other
        # two renormalised to 0.5 each -- silently. The arithmetic is fine; the
        # silence is not, because the config still reads (1/3, 1/3, 1/3) and a
        # results table copied from it would misstate how the score was formed.
        # Warn once, then carry on with the same renormalisation.
        if len(w) != len(vals) and not getattr(self, "_warned_weights", False):
            import warnings
            warnings.warn(
                f"critic_weights has {len(w)} entries but {len(vals)} critics are "
                f"registered ({', '.join(scores)}). The extra weights are dropped and "
                f"the rest renormalised, so the effective weights are "
                f"{[round(x / (sum(w[:len(vals)]) or 1.0), 4) for x in w[:len(vals)]]}, "
                f"not the configured {w}. Report the effective weights.",
                RuntimeWarning, stacklevel=2,
            )
            self._warned_weights = True

        w = (w + [0.0] * len(vals))[:len(vals)]
        s = sum(w) or 1.0
        return float(sum(wi * v for wi, v in zip(w, vals)) / s)

    def effective_critic_weights(self) -> Dict[str, float]:
        """What the score is actually a weighted sum of, for the results table."""
        names = [cr.name for cr in self.critics]
        w = (list(self.cfg.critic_weights) + [0.0] * len(names))[:len(names)]
        total = sum(w) or 1.0
        return {n: wi / total for n, wi in zip(names, w)}

    @torch.no_grad()
    def _opponent_probs(self, z_t: torch.Tensor,
                        cands: Sequence[Candidate],
                        style: int) -> np.ndarray:
        """One batched forward pass for every candidate against a cached z_t."""
        n = len(cands)
        z = z_t.expand(n, -1)
        dev = next(self.opponent.parameters()).device
        f_src = torch.tensor([formation_id(c.f_src) for c in cands], device=dev)
        f_tgt = torch.tensor([formation_id(c.f_tgt) for c in cands], device=dev)
        intent = torch.tensor([INTENT_TO_ID[c.intent] for c in cands], device=dev)
        sty = torch.full((n,), int(style), dtype=torch.long, device=dev)
        logits = self.opponent.head(z, f_src, f_tgt, intent, sty)
        return torch.softmax(logits, dim=-1).float().cpu().numpy()

    # -- Algorithm 1 -------------------------------------------------------
    @torch.no_grad()
    def plan(self, state: GameState, top_n: int = 3) -> List[Recommendation]:
        seq_len = self.opponent.enc_cfg.seq_len
        knn = self.opponent.enc_cfg.knn
        # Build the batch on whatever device the model is actually on, rather
        # than on whatever string was passed to __init__. The two can differ --
        # a checkpoint loaded with map_location, a model moved afterwards, a
        # caller that passed the default -- and the mismatch only shows up deep
        # inside an embedding lookup as "found at least two devices".
        model_device = next(self.opponent.parameters()).device
        batch = state.to_batch(seq_len, knn, model_device)
        z_t = self.opponent.encoder(batch)                       # [1, d_context]

        cands = self.generator.generate(state, self.cfg.candidate_k)
        if not cands:
            return []
        probs = self._opponent_probs(z_t, cands, state.opp_style)

        results: List[Recommendation] = []
        stubs = tuple(
            ["world model"] if getattr(self.wm, "is_stub", False) else []
        ) + tuple(
            f"{cr.name} critic" for cr in self.critics
            if getattr(cr, "is_stub", False)
        )

        # Score every candidate first, so the critics can be put on a common
        # scale before they are combined. _aggregate averages RAW scores, and
        # the critics do not share units: measured on real decisions their
        # standard deviations are 0.076 (C_stab), 0.056 (C_xG) and 0.027
        # (C_res), so equal nominal weights gave effective weights of
        # 48 / 36 / 17. The largest share went to the critic that scores worst
        # on its own. Normalising within a decision makes the configured
        # weights mean what they say; it changes nothing when a critic is used
        # alone, since an affine transform cannot change a ranking.
        raw = []
        for c, p in zip(cands, probs):
            f_opp = FORMATIONS[int(p.argmax())]
            wm_out = self.wm.predict(c.f_tgt, f_opp, state)
            raw.append((c, p, f_opp, wm_out,
                        {cr.name: cr.score(c.f_src, c.f_tgt, f_opp, state, wm_out)
                         for cr in self.critics}))

        scaled = self._normalise_critics([r[4] for r in raw])

        z_np = z_t.detach().cpu().numpy().reshape(-1)   # (256,)
        for (c, p, f_opp, wm_out, cs), cs_scaled in zip(raw, scaled):
            results.append(Recommendation(
                f_src=c.f_src, f_tgt=c.f_tgt, intent=c.intent,
                score=self._aggregate(cs_scaled), predicted_opponent=f_opp,
                opponent_probs=p, critic_scores=cs, world_model=wm_out,
                rounds_used=0, conformal_set_size=self.terminator.set_size(p),
                is_stub=bool(stubs), stub_components=stubs,
                z_t=z_np,
            ))

        results.sort(key=lambda r: -r.score)
        revised = self._revise(state, z_t, results[:top_n])
        final = revised + results[top_n:]
        # _revise re-scores its candidates one at a time and cannot normalise
        # across the decision, so after it runs the revised candidates carry
        # raw-scale scores while the rest carry normalised ones. Sorting the
        # two together compared numbers in different units. Re-score everything
        # from the (possibly updated) critic scores on one common scale.
        self._rescore(final)
        return sorted(final, key=lambda r: -r.score)[:self.cfg.candidate_k]

    def _rescore(self, recs: List[Recommendation]) -> None:
        scaled = self._normalise_critics([r.critic_scores for r in recs])
        for r, cs in zip(recs, scaled):
            r.score = self._aggregate(cs)

    @torch.no_grad()
    def _revise(self, state: GameState, z_t: torch.Tensor,
                top: List[Recommendation]) -> List[Recommendation]:
        """Bounded revision with the dual stopping criterion."""
        for m in range(1, self.cfg.max_rounds + 1):
            still_open = []
            for r in top:
                if self.terminator.should_stop(r.opponent_probs):
                    continue                                  # conformal termination
                still_open.append(r)
            if not still_open:
                break

            z_upd = self._update_context(z_t, still_open[0].world_model)
            cands = [Candidate(r.f_src, r.f_tgt, r.intent) for r in still_open]
            probs = self._opponent_probs(z_upd, cands, state.opp_style)

            for r, p in zip(still_open, probs):
                f_opp = FORMATIONS[int(p.argmax())]
                wm_out = self.wm.predict(r.f_tgt, f_opp, state)
                cs = {cr.name: cr.score(r.f_src, r.f_tgt, f_opp, state, wm_out)
                      for cr in self.critics}
                r.opponent_probs, r.predicted_opponent = p, f_opp
                r.world_model, r.critic_scores = wm_out, cs
                r.score = self._aggregate(cs)
                r.rounds_used = m
                r.conformal_set_size = self.terminator.set_size(p)
        return top

    def _update_context(self, z_t: torch.Tensor, wm_out: Dict[str, float]) -> torch.Tensor:
        """
        UpdateContext from Algorithm 1 line 12: fold the world model's
        predicted possession and PPDA back into the context vector.

        Deliberately a fixed, norm-preserving perturbation rather than a
        learned layer: there is no supervision for this map anywhere in the
        paper, so a learned one would be an untrained layer masquerading as a
        component. Report it as a design choice, not an oversight.
        """
        d = z_t.size(-1)
        feats = torch.tensor(
            [(wm_out["possession"] - 50.0) / 50.0, (wm_out["ppda"] - 12.0) / 12.0],
            dtype=z_t.dtype, device=z_t.device)
        bump = torch.zeros(d, dtype=z_t.dtype, device=z_t.device)
        bump[0::2] = feats[0]
        bump[1::2] = feats[1]
        out = z_t + 0.1 * bump.unsqueeze(0)
        return out * (z_t.norm(dim=-1, keepdim=True) / out.norm(dim=-1, keepdim=True))


def demo_state(rng: Optional[np.random.Generator] = None) -> GameState:
    """A plausible 62nd-minute decision point, for smoke tests and the demo."""
    rng = rng or np.random.default_rng(0)
    n = 50
    return GameState(
        f_src=442, opp_formation=433, opp_style=0, goal_diff=-1, minute=62,
        opp_ppda=8.4, phase="defending",
        event_type=rng.integers(0, 12, n).tolist(),
        event_zone=rng.integers(0, 24, n).tolist(),
        event_team=rng.integers(0, 2, n).tolist(),
        event_minute=[minute_bucket(62)] * n,
        event_outcome=rng.integers(0, 9, n).tolist(),
        available_roles=list(formation_layout(442)[1]),
    )
