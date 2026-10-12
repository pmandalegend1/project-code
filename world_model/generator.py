"""
Formation Transition Generator
================================
Enumerates the feasible subset of T = V x V x I for a given game state
and returns the top-K=20 candidates scored by a lightweight MLP.

Full space: |T| = |V|^2 * |I| = 15*15*6 = 1350  (all (f_src, f_tgt, intent) triples)
Per-inference space (fixed f_src): |V| * |I| = 15*6 = 90
After grammar filter: ~50-100
After top-K: 20
"""

import yaml
import torch
import torch.nn as nn
import numpy as np
from collections import namedtuple
from pathlib import Path
from typing import List, Optional

# ── Types ──────────────────────────────────────────────────────────────────────
Transition = namedtuple("Transition", ["f_src", "f_tgt", "intent", "score"])

# ── Intent vocabulary (Section 5.4) ────────────────────────────────────────────
INTENTS = ["offensive", "defensive", "wide", "narrow", "pressing", "conservative"]
INTENT_TO_IDX = {v: i for i, v in enumerate(INTENTS)}


# ── Prior (B5) ─────────────────────────────────────────────────────────────────
# Used when no trained scorer checkpoint exists. See build_generator().
# score(τ, st) = -0.02 * Δstruct(f_src, f_tgt)
#              + 0.15 * goal_diff * attackers_change(f_src, f_tgt)
#
# Deterministic, non-learnable. Strictly more informative than a
# randomly-initialised MLP, which is what the code was silently using
# whenever build_generator() was called without a checkpoint.

def _attacker_count(formation_code: int) -> int:
    s = str(int(formation_code))
    try:
        if len(s) == 3:
            return int(s[-1])
        if len(s) == 4:
            return int(s[-2]) + int(s[-1])
        return int(s[-1])
    except ValueError:
        return 2


def _structural_distance(a_code: int, b_code: int) -> float:
    """L2 distance between (defenders, midfielders, attackers) vectors.
    Mirrors critics.structural_distance; defined locally so this module has
    no import-order dependency on critics.py."""
    def _vec(code):
        s = str(int(code))
        try:
            if len(s) == 3:
                return np.array([int(s[0]), int(s[1]), int(s[2])], dtype=np.float32)
            if len(s) == 4:
                return np.array([int(s[0]), int(s[1]) + int(s[2]), int(s[3])],
                                dtype=np.float32)
        except ValueError:
            pass
        return np.array([4.0, 4.0, 2.0], dtype=np.float32)
    return float(np.linalg.norm(_vec(a_code) - _vec(b_code)))


def _prior_score(f_src_code: int, f_tgt_code: int, goal_diff: int) -> float:
    d_struct   = _structural_distance(f_src_code, f_tgt_code)
    att_change = _attacker_count(f_tgt_code) - _attacker_count(f_src_code)
    return -0.02 * d_struct - 0.15 * goal_diff * att_change

# ── Embedding dims (Table 9) ───────────────────────────────────────────────────
FORMATION_EMB_DIM = 32   # emb(f_src), emb(f_tgt) each ∈ R^32
INTENT_EMB_DIM    = 16   # emb(ι)      ∈ R^16
CONTEXT_DIM       = 256  # z_t         ∈ R^256
MLP_HIDDEN        = 128
TOP_K             = 20


# ── Grammar-filter helper ───────────────────────────────────────────────────────
class TacticalGrammarFilter:
    """
    Encodes at least 8 football-tactics rules that mark certain (f_src, f_tgt, intent)
    triples as INFEASIBLE for a given game state.

    Rules are also stored in grammar_rules.yaml for coaching-staff transparency.
    Each rule returns True  → FEASIBLE (passes)
                         False → INFEASIBLE (filtered out)

    All rule rationale comments reference standard football tactics literature
    (e.g., Forcher et al. 2022; Bauer et al. 2023).
    """

    def __init__(self, rules_path: str = "models/transition_generator/grammar_rules.yaml"):
        with open(rules_path) as f:
            self.rules_cfg = yaml.safe_load(f)

    def is_feasible(
        self,
        f_src: int,
        f_tgt: int,
        intent: str,
        scoreline: tuple,      # (home_score, away_score) from team-a perspective
        minute: int,
        period: int,
        team_is_home: bool,
    ) -> bool:
        """Returns False if any grammar rule marks this transition infeasible."""
        score_diff = (scoreline[0] - scoreline[1]) if team_is_home else (scoreline[1] - scoreline[0])
        # score_diff > 0 → winning

        # Rule 1: No offensive intent when winning ≥2 and minute ≥75.
        # Rationale: protecting a lead; offensive exposure increases counter-attack risk.
        if intent == "offensive" and score_diff >= 2 and minute >= 75:
            return False

        # Rule 2: No conservative intent when losing ≥2 and minute < 60.
        # Rationale: dropping deep while 2+ goals down too early is tactically untenable.
        if intent == "conservative" and score_diff <= -2 and minute < 60:
            return False

        # Rule 3: No pressing intent when losing ≥2 and minute ≥80.
        # Rationale: high-press at this juncture drains energy without realistic return.
        if intent == "pressing" and score_diff <= -2 and minute >= 80:
            return False

        # Rule 4: No defensive intent when losing ≥2 and minute ≥70.
        # Rationale: sitting deep when 2+ goals behind with little time is suicidal.
        if intent == "defensive" and score_diff <= -2 and minute >= 70:
            return False

        # Rule 5: No transition to the SAME formation (no-op transition).
        # Rationale: a Tactical Shift to the current formation is meaningless.
        if f_src == f_tgt:
            return False

        # Rule 6: No narrow intent when already in a 3-at-the-back system (f_src starts with 3).
        # Rationale: a 3-back setup is already compact; narrow intent double-compresses
        #            and removes width needed for wing-back overlaps.
        if intent == "narrow" and str(f_src).startswith("3"):
            return False

        # Rule 7: No offensive intent in the first 3 minutes of period 2
        # (minute 45-48), allowing shape to settle after half-time.
        # Rationale: players are still repositioning; immediate offensive intent risks
        #            structural disorganisation at the back.
        if intent == "offensive" and period == 2 and minute <= 48:
            return False

        # Rule 8: No pressing intent when winning ≥2 goals.
        # Rationale: unnecessary energy expenditure when managing a lead.
        if intent == "pressing" and score_diff >= 2:
            return False

        # Rule 9: No wide intent in extra time (minute > 90).
        # Rationale: width is a long-term positional investment; extra time demands
        #            central compactness and quick transitions.
        if intent == "wide" and minute > 90:
            return False

        # Rule 10: Conservative must not be combined with a switch to an attacking 3-forward system.
        # Detect "3 forwards" heuristic: last digit of formation code == 3 and it's not 3-digit (e.g. 433).
        f_tgt_str = str(f_tgt)
        has_three_forwards = (f_tgt_str[-1] == "3")
        if intent == "conservative" and has_three_forwards:
            return False

        return True


# ── MLP Scorer  (Equation 15) ──────────────────────────────────────────────────
class TransitionScorerMLP(nn.Module):
    """
    s(τ) = MLP([z_t; emb(f_src); emb(f_tgt); emb(ι)])
    Input dim: 256 + 32 + 32 + 16 = 336
    Architecture: Linear(336, 128) -> GELU -> Linear(128, 1)
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.formation_emb = nn.Embedding(vocab_size, FORMATION_EMB_DIM)
        self.intent_emb    = nn.Embedding(len(INTENTS), INTENT_EMB_DIM)

        in_dim = CONTEXT_DIM + FORMATION_EMB_DIM + FORMATION_EMB_DIM + INTENT_EMB_DIM
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, MLP_HIDDEN),
            nn.GELU(),
            nn.Linear(MLP_HIDDEN, 1),
        )

    def forward(
        self,
        z_t: torch.Tensor,              # (B, 256) or (256,)
        f_src_indices: torch.Tensor,    # (B,) or scalar
        f_tgt_indices: torch.Tensor,    # (B,)
        intent_indices: torch.Tensor,   # (B,)
    ) -> torch.Tensor:
        e_src    = self.formation_emb(f_src_indices)   # (B, 32)
        e_tgt    = self.formation_emb(f_tgt_indices)   # (B, 32)
        e_intent = self.intent_emb(intent_indices)     # (B, 16)

        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0).expand(e_src.shape[0], -1)

        x = torch.cat([z_t, e_src, e_tgt, e_intent], dim=-1)  # (B, 336)
        return self.mlp(x).squeeze(-1)                          # (B,)


# ── Formation Vocabulary Wrapper ───────────────────────────────────────────────
class FormationVocab:
    """
    Thin wrapper around formation_vocab.json providing consistent
    code <-> index mapping across all modules.
    Import from pipeline/features/vocab.py in production; this is a
    local fallback for unit-testing without the full data pipeline.
    """

    def __init__(self, vocab_path: Optional[str] = None):
        if vocab_path and Path(vocab_path).exists():
            import json
            with open(vocab_path) as f:
                data = json.load(f)
            self.vocab: List[int] = data["vocab"]
        else:
            # Fallback placeholder; replace with the actual vocab after the
            # data pipeline runs. No hard length assert: the real vocab is
            # 15-22 formations, and the old `assert len == 15` fired on the
            # current data (B6).
            self.vocab = [4231, 433, 442, 4141, 352, 4321, 4312, 343,
                          451, 4411, 532, 541, 3412, 4222, 3421]

        self._code_to_idx = {code: i for i, code in enumerate(self.vocab)}
        self._idx_to_code = {i: code for i, code in enumerate(self.vocab)}

    def code_to_idx(self, code: int) -> int:
        return self._code_to_idx[code]

    def idx_to_code(self, idx: int) -> int:
        return self._idx_to_code[idx]

    def __len__(self) -> int:
        return len(self.vocab)


# ── Main Generator ─────────────────────────────────────────────────────────────
class FormationTransitionGenerator:
    """
    Generates the top-K feasible formation transition candidates for a given
    game state context vector z_t and source formation f_src.

    Usage:
        gen = FormationTransitionGenerator(scorer, vocab, grammar)
        candidates = gen.generate_candidates(z_t, f_src_code=4231,
                                             scoreline=(1,0), minute=65,
                                             period=2, team_is_home=True)
    """

    def __init__(
        self,
        scorer: Optional[TransitionScorerMLP],   # None -> use the deterministic prior
        vocab: FormationVocab,
        grammar: TacticalGrammarFilter,
        top_k: int = TOP_K,
    ):
        self.scorer  = scorer
        self.vocab   = vocab
        self.grammar = grammar
        self.top_k   = top_k

    @torch.no_grad()
    def _diverse_top_k(self, feasible_f_tgt_code, scores, k):
        """Take the best intent per target formation first, then the second-
        best, and so on, until k are taken.

        The deterministic prior is intent-independent, so a straight top-k
        returns six intents of the single highest-prior formation and starves
        the target-formations that could actually have been observed. Grouping
        by target and interleaving by intent rank gives up to k distinct
        formations inside the same budget. The paper's coverage table calls
        this 'diversity-aware' selection: k = 20 reaches 16/17 distinct
        formations versus 4/17 for a straight top-k.
        """
        import collections
        by_target = collections.OrderedDict()
        for i, code in enumerate(feasible_f_tgt_code):
            by_target.setdefault(code, []).append(i)

        for code in by_target:
            by_target[code].sort(key=lambda i: -float(scores[i]))

        picks, position = [], 0
        while len(picks) < k:
            added = False
            for code, idx_list in by_target.items():
                if position < len(idx_list):
                    picks.append(idx_list[position])
                    added = True
                    if len(picks) >= k:
                        break
            if not added:
                break
            position += 1
        return picks

    @torch.no_grad()
    def generate_candidates(
        self,
        z_t: torch.Tensor,
        f_src_code: int,
        scoreline: tuple = (0, 0),
        minute: int = 45,
        period: int = 1,
        team_is_home: bool = True,
    ) -> List[Transition]:
        f_src_idx = self.vocab.code_to_idx(f_src_code)

        feasible_f_tgt_code   = []
        feasible_f_tgt_idx    = []
        feasible_intent       = []

        for f_tgt_code in self.vocab.vocab:
            for intent_str in INTENTS:
                if self.grammar.is_feasible(
                    f_src=f_src_code, f_tgt=f_tgt_code, intent=intent_str,
                    scoreline=scoreline, minute=minute, period=period,
                    team_is_home=team_is_home,
                ):
                    feasible_f_tgt_code.append(f_tgt_code)
                    feasible_f_tgt_idx.append(self.vocab.code_to_idx(f_tgt_code))
                    feasible_intent.append(INTENT_TO_IDX[intent_str])

        if not feasible_f_tgt_code:
            return []

        if self.scorer is None:
            # B5: deterministic prior ranking, no learned scorer.
            goal_diff = (scoreline[0] - scoreline[1]) if team_is_home \
                        else (scoreline[1] - scoreline[0])
            scores = torch.tensor(
                [_prior_score(f_src_code, c, goal_diff) for c in feasible_f_tgt_code],
                dtype=torch.float,
            )
        else:
            f_src_tensor   = torch.tensor(f_src_idx, dtype=torch.long)
            f_tgt_tensor   = torch.tensor(feasible_f_tgt_idx, dtype=torch.long)
            intent_tensor  = torch.tensor(feasible_intent, dtype=torch.long)
            f_src_expanded = f_src_tensor.expand(len(feasible_f_tgt_idx))
            scores = self.scorer(z_t, f_src_expanded, f_tgt_tensor, intent_tensor)

        k = min(self.top_k, len(scores))
        top_indices = self._diverse_top_k(feasible_f_tgt_code, scores, k)

        candidates = []
        for orig_idx in top_indices:
            candidates.append(Transition(
                f_src  = f_src_code,
                f_tgt  = feasible_f_tgt_code[orig_idx],
                intent = INTENTS[feasible_intent[orig_idx]],
                score  = float(scores[orig_idx]),
            ))
        return candidates


# ── Factory helper ─────────────────────────────────────────────────────────────
def build_generator(
    vocab_path: Optional[str] = "data/processed/formation_vocab.json",
    grammar_path: str = "models/transition_generator/grammar_rules.yaml",
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> FormationTransitionGenerator:
    """
    Instantiates vocab, grammar, scorer, and wraps them in a Generator.

    If checkpoint_path is None or missing, the generator falls back to the
    deterministic prior (-0.02*Δstruct + 0.15*goal_diff*attackers_change).
    It does NOT silently use a randomly-initialised MLP: doing so made the
    top-K indistinguishable from a random draw over the feasible set (B5).
    """
    vocab   = FormationVocab(vocab_path)
    grammar = TacticalGrammarFilter(grammar_path)

    if checkpoint_path and Path(checkpoint_path).exists():
        scorer = TransitionScorerMLP(vocab_size=len(vocab)).to(device)
        state  = torch.load(checkpoint_path, map_location=device)
        scorer.load_state_dict(state)
        scorer.eval()
        print(f"Loaded trained scorer from {checkpoint_path}")
    else:
        if checkpoint_path:
            print(f"WARNING: scorer checkpoint {checkpoint_path} not found.")
        print("WARNING: no trained scorer checkpoint; using the deterministic "
              "prior for top-K ranking. Train the scorer or pass a checkpoint "
              "to use learned scores.")
        scorer = None

    return FormationTransitionGenerator(scorer, vocab, grammar)
# ── Quick smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    gen = build_generator(vocab_path=None, grammar_path="grammar_rules.yaml")
    z_t = torch.zeros(256)  # synthetic context vector

    candidates = gen.generate_candidates(
        z_t,
        f_src_code   = 4231,
        scoreline    = (1, 0),
        minute       = 65,
        period       = 2,
        team_is_home = True,
    )

    print(f"Candidates returned: {len(candidates)}  (expected ≤ 20)")
    for i, c in enumerate(candidates[:5]):
        print(f"  [{i+1}] {c.f_src} → {c.f_tgt} | {c.intent:15s} | score={c.score:.4f}")
