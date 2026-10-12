# RAMFP — state encoder, opponent co-behavior model, training, inference

My slice of the RAMFP pipeline: steps 07, 09 and the Algorithm 1 inference
loop. It runs today on synthetic data with a known Bayes ceiling, and switches
to real StatsBomb triples by pointing one flag at a parquet file.

```
pip install -r requirements.txt

# 1. pretrain the state encoder on next-event-type prediction
python -m ramfp.train.pretrain_encoder --config configs/default.yaml

# 2. train the opponent model, with the no-style ablation
python -m ramfp.train.train_opponent --config configs/default.yaml \
    --encoder models/encoder.pt --ablate-style

# 3. the E1 baselines
python -m ramfp.train.baselines --config configs/default.yaml

# 4. the inference loop
python scripts/demo_inference.py --config configs/default.yaml \
    --opponent models/opponent.pt
```

Add `--sequences data/processed/sequences.parquet` and
`--triples data/processed/triples.parquet` once the data team ships them.
Nothing else changes.

## What is real and what is a placeholder

| Component | Status |
|---|---|
| State encoder (step 07) | complete |
| Formation graph + role vocabulary (N5) | complete |
| Opponent co-behavior model (step 09) | complete |
| Training, CV, metrics, E1 baselines | complete |
| Conformal termination | complete |
| Inference loop / Algorithm 1 | complete |
| World model (step 10) | **stub** — teammate's module, enters via the `WorldModel` protocol |
| Stability & counter-resistance critics (step 12) | **stub** — enter via the `Critic` protocol |
| Learned transition scorer (step 11) | heuristic prior; swap in the trained MLP |

Anything that touched a stub is flagged `is_stub=True` on the result object and
prints a `[STUB WORLD MODEL — not a result]` marker. Never copy a stubbed
number into a results table.

## Synthetic data and the Bayes ceiling

`ramfp/data/synthetic.py` samples responses from an explicit conditional
distribution and reports the accuracy a perfect model would achieve. This is
the point of it: when the real data lands and you score 0.20 Top-1, the
synthetic benchmark is how you tell a broken model from a noisy problem.

```
majority-class Top-1 : 0.087   <- beat this or the model learned nothing
Bayes-optimal  Top-1 : 0.311   <- a correct implementation approaches this
```

Sweep `--temperature` (0.3 easy, 0.85 hard) and check your model tracks the
ceiling rather than one operating point.

## Design decisions worth defending in the paper

**Causal transformer, not bidirectional.** The pretraining objective is
next-event-type prediction, so a causal mask makes the pretrained weights and
the downstream encoder the same function. `z_trans` is the hidden state at the
last real token — exactly the position the objective optimises. A bidirectional
encoder would need a separate finetuning story that the draft does not have.

**Role vocabulary instead of a per-formation adjacency table.** Formation codes
are decoded into lines, players placed at canonical pitch positions, and each
assigned a role from a fixed 6 band x 5 lateral grid. Two formations sharing a
back four therefore share those four role embeddings. This is contribution N5,
it needs no hand-written taxonomy, it generalises to unseen formation codes,
and it is deterministic enough to unit-test. Adjacency is k-nearest-neighbour
over the same positions.

**Style ablation removes the key-value entry, not the values.** Setting
`style_conditioned=False` drops `e_phi` from the cross-attention input
altogether rather than zeroing it, so E1 baseline 4 is a true architectural
ablation.

**GroupKFold by match_id.** Two triples from the same match share a game state
and usually the teams. A random split leaks and inflates Top-1 by several
points. With ~1,500 examples, one split is not a result — report mean ± std
over folds × seeds.

**KL(p ‖ uniform) is a calibration regulariser.** It equals `log|V| − H(p)`, so
minimising it maximises predictive entropy. With 15 classes and ~100 examples
each the model otherwise becomes confidently wrong, which wrecks the ECE column
and, downstream, breaks the conformal stopping rule in the planner.

**The candidate space is 90, not 1,350.** `f_src` is fixed by the current
state, so the real per-decision space is |V| × |I|. The draft's 1,350 is the
full product space V × V × I. Say so explicitly — a reviewer who multiplies it
out will notice.

**`UpdateContext` is a fixed perturbation, not a learned layer.** Nothing in
the paper supervises this map, so a learned version would be an untrained layer
presented as a component. It is norm-preserving and documented as a design
choice.

## Things to ask the team

1. **Intent labels.** The draft never says how the six labels
   (offensive / defensive / wide / narrow / pressing / conservative) are
   assigned. Step 05 owns this. Until there is a written rule table, the
   `intent` column is undefined and my query vector is one third guesswork.
2. **PPDA bucket edges.** `vocab.PPDA_EDGES` are placeholders. Replace with
   real quantiles from step 03 or the bucket embedding is close to noise.
3. **Formation vocabulary.** `vocab.FORMATIONS` is my guess at the top 15.
   Replace with the real counts from step 02.
4. **Calibration split.** The conformal terminator needs triples the opponent
   model never trained on. Reserve them at split time, not afterwards.

## Layout

```
ramfp/
  vocab.py                formations, roles, geometry, adjacency, zones
  config.py               typed config, loaded from YAML
  data/
    schema.py             the parquet contract with the data team
    synthetic.py          synthetic triples + Bayes ceiling
    datasets.py           datasets, collation, mirror augmentation
  models/
    layers.py             DenseGAT (pure torch) + optional PyG wrapper
    state_encoder.py      step 07
    opponent.py           step 09
  train/
    pretrain_encoder.py   next-event-type pretraining
    train_opponent.py     E1 training with GroupKFold CV
    baselines.py          E1 baselines 1-3
  eval/metrics.py         Top-k, ECE, reliability, split conformal
  inference.py            Algorithm 1, protocols for the teammates' modules
scripts/demo_inference.py end-to-end demo
tests/test_smoke.py       shapes, masking, determinism, no-leak checks
```

## Verified runs (synthetic, temperature 0.5, GroupKFold, seed 0)

| System | Top-1 | Top-3 | ECE |
|---|---|---|---|
| Random | 0.071 | 0.206 | 0.013 |
| Majority class | 0.082 | 0.244 | 0.009 |
| Logistic (state + style) | 0.184 | 0.405 | 0.049 |
| Transformer, no style conditioning | 0.169 | 0.397 | 0.116 |
| **Full model** | **0.230** | **0.516** | 0.121 |
| *Bayes ceiling* | *0.319* | *0.637* | — |

Two things this table tells you before the real data arrives:

**Style conditioning is worth about +6 points of Top-1.** That is the E1
ablation working as intended, on data where we know the style label genuinely
matters. If it is worth nothing on real data, suspect the step-04 clusters
before you suspect the architecture.

**The logistic baseline is strong here** because my synthetic rule is close to
linear in style and scoreline. Do not read that as a result about football —
it is a property of the generator. On real data the transition input should
carry more, and if it does not, that is a finding worth reporting honestly.

## Two findings from building this that affect the paper

**1. Select on NLL, not Top-1.** Early-stopping on validation Top-1 let the
model keep sharpening a confidently-wrong distribution long after accuracy
plateaued. Same architecture, same data, only the selection metric changed:

| Selection metric | Top-1 | ECE |
|---|---|---|
| Top-1 | 0.215 | 0.252 |
| NLL | **0.230** | **0.121** |

Table 3 reports ECE, and the planner's conformal rule reads this model's
probabilities, so a miscalibrated model degrades two results at once. Default
is now `train.select_on: nll`.

**2. `theta_conf` cannot be picked as a round number.** With 15 classes and a
Bayes ceiling near 0.32, a 90% conformal prediction set is naturally about 9-10
classes wide. The draft's implied threshold of 6 sits below the median, so
every candidate terminates at round 0 and the revision loop — contribution N7 —
never actually runs. `scripts/demo_inference.py` now prints the set-size
distribution and warns when the threshold is below the median. Calibrate
`theta_conf` against that distribution and report both the value and the
distribution, or the ablation over M is measuring nothing.
