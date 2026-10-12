"""Typed configuration. Every script takes --config path/to.yaml."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class EncoderConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dim_ff: int = 256
    dropout: float = 0.1
    seq_len: int = 50              # N in the paper
    gat_layers: int = 2
    gat_heads: int = 4
    gat_dim: int = 128
    knn: int = 3                   # intra-formation adjacency degree
    d_context: int = 256           # dimension of z_t
    use_pyg: bool = False          # optional torch-geometric path


@dataclass
class OpponentConfig:
    d_model: int = 256             # must equal EncoderConfig.d_context
    n_heads: int = 4
    d_style: int = 32
    dropout: float = 0.1
    kl_lambda: float = 0.1         # KL(p || uniform) calibration weight
    label_smoothing: float = 0.0
    freeze_encoder: bool = False   # True = train only the cross-attention head
    # Second head: P(the opponent changes shape at all).
    respond_weight: float = 1.0    # weight of the respond BCE in the total loss
    respond_pos_weight: float = 0.0  # 0 = balance from the batch's own base rate


@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    epochs: int = 40
    patience: int = 8
    # "loss" | "nll" -- see validation_objective() in train_opponent.
    # "loss" is the two-head training objective on held-out matches and is
    # the default; "nll" is the formation head's NLL on the decisions where
    # the shape actually moved. Single-head names like "top1" now warn and
    # fall back to "loss".
    select_on: str = "loss"
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    mirror_augment: bool = True
    n_folds: int = 5
    seeds: tuple = (0, 1, 2)
    num_workers: int = 0
    amp: bool = True               # mixed precision when CUDA is available


@dataclass
class PretrainConfig:
    lr: float = 3e-4
    batch_size: int = 128
    epochs: int = 5
    max_sequences: int | None = None   # cap for a quick smoke run


@dataclass
class InferenceConfig:
    candidate_k: int = 20          # K in the paper
    max_rounds: int = 2            # M
    # "zscore" puts the critics on a common scale within each decision before
    # weighting, so the configured weights are the influence each actually has.
    # "none" restores the original behaviour, in which a critic with a wider
    # raw spread counted for more than its weight.
    critic_normalisation: str = "zscore"
    conformal_alpha: float = 0.10
    # theta_conf: stop revising once the prediction set grows past this.
    #
    # This was 6, chosen before the terminator had ever been fitted. Measured
    # on the fold-0 held-out triples at alpha = 0.10, every prediction set is
    # of size 3 or less (60.4% are singletons, 97.2% are <= 2), so a threshold
    # of 6 fires on 0.0% of decisions: the rule was in the loop and could never
    # trigger, and every run to date spent its full revision budget.
    #
    # 1 makes the rule active -- it stops whenever the model is not down to a
    # single shape, which is 39.6% of decisions. 2 fires on 2.8%. See
    # eval_conformal.py and results/conformal.json; re-derive this if alpha or
    # the opponent model changes, rather than carrying the number forward.
    conformal_threshold: int = 1
    critic_weights: tuple = (1 / 3, 1 / 3, 1 / 3)


@dataclass
class Paths:
    processed: str = "data/processed"
    models: str = "models"
    results: str = "results"

    def ensure(self) -> None:
        for p in (self.processed, self.models, self.results):
            Path(p).mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    opponent: OpponentConfig = field(default_factory=OpponentConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    pretrain: PretrainConfig = field(default_factory=PretrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    paths: Paths = field(default_factory=Paths)
    device: str = "auto"           # "auto" | "cuda" | "cpu"
    seed: int = 0

    # -- io ---------------------------------------------------------------
    @staticmethod
    def load(path: str | Path | None) -> "Config":
        cfg = Config()
        if path is None:
            return cfg
        raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
        for section, values in raw.items():
            if not hasattr(cfg, section):
                raise KeyError(f"unknown config section: {section}")
            target = getattr(cfg, section)
            if dataclasses.is_dataclass(target):
                for k, v in (values or {}).items():
                    if not hasattr(target, k):
                        raise KeyError(f"unknown key {section}.{k}")
                    setattr(target, k, tuple(v) if isinstance(v, list) else v)
            else:
                setattr(cfg, section, values)
        # invariant the models rely on
        assert cfg.opponent.d_model == cfg.encoder.d_context, (
            "opponent.d_model must equal encoder.d_context "
            f"({cfg.opponent.d_model} != {cfg.encoder.d_context})"
        )
        return cfg

    def resolved_device(self) -> str:
        """Pick the fastest device actually present.

        This used to test only for CUDA, so on an Apple-silicon Mac it fell
        through to the CPU while a usable GPU sat idle -- which is most of
        the machines this is developed on. Mixed precision stays CUDA-only
        (the training scripts guard it with device.startswith("cuda")):
        MPS autocast is not equivalent and would change results silently.

        If an MPS run dies on an unimplemented operator, re-run it with
        PYTORCH_ENABLE_MPS_FALLBACK=1, which sends just those ops to the
        CPU. Compare one fold against device: cpu before trusting a full
        MPS run -- the backend is younger than the CUDA one.
        """
        if self.device != "auto":
            return self.device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available() and mps.is_built():
            return "mps"
        return "cpu"

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)
