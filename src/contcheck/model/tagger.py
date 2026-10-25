"""``ContextTagger``: an encoder with a token classification head and an optional
document vector fused into every token."""

from dataclasses import dataclass
from typing import override

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.nn import functional
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

from contcheck.model.models import IGNORE_INDEX
from contcheck.taxonomy.labels import LABEL2ID, LABELS, OUTSIDE

DOC_DIM = 392
"""384 embedding dims, 4 tense shares and 4 document stats (M4)."""


@dataclass(frozen=True, slots=True)
class TaggerOutput:
    """``(batch, tokens, labels)`` logits, and the loss when labels were given."""

    logits: torch.Tensor
    loss: torch.Tensor | None


def class_weights(outside_weight: float) -> torch.Tensor:
    """Loss weight per label: ``outside_weight`` for ``O``, 1 for every error label."""
    weights = torch.ones(len(LABELS))
    weights[LABEL2ID[OUTSIDE]] = outside_weight
    return weights


class ContextTagger(nn.Module):
    """Tags every token with a BIO label.

    With ``use_doc`` a projected, layer-normalized document vector is concatenated to
    every token before the head; the ``LayerNorm`` matters because embedding values and
    histogram shares are on different scales.
    """

    def __init__(
        self,
        encoder: PreTrainedModel,
        *,
        use_doc: bool = False,
        doc_dim: int = DOC_DIM,
        fuse_dim: int = 128,
        outside_weight: float = 0.2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        hidden = _hidden_size(encoder.config)
        self.use_doc = use_doc
        self.doc_proj = (
            nn.Sequential(nn.LayerNorm(doc_dim), nn.Linear(doc_dim, fuse_dim), nn.GELU())
            if use_doc
            else None
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + (fuse_dim if use_doc else 0), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, len(LABELS)),
        )
        self.class_weights: torch.Tensor
        self.register_buffer("class_weights", class_weights(outside_weight))

    @override
    def forward(  # ty: ignore[invalid-mutable-override] -- torch types forward as a mutable attribute
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        doc_vec: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> TaggerOutput:
        tokens = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if self.doc_proj is not None:
            if doc_vec is None:
                raise ValueError("this tagger was built with use_doc and needs doc_vec")
            fused = self.doc_proj(doc_vec).unsqueeze(1).expand(-1, tokens.size(1), -1)
            tokens = torch.cat([tokens, fused], dim=-1)
        logits = self.head(tokens)
        loss = None
        if labels is not None:
            loss = functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                weight=self.class_weights,
                ignore_index=IGNORE_INDEX,
            )
        return TaggerOutput(logits, loss)

    def parameter_groups(self, *, encoder_lr: float, head_lr: float) -> list[dict[str, object]]:
        """Optimizer groups: the pretrained encoder learns slowly, the new head fast."""
        new = [p for name, p in self.named_parameters() if not name.startswith("encoder.")]
        return [
            {"params": list(self.encoder.parameters()), "lr": encoder_lr},
            {"params": new, "lr": head_lr},
        ]


def _hidden_size(config: PretrainedConfig) -> int:
    hidden = getattr(config, "hidden_size", None)
    if not isinstance(hidden, int):
        raise ValueError("encoder config has no hidden_size")
    return hidden


def pretrained_encoder(name: str) -> PreTrainedModel:
    """Load pretrained encoder weights by Hub name (downloaded once) or from a directory."""
    return _as_model(AutoModel.from_pretrained(name))


def encoder_from_config(config: PretrainedConfig) -> PreTrainedModel:
    """An encoder with random weights, for tests or before loading a checkpoint."""
    return _as_model(AutoModel.from_config(config))


def encoder_config(name_or_path: str) -> PretrainedConfig:
    """The architecture config of an encoder, without its weights."""
    config: object = AutoConfig.from_pretrained(name_or_path)
    if not isinstance(config, PretrainedConfig):
        raise TypeError(f"{name_or_path} did not load a model config")
    return config


def _as_model(model: object) -> PreTrainedModel:
    if not isinstance(model, PreTrainedModel):
        raise TypeError(f"expected a PreTrainedModel, got {type(model).__name__}")
    return model


class TorchTagger:
    """The ``Tagger`` protocol over a ``ContextTagger`` on one device."""

    def __init__(self, model: ContextTagger, device: torch.device | str) -> None:
        self._model = model.to(device).eval()
        self._device = torch.device(device)

    def logits(
        self, input_ids: npt.NDArray[np.int64], attention_mask: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]:
        with torch.inference_mode():
            # forward, not __call__: no hooks are used, and it keeps the output typed.
            output = self._model.forward(
                torch.from_numpy(input_ids).to(self._device),
                torch.from_numpy(attention_mask).to(self._device),
            )
        return output.logits.float().cpu().numpy()
