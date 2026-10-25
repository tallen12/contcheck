"""Saving and loading a trained tagger as a self-contained directory.

A checkpoint holds the weights, the encoder's architecture config and the tokenizer
files, so it loads offline without the Hub.
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from pydantic import BaseModel, ConfigDict

from contcheck.model.hf_tokenizer import HuggingFaceTokenizer, load_tokenizer
from contcheck.model.tagger import ContextTagger, encoder_config, encoder_from_config
from contcheck.taxonomy.labels import LABELS

WEIGHTS = "model.pt"
INFO = "checkpoint.json"
ENCODER = "encoder"
TOKENIZER = "tokenizer"


class CheckpointInfo(BaseModel):
    """What is needed to rebuild the model around the saved weights."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    encoder: str
    labels: tuple[str, ...]
    use_doc: bool
    fuse_dim: int
    window: int
    stride: int
    epoch: int
    val_f05: float


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """A model in eval mode with its tokenizer and settings."""

    model: ContextTagger
    tokenizer: HuggingFaceTokenizer
    info: CheckpointInfo


def save_checkpoint(
    directory: Path,
    *,
    model: ContextTagger,
    state: dict[str, torch.Tensor],
    tokenizer: HuggingFaceTokenizer,
    info: CheckpointInfo,
) -> None:
    """Write ``state`` (the weights to keep), the encoder config, tokenizer and info."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(state, directory / WEIGHTS)
    model.encoder.config.save_pretrained(directory / ENCODER)
    tokenizer.save(directory / TOKENIZER)
    (directory / INFO).write_text(info.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_checkpoint(directory: Path, *, device: torch.device | str = "cpu") -> LoadedCheckpoint:
    """Rebuild the model from a checkpoint directory. Raises if its labels differ from ours."""
    info = CheckpointInfo.model_validate_json((directory / INFO).read_text(encoding="utf-8"))
    if info.labels != LABELS:
        raise ValueError(f"{directory} was trained with a different label set")
    encoder = encoder_from_config(encoder_config(str(directory / ENCODER)))
    model = ContextTagger(encoder, use_doc=info.use_doc, fuse_dim=info.fuse_dim)
    state = torch.load(directory / WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return LoadedCheckpoint(model, load_tokenizer(directory / TOKENIZER), info)
