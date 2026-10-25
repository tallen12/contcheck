from pathlib import Path

import pytest
import torch
from transformers import BertTokenizer, ElectraConfig

from contcheck.model.checkpoint import INFO, CheckpointInfo, load_checkpoint, save_checkpoint
from contcheck.model.hf_tokenizer import HuggingFaceTokenizer
from contcheck.model.tagger import ContextTagger, encoder_from_config
from contcheck.taxonomy.labels import LABELS

VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "i", "went", "home"]


def info(**changes: object) -> CheckpointInfo:
    data: dict[str, object] = {
        "encoder": "tiny",
        "labels": LABELS,
        "use_doc": False,
        "fuse_dim": 8,
        "window": 16,
        "stride": 8,
        "epoch": 2,
        "val_f05": 0.5,
    }
    data.update(changes)
    return CheckpointInfo.model_validate(data)


@pytest.fixture
def saved(tmp_path: Path, tiny_encoder_config: ElectraConfig) -> tuple[Path, ContextTagger]:
    torch.manual_seed(0)
    model = ContextTagger(encoder_from_config(tiny_encoder_config), fuse_dim=8)
    tokenizer = HuggingFaceTokenizer(BertTokenizer(vocab={t: i for i, t in enumerate(VOCAB)}))
    save_checkpoint(
        tmp_path, model=model, state=model.state_dict(), tokenizer=tokenizer, info=info()
    )
    return tmp_path, model


def test_checkpoint_round_trips_weights_tokenizer_and_info(
    saved: tuple[Path, ContextTagger],
) -> None:
    directory, model = saved
    input_ids = torch.tensor([[2, 5, 6, 7, 3]])
    mask = torch.ones_like(input_ids)

    loaded = load_checkpoint(directory)

    assert loaded.info == info()
    assert loaded.tokenizer.encode_words(["went"]).input_ids == (6,)
    assert not loaded.model.training
    model.eval()
    assert torch.allclose(
        loaded.model.forward(input_ids, mask).logits, model.forward(input_ids, mask).logits
    )


def test_checkpoint_with_other_labels_is_rejected(saved: tuple[Path, ContextTagger]) -> None:
    directory, _ = saved
    (directory / INFO).write_text(info(labels=("O",)).model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="label set"):
        load_checkpoint(directory)


@pytest.mark.parametrize(
    "changes", [{"window": 16, "stride": 15}, {"window": 4, "stride": 1}, {"stride": 0}]
)
def test_info_rejects_windows_that_leave_gaps(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        info(**changes)


def test_checkpoint_with_document_vectors_is_rejected(saved: tuple[Path, ContextTagger]) -> None:
    directory, _ = saved
    (directory / INFO).write_text(info(use_doc=True).model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="document vectors"):
        load_checkpoint(directory)
