import numpy as np
import pytest
import torch
from transformers import ElectraConfig

from contcheck.model.models import IGNORE_INDEX
from contcheck.model.protocols import Tagger
from contcheck.model.tagger import (
    DOC_DIM,
    ContextTagger,
    TorchTagger,
    class_weights,
    encoder_from_config,
)
from contcheck.taxonomy.labels import LABEL2ID, LABELS


@pytest.fixture
def inputs() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    input_ids = torch.randint(3, 100, (2, 7))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, 5:] = 0
    return input_ids, attention_mask


def test_logits_have_one_score_per_label(
    tiny_encoder_config: ElectraConfig, inputs: tuple[torch.Tensor, torch.Tensor]
) -> None:
    model = ContextTagger(encoder_from_config(tiny_encoder_config))

    output = model.forward(*inputs)

    assert output.logits.shape == (2, 7, len(LABELS))
    assert output.loss is None


def test_loss_skips_ignored_labels(
    tiny_encoder_config: ElectraConfig, inputs: tuple[torch.Tensor, torch.Tensor]
) -> None:
    model = ContextTagger(encoder_from_config(tiny_encoder_config)).eval()
    labels = torch.full((2, 7), IGNORE_INDEX)
    labels[0, 1] = LABEL2ID["B-Spelling"]
    changed = labels.clone()
    changed[1, 6] = IGNORE_INDEX

    loss = model.forward(*inputs, labels=labels).loss
    same = model.forward(*inputs, labels=changed).loss

    assert loss is not None and same is not None
    assert torch.isfinite(loss)
    assert torch.equal(loss, same)


def test_outside_label_weighs_less() -> None:
    weights = class_weights(0.2)

    assert weights[LABEL2ID["O"]] == pytest.approx(0.2)
    assert weights.sum() == pytest.approx(len(LABELS) - 0.8)


def test_doc_vector_path(
    tiny_encoder_config: ElectraConfig, inputs: tuple[torch.Tensor, torch.Tensor]
) -> None:
    model = ContextTagger(encoder_from_config(tiny_encoder_config), use_doc=True, fuse_dim=8).eval()
    doc_vec = torch.randn(2, DOC_DIM)

    first = model.forward(*inputs, doc_vec=doc_vec).logits
    second = model.forward(*inputs, doc_vec=torch.randn(2, DOC_DIM)).logits

    assert first.shape == (2, 7, len(LABELS))
    assert not torch.allclose(first, second)
    with pytest.raises(ValueError, match="doc_vec"):
        model.forward(*inputs)


def test_parameter_groups_split_encoder_and_head(tiny_encoder_config: ElectraConfig) -> None:
    model = ContextTagger(encoder_from_config(tiny_encoder_config), use_doc=True)

    encoder, head = model.parameter_groups(encoder_lr=1e-5, head_lr=1e-3)

    assert (encoder["lr"], head["lr"]) == (1e-5, 1e-3)
    params = encoder["params"]
    new = head["params"]
    assert isinstance(params, list) and isinstance(new, list)
    assert len(params) + len(new) == len(list(model.parameters()))


def test_torch_tagger_returns_numpy_logits(
    tiny_encoder_config: ElectraConfig, inputs: tuple[torch.Tensor, torch.Tensor]
) -> None:
    tagger: Tagger = TorchTagger(ContextTagger(encoder_from_config(tiny_encoder_config)), "cpu")
    input_ids, attention_mask = (tensor.numpy().astype(np.int64) for tensor in inputs)

    logits = tagger.logits(input_ids, attention_mask)

    assert logits.shape == (2, 7, len(LABELS))
    assert logits.dtype == np.float32
