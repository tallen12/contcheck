import numpy as np
import numpy.typing as npt

from contcheck.infer.predict import inference_windows, predict_paragraphs
from contcheck.model.models import IGNORE_INDEX
from contcheck.taxonomy.labels import LABEL2ID, LABELS
from tests.conftest import FakeTokenizer

MARKED = FakeTokenizer().piece_id("bad")


class MarkingTagger:
    """Says B-Spelling for the subword ``bad`` and O for everything else."""

    def __init__(self) -> None:
        self.batches: list[tuple[int, ...]] = []

    def logits(
        self, input_ids: npt.NDArray[np.int64], attention_mask: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]:
        self.batches.append(input_ids.shape)
        logits = np.zeros((*input_ids.shape, len(LABELS)), dtype=np.float32)
        logits[..., LABEL2ID["O"]] = 5.0
        logits[input_ids == MARKED, LABEL2ID["B-Spelling"]] = 10.0
        return logits


def test_inference_windows_mark_first_subwords(tokenizer: FakeTokenizer) -> None:
    (window,) = inference_windows(["abcdef", "x"], tokenizer, window=16, stride=8)

    assert window.labels == (IGNORE_INDEX, 0, IGNORE_INDEX, 0, IGNORE_INDEX)


def test_predicts_every_word_of_every_paragraph(tokenizer: FakeTokenizer) -> None:
    tagger = MarkingTagger()
    long = ["ok"] * 30 + ["bad"] + ["ok"] * 30
    paragraphs = [["a", "bad", "word"], long, []]

    results = predict_paragraphs(
        paragraphs, tagger=tagger, tokenizer=tokenizer, window=16, stride=8, batch_size=4
    )

    assert [r.shape for r in results] == [(3, len(LABELS)), (61, len(LABELS)), (0, len(LABELS))]
    assert [LABELS[i] for i in results[0].argmax(axis=-1)] == ["O", "B-Spelling", "O"]
    assert LABELS[int(results[1][30].argmax())] == "B-Spelling"
    assert all(shape[0] <= 4 for shape in tagger.batches)
