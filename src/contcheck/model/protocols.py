"""Interfaces for subword tokenization and the tagger."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import numpy.typing as npt

from contcheck.model.models import Encoding


class Tokenizer(Protocol):
    """Splits pre-split words into subword ids."""

    @property
    def cls_id(self) -> int:
        """Id of the token that starts every window."""
        ...

    @property
    def sep_id(self) -> int:
        """Id of the token that ends every window."""
        ...

    @property
    def pad_id(self) -> int:
        """Id used to pad windows in a batch."""
        ...

    def encode_words(self, words: Sequence[str]) -> Encoding:
        """Subwords of ``words`` in order, without special tokens."""
        ...


class Tagger(Protocol):
    """Scores every token of a padded batch of windows against every label."""

    def logits(
        self, input_ids: npt.NDArray[np.int64], attention_mask: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.float32]:
        """Logits shaped ``(batch, tokens, labels)`` for ``(batch, tokens)`` inputs."""
        ...
