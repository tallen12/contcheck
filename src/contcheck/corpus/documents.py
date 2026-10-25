"""Clean paragraphs and the documents they are grouped into."""

import hashlib
import random
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import groupby

from pydantic import BaseModel, ConfigDict, Field

from contcheck.corpus.record import Split

MIN_PARAGRAPHS = 2
MAX_PARAGRAPHS = 5


class RawParagraph(BaseModel):
    """One clean paragraph; consecutive paragraphs with one ``source_id`` belong together.

    This is also the line format of raw JSONL files.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class Document:
    """A few consecutive paragraphs from one source, with its own split."""

    doc_id: str
    split: Split
    paragraphs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SplitShares:
    """Fractions of documents for validation and test; the rest is training data."""

    val: float = 0.05
    test: float = 0.05

    def __post_init__(self) -> None:
        if self.val < 0 or self.test < 0 or self.val + self.test >= 1:
            raise ValueError("val and test shares must be non-negative and leave room for train")


def assign_split(doc_id: str, shares: SplitShares) -> Split:
    """Split by a hash of ``doc_id``, so a document always lands in the same split."""
    digest = hashlib.sha256(doc_id.encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") / 2**64
    if position < shares.val:
        return Split.VAL
    if position < shares.val + shares.test:
        return Split.TEST
    return Split.TRAIN


def group_documents(
    paragraphs: Iterable[RawParagraph], *, seed: int, shares: SplitShares
) -> Iterator[Document]:
    """Group consecutive paragraphs of each source into documents of 2 to 5 paragraphs.

    Group sizes come from a generator seeded by ``seed`` and the source id, so the result
    does not depend on which other sources are read. A source's last document may hold a
    single paragraph.

    Raises ``ValueError`` if a source's paragraphs are not consecutive: grouping it again
    would reuse its document ids and seeds for different text.
    """
    seen: set[str] = set()
    for source_id, group in groupby(paragraphs, key=lambda paragraph: paragraph.source_id):
        if source_id in seen:
            raise ValueError(f"paragraphs of source {source_id!r} are not consecutive")
        seen.add(source_id)
        texts = [paragraph.text for paragraph in group]
        rng = random.Random(f"{seed}:{source_id}")
        start = 0
        index = 0
        while start < len(texts):
            size = rng.randint(MIN_PARAGRAPHS, MAX_PARAGRAPHS)
            doc_id = f"{source_id}-{index:04d}"
            yield Document(doc_id, assign_split(doc_id, shares), tuple(texts[start : start + size]))
            start += size
            index += 1
