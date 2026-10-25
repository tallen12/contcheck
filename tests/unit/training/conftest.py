"""Records shared by training tests."""

import pytest

from contcheck.corpus.record import EditRecord, Record, RecordSource, Split
from contcheck.taxonomy.kinds import ErrorKind, Operation


def make_record(index: int, *, noisy: bool) -> Record:
    words = ("I", "go" if noisy else "went", "to", "the", "store", "yesterday", ".")
    labels = ("O", "B-VerbTense" if noisy else "O", "O", "O", "O", "O", "O")
    edits = (
        (
            EditRecord(
                start=1, end=2, kind=ErrorKind.VERB_TENSE, op=Operation.REPLACEMENT, orig=("went",)
            ),
        )
        if noisy
        else ()
    )
    return Record(
        doc_id=f"doc-{index}",
        paragraph=0,
        variant=0,
        split=Split.TRAIN,
        source=RecordSource.RULE,
        clean="I went to the store yesterday.",
        noisy=" ".join(words),
        words=words,
        labels=labels,
        edits=edits,
    )


@pytest.fixture
def records() -> list[Record]:
    return [make_record(i, noisy=i % 2 == 0) for i in range(16)]
