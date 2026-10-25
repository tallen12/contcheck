from contcheck.corpus.merge import merge_records
from contcheck.corpus.record import Record, RecordSource, Split


def record(doc_id: str, paragraph: int, variant: int, noisy: str, split: Split) -> Record:
    return Record(
        doc_id=doc_id,
        paragraph=paragraph,
        variant=variant,
        split=split,
        source=RecordSource.RULE,
        clean="a b",
        noisy=noisy,
        words=tuple(noisy.split()),
        labels=("O",) * len(noisy.split()),
        edits=(),
    )


def test_merge_dedupes_and_keeps_splits() -> None:
    records = [
        record("d1", 0, 0, "a b", Split.TRAIN),
        record("d1", 0, 1, "a b", Split.TRAIN),
        record("d1", 0, 2, "a c", Split.TRAIN),
        record("d2", 0, 0, "a b", Split.TEST),
    ]

    merged = merge_records(records, seed=1)

    assert sorted(r.noisy for r in merged[Split.TRAIN]) == ["a b", "a c"]
    assert [r.doc_id for r in merged[Split.TEST]] == ["d2"]
    assert merged[Split.VAL] == []


def test_merge_shuffles_deterministically() -> None:
    records = [record(f"d{i}", 0, 0, f"w{i}", Split.TRAIN) for i in range(50)]

    first = [r.doc_id for r in merge_records(records, seed=3)[Split.TRAIN]]
    again = [r.doc_id for r in merge_records(records, seed=3)[Split.TRAIN]]

    assert first == again
    assert first != [r.doc_id for r in records]
