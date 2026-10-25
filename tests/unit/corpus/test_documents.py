from collections import Counter

import pytest

from contcheck.corpus.documents import (
    RawParagraph,
    SplitShares,
    assign_split,
    group_documents,
)
from contcheck.corpus.record import Split


def paragraphs(source: str, count: int) -> list[RawParagraph]:
    return [RawParagraph(source_id=source, text=f"{source} paragraph {i}.") for i in range(count)]


def test_documents_hold_consecutive_paragraphs_of_one_source() -> None:
    raw = paragraphs("a", 12) + paragraphs("b", 3)

    documents = list(group_documents(raw, seed=1, shares=SplitShares()))

    assert [p for d in documents for p in d.paragraphs] == [p.text for p in raw]
    assert all(len(d.paragraphs) <= 5 for d in documents)
    from_a = [d for d in documents if d.doc_id.startswith("a-")]
    assert all(len(d.paragraphs) >= 2 for d in from_a[:-1])
    assert {d.doc_id.rsplit("-", 1)[0] for d in documents} == {"a", "b"}
    assert documents[0].doc_id == "a-0000"


def test_grouping_is_deterministic_per_source() -> None:
    alone = list(group_documents(paragraphs("b", 9), seed=1, shares=SplitShares()))
    mixed = list(
        group_documents(paragraphs("a", 7) + paragraphs("b", 9), seed=1, shares=SplitShares())
    )

    assert [d for d in mixed if d.doc_id.startswith("b")] == alone


def test_split_is_stable_and_roughly_follows_shares() -> None:
    shares = SplitShares(val=0.1, test=0.2)

    splits = Counter(assign_split(f"doc-{i}", shares) for i in range(5000))

    assert assign_split("doc-1", shares) == assign_split("doc-1", shares)
    assert splits[Split.VAL] == pytest.approx(500, rel=0.15)
    assert splits[Split.TEST] == pytest.approx(1000, rel=0.15)


@pytest.mark.parametrize(("val", "test"), [(-0.1, 0.1), (0.5, 0.5)])
def test_invalid_shares(val: float, test: float) -> None:
    with pytest.raises(ValueError, match="shares"):
        SplitShares(val=val, test=test)
