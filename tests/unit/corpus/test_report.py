from contcheck.corpus.record import EditRecord, Record, RecordSource, Split
from contcheck.corpus.report import build_report, format_report, round_trips
from contcheck.taxonomy.kinds import ErrorKind, Operation

TENSE = EditRecord(
    start=1, end=2, kind=ErrorKind.VERB_TENSE, op=Operation.REPLACEMENT, orig=("went",)
)
DROP = EditRecord(start=3, end=3, kind=ErrorKind.DETERMINER, op=Operation.MISSING, orig=("the",))


def record(
    edits: tuple[EditRecord, ...], *, split: Split = Split.TRAIN, clean: str = "I went to the shop"
) -> Record:
    words = ("I", "go", "to", "shop")
    labels = ["O"] * 4
    for edit in edits:
        labels[min(edit.start, 3)] = f"B-{edit.kind.value}"
    return Record(
        doc_id="d",
        paragraph=0,
        variant=0,
        split=split,
        source=RecordSource.RULE,
        clean=clean,
        noisy=" ".join(words),
        words=words,
        labels=tuple(labels),
        edits=edits,
    )


def test_round_trip_ignores_whitespace() -> None:
    assert round_trips(record((TENSE, DROP)))
    assert not round_trips(record((TENSE,)))


def test_report_counts_kinds_errors_and_failures() -> None:
    report = build_report(
        [record((TENSE, DROP)), record((TENSE, DROP), split=Split.VAL), record((TENSE,))]
    )

    assert report.records == 3
    assert report.words == 12
    assert report.splits == {Split.TRAIN: 2, Split.VAL: 1}
    assert report.kinds == {ErrorKind.VERB_TENSE: 3, ErrorKind.DETERMINER: 2}
    assert report.errors_per_paragraph == {2: 2, 1: 1}
    assert report.round_trip_failures == 1


def test_format_shows_targets_next_to_shares() -> None:
    report = build_report([record((TENSE, DROP))])

    text = format_report(
        report,
        target_errors={0: 0.5, 2: 0.5},
        target_kinds={ErrorKind.VERB_TENSE: 3.0, ErrorKind.DETERMINER: 1.0},
    )

    assert "round-trip failures: 0" in text
    assert "     0           0    0.0%   50.0%" in text
    assert "VerbTense                   1   50.0%   75.0%" in text


def test_format_without_targets() -> None:
    text = format_report(build_report([]))

    assert "records: 0" in text
    assert "VerbTense" not in text
