import pytest
from pydantic import ValidationError

from contcheck.corpus.record import EditRecord, Record, RecordSource, Split, revert_edits
from contcheck.taxonomy.kinds import ErrorKind, Formality, Operation


def record(**changes: object) -> Record:
    data: dict[str, object] = {
        "doc_id": "wiki-1-0000",
        "paragraph": 0,
        "variant": 0,
        "split": "train",
        "source": "rule",
        "clean": "Yesterday I went home.",
        "noisy": "Yesterday I go home.",
        "words": ["Yesterday", "I", "go", "home", "."],
        "labels": ["O", "O", "B-VerbTense", "O", "O"],
        "edits": [{"start": 2, "end": 3, "kind": "VerbTense", "op": "R", "orig": ["went"]}],
    }
    data.update(changes)
    return Record.model_validate(data)


def test_record_round_trips_through_json() -> None:
    original = record()

    assert Record.model_validate_json(original.model_dump_json()) == original
    assert original.split is Split.TRAIN
    assert original.source is RecordSource.RULE
    assert original.edits[0].kind is ErrorKind.VERB_TENSE


@pytest.mark.parametrize(
    "changes",
    [
        {"labels": ["O"]},
        {"labels": ["O", "O", "B-Nope", "O", "O"]},
        {"edits": [{"start": 4, "end": 6, "kind": "Spelling", "op": "R", "orig": ["x"]}]},
        {"split": "dev"},
    ],
)
def test_invalid_records_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        record(**changes)


def test_only_missing_edits_have_an_empty_range() -> None:
    EditRecord(start=1, end=1, kind=ErrorKind.DETERMINER, op=Operation.MISSING, orig=("the",))
    with pytest.raises(ValidationError, match="Missing"):
        EditRecord(start=1, end=1, kind=ErrorKind.DETERMINER, op=Operation.REPLACEMENT, orig=())
    with pytest.raises(ValidationError, match="Missing"):
        EditRecord(start=1, end=2, kind=ErrorKind.DETERMINER, op=Operation.MISSING, orig=())
    with pytest.raises(ValidationError, match="invalid range"):
        EditRecord(start=2, end=1, kind=ErrorKind.DETERMINER, op=Operation.REPLACEMENT, orig=())


def test_formality_defaults_to_error_for_older_records() -> None:
    edit = EditRecord.model_validate(
        {"start": 2, "end": 3, "kind": "VerbTense", "op": "R", "orig": ["went"]}
    )
    informal = EditRecord.model_validate(
        {
            "start": 2,
            "end": 3,
            "kind": "VerbTense",
            "op": "R",
            "orig": ["went"],
            "formality": "informal",
        }
    )

    assert edit.formality is Formality.ERROR
    assert informal.formality is Formality.INFORMAL


def test_revert_undoes_a_replacement_before_a_missing_edit_at_its_start() -> None:
    edits = [
        EditRecord(start=1, end=1, kind=ErrorKind.PREPOSITION, op=Operation.MISSING, orig=("IN",)),
        EditRecord(
            start=1, end=2, kind=ErrorKind.SPELLING, op=Operation.REPLACEMENT, orig=("Rehab",)
        ),
    ]

    assert revert_edits(["track", "INRihab", "."], edits) == ("track", "IN", "Rehab", ".")
