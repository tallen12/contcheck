import random
from pathlib import Path

import pytest

from contcheck.analysis.protocols import Inflector, TextAnalyzer
from contcheck.annotation.models import Annotated
from contcheck.annotation.protocols import Annotator
from contcheck.corpus.record import EditRecord, Record
from contcheck.corrupt.protocols import Rng
from contcheck.llm.models import ChatMessage, ChatOptions, LLMError, Role
from contcheck.llm.protocols import LLMClient
from contcheck.model.checkpoint import load_checkpoint
from contcheck.spelling.protocols import Dictionary
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import (
    FakeAnalyzer,
    FakeAnnotator,
    FakeDictionary,
    FakeInflector,
    FakeLLMClient,
    edit_distance,
)


def test_fakes_implement_their_protocols(
    analyzer: FakeAnalyzer,
    inflector: FakeInflector,
    dictionary: FakeDictionary,
    annotator: FakeAnnotator,
    rng: random.Random,
) -> None:
    implementations: tuple[TextAnalyzer, Inflector, Dictionary, Annotator, Rng] = (
        analyzer,
        inflector,
        dictionary,
        annotator,
        rng,
    )

    assert len(implementations) == 5


def test_fake_annotator_finds_missing_unnecessary_and_unchanged_spans(
    annotator: FakeAnnotator,
) -> None:
    missing = annotator.annotate(("a", "dog"), ("a", "big", "dog"))
    extra = annotator.annotate(("a", "big", "dog"), ("a", "dog"))

    assert missing == Annotated(
        (EditRecord(start=1, end=1, kind=ErrorKind.REAL_WORD, op=Operation.MISSING, orig=("big",)),)
    )
    assert extra == Annotated(
        (EditRecord(start=1, end=2, kind=ErrorKind.REAL_WORD, op=Operation.UNNECESSARY, orig=()),)
    )
    assert annotator.annotate(("a",), ("a",)) == Annotated(())


def test_fake_analyzer_tags_from_the_lexicon_and_keeps_spacing(analyzer: FakeAnalyzer) -> None:
    analyzed = analyzer.analyze("Yesterday I went home. She walks.")

    assert analyzed.words == ("Yesterday", "I", "went", "home", ".", "She", "walks", ".")
    assert [t.tag for t in analyzed.tokens] == ["NN", "PRP", "VBD", "NN", ".", "PRP", "VBZ", "."]
    assert [t.lemma for t in analyzed.tokens][2] == "go"
    assert [t.sentence_start for t in analyzed.tokens] == [
        True, False, False, False, False, True, False, False,
    ]  # fmt: skip
    assert [t.finite for t in analyzed.tokens].count(True) == 2
    assert analyzed.text == "Yesterday I went home. She walks."


def test_fake_analyzer_analyzes_many_in_order(analyzer: FakeAnalyzer) -> None:
    results = list(analyzer.analyze_many(["a dog", "the dogs"]))

    assert [r.words for r in results] == [("a", "dog"), ("the", "dogs")]


def test_fake_inflector_returns_table_forms(inflector: FakeInflector) -> None:
    assert inflector.inflect("go", "VBZ") == ("goes",)
    assert inflector.inflect("fly", "VBZ") == ()


def test_fake_dictionary_looks_up_ignoring_case(dictionary: FakeDictionary) -> None:
    assert dictionary.contains("The")
    assert not dictionary.contains("teh")


def test_fake_dictionary_suggests_closest_then_most_frequent(dictionary: FakeDictionary) -> None:
    suggestions = dictionary.suggestions("teh", max_distance=2)

    assert suggestions[0].term == "the"
    assert all(s.distance <= 2 for s in suggestions)
    assert [s.distance for s in suggestions] == sorted(s.distance for s in suggestions)


def test_edit_distance() -> None:
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_distance("same", "same") == 0


def test_records_alternate_noisy_and_clean(records: list[Record]) -> None:
    assert len(records) == 16
    assert [bool(r.edits) for r in records[:4]] == [True, False, True, False]
    assert records[0].labels[1] == "B-VerbTense"


def test_checkpoint_fixture_loads(checkpoint: Path) -> None:
    loaded = load_checkpoint(checkpoint)

    assert loaded.info.window == 16
    assert loaded.tokenizer.encode_words(["go"]).input_ids == (6,)


def test_fake_llm_client_replays_its_script_and_records_requests() -> None:
    fake = FakeLLMClient(["first", LLMError("down"), "last"])
    client: LLMClient = fake
    messages = [ChatMessage(Role.USER, "hi")]

    first = client.complete(messages, options=ChatOptions())
    with pytest.raises(LLMError):
        client.complete(messages, options=ChatOptions())
    last = [client.complete(messages, options=ChatOptions()) for _ in range(2)]

    assert first == "first"
    assert last == ["last", "last"]
    assert len(fake.requests) == 4
    assert fake.requests[0][0] == (ChatMessage(Role.USER, "hi"),)
