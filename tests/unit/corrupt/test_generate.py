import pytest

from contcheck.corpus.documents import Document
from contcheck.corpus.record import Split, revert_edits
from contcheck.corrupt.config import GenConfig
from contcheck.corrupt.context import CorruptionContext
from contcheck.corrupt.determiner import DeterminerCorruptor
from contcheck.corrupt.edit import Edit
from contcheck.corrupt.generate import RoundTripError, derive_seed, generate_records
from contcheck.corrupt.protocols import Rng
from contcheck.corrupt.real_word import RealWordCorruptor, parse_confusion_sets
from contcheck.corrupt.registry import CorruptorRegistry
from contcheck.corrupt.spelling import SpellingCorruptor
from contcheck.corrupt.verb_tense import VerbTenseCorruptor
from contcheck.taxonomy.bio import labels_to_spans
from contcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import FakeAnalyzer, FakeDictionary, FakeInflector

DOCUMENTS = [
    Document(
        "story-0000",
        Split.TRAIN,
        ("Yesterday she walked to the store and bought apples.", "Then they went to the park."),
    ),
    Document("story-0001", Split.VAL, ("The dog is in the garden.", "It walks in the morning.")),
]
CONFIG = GenConfig(
    seed=13,
    variants_per_paragraph=3,
    errors_per_paragraph={0: 0.2, 1: 0.4, 2: 0.4},
    kind_weights={"default": 1},
)


@pytest.fixture
def registry(dictionary: FakeDictionary, inflector: FakeInflector) -> CorruptorRegistry:
    return CorruptorRegistry(
        [
            SpellingCorruptor(dictionary),
            RealWordCorruptor(parse_confusion_sets(["then than", "form from", "to too"])),
            DeterminerCorruptor(),
            VerbTenseCorruptor(inflector),
        ]
    )


def test_records_are_labeled_and_round_trip(
    analyzer: FakeAnalyzer, registry: CorruptorRegistry
) -> None:
    records = list(generate_records(DOCUMENTS, analyzer=analyzer, registry=registry, config=CONFIG))

    assert len(records) == 4 * 3
    assert {(r.doc_id, r.split) for r in records} == {
        ("story-0000", Split.TRAIN),
        ("story-0001", Split.VAL),
    }
    for record in records:
        clean_words = analyzer.analyze(record.clean).words
        assert revert_edits(record.words, record.edits) == clean_words
        assert analyzer.analyze(record.noisy).words == record.words
        spans = labels_to_spans(record.labels)
        assert len(spans) == len(record.edits)
        assert [s.kind for s in spans] == [e.kind for e in record.edits]
    assert any(record.edits for record in records)


def test_generation_is_deterministic(analyzer: FakeAnalyzer, registry: CorruptorRegistry) -> None:
    def run() -> list[str]:
        records = generate_records(DOCUMENTS, analyzer=analyzer, registry=registry, config=CONFIG)
        return [r.model_dump_json() for r in records]

    assert run() == run()


def test_seed_changes_the_output(analyzer: FakeAnalyzer, registry: CorruptorRegistry) -> None:
    other = CONFIG.model_copy(update={"seed": 14})

    first = [
        r.noisy
        for r in generate_records(DOCUMENTS, analyzer=analyzer, registry=registry, config=CONFIG)
    ]
    second = [
        r.noisy
        for r in generate_records(DOCUMENTS, analyzer=analyzer, registry=registry, config=other)
    ]

    assert first != second


def test_derive_seed_depends_on_every_part() -> None:
    seeds = {derive_seed(1, "d", 0, 0), derive_seed(2, "d", 0, 0), derive_seed(1, "e", 0, 0)}
    seeds |= {derive_seed(1, "d", 1, 0), derive_seed(1, "d", 0, 1)}

    assert len(seeds) == 5


class Broken:
    """Claims a deletion of one word but reports a different original word."""

    kind = ErrorKind.MISSING_WORD

    def corrupt(self, context: CorruptionContext, rng: Rng) -> Edit | None:
        return Edit(1, 2, self.kind, Operation.MISSING, ("nonsense",), ())


def test_round_trip_failures_raise(analyzer: FakeAnalyzer) -> None:
    config = CONFIG.model_copy(update={"errors_per_paragraph": {1: 1.0}})

    with pytest.raises(RoundTripError, match="story-0000"):
        list(
            generate_records(
                DOCUMENTS, analyzer=analyzer, registry=CorruptorRegistry([Broken()]), config=config
            )
        )
