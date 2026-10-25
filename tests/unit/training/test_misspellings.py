import pytest

from contcheck.corpus.record import Record
from contcheck.taxonomy.bio import WordSpan
from contcheck.taxonomy.kinds import ErrorKind
from contcheck.training.misspellings import (
    MisspellingProbe,
    build_probes,
    load_misspellings,
    parse_misspellings,
    probe_recall,
)
from tests.conftest import make_record


def test_parse_maps_correct_words_to_misspellings() -> None:
    parsed = parse_misspellings(["# comment", "", "Tomorrow: tomorow tommorrow  # note"])

    assert parsed == {"tomorrow": ("tomorow", "tommorrow")}


@pytest.mark.parametrize("line", ["no colon here", ": orphan", "word:"])
def test_parse_rejects_malformed_lines(line: str) -> None:
    with pytest.raises(ValueError, match="expected"):
        parse_misspellings([line])


def test_bundled_list_loads() -> None:
    misspellings = load_misspellings()

    assert misspellings["occurred"] == ("occured",)
    assert len(misspellings) > 80


def test_probes_skip_words_in_or_next_to_errors() -> None:
    noisy = make_record(0, noisy=True)
    clean = make_record(1, noisy=False)
    misspellings = {"store": ("stor",), "went": ("wnet",), "i": ("ii",)}

    probes = build_probes([noisy, clean], misspellings)

    # In the noisy record "go" (an error) sits next to "I" and "to"; "store" is free.
    assert [(p.site, p.misspelling) for p in probes] == [(4, "stor"), (1, "wnet"), (4, "stor")]
    assert probes[0].words[4] == "stor"
    assert all(p.words[p.site] == p.misspelling for p in probes)


def test_probes_cycle_variants_and_cap_per_word(records: list[Record]) -> None:
    probes = build_probes(records, {"store": ("stor", "stoer")}, per_word=3)

    assert [p.misspelling for p in probes] == ["stor", "stoer", "stor"]


def test_recall_counts_spelling_spans_on_the_site() -> None:
    probes = [MisspellingProbe(("a", "b"), 1, "b"), MisspellingProbe(("a", "b"), 0, "a")]
    predicted = [
        (WordSpan(1, 2, ErrorKind.SPELLING),),
        (WordSpan(0, 1, ErrorKind.REAL_WORD),),
    ]

    assert probe_recall(probes, predicted) == 0.5
    assert probe_recall([], []) == 0.0
