from contcheck.corpus.record import Record
from contcheck.taxonomy.bio import WordSpan
from contcheck.taxonomy.kinds import ErrorKind
from contcheck.training.probes import Probe, ProbeSet, score_probes, substitution_probes
from tests.conftest import make_record


def test_probes_skip_words_in_or_next_to_errors() -> None:
    noisy = make_record(0, noisy=True)
    clean = make_record(1, noisy=False)
    substitutions = {"store": ("stor",), "went": ("wnet",), "to": ("too",)}

    probes = substitution_probes([noisy, clean], substitutions)

    # In the noisy record "to" sits next to the error on "go"; "store" is free.
    assert [(p.site, p.replacement) for p in probes] == [
        (4, "stor"),
        (1, "wnet"),
        (2, "too"),
        (4, "stor"),
    ]
    assert all(p.words[p.site] == p.replacement for p in probes)


def test_probes_cycle_replacements_and_cap_per_word(records: list[Record]) -> None:
    probes = substitution_probes(records, {"store": ("stor", "stoer")}, per_word=3)

    assert [p.replacement for p in probes] == ["stor", "stoer", "stor"]


def test_capitalized_occurrences_are_skipped(records: list[Record]) -> None:
    assert substitution_probes(records, {"i": ("me",)}) == ()


def test_score_counts_any_flag_and_the_right_kind() -> None:
    probes = (
        Probe(("a", "b"), 1, "b"),
        Probe(("a", "b"), 0, "a"),
        Probe(("a", "b"), 0, "a"),
    )
    predicted = [
        (WordSpan(1, 2, ErrorKind.REAL_WORD),),
        (WordSpan(0, 1, ErrorKind.SPELLING),),
        (WordSpan(1, 2, ErrorKind.REAL_WORD),),
    ]

    score = score_probes(ProbeSet(ErrorKind.REAL_WORD, probes), predicted)

    assert score.probes == 3
    assert score.detected == 2 / 3
    assert score.correct_kind == 1 / 3


def test_empty_probe_set_scores_zero() -> None:
    score = score_probes(ProbeSet(ErrorKind.REAL_WORD, ()), [])

    assert (score.probes, score.detected, score.correct_kind) == (0, 0.0, 0.0)
