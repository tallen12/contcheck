import pytest

from contcheck.corpus.cleanliness import (
    ACCEPT_ALL,
    CleanlinessConfig,
    CleanlinessFilter,
    CountingScreen,
    caps_share,
    is_checked,
    longest_sentence,
    spelling_forms,
)
from contcheck.corpus.models import Clean, Rejected, RejectReason
from tests.conftest import FakeDictionary

WORDS: dict[str, int] = {
    w: 1 for w in "the cat sat on a mat and it was very happy there today".split()
}


@pytest.fixture
def screen() -> CleanlinessFilter:
    return CleanlinessFilter(FakeDictionary(WORDS), CleanlinessConfig(max_non_word_share=0.1))


def test_clean_prose_passes(screen: CleanlinessFilter) -> None:
    assert screen.screen("The cat sat on a mat, and Tom was very happy there today.") == Clean()


def test_names_and_contractions_are_not_looked_up(screen: CleanlinessFilter) -> None:
    assert screen.screen("Tom's cat sat there, didn't it? Paris was happy.") == Clean()


def test_too_many_non_words_are_rejected(screen: CleanlinessFilter) -> None:
    verdict = screen.screen("The cat sat on teh mat and it was vrey happy.")

    assert verdict == Rejected(RejectReason.NON_WORDS)


@pytest.mark.parametrize(
    ("paragraph", "reason"),
    [
        ("See <b>the</b> cat.", RejectReason.MARKUP),
        ("The cat {{citation needed}} sat.", RejectReason.MARKUP),
        ("Visit https://example.org today.", RejectReason.MARKUP),
        ("The cat &amp; the mat.", RejectReason.MARKUP),
        ("The cat | the mat.", RejectReason.MARKUP),
        ("The _cat_ sat.", RejectReason.MARKUP),
        ("- the cat sat on a mat.", RejectReason.LIST),
        ("2. The cat sat on a mat.", RejectReason.LIST),
        ("Cats • mats • hats.", RejectReason.LIST),
        ("THE CAT SAT ON A MAT today.", RejectReason.ALL_CAPS),
        (" ".join(["the cat sat"] * 30) + ".", RejectReason.LONG_SENTENCE),
    ],
)
def test_structural_checks(screen: CleanlinessFilter, paragraph: str, reason: RejectReason) -> None:
    assert screen.screen(paragraph) == Rejected(reason)


def test_non_word_share_counts_lowercase_alphabetic_words_only() -> None:
    screen = CleanlinessFilter(FakeDictionary(WORDS), CleanlinessConfig())

    assert screen.non_word_share(["The", "cat", "zzz", "it's", "42", "NASA", "km"]) == 0.5
    assert screen.non_word_share(["Paris"]) == 0.0


def test_caps_share_and_longest_sentence() -> None:
    assert caps_share(["NASA", "sent", "a", "probe", "OK"]) == 0.5
    assert caps_share([]) == 0.0
    assert longest_sentence('One two. "Three four five!" Six.') == 3


def test_config_rejects_impossible_thresholds() -> None:
    with pytest.raises(ValueError, match="shares"):
        CleanlinessConfig(max_non_word_share=1.5)
    with pytest.raises(ValueError, match="positive"):
        CleanlinessConfig(max_sentence_words=0)


def test_accept_all_keeps_everything() -> None:
    assert ACCEPT_ALL.screen("<b>teh</b>") == Clean()


def test_counting_screen_tallies_reasons(screen: CleanlinessFilter) -> None:
    counting = CountingScreen(screen)

    for paragraph in ["The cat sat.", "See <b>it</b>.", "Teh vrey zzz."]:
        counting.screen(paragraph)

    assert counting.screened == 3
    assert counting.rejected == {RejectReason.MARKUP: 1, RejectReason.NON_WORDS: 1}
    assert counting.summary().startswith("screened 3, rejected 2 (66.7%): non_words 1, markup 1")


def test_counting_screen_summary_with_nothing_screened() -> None:
    assert CountingScreen(ACCEPT_ALL).summary().startswith("screened 0, rejected 0 (0.0%)")


@pytest.mark.parametrize(
    ("word", "british"),
    [
        ("neighborhood", "neighbourhood"),
        ("favorite", "favourite"),
        ("theater", "theatre"),
        ("kilometers", "kilometres"),
        ("traveled", "travelled"),
        ("willful", "wilful"),
        ("offenses", "offences"),
        ("catalog", "catalogue"),
        ("skeptical", "sceptical"),
        ("practicing", "practising"),
        ("aluminum", "aluminium"),
        ("sulfide", "sulphide"),
        ("distill", "distil"),
        ("moms", "mums"),
    ],
)
def test_american_spellings_have_a_british_form(word: str, british: str) -> None:
    assert british in spelling_forms(word)


def test_american_spellings_count_as_known_when_the_british_form_is() -> None:
    screen = CleanlinessFilter(FakeDictionary({"neighbourhood": 1}), CleanlinessConfig())

    assert screen.knows("neighborhood")
    assert not screen.knows("nieghborhood")


def test_only_long_lowercase_ascii_words_are_checked() -> None:
    assert is_checked("cat")
    assert not is_checked("km")
    assert not is_checked("Paris")
    assert not is_checked("café")
