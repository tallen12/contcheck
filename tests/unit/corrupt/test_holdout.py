import pytest

from contcheck.corrupt.holdout import pair_position, split_confusions

CONFUSIONS = {
    "then": ("than",),
    "than": ("then",),
    "cite": ("site", "sight"),
    "site": ("cite", "sight"),
    "sight": ("cite", "site"),
}


def test_pair_position_ignores_order() -> None:
    assert pair_position("then", "than", seed=1) == pair_position("than", "then", seed=1)
    assert pair_position("then", "than", seed=1) != pair_position("then", "than", seed=2)


def test_split_moves_whole_pairs_in_both_directions() -> None:
    for seed in range(20):
        split = split_confusions(CONFUSIONS, share=0.5, seed=seed)
        for word, alternatives in split.held_out.items():
            for alternative in alternatives:
                assert word in split.held_out[alternative]
                assert alternative not in split.train.get(word, ())
        pairs = sum(len(a) for a in split.train.values()) + sum(
            len(a) for a in split.held_out.values()
        )
        assert pairs == sum(len(a) for a in CONFUSIONS.values())


def test_no_share_holds_nothing_out() -> None:
    split = split_confusions(CONFUSIONS, share=0.0, seed=1)

    assert split.held_out == {}
    assert split.train == CONFUSIONS


def test_share_must_leave_training_pairs() -> None:
    with pytest.raises(ValueError, match="share"):
        split_confusions(CONFUSIONS, share=1.0, seed=1)
