from pathlib import Path

import pytest

from contcheck.corrupt.confusables import Confusion, ConfusionMap, ConfusionSource
from contcheck.corrupt.confusion_cache import cache_key, load_confusion_map, save_confusion_map

MAP = ConfusionMap.of(
    [
        Confusion("then", "than", 2.5, ConfusionSource.HAND_WRITTEN),
        Confusion("than", "then", 2.5, ConfusionSource.HAND_WRITTEN),
        Confusion("form", "from", 0.75, ConfusionSource.LOOK_ALIKE),
    ]
)


def test_saved_map_loads_back_with_its_key(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "confusions.json"
    save_confusion_map(path, MAP, key="k1")

    assert load_confusion_map(path, key="k1") == MAP


def test_missing_file_or_other_key_is_a_miss(tmp_path: Path) -> None:
    path = tmp_path / "confusions.json"

    assert load_confusion_map(path, key="k1") is None
    save_confusion_map(path, MAP, key="k1")
    assert load_confusion_map(path, key="k2") is None


def test_a_file_that_is_not_a_map_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "confusions.json"
    path.write_text('{"nope": 1}', encoding="utf-8")

    with pytest.raises(ValueError):
        load_confusion_map(path, key="k1")


def test_cache_key_depends_on_every_part_and_their_order() -> None:
    assert cache_key(["a", "b"]) == cache_key(["a", "b"])
    assert cache_key(["a", "b"]) != cache_key(["b", "a"])
    assert cache_key(["ab"]) != cache_key(["a", "b"])
