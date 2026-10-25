from pathlib import Path

import pytest
from pydantic import ValidationError

from contcheck.corrupt.config import GenConfig, load_gen_config
from contcheck.taxonomy.kinds import ErrorKind

CONFIG = """\
seed: 13
variants_per_paragraph: 2
errors_per_paragraph: {0: 0.15, 1: 0.40, 2: 0.30, 3: 0.15}
kind_weights:
  Spelling: 3
  RealWord: 2
  default: 1
"""


def test_load_reads_yaml(tmp_path: Path) -> None:
    path = tmp_path / "gen.yaml"
    path.write_text(CONFIG, encoding="utf-8")

    config = load_gen_config(path)

    assert config.seed == 13
    assert config.errors_per_paragraph == {0: 0.15, 1: 0.40, 2: 0.30, 3: 0.15}
    assert config.weight(ErrorKind.SPELLING) == 3
    assert config.weight(ErrorKind.VERB_TENSE) == 1
    assert config.adjacent_share == 0.0


def test_weight_is_zero_without_default() -> None:
    config = GenConfig(
        seed=1,
        variants_per_paragraph=1,
        errors_per_paragraph={1: 1.0},
        kind_weights={"Spelling": 1},
    )

    assert config.weight(ErrorKind.DETERMINER) == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"kind_weights": {"Nope": 1}},
        {"kind_weights": {"Spelling": -1}},
        {"kind_weights": {"Spelling": 0}},
        {"errors_per_paragraph": {}},
        {"errors_per_paragraph": {-1: 1.0}},
        {"variants_per_paragraph": 0},
        {"adjacent_share": 1.5},
        {"extra": True},
    ],
)
def test_invalid_configs_are_rejected(changes: dict[str, object]) -> None:
    data: dict[str, object] = {
        "seed": 1,
        "variants_per_paragraph": 1,
        "errors_per_paragraph": {1: 1.0},
        "kind_weights": {"default": 1},
    }
    data.update(changes)

    with pytest.raises(ValidationError):
        GenConfig.model_validate(data)
