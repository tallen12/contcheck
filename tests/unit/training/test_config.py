from pathlib import Path

import pytest
from pydantic import ValidationError

from contcheck.training.config import TrainConfig, load_train_config


def test_repository_config_loads() -> None:
    config = load_train_config(Path(__file__).parents[3] / "configs" / "train.yaml")

    assert config.encoder == "google/electra-small-discriminator"
    assert config.window == 256
    assert not config.use_doc


def test_defaults_match_the_plan() -> None:
    config = TrainConfig()

    assert (config.encoder_lr, config.head_lr, config.batch_size) == (3e-5, 1e-3, 32)
    assert config.outside_weight == 0.2


@pytest.mark.parametrize("changes", [{"stride": 255, "window": 256}, {"epochs": 0}, {"nope": 1}])
def test_invalid_configs(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(changes)
