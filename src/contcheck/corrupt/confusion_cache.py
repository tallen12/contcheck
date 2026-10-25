"""A built ``ConfusionMap`` saved to disk, so commands that need it skip the rebuild.

Building the map runs a look-alike search over the 25,000 most frequent words, which
``corrupt`` and ``evaluate`` would otherwise repeat on every run. The file records a key
naming everything the map was built from; a different key means a stale file.
"""

import hashlib
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from contcheck.corrupt.confusables import Confusion, ConfusionMap, ConfusionSource


class _CachedConfusion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    word: str
    alternative: str
    weight: float
    source: ConfusionSource


class _CachedMap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str
    confusions: tuple[_CachedConfusion, ...]


def cache_key(parts: Iterable[str]) -> str:
    """A digest of every input the map depends on, such as versions and data files."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def save_confusion_map(path: Path, confusions: ConfusionMap, *, key: str) -> None:
    """Write ``confusions`` with the ``key`` of the inputs they were built from.

    The file appears whole or not at all: it is written beside ``path`` and then moved
    into place, so a concurrent reader or an interrupted write never sees half a map.
    """
    cached = _CachedMap(
        key=key,
        confusions=tuple(
            _CachedConfusion(
                word=c.word, alternative=c.alternative, weight=c.weight, source=c.source
            )
            for c in confusions
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(cached.model_dump_json() + "\n")
        Path(temporary).replace(path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_confusion_map(path: Path, *, key: str) -> ConfusionMap | None:
    """The map saved at ``path``, or ``None`` when there is none or it has another key.

    Raises ``ValueError`` if the file exists but is not a saved map.
    """
    if not path.exists():
        return None
    cached = _CachedMap.model_validate_json(path.read_text(encoding="utf-8"))
    if cached.key != key:
        return None
    return ConfusionMap.of(
        Confusion(c.word, c.alternative, c.weight, c.source) for c in cached.confusions
    )
