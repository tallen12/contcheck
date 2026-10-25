"""Reading and writing JSONL files of validated models, one model per line."""

from collections.abc import Iterable, Iterator
from pathlib import Path

from pydantic import BaseModel


def read_jsonl[M: BaseModel](path: Path, model: type[M]) -> Iterator[M]:
    """Yield one validated ``model`` per non-blank line of ``path``.

    Raises ``ValueError`` naming the file and line when a line does not validate.
    """
    with path.open(encoding="utf-8") as file:
        for number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield model.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error


def write_jsonl(path: Path, items: Iterable[BaseModel]) -> int:
    """Write ``items`` to ``path``, creating parent directories. Returns the count."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for item in items:
            file.write(item.model_dump_json())
            file.write("\n")
            count += 1
    return count
