"""The bundled list of common real misspellings, for the real-misspelling probe check.

Non-word misspellings are the SymSpell pre-pass's job, so this check mainly shows how
little the tagger relies on spelling knowledge; see ``design-docs/poc-plan.md``.
"""

from collections.abc import Iterable
from importlib.resources import files

MISSPELLINGS = "natural_misspellings.txt"


def parse_misspellings(lines: Iterable[str]) -> dict[str, tuple[str, ...]]:
    """Map each correct word to its misspellings, from ``correct: miss miss`` lines."""
    misspellings: dict[str, tuple[str, ...]] = {}
    for line in lines:
        content = line.split("#", 1)[0].strip()
        if not content:
            continue
        correct, sep, rest = content.partition(":")
        variants = tuple(rest.split())
        if not sep or not correct.strip() or not variants:
            raise ValueError(f"expected 'correct: misspelling ...', got {line!r}")
        misspellings[correct.strip().lower()] = variants
    return misspellings


def load_misspellings() -> dict[str, tuple[str, ...]]:
    """The misspelling list bundled with the package."""
    text = (files("contcheck.training") / "data" / MISSPELLINGS).read_text(encoding="utf-8")
    return parse_misspellings(text.splitlines())
