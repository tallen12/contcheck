"""Distribution and quality report over a record file."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from contcheck.corpus.record import Record, Split, revert_edits
from contcheck.taxonomy.kinds import ErrorKind, Formality


@dataclass(frozen=True, slots=True, kw_only=True)
class Report:
    """Counts per split, per kind and per number of errors in a paragraph."""

    records: int
    splits: Mapping[Split, int]
    kinds: Mapping[ErrorKind, int]
    errors_per_paragraph: Mapping[int, int]
    words: int
    round_trip_failures: int
    informal_edits: int = 0
    """Edits marked ``Formality.INFORMAL``, out of all edits in ``kinds``."""


def round_trips(record: Record) -> bool:
    """Whether undoing the edits gives the clean text, ignoring whitespace."""
    restored = revert_edits(record.words, record.edits)
    return "".join(restored) == "".join(record.clean.split())


def build_report(records: Iterable[Record]) -> Report:
    """Count everything the report shows in one pass."""
    count = 0
    words = 0
    failures = 0
    informal = 0
    splits: Counter[Split] = Counter()
    kinds: Counter[ErrorKind] = Counter()
    errors: Counter[int] = Counter()
    for record in records:
        count += 1
        words += len(record.words)
        splits[record.split] += 1
        kinds.update(edit.kind for edit in record.edits)
        informal += sum(edit.formality is Formality.INFORMAL for edit in record.edits)
        errors[len(record.edits)] += 1
        failures += not round_trips(record)
    return Report(
        records=count,
        splits=dict(splits),
        kinds=dict(kinds),
        errors_per_paragraph=dict(errors),
        words=words,
        round_trip_failures=failures,
        informal_edits=informal,
    )


def _share(part: float, whole: float) -> str:
    return f"{part / whole:6.1%}" if whole else "     -"


def format_report(
    report: Report,
    *,
    target_errors: Mapping[int, float] | None = None,
    target_kinds: Mapping[ErrorKind, float] | None = None,
) -> str:
    """A plain-text table, with the config's target shares next to actual ones if given."""
    target_errors = dict[int, float]() if target_errors is None else target_errors
    target_kinds = dict[ErrorKind, float]() if target_kinds is None else target_kinds
    lines = [
        f"records: {report.records}   words: {report.words}   "
        f"round-trip failures: {report.round_trip_failures}   "
        f"informal edits: {report.informal_edits}",
        "",
        "split    records",
    ]
    lines += [f"{split.value:<8} {report.splits.get(split, 0):>7}" for split in Split]

    total_errors = sum(report.errors_per_paragraph.values())
    error_total_target = sum(target_errors.values())
    lines += ["", "errors  paragraphs   share  target"]
    for n in sorted(set(report.errors_per_paragraph) | set(target_errors)):
        actual = report.errors_per_paragraph.get(n, 0)
        target = _share(target_errors.get(n, 0.0), error_total_target)
        lines.append(f"{n:>6}  {actual:>10}  {_share(actual, total_errors)}  {target}")

    total_kinds = sum(report.kinds.values())
    kind_total_target = sum(target_kinds.values())
    lines += ["", "kind                    edits   share  target"]
    for kind in ErrorKind:
        actual = report.kinds.get(kind, 0)
        weight = target_kinds.get(kind, 0.0)
        if actual or weight:
            target = _share(weight, kind_total_target)
            lines.append(f"{kind.value:<22} {actual:>6}  {_share(actual, total_kinds)}  {target}")
    return "\n".join(lines)
