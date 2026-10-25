"""Per-kind agreement between a corruptor's labels and ERRANT's mapped kinds.

The label-agreement check from the implementation plan (Phase 1, target >= 90%): revert
each record's edits to get the clean words, run ERRANT on (noisy, clean), and compare
every corruptor edit with the ERRANT edit on the same noisy range.

Run with ``uv run python scripts/errant_agreement.py RECORDS.jsonl [--limit N]``.
"""

import argparse
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from contcheck.annotation.convert import to_annotation
from contcheck.annotation.errant_annotator import load_errant_annotator
from contcheck.annotation.models import Unmappable
from contcheck.corpus.record import Record, revert_edits
from contcheck.taxonomy.errant_map import Dropped, Mapped, map_edit

EXAMPLES_PER_KIND = 4


def read_records(path: Path, limit: int) -> list[Record]:
    """Every ``len // limit``-th record with edits, so the sample spans the file."""
    with path.open(encoding="utf-8") as lines:
        records = [Record.model_validate_json(line) for line in lines]
    with_edits = [record for record in records if record.edits]
    step = max(1, len(with_edits) // limit)
    return with_edits[::step][:limit]


def outcomes(records: list[Record], rejected: list[Record]) -> Iterator[tuple[str, str, str]]:
    """Yield (our kind, ERRANT's outcome, example) per corruptor edit.

    Records the annotator would reject as unmappable are appended to ``rejected``.
    """
    annotator = load_errant_annotator()
    pairs = [(record.words, revert_edits(record.words, record.edits)) for record in records]
    for record, (edits, noisy, clean) in zip(
        records, annotator.typed_edits_many(pairs), strict=True
    ):
        if isinstance(to_annotation(edits, noisy, clean), Unmappable):
            rejected.append(record)
        by_range = {(edit.noisy_start, edit.noisy_end): edit for edit in edits}
        for ours in record.edits:
            example = (
                f"{' '.join(noisy[ours.start : ours.end]) or '∅'} -> {' '.join(ours.orig) or '∅'}"
            )
            theirs = by_range.get((ours.start, ours.end))
            if theirs is None:
                overlapping = [
                    e.code for e in edits if e.noisy_start <= ours.end and ours.start <= e.noisy_end
                ]
                yield ours.kind, f"span differs {overlapping}", example
                continue
            match map_edit(theirs.code, theirs.evidence):
                case Mapped(kind=kind):
                    outcome = "agree" if kind == ours.kind else f"{kind} ({theirs.code})"
                case Dropped():
                    outcome = f"dropped ({theirs.code})"
            yield ours.kind, outcome, f"{example} [{theirs.code}]"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", type=Path)
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    records = read_records(args.records, args.limit)
    totals: Counter[str] = Counter()
    agree: Counter[str] = Counter()
    misses: defaultdict[str, Counter[str]] = defaultdict(Counter)
    examples: defaultdict[str, list[str]] = defaultdict(list)
    rejected: list[Record] = []
    for kind, outcome, example in outcomes(records, rejected):
        totals[kind] += 1
        if outcome == "agree":
            agree[kind] += 1
            continue
        misses[kind][outcome] += 1
        if len(examples[kind]) < EXAMPLES_PER_KIND:
            examples[kind].append(f"{example} => {outcome}")
    edits = sum(totals.values())
    print(f"{len(records)} records, {edits} edits, {sum(agree.values()) / edits:.1%} agree")
    print(f"{len(rejected)} records ({len(rejected) / len(records):.1%}) would be unmappable\n")
    print("| Kind | Edits | Agree | Most common disagreement |")
    print("| --- | ---: | ---: | --- |")
    for kind in sorted(totals, key=lambda k: agree[k] / totals[k]):
        common = ", ".join(f"{o} x{n}" for o, n in misses[kind].most_common(3))
        print(f"| {kind} | {totals[kind]} | {agree[kind] / totals[kind]:.0%} | {common} |")
    print()
    for kind in sorted(examples, key=lambda k: agree[k] / totals[k]):
        print(f"{kind}:")
        for example in examples[kind]:
            print(f"  {example}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
