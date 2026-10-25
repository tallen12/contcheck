"""Scoring a tagger or a baseline on a record file, and the results log."""

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from contcheck.corpus.record import Record
from contcheck.infer.decode import DEFAULT_THRESHOLD, decode
from contcheck.infer.predict import predict_paragraphs
from contcheck.model.protocols import Tagger, Tokenizer
from contcheck.spelling.protocols import Dictionary
from contcheck.taxonomy.bio import WordSpan, labels_to_spans
from contcheck.taxonomy.kinds import ErrorKind
from contcheck.training.metrics import ParagraphSpans, score
from contcheck.training.models import EvalResult, Scores
from contcheck.training.probes import CheckName, ProbeSet, score_probes


def evaluate_tagger(
    records: Sequence[Record],
    *,
    tagger: Tagger,
    tokenizer: Tokenizer,
    window: int,
    stride: int,
    batch_size: int,
    name: str,
    dataset: str,
    thresholds: Mapping[ErrorKind, float] | None = None,
    default_threshold: float = DEFAULT_THRESHOLD,
    checks: Mapping[CheckName, ProbeSet] | None = None,
) -> EvalResult:
    """Tag every record's noisy words and score the spans against its labels.

    Each of ``checks`` is a probe check, scored on its own probes.
    """

    def spans(paragraphs: Sequence[Sequence[str]]) -> list[tuple[WordSpan, ...]]:
        probabilities = predict_paragraphs(
            paragraphs,
            tagger=tagger,
            tokenizer=tokenizer,
            window=window,
            stride=stride,
            batch_size=batch_size,
        )
        return [
            tuple(
                WordSpan(e.start, e.end, e.kind)
                for e in decode(p, thresholds=thresholds, default_threshold=default_threshold)
            )
            for p in probabilities
        ]

    predicted = spans([record.words for record in records])
    paragraphs = (
        ParagraphSpans(len(record.words), tuple(labels_to_spans(record.labels)), found)
        for record, found in zip(records, predicted, strict=True)
    )
    result = score(paragraphs, name=name, dataset=dataset)
    return _with_checks(
        result, checks, lambda probe_set: spans([probe.words for probe in probe_set.probes])
    )


def symspell_baseline(
    records: Sequence[Record],
    *,
    dictionary: Dictionary,
    dataset: str,
    checks: Mapping[CheckName, ProbeSet] | None = None,
) -> EvalResult:
    """Flag every alphabetic, lowercase-initial word the dictionary does not know as Spelling.

    This is what a dictionary-only checker does; capitalized words are skipped as likely
    names. Only its Spelling scores are comparable with the tagger's.
    """

    def spans(words: Sequence[str]) -> tuple[WordSpan, ...]:
        return tuple(
            WordSpan(i, i + 1, ErrorKind.SPELLING)
            for i, word in enumerate(words)
            if word.isalpha() and word[0].islower() and not dictionary.contains(word)
        )

    paragraphs = (
        ParagraphSpans(
            len(record.words), tuple(labels_to_spans(record.labels)), spans(record.words)
        )
        for record in records
    )
    result = score(paragraphs, name="symspell-baseline", dataset=dataset)
    return _with_checks(
        result, checks, lambda probe_set: [spans(probe.words) for probe in probe_set.probes]
    )


def _with_checks(
    result: EvalResult,
    checks: Mapping[CheckName, ProbeSet] | None,
    predict: Callable[[ProbeSet], Sequence[Sequence[WordSpan]]],
) -> EvalResult:
    if checks is None or len(checks) == 0:
        return result
    scores = {
        name: score_probes(probe_set, predict(probe_set)) for name, probe_set in checks.items()
    }
    return result.model_copy(update={"checks": scores})


def append_result(path: Path, result: EvalResult) -> None:
    """Append ``result`` as one JSON line to the results log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(result.model_dump_json() + "\n")


def _row(label: str, scores: Scores) -> str:
    gold = scores.true_positives + scores.false_negatives
    return f"{label:<22} {scores.precision:6.3f} {scores.recall:6.3f} {scores.f05:6.3f} {gold:>7}"


def format_result(result: EvalResult) -> str:
    """Per-kind precision, recall and F0.5, then overall, detection and false alarms."""
    lines = [
        f"{result.name} on {result.dataset}",
        f"{'kind':<22} {'P':>6} {'R':>6} {'F0.5':>6} {'gold':>7}",
    ]
    lines += [
        _row(kind.value, result.per_kind[kind]) for kind in ErrorKind if kind in result.per_kind
    ]
    lines += [
        _row("overall", result.overall),
        _row("detection only", result.detection),
        f"false alarms per 1,000 clean words: {result.false_alarms_per_1000:.2f} "
        f"({result.clean_words} words)",
    ]
    for name, check in result.checks.items():
        lines.append(
            f"{name}: flagged {check.detected:.3f}, with the right kind {check.correct_kind:.3f} "
            f"({check.probes} probes)"
        )
    return "\n".join(lines)
