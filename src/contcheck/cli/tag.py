"""``contcheck tag``: flag errors in text with a PyTorch checkpoint.

This is the proof of concept's way to try the model; M6's ONNX ``check()`` replaces it
for production use, and it stays as a debugging command.
"""

from collections.abc import Sequence
from pathlib import Path

from contcheck.analysis.spacy_analyzer import load_spacy_analyzer
from contcheck.cli.device import CPU
from contcheck.corpus.readers import PARAGRAPH_BREAK, normalize_whitespace
from contcheck.infer.decode import DEFAULT_THRESHOLD, decode
from contcheck.infer.models import SpanError
from contcheck.infer.predict import predict_paragraphs
from contcheck.model.checkpoint import load_checkpoint
from contcheck.model.tagger import TorchTagger


def split_paragraphs(text: str) -> list[str]:
    """Paragraphs of ``text``, split on blank lines as the corpus readers do, since the
    model was trained on one paragraph at a time."""
    blocks = (normalize_whitespace(block) for block in PARAGRAPH_BREAK.split(text))
    return [block for block in blocks if block]


def format_flags(words: Sequence[str], errors: Sequence[SpanError]) -> str:
    """One line per flagged span: its words, kind and probability; or a note if none."""
    if not errors:
        return "no errors flagged"
    width = max(len(" ".join(words[e.start : e.end])) for e in errors)
    return "\n".join(
        f"{' '.join(words[e.start : e.end]):<{width}}  {e.kind.value:<22} {e.probability:.2f}"
        for e in errors
    )


def tag(
    text: str,
    /,
    *,
    checkpoint: Path = Path("runs/baseline"),
    threshold: float = DEFAULT_THRESHOLD,
    device: str = CPU,
) -> None:
    """Flag the errors the tagger finds in ``text``.

    Parameters
    ----------
    text
        Text to check; paragraphs separated by blank lines are checked one at a time.
    checkpoint
        Checkpoint directory written by ``contcheck train``.
    threshold
        Minimum span probability to flag.
    device
        PyTorch device; the CPU is fast enough for a few sentences.
    """
    loaded = load_checkpoint(checkpoint, device=device)
    analyzed = load_spacy_analyzer().analyze_many(split_paragraphs(text))
    paragraphs = [list(paragraph.words) for paragraph in analyzed]
    probabilities = predict_paragraphs(
        paragraphs,
        tagger=TorchTagger(loaded.model, device),
        tokenizer=loaded.tokenizer,
        window=loaded.info.window,
        stride=loaded.info.stride,
        batch_size=8,
    )
    flags = [
        format_flags(words, decode(found, default_threshold=threshold))
        for words, found in zip(paragraphs, probabilities, strict=True)
    ]
    print("\n\n".join(flags) if flags else format_flags([], []))
