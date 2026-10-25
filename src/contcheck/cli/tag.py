"""``contcheck tag``: flag errors in text with a PyTorch checkpoint.

This is the proof of concept's way to try the model; M6's ONNX ``check()`` replaces it
for production use, and it stays as a debugging command.
"""

from collections.abc import Sequence
from pathlib import Path

from contcheck.analysis.spacy_analyzer import load_spacy_analyzer
from contcheck.cli.device import CPU
from contcheck.infer.decode import decode
from contcheck.infer.models import SpanError
from contcheck.infer.predict import predict_paragraphs
from contcheck.model.checkpoint import load_checkpoint
from contcheck.model.tagger import TorchTagger


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
    threshold: float = 0.5,
    device: str = CPU,
) -> None:
    """Flag the errors the tagger finds in ``text``.

    Parameters
    ----------
    text
        Text to check; one paragraph or more.
    checkpoint
        Checkpoint directory written by ``contcheck train``.
    threshold
        Minimum span probability to flag.
    device
        PyTorch device; the CPU is fast enough for a few sentences.
    """
    loaded = load_checkpoint(checkpoint, device=device)
    words = list(load_spacy_analyzer().analyze(text).words)
    (probabilities,) = predict_paragraphs(
        [words],
        tagger=TorchTagger(loaded.model, device),
        tokenizer=loaded.tokenizer,
        window=loaded.info.window,
        stride=loaded.info.stride,
        batch_size=8,
    )
    print(format_flags(words, decode(probabilities, default_threshold=threshold)))
