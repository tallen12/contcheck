"""Document statistics that separate dialogue, narration and exposition."""

from collections.abc import Sequence

from contcheck.analysis.models import AnalyzedText, DocStats

FIRST_PERSON = frozenset({"i", "me", "my", "mine", "myself", "we", "us", "our", "ours"})
QUOTE_TAGS = frozenset({"``", "''"})
QUOTE_CHARS = frozenset({'"', "“", "”"})


def doc_stats(paragraphs: Sequence[AnalyzedText]) -> DocStats:
    """Sentence count, mean sentence length in words, and the share of words that are
    first-person pronouns or inside double quotes."""
    sentences = 0
    words = 0
    first_person = 0
    quoted = 0
    for paragraph in paragraphs:
        inside_quote = False
        for token in paragraph.tokens:
            if token.tag in QUOTE_TAGS or token.text in QUOTE_CHARS:
                inside_quote = not inside_quote
                continue
            if token.sentence_start:
                sentences += 1
            if not any(char.isalnum() for char in token.text):
                continue
            words += 1
            first_person += token.text.lower() in FIRST_PERSON
            quoted += inside_quote
    if words == 0:
        return DocStats(sentences, 0.0, 0.0, 0.0)
    return DocStats(
        sentence_count=sentences,
        mean_sentence_length=words / max(sentences, 1),
        first_person_share=first_person / words,
        quoted_share=quoted / words,
    )
