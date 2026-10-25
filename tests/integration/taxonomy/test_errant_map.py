import inspect
import re

from errant.en import classifier

from contcheck.taxonomy.errant_map import ERRANT_CODES, UNKNOWN_CODE

# Types the classifier returns as literals, besides its POS- and dependency-based ones.
ONE_SIDED_SPECIAL = {"NOUN:POSS", "CONTR", "VERB:FORM", "VERB:TENSE", "VERB", "OTHER"}
REPLACEMENT_SPECIAL = ONE_SIDED_SPECIAL | {
    "ORTH",
    "WO",
    "VERB:SVA",
    "NOUN:INFL",
    "VERB:INFL",
    "MORPH",
    "SPELL",
    "ADJ:FORM",
    "NOUN:NUM",
    "PART",
    "PRON",
    "DET",
}


def errant_codes() -> set[str]:
    """Every code the installed ERRANT classifier can produce."""
    pos_types = set(classifier.pos_map.values()) - classifier.rare_pos
    common = pos_types | set(classifier.dep_map.values())
    one_sided = common | ONE_SIDED_SPECIAL
    replacement = common | REPLACEMENT_SPECIAL
    return (
        {f"{op}:{type_}" for op in ("M", "U") for type_ in one_sided}
        | {f"R:{type_}" for type_ in replacement}
        | {UNKNOWN_CODE}
    )


def test_map_lists_exactly_the_codes_errant_can_emit() -> None:
    assert errant_codes() == ERRANT_CODES


def test_special_types_cover_every_literal_the_classifier_returns() -> None:
    source = inspect.getsource(classifier)
    literals = set(re.findall(r'return "([A-Z:]+)"', source))
    # NOUN:INFL and VERB:INFL are built as o_pos[0] + ":INFL".
    assert '+":INFL"' in source.replace(" ", "")

    assert literals <= REPLACEMENT_SPECIAL | set(classifier.dep_map.values())
