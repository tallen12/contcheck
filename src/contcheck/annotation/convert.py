"""Turning ERRANT's typed edits into our edit records: pure functions."""

from collections.abc import Sequence

from contcheck.annotation.models import AlignmentError, Annotated, Annotation, TypedEdit, Unmappable
from contcheck.corpus.record import EditRecord, revert_edits
from contcheck.taxonomy.errant_map import ERRANT_CODES, Dropped, Mapped, map_edit
from contcheck.taxonomy.kinds import Operation


def shape_operation(edit: TypedEdit) -> Operation:
    """The operation the edit's ranges show.

    Usually the same as the code's prefix. ERRANT types an edit whose last word only
    changes case (``Doctor`` to ``The doctor``) as if that word were not there, giving
    ``M:DET`` for an edit that replaces a word; our records need the range's shape.
    """
    if edit.noisy_start == edit.noisy_end:
        return Operation.MISSING
    if edit.clean_start == edit.clean_end:
        return Operation.UNNECESSARY
    return Operation.REPLACEMENT


def code_for_shape(code: str, op: Operation) -> str:
    """``code`` with its operation prefix set to ``op``, when it has one.

    ``M:ADJ`` on an edit that replaces words becomes ``R:ADJ``, so the kind follows what
    the edit does: a word missing from the noisy side is MissingWord, a different word in
    its place is RealWord. Codes without a prefix (``UNK``), and types ERRANT never gives
    that operation (``ADJ:FORM`` is replacement only), stay as they are.
    """
    prefix, separator, type_ = code.partition(":")
    if not separator or prefix not in {member.value for member in Operation}:
        return code
    shaped = f"{op.value}:{type_}"
    return shaped if shaped in ERRANT_CODES else code


def to_annotation(
    edits: Sequence[TypedEdit], noisy: Sequence[str], clean: Sequence[str]
) -> Annotation:
    """Map ``edits`` through the ERRANT map into records in noisy word indices.

    Each edit is mapped under the operation its ranges show (see :func:`shape_operation`),
    so its kind and operation agree. Returns :class:`Unmappable` for the first edit whose
    code is dropped. Raises
    :class:`AlignmentError` if the records do not revert ``noisy`` to ``clean``.
    """
    records: list[EditRecord] = []
    for edit in sorted(edits, key=lambda e: (e.noisy_start, e.noisy_end)):
        op = shape_operation(edit)
        match map_edit(code_for_shape(edit.code, op), edit.evidence):
            case Dropped(reason=reason):
                return Unmappable(
                    edit.code,
                    tuple(noisy[edit.noisy_start : edit.noisy_end]),
                    tuple(clean[edit.clean_start : edit.clean_end]),
                    reason,
                )
            case Mapped(kind=kind):
                records.append(
                    EditRecord(
                        start=edit.noisy_start,
                        end=edit.noisy_end,
                        kind=kind,
                        op=op,
                        orig=tuple(clean[edit.clean_start : edit.clean_end]),
                    )
                )
    if revert_edits(noisy, records) != tuple(clean):
        raise AlignmentError(f"edits {records} do not turn {list(noisy)} into {list(clean)}")
    return Annotated(tuple(records))
