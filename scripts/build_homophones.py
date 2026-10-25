"""Build ``src/contcheck/corrupt/data/homophones.txt`` from the CMU Pronouncing Dictionary.

Download ``cmudict.dict`` and ``LICENSE`` from https://github.com/cmusphinx/cmudict
(BSD-2-Clause), then run::

    uv run python scripts/build_homophones.py path/to/cmudict.dict path/to/LICENSE

Only words in symspellpy's English frequency list are kept. The output is deterministic
and carries the CMU license, which the BSD terms require.
"""

import sys
from pathlib import Path

from contcheck.corrupt.confusables import HOMOPHONES, homophone_groups
from contcheck.spelling.symspell_dictionary import load_symspell_dictionary

OUTPUT = Path(__file__).resolve().parents[1] / "src" / "contcheck" / "corrupt" / "data" / HOMOPHONES
HEADER = """\
# Homophone groups, one per line, from the CMU Pronouncing Dictionary
# (https://github.com/cmusphinx/cmudict), limited to words in symspellpy's English
# frequency list. Regenerate with scripts/build_homophones.py; do not edit by hand.
#
"""


def main(cmudict: Path, license_file: Path) -> None:
    vocabulary = load_symspell_dictionary().frequencies().keys()
    lines = cmudict.read_text(encoding="utf-8").splitlines()
    groups = homophone_groups(lines, vocabulary)
    notice = "".join(
        f"# {line}".rstrip() + "\n"
        for line in license_file.read_text(encoding="utf-8").splitlines()
    )
    body = "".join(" ".join(group) + "\n" for group in groups)
    OUTPUT.write_text(HEADER + notice + body, encoding="utf-8")
    print(f"wrote {len(groups)} groups to {OUTPUT}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
