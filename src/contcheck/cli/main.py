"""The root ``contcheck`` app.

Each domain's commands live in their own ``cli`` module, which this root app registers.
CLI modules are the composition roots: they build the concrete adapters and pass them in.
"""

from collections.abc import Sequence
from importlib.metadata import version

from cyclopts import App


def build_app() -> App:
    """Build the root app with every command registered."""
    return App(
        name="contcheck",
        help="Context-aware spelling and grammar tagger.",
        version=version("contcheck"),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the ``contcheck`` command with ``argv``, or the process arguments."""
    build_app()(argv)
