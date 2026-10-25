"""Value types for clean-text sources."""

from pydantic import BaseModel, ConfigDict


class Article(BaseModel):
    """One article from a source corpus, with paragraphs separated by blank lines."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    title: str
    text: str
