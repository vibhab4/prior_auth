from typing import Optional

from pydantic import BaseModel, Field


class PolicyChunk(BaseModel):
    """A retrieved policy excerpt with enough metadata for auditable citation."""

    source_document: str = Field(description="Filename of the source policy document")
    source_title: str = Field(description="Human-readable title of the policy document")
    section: Optional[str] = Field(
        default=None, description="Top-level section name (## header in source)"
    )
    subsection: Optional[str] = Field(
        default=None, description="Subsection name (### header in source)"
    )
    chunk_text: str = Field(description="The actual retrieved policy text")
    relevance_score: Optional[float] = Field(
        default=None, description="Cosine similarity score (0-1; higher = more relevant)"
    )
