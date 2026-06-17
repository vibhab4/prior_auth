from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from prior_auth.schemas.policy import PolicyChunk
from prior_auth.state import PriorAuthState

# Resolve project root from this file's location (src/prior_auth/nodes/ -> project root)
_PROJECT_ROOT = Path(__file__).parents[3]
VECTORSTORE_DIR = _PROJECT_ROOT / "data" / "vectorstore"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
K = 4  # number of chunks to retrieve per query
MIN_RELEVANCE_SCORE = 0.65  # below this, flag "no_relevant_policy_found"


@lru_cache(maxsize=1)
def _get_vectorstore() -> Chroma:
    """Lazy-load the persisted Chroma store once per Python process.

    The embedding model + Chroma client are expensive to initialize, so we
    cache the result. lru_cache(maxsize=1) is sufficient since there's only
    one collection and one configuration.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return Chroma(
        persist_directory=str(VECTORSTORE_DIR),
        embedding_function=embeddings,
        collection_name="policy_corpus",
    )


def retrieve_policy(state: PriorAuthState) -> dict:
    """Retrieve relevant coverage policy excerpts for the extracted PA request.

    Queries the pre-built Chroma vector store with a single combined query
    constructed from the request's procedure, diagnosis, and clinical
    justification fields. Returns the k most similar policy chunks with their
    citation metadata.

    Follows the same flag-don't-fail philosophy as extract_request: always
    returns a result, never raises. Low-relevance retrieval is surfaced via
    retrieval_flags rather than an exception.
    """
    req = state["extracted_request"]

    # Single combined query: procedure + diagnosis + clinical context.
    # This gives the embedding model the full clinical picture to match against
    # coverage-criteria paragraphs (e.g. "MRI for radiculopathy after failed PT").
    query = (
        f"{req.procedure_description} ({req.procedure_code or ''}) "
        f"for {req.diagnosis_description} ({req.diagnosis_code or ''}). "
        f"{req.clinical_justification}"
    )

    try:
        vectorstore = _get_vectorstore()
        raw_results = vectorstore.similarity_search_with_relevance_scores(query, k=K)
    except Exception:
        # Retrieval failure (vectorstore not built, file missing, etc.) is treated
        # as a flag rather than a crash -- the graph can still route this case for
        # human review.
        return {
            "retrieved_policy_chunks": [],
            "retrieval_flags": ["retrieval_error"],
        }

    chunks = [
        PolicyChunk(
            source_document=doc.metadata.get("source_document", "unknown"),
            source_title=doc.metadata.get("source_title", "Unknown Policy"),
            section=doc.metadata.get("section"),
            subsection=doc.metadata.get("subsection"),
            chunk_text=doc.page_content,
            relevance_score=round(score, 4),
        )
        for doc, score in raw_results
    ]

    flags: list[str] = []
    if not chunks or chunks[0].relevance_score < MIN_RELEVANCE_SCORE:
        flags.append("no_relevant_policy_found")

    return {"retrieved_policy_chunks": chunks, "retrieval_flags": flags}
