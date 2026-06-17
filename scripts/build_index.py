"""
Build the policy corpus vector store.

Run once (or whenever data/policy_corpus/ changes) before using the agent:
    python scripts/build_index.py

Output: data/vectorstore/ (Chroma persistent store, gitignored).
"""

import os
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

CORPUS_DIR = Path(__file__).parent.parent / "data" / "policy_corpus"
VECTORSTORE_DIR = Path(__file__).parent.parent / "data" / "vectorstore"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Markdown headers to split on -- matches the ## / ### headers written in our corpus files.
# Each level's text is stored as chunk metadata (e.g. {"section": "INDICATIONS FOR LUMBAR SPINE MRI"}).
HEADERS_TO_SPLIT_ON = [
    ("##", "section"),
    ("###", "subsection"),
]


def _extract_source_title(text: str) -> str:
    """Pull the human-readable title from the '# Source:' header line."""
    for line in text.splitlines():
        if line.startswith("# Source:"):
            return line.replace("# Source:", "").strip()
    return "Unknown Policy Document"


def load_and_split(filepath: Path) -> list:
    loader = TextLoader(str(filepath), encoding="utf-8")
    raw_docs = loader.load()
    raw_text = raw_docs[0].page_content

    source_title = _extract_source_title(raw_text)
    source_document = filepath.name

    # Pass 1: split on markdown section headers.
    # Each resulting chunk carries {"section": "...", "subsection": "..."} metadata.
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False,  # keep the header text in the chunk for context
    )
    header_chunks = md_splitter.split_text(raw_text)

    # Pass 2: split any oversized sections into smaller pieces.
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    final_chunks = char_splitter.split_documents(header_chunks)

    # Attach file-level metadata to every chunk.
    for chunk in final_chunks:
        chunk.metadata["source_document"] = source_document
        chunk.metadata["source_title"] = source_title

    return final_chunks


def main():
    corpus_files = sorted(CORPUS_DIR.glob("*.txt"))
    if not corpus_files:
        raise FileNotFoundError(f"No .txt files found in {CORPUS_DIR}")

    print(f"Loading {len(corpus_files)} document(s) from {CORPUS_DIR}")

    all_chunks = []
    for filepath in corpus_files:
        chunks = load_and_split(filepath)
        print(f"  {filepath.name}: {len(chunks)} chunks")
        all_chunks.append((filepath.name, chunks))

    all_docs = [chunk for _, chunks in all_chunks for chunk in chunks]
    print(f"\nTotal chunks: {len(all_docs)}")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    print("(First run downloads the model weights -- may take a minute.)")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # cosine similarity via dot product
    )

    print(f"\nBuilding Chroma vector store at: {VECTORSTORE_DIR}")
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_name="policy_corpus",
    )

    print(f"\nIndex built successfully. {vectorstore._collection.count()} vectors stored.")

    # Quick sanity check: retrieve a sample query and show the top result.
    print("\n--- Sanity check: sample query ---")
    query = "lumbar MRI coverage criteria radiculopathy failed conservative treatment"
    results = vectorstore.similarity_search_with_relevance_scores(query, k=1)
    if results:
        doc, score = results[0]
        print(f"Query: '{query}'")
        print(f"Top result score: {score:.3f}")
        print(f"Source: {doc.metadata.get('source_document', 'unknown')}")
        print(f"Section: {doc.metadata.get('section', 'unknown')}")
        print(f"Subsection: {doc.metadata.get('subsection', 'N/A')}")
        print(f"Snippet: {doc.page_content[:200]}...")


if __name__ == "__main__":
    main()
