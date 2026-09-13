"""Load documents from DOCUMENTS_DIR, chunk them, and build a local TF-IDF index.

No network calls, no embedding model downloads — everything here runs with
plain scikit-learn so the pipeline works offline out of the box. Swap in a
local embedding model later for better retrieval quality (see README roadmap).
"""
import os
import json
import sys
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from rich.console import Console

import config

console = Console()


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def read_pdf_file(path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def load_documents(documents_dir: str):
    """Yield (source_path, full_text) for every supported file in documents_dir."""
    if not os.path.isdir(documents_dir):
        console.print(f"[red]Documents folder '{documents_dir}' not found.[/red]")
        return

    found_any = False
    for root, _, files in os.walk(documents_dir):
        for name in sorted(files):
            path = os.path.join(root, name)
            ext = os.path.splitext(name)[1].lower()
            try:
                if ext == ".pdf":
                    text = read_pdf_file(path)
                elif ext in (".txt", ".md", ".markdown"):
                    text = read_text_file(path)
                else:
                    continue
            except Exception as e:
                console.print(f"[yellow]Skipping {path}: {e}[/yellow]")
                continue

            if text.strip():
                found_any = True
                yield path, text

    if not found_any:
        console.print(
            "[yellow]No .pdf, .txt, or .md files found. "
            "Add some documents to the 'documents/' folder first.[/yellow]"
        )


def chunk_text(text: str, chunk_size: int, overlap: int):
    """Simple sliding-window character chunker."""
    # Validate parameters to prevent infinite loops
    if chunk_size <= 0:
        raise ValueError(f"CHUNK_SIZE must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"CHUNK_OVERLAP must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"CHUNK_OVERLAP ({overlap}) must be less than CHUNK_SIZE ({chunk_size})")

    chunks = []
    start = 0
    text = text.replace("\r\n", "\n")
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap
        if end >= len(text):
            break
    return chunks


def build_index():
    all_chunks = []
    all_sources = []

    for source, text in load_documents(config.DOCUMENTS_DIR):
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        console.print(f"[green]{source}[/green]: {len(chunks)} chunks")
        all_chunks.extend(chunks)
        all_sources.extend([source] * len(chunks))

    if not all_chunks:
        console.print("[red]No chunks produced. Nothing to index.[/red]")
        sys.exit(1)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=50_000)
    matrix = vectorizer.fit_transform(all_chunks)

    import json
    import numpy as np

    # Convert vectorizer and matrix to JSON-serializable format
    # Handle stop_words which might be None or a set
    stop_words = vectorizer.stop_words
    if stop_words is not None:
        stop_words = list(stop_words)

    # Convert vocabulary values from numpy types to Python int
    vocabulary = {k: int(v) for k, v in vectorizer.vocabulary_.items()}

    vectorizer_data = {
        "stop_words": stop_words,
        "max_features": int(vectorizer.max_features) if vectorizer.max_features is not None else None,
        "vocabulary": vocabulary,
        "idf": [float(x) for x in vectorizer.idf_.tolist()]
    }

    # Convert sparse matrix to CSC format for efficient column slicing if needed, but we'll use CSR
    # Ensure matrix is in CSR format
    if not hasattr(matrix, "format") or matrix.format != "csr":
        matrix = matrix.tocsr()

    # Explicitly convert numpy types to Python native types
    matrix_data = {
        "data": [float(x) for x in np.asarray(matrix.data).tolist()],
        "indices": [int(x) for x in np.asarray(matrix.indices).tolist()],
        "indptr": [int(x) for x in np.asarray(matrix.indptr).tolist()],
        "shape": [int(x) for x in np.asarray(matrix.shape).tolist()]
    }

    data_to_save = {
        "vectorizer": vectorizer_data,
        "matrix": matrix_data,
        "chunks": all_chunks,
        "sources": all_sources
    }

    with open(config.INDEX_PATH, "w") as f:
        json.dump(data_to_save, f)

    console.print(
        f"\n[bold green]Indexed {len(all_chunks)} chunks from "
        f"{len(set(all_sources))} document(s) -> {config.INDEX_PATH}[/bold green]"
    )


if __name__ == "__main__":
    build_index()
