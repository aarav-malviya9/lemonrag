"""Shared configuration for LemonRAG, all overridable via environment variables."""
import os

LEMONADE_BASE_URL = os.environ.get("LEMONADE_BASE_URL", "http://localhost:8000/api/v1")
LEMONADE_MODEL = os.environ.get("LEMONADE_MODEL", "Llama-3.2-1B-Instruct-Hybrid")
# Lemonade Server doesn't require a real API key, but the OpenAI client needs *something*.
LEMONADE_API_KEY = os.environ.get("LEMONADE_API_KEY", "lemonade")

TOP_K = int(os.environ.get("TOP_K", "4"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "800"))  # characters per chunk
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))

DOCUMENTS_DIR = os.environ.get("DOCUMENTS_DIR", "documents")
INDEX_PATH = os.environ.get("INDEX_PATH", "index.pkl")
