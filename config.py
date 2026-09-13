"""Shared configuration for LemonRAG, all overridable via environment variables."""
import os

def _get_int_env(var_name: str, default: str) -> int:
    """Get integer environment variable with validation."""
    value = os.environ.get(var_name, default)
    try:
        return int(value)
    except ValueError:
        print(f"Warning: Invalid value for {var_name}='{value}', using default {default}")
        return int(default)

LEMONADE_BASE_URL = os.environ.get("LEMONADE_BASE_URL", "http://127.0.0.1:13305/v1")
LEMONADE_MODEL = os.environ.get("LEMONADE_MODEL", "Llama-3.2-1B-Instruct-GGUF")
# Lemonade Server doesn't require a real API key, but the OpenAI client needs *something*.
LEMONADE_API_KEY = os.environ.get("LEMONADE_API_KEY", "lemonade")

TOP_K = _get_int_env("TOP_K", "4")
CHUNK_SIZE = _get_int_env("CHUNK_SIZE", "800")  # characters per chunk
CHUNK_OVERLAP = _get_int_env("CHUNK_OVERLAP", "150")

# Additional validation for chunk parameters
if CHUNK_SIZE <= 0:
    print(f"Warning: CHUNK_SIZE must be positive, got {CHUNK_SIZE}, using default 800")
    CHUNK_SIZE = 800
if CHUNK_OVERLAP < 0:
    print(f"Warning: CHUNK_OVERLAP must be non-negative, got {CHUNK_OVERLAP}, using default 150")
    CHUNK_OVERLAP = 150
if CHUNK_OVERLAP >= CHUNK_SIZE:
    print(f"Warning: CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({CHUNK_SIZE}), adjusting")
    CHUNK_OVERLAP = max(0, CHUNK_SIZE // 2)  # Set to half of chunk size or 0

DOCUMENTS_DIR = os.environ.get("DOCUMENTS_DIR", "documents")
INDEX_PATH = os.environ.get("INDEX_PATH", "index.json")
