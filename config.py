"""Shared settings for the Historia Naturalis RAG pipeline."""
from pathlib import Path

DATA_CSV = Path("all_chunks.csv")          # prepared chunks
INDEX_PATH = Path("pliny.index")           # FAISS index (vectors only)
META_PATH = Path("pliny_meta.pkl")         # chunk text + metadata, same order as index

EMBED_MODEL = "BAAI/bge-base-en-v1.5"
# bge models work best when *queries* (not documents) get this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "llama3.1:8b"                  # any model you've pulled with `ollama pull`
TOP_K = 5
