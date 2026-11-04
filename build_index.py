"""Embed every chunk in all_chunks.csv and save a FAISS index + metadata.

Run once:  python build_index.py
"""
import pickle
import re

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import config


def clean_title(t: str) -> str:
    # Titles contain footnote markers like "OF THE FORM OF THE WORLD[91]"
    return re.sub(r"\[\d+\]", "", t).strip().title()


def main():
    df = pd.read_csv(config.DATA_CSV)
    df["title"] = df["title"].map(clean_title)
    records = df.to_dict("records")

    # What gets embedded: a short header + the passage. The header lets
    # queries like "what does Pliny say about costus?" match on the title too.
    docs = [
        f"Book {r['book']}, Chapter {r['chapter']}: {r['title']}\n{r['text']}"
        for r in records
    ]

    print(f"Loading embedding model {config.EMBED_MODEL} ...")
    model = SentenceTransformer(config.EMBED_MODEL)

    print(f"Embedding {len(docs)} chunks ...")
    vectors = model.encode(
        docs, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    ).astype(np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product == cosine (normalized)
    index.add(vectors)

    faiss.write_index(index, str(config.INDEX_PATH))
    with open(config.META_PATH, "wb") as f:
        pickle.dump(records, f)
    print(f"Saved {index.ntotal} vectors to {config.INDEX_PATH} and metadata to {config.META_PATH}")


if __name__ == "__main__":
    main()
