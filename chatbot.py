"""Ask questions about Pliny's Natural History, answered by a local model via Ollama.

Run:  python chatbot.py                (interactive)
      python chatbot.py -q "Did Pliny think elephants were intelligent?"
"""
import argparse
import json
import pickle

import faiss
import requests
from sentence_transformers import SentenceTransformer

import config

SYSTEM_PROMPT = """You are Pliny the Elder and everything you know comes from passages from Natural History \
(Bostock & Riley translation).

Rules:
1. Answer in the first person.
2. If you add anything not stated in the passages (background, interpretation, \
connections between passages), put it in a separate final section headed \
"Beyond the text:" and keep it brief. If there is nothing, omit the section.
3. Pliny's text includes ancient beliefs that are false by modern standards. That is fine, \
assert it as fact. The point isn't to be factually accurate, but rather faithful to his beliefs."""

# 1. Answer using ONLY the passages. Cite every claim with its passage number, e.g. [2].
# 2. If the passages do not contain the answer, say so plainly. Do not invent details.


class PlinyRAG:
    def __init__(self):
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.index = faiss.read_index(str(config.INDEX_PATH))
        with open(config.META_PATH, "rb") as f:
            self.meta = pickle.load(f)

    def retrieve(self, question: str, k: int = config.TOP_K):
        q = self.embedder.encode(
            [config.QUERY_PREFIX + question], normalize_embeddings=True
        ).astype("float32")
        scores, ids = self.index.search(q, k)
        return [(self.meta[i], float(s)) for i, s in zip(ids[0], scores[0])]

    @staticmethod
    def build_context(hits) -> str:
        blocks = []
        for n, (r, _) in enumerate(hits, 1):
            part = f" (part {r['part']}/{r['total_parts']})" if r["total_parts"] > 1 else ""
            blocks.append(
                f"[{n}] Book {r['book']}, Chapter {r['chapter']} - {r['title']}{part}\n{r['text']}"
            )
        return "\n\n".join(blocks)

    def ask(self, question: str, k: int = config.TOP_K, stream: bool = True):
        hits = self.retrieve(question, k)
        user_msg = f"Passages:\n\n{self.build_context(hits)}\n\nQuestion: {question}"
        payload = {
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "stream": stream,
            "options": {"temperature": 0.2},  # low temperature = stays closer to the passages
        }
        resp = requests.post(config.OLLAMA_URL, json=payload, stream=stream, timeout=300)
        resp.raise_for_status()

        answer = ""
        if stream:
            for line in resp.iter_lines():
                if line:
                    piece = json.loads(line).get("message", {}).get("content", "")
                    print(piece, end="", flush=True)
                    answer += piece
            print()
        else:
            answer = resp.json()["message"]["content"]
        return answer, hits


def print_sources(hits):
    print("\nSources retrieved:")
    for n, (r, score) in enumerate(hits, 1):
        print(f"  [{n}] Book {r['book']}, Ch. {r['chapter']} - {r['title']}  (similarity {score:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-q", "--question")
    ap.add_argument("-k", type=int, default=config.TOP_K)
    args = ap.parse_args()

    rag = PlinyRAG()
    if args.question:
        _, hits = rag.ask(args.question, args.k)
        print_sources(hits)
        return

    print("Ask about Pliny's Natural History (empty line to quit).")
    while True:
        q = input("\n> ").strip()
        if not q:
            break
        _, hits = rag.ask(q, args.k)
        print_sources(hits)


if __name__ == "__main__":
    main()
