"""Core retrieval + generation logic, shared by cli.py and app.py."""
import os
import pickle

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

import config

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the provided "
    "context excerpts. If the answer isn't in the context, say you don't know — "
    "don't make things up. Cite which source(s) you used."
)


class LemonRAG:
    def __init__(self, index_path: str = None):
        index_path = index_path or config.INDEX_PATH
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"No index found at '{index_path}'. Run `python ingest.py` first."
            )

        with open(index_path, "rb") as f:
            data = pickle.load(f)

        self.vectorizer = data["vectorizer"]
        self.matrix = data["matrix"]
        self.chunks = data["chunks"]
        self.sources = data["sources"]

        self.client = OpenAI(
            base_url=config.LEMONADE_BASE_URL,
            api_key=config.LEMONADE_API_KEY,
        )

    def retrieve(self, query: str, top_k: int = None):
        top_k = top_k or config.TOP_K
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        ranked = scores.argsort()[::-1][:top_k]
        return [
            {"text": self.chunks[i], "source": self.sources[i], "score": float(scores[i])}
            for i in ranked
            if scores[i] > 0
        ]

    def build_prompt(self, query: str, retrieved: list):
        context = "\n\n".join(
            f"[Source: {r['source']}]\n{r['text']}" for r in retrieved
        )
        return (
            f"Context excerpts:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer using only the context above."
        )

    def answer(self, query: str, top_k: int = None, stream: bool = False):
        retrieved = self.retrieve(query, top_k)

        if not retrieved:
            return {
                "answer": "I couldn't find anything relevant in the indexed documents.",
                "sources": [],
            }

        user_prompt = self.build_prompt(query, retrieved)

        response = self.client.chat.completions.create(
            model=config.LEMONADE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=stream,
        )

        if stream:
            return response  # caller iterates over chunks

        answer_text = response.choices[0].message.content
        return {
            "answer": answer_text,
            "sources": sorted(set(r["source"] for r in retrieved)),
        }
