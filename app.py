"""Minimal web chat UI for LemonRAG, backed by Lemonade Server."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag import LemonRAG

app = FastAPI(title="LemonRAG")

_rag = None


def get_rag():
    global _rag
    if _rag is None:
        _rag = LemonRAG()
    return _rag


class Query(BaseModel):
    question: str


@app.post("/api/ask")
def ask(query: Query):
    try:
        rag = get_rag()
    except FileNotFoundError as e:
        return {"answer": str(e), "sources": []}

    try:
        return rag.answer(query.question)
    except Exception as e:
        return {
            "answer": f"Error talking to Lemonade Server: {e}. Is it running?",
            "sources": [],
        }


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
