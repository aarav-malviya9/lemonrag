"""Minimal web chat UI for LemonRAG, backed by Lemonade Server."""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from rag import LemonRAG

app = FastAPI(title="LemonRAG")

# Simple module-level instantiation - creates singleton on import
_rag = LemonRAG()


class Query(BaseModel):
    question: str


@app.post("/api/ask")
def ask(query: Query):
    try:
        return _rag.answer(query.question)
    except FileNotFoundError as e:
        return {"answer": str(e), "sources": []}
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

    uvicorn.run(app, host="0.0.0.0", port=8085)
