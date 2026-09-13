


https://github.com/user-attachments/assets/dd885ce4-fb24-41dc-a92e-aa64fd0ac295






# 🍋 LemonRAG

**A fully local, private RAG chat assistant for your own documents — powered end-to-end by [Lemonade Server](https://github.com/lemonade-sdk/lemonade), AMD's local-first LLM runtime.**

Ask questions about your PDFs, notes, or docs. Retrieval and generation both happen on your machine. Nothing is sent to the cloud.

Built for the [AMD Lemonade Developer Challenge](https://www.amd.com/en/developer/resources/technical-articles/2026/join-the-lemonade-developer-challenge.html).

---

## Contents

- [Why](#why)
- [Features](#features)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Benchmarking](#benchmarking)
- [Project layout](#project-layout)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Why

Cloud LLM APIs are great until you have documents you'd rather not upload, or you want
zero marginal cost per query, or you just want something that works offline. LemonRAG
is a small, readable reference implementation of "chat with your docs" that runs
entirely through Lemonade Server, so it works on whatever local AI backend Lemonade
has configured for your machine — CPU, Vulkan GPU, or NPU/Hybrid.

## Features

- 📄 **Local ingestion** — point it at a folder of PDF / Markdown / plain-text files.
- 🔍 **Local retrieval** — TF-IDF search, no embedding model download required to get started.
- 🤖 **Local generation** — talks to Lemonade Server's OpenAI-compatible API, so any
  model Lemonade supports works here unmodified.
- 💬 **Two interfaces** — a terminal chat (`cli.py`) and a minimal web UI (`app.py`).
- 📊 **Built-in benchmarking** — `benchmark.py` measures latency and throughput across
  backends so you can compare, e.g., CPU vs. Hybrid/NPU on your own hardware.
- 🔓 **MIT licensed** — fork it, extend it, ship it.

## Architecture

```
documents/*.pdf,*.md,*.txt
        │
        ▼
   ingest.py  ──►  index.pkl  (TF-IDF vectors, stored locally)
        │
        ▼
    rag.py  (retrieve top-k chunks, build grounded prompt)
        │
        ▼
  Lemonade Server  (http://localhost:8000/api/v1)  ── OpenAI-compatible chat API
        │
        ▼
  cli.py   or   app.py + static/index.html   (answer + cited sources)
```

## Quickstart

**1. Install and start Lemonade Server.**
Follow the install instructions for your OS in the [Lemonade SDK repo](https://github.com/lemonade-sdk/lemonade)
(Windows, Linux, macOS, and Docker are all supported), then pull a chat model:

```bash
lemonade run Llama-3.2-1B-Instruct-Hybrid
```

By default this serves an OpenAI-compatible API at `http://localhost:8000/api/v1`.

**2. Clone this repo and install dependencies.**

```bash
git clone https://github.com/aarav-malviya9/lemonrag
cd lemonrag
pip install -r requirements.txt
```

**3. Add your documents.**

```bash
cp ~/some-folder/*.pdf documents/
```

**4. Build the index.**

```bash
python ingest.py
```

**5. Chat.**

```bash
python cli.py          # terminal chat
# or
python app.py           # then open http://localhost:8080
```

## Configuration

All optional, set as environment variables:

| Variable            | Default                          | Purpose                        |
|----------------------|-----------------------------------|---------------------------------|
| `LEMONADE_BASE_URL`  | `http://localhost:8000/api/v1`   | Lemonade Server endpoint       |
| `LEMONADE_MODEL`     | `Llama-3.2-1B-Instruct-Hybrid`   | Model to use for generation     |
| `TOP_K`              | `4`                               | Number of chunks retrieved per query |
| `CHUNK_SIZE`         | `800`                             | Characters per chunk            |
| `CHUNK_OVERLAP`      | `150`                             | Character overlap between chunks |

## Benchmarking

`benchmark.py` runs a fixed set of prompts against one or more models/backends and
reports time-to-first-token, total latency, and approximate tokens/sec:

```bash
python benchmark.py --models Llama-3.2-1B-Instruct-Hybrid,Llama-3.2-1B-Instruct-CPU --repeats 3
```

This writes `benchmark_report.md` with a comparison table — useful on its own as a
"deep-dive performance eval" submission, or alongside the app.

## Project layout

```
lemonrag/
├── config.py         # env-var configuration
├── ingest.py          # document loading, chunking, TF-IDF indexing
├── rag.py              # retrieval + Lemonade generation
├── cli.py               # terminal chat interface
├── app.py               # FastAPI web server
├── static/index.html    # web chat UI
├── benchmark.py         # latency/throughput benchmarking
├── documents/            # put your files here (gitignored)
├── requirements.txt
└── LICENSE
```

## Roadmap

- [ ] Swap TF-IDF for a local embedding model for better retrieval quality
- [ ] Streaming responses in the web UI
- [ ] Support `.docx` and `.csv` ingestion
- [ ] Multi-turn conversation memory
- [ ] Source-passage highlighting in the UI

Contributions on any of these are welcome — see below.

## Contributing

Issues and PRs are welcome. This is a small, readable codebase on purpose — if you're
extending it, try to keep new features in their own module rather than growing the
existing files, so it stays easy for the next person to follow.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

Built on [Lemonade](https://github.com/lemonade-sdk/lemonade), AMD's open local-first
AI runtime, for the AMD Lemonade Developer Challenge.
