"""Benchmark Lemonade Server latency & throughput across one or more models/backends.

This is meant for the "deep-dive performance eval" submission track: run it against
each backend variant you have available (e.g. a CPU build and a Hybrid/NPU build of
the same model) and it produces a reproducible markdown report comparing them.

Usage:
    python benchmark.py --models Llama-3.2-1B-Instruct-Hybrid,Llama-3.2-1B-Instruct-CPU
    python benchmark.py --models Llama-3.2-1B-Instruct-Hybrid --prompts 5 --repeats 3
"""
import argparse
import statistics
import time

from openai import OpenAI

import config

DEFAULT_PROMPTS = [
    "Explain what a hash map is, in two sentences.",
    "Write a haiku about local-first software.",
    "List three benefits of running an LLM on-device instead of in the cloud.",
    "Summarize the plot of a story about a lighthouse keeper, in three sentences.",
    "What's the difference between latency and throughput?",
]


def run_single(client, model, prompt):
    """Returns (time_to_first_token_s, total_time_s, completion_tokens)."""
    start = time.perf_counter()
    first_token_time = None
    completion_tokens = 0

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            completion_tokens += 1  # rough proxy: counts stream deltas, not real tokens

    end = time.perf_counter()
    ttft = (first_token_time - start) if first_token_time else None
    total = end - start
    return ttft, total, completion_tokens


def benchmark_model(client, model, prompts, repeats):
    ttfts, totals, tps_list = [], [], []

    print(f"\n== {model} ==")
    for i, prompt in enumerate(prompts):
        for r in range(repeats):
            try:
                ttft, total, tokens = run_single(client, model, prompt)
            except Exception as e:
                print(f"  ! error on prompt {i+1}, repeat {r+1}: {e}")
                continue

            tps = tokens / total if total > 0 else 0
            ttfts.append(ttft or 0)
            totals.append(total)
            tps_list.append(tps)
            print(
                f"  prompt {i+1}/{len(prompts)} rep {r+1}/{repeats}: "
                f"ttft={ttft:.3f}s total={total:.3f}s ~tok/s={tps:.1f}"
            )

    if not totals:
        return None

    return {
        "model": model,
        "avg_ttft": statistics.mean(ttfts),
        "avg_total": statistics.mean(totals),
        "avg_tps": statistics.mean(tps_list),
        "p50_total": statistics.median(totals),
        "runs": len(totals),
    }


def write_report(results, path="benchmark_report.md"):
    lines = [
        "# LemonRAG Benchmark Report",
        "",
        f"Endpoint: `{config.LEMONADE_BASE_URL}`",
        "",
        "| Model | Avg TTFT (s) | Avg total (s) | Median total (s) | Approx tok/s | Runs |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        if not r:
            continue
        lines.append(
            f"| {r['model']} | {r['avg_ttft']:.3f} | {r['avg_total']:.3f} | "
            f"{r['p50_total']:.3f} | {r['avg_tps']:.1f} | {r['runs']} |"
        )

    lines += [
        "",
        "_TTFT = time to first token. Token counts are approximated from stream chunk "
        "counts, not exact tokenizer counts — treat tok/s as directionally useful, "
        "not an exact figure. Run on the same machine, same prompts, back-to-back for "
        "a fair comparison between backends._",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nReport written to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default=config.LEMONADE_MODEL,
        help="Comma-separated list of Lemonade model names to benchmark",
    )
    parser.add_argument("--prompts", type=int, default=len(DEFAULT_PROMPTS))
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    client = OpenAI(base_url=config.LEMONADE_BASE_URL, api_key=config.LEMONADE_API_KEY)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    prompts = DEFAULT_PROMPTS[: args.prompts]

    results = [benchmark_model(client, m, prompts, args.repeats) for m in models]
    write_report(results)


if __name__ == "__main__":
    main()
