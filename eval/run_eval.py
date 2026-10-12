"""Run the audit graph over labelled cases and report precision/recall, groundedness,
latency and token cost.

    uv run python -m eval.run_eval --offline          # fake components, checks the harness
    uv run python -m eval.run_eval                    # Azure Search + Azure OpenAI from .env
    uv run python -m eval.run_eval --input-price 2.5 --output-price 10   # USD per 1M tokens

Cases are JSONL: {"id", "transcript", "ocr_text", "expected": [category, ...]}.
A case with "video_url" and no transcript is sent through Azure Video Indexer (live only).
"""

import argparse
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from backend.src.graph.workflow import create_graph
from backend.src.services.fakes import FakeChatModel, FakeIndexer, FakeRetriever

HERE = Path(__file__).parent
PLACEHOLDER_URL = "https://youtu.be/eval"


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def is_grounded(quote: str, guidelines: list[str]) -> bool:
    quote = normalize(quote)
    return bool(quote) and any(quote in normalize(g) for g in guidelines)


# Offline auditor: keyword rules standing in for the model so the harness can run without
# credentials. Its scores say nothing about model quality.
OFFLINE_RULES = {
    "Disclosure": (
        r"\b(free|gifted|sent me)\b",
        r"\b(sponsored|#ad|paid partnership)\b",
        "disclosure",
    ),
    "Claim Validation": (r"\b(cures?|clinically proven)\b", None, "proof"),
}


def offline_respond(messages) -> str:
    system, user = messages[0].content, messages[1].content.lower()
    guidelines = system.split("Guidelines:\n", 1)[1].split("\n\nReturn a single JSON", 1)[0]
    sentences = re.split(r"(?<=[.!?])\s+", guidelines)
    issues = []
    for category, (trigger, waiver, anchor) in OFFLINE_RULES.items():
        if re.search(trigger, user) and not (waiver and re.search(waiver, user)):
            quote = next((s for s in sentences if anchor in s.lower()), anchor)
            issues.append(
                {
                    "category": category,
                    "severity": "High",
                    "timestamp": None,
                    "description": f"matched {trigger!r}",
                    "guideline_quote": quote,
                }
            )
    return json.dumps({"compliance_issues": issues, "final_report": f"{len(issues)} issue(s)."})


def build_components(offline: bool):
    if offline:
        passages = (HERE / "guidelines.txt").read_text().strip().split("\n\n")
        return FakeRetriever(passages), FakeChatModel(respond=offline_respond), None
    from backend.src.config import get_settings
    from backend.src.services.azure import (
        AzureSearchRetriever,
        build_chat_model,
        build_vector_store,
    )
    from backend.src.services.video_indexer import VideoIndexerService

    settings = get_settings()
    return (
        AzureSearchRetriever(build_vector_store(settings)),
        build_chat_model(settings),
        lambda: VideoIndexerService(settings),
    )


def run_case(case: dict, retriever, llm, make_indexer) -> dict:
    if case.get("transcript") or case.get("ocr_text"):
        indexer = FakeIndexer(case.get("transcript", ""), case.get("ocr_text", []))
    elif make_indexer:
        indexer = make_indexer()
    else:
        raise ValueError(f"case {case['id']} needs a transcript in offline mode")

    graph = create_graph(indexer, retriever, llm)
    start = time.perf_counter()
    result = graph.invoke(
        {"video_url": case.get("video_url", PLACEHOLDER_URL), "video_id": case["id"]}
    )
    latency = time.perf_counter() - start

    issues = result.get("compliance_results", [])
    guidelines = result.get("retrieved_guidelines", [])
    usage = result.get("token_usage", {})
    return {
        "id": case["id"],
        "status": result.get("final_status"),
        "expected": sorted({c.lower() for c in case["expected"]}),
        "predicted": sorted({i.category.lower() for i in issues}),
        "grounded": [is_grounded(i.guideline_quote, guidelines) for i in issues],
        "latency_s": latency,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "errors": result.get("errors", []),
    }


def summarize(rows: list[dict], input_price: float | None, output_price: float | None) -> dict:
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for r in rows:
        expected, predicted = set(r["expected"]), set(r["predicted"])
        for c in expected & predicted:
            counts[c]["tp"] += 1
        for c in predicted - expected:
            counts[c]["fp"] += 1
        for c in expected - predicted:
            counts[c]["fn"] += 1

    def ratio(a: int, b: int) -> float | None:
        return a / b if b else None

    per_category = {
        c: {
            **n,
            "precision": ratio(n["tp"], n["tp"] + n["fp"]),
            "recall": ratio(n["tp"], n["tp"] + n["fn"]),
        }
        for c, n in sorted(counts.items())
    }
    grounded = [g for r in rows for g in r["grounded"]]
    tokens_in = sum(r["input_tokens"] for r in rows)
    tokens_out = sum(r["output_tokens"] for r in rows)
    cost = None
    if input_price is not None and output_price is not None:
        cost = (tokens_in * input_price + tokens_out * output_price) / 1_000_000
    latencies = [r["latency_s"] for r in rows]
    return {
        "cases": len(rows),
        "errors": sum(1 for r in rows if r["errors"]),
        "per_category": per_category,
        "groundedness": ratio(sum(grounded), len(grounded)),
        "latency_s": {"mean": statistics.mean(latencies), "max": max(latencies)},
        "input_tokens": tokens_in,
        "output_tokens": tokens_out,
        "cost_usd": cost,
    }


def fmt(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.2f}"


def print_summary(summary: dict, offline: bool) -> None:
    if offline:
        print("OFFLINE HARNESS CHECK: fake retriever and keyword auditor, not model quality.\n")
    print(f"{'category':<20}{'tp':>4}{'fp':>4}{'fn':>4}{'precision':>11}{'recall':>8}")
    for c, m in summary["per_category"].items():
        print(
            f"{c:<20}{m['tp']:>4}{m['fp']:>4}{m['fn']:>4}"
            f"{fmt(m['precision']):>11}{fmt(m['recall']):>8}"
        )
    tokens = "word counts from the fake model" if offline else "tokens"
    cost = "n/a" if summary["cost_usd"] is None else f"${summary['cost_usd']:.4f}"
    latency = summary["latency_s"]
    print(
        f"\ncases {summary['cases']}  errors {summary['errors']}  "
        f"groundedness {fmt(summary['groundedness'])}"
    )
    print(f"latency mean {latency['mean']:.3f}s  max {latency['max']:.3f}s")
    print(f"{tokens}: in {summary['input_tokens']}  out {summary['output_tokens']}  cost {cost}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cases", type=Path, default=HERE / "cases.jsonl")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--input-price", type=float, help="USD per 1M input tokens")
    parser.add_argument("--output-price", type=float, help="USD per 1M output tokens")
    parser.add_argument("--out", type=Path, help="write per-case rows and summary as JSON")
    args = parser.parse_args()

    retriever, llm, make_indexer = build_components(args.offline)
    rows = [run_case(c, retriever, llm, make_indexer) for c in load_cases(args.cases)]
    summary = summarize(rows, args.input_price, args.output_price)
    print_summary(summary, args.offline)
    if args.out:
        args.out.write_text(
            json.dumps({"offline": args.offline, "summary": summary, "cases": rows}, indent=2)
        )


if __name__ == "__main__":
    main()
