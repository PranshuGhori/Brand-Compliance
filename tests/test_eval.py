from eval.run_eval import (
    HERE,
    build_components,
    is_grounded,
    load_cases,
    run_case,
    summarize,
)


def row(expected, predicted, grounded=(), tokens=(0, 0)):
    return {
        "expected": expected,
        "predicted": predicted,
        "grounded": list(grounded),
        "latency_s": 0.5,
        "input_tokens": tokens[0],
        "output_tokens": tokens[1],
        "errors": [],
    }


def test_is_grounded_ignores_case_and_punctuation():
    guidelines = ["You can't make up claims about a product that would require proof."]
    assert is_grounded("you can't make up CLAIMS about a product", guidelines)
    assert not is_grounded("products must be tested", guidelines)
    assert not is_grounded("", guidelines)


def test_summarize_precision_recall_and_cost():
    rows = [
        row(["disclosure"], ["disclosure"], [True], (1000, 100)),
        row(["disclosure"], [], [], (1000, 100)),
        row([], ["disclosure", "claim validation"], [True, False], (1000, 100)),
    ]
    s = summarize(rows, input_price=2.0, output_price=10.0)
    d = s["per_category"]["disclosure"]
    assert (d["tp"], d["fp"], d["fn"]) == (1, 1, 1)
    assert d["precision"] == 0.5 and d["recall"] == 0.5
    assert s["per_category"]["claim validation"]["recall"] is None
    assert s["groundedness"] == 2 / 3
    assert s["cost_usd"] == (3000 * 2.0 + 300 * 10.0) / 1_000_000


def test_cost_is_unknown_without_prices():
    assert summarize([row([], [])], None, None)["cost_usd"] is None


def test_offline_harness_runs_every_case():
    retriever, llm, make_indexer = build_components(offline=True)
    cases = load_cases(HERE / "cases.jsonl")
    rows = [run_case(c, retriever, llm, make_indexer) for c in cases]
    assert len(rows) == len(cases)
    assert all(r["status"] in {"PASS", "FAIL"} and not r["errors"] for r in rows)
    assert all(r["input_tokens"] > 0 for r in rows)
