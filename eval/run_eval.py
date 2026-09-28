#!/usr/bin/env python
"""Retrieval evaluation sweep.

Answers the question the three chunking strategies were built to raise and
never measured: which one actually retrieves the right passage?

    python eval/run_eval.py                      # default sweep
    python eval/run_eval.py --methods recursive --sizes 500 1000 --k 1 3 5
    python eval/run_eval.py --judge              # also score answer quality (uses the LLM)

Put PDFs in eval/corpus/ and ground truth in eval/questions.json first.
Embedding is local and free, so the sweep costs nothing but time.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.metrics import aggregate, is_relevant, score  # noqa: E402
from src.extract_from_pdf import load_and_chunk  # noqa: E402
from src.vectorstore import build_vectorstore  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
CORPUS = EVAL_DIR / "corpus"
RESULTS = EVAL_DIR / "results"


def load_questions(path: Path):
    if not path.exists():
        sys.exit(
            f"No question set at {path}.\n"
            f"Copy eval/questions.example.json to eval/questions.json and edit it, "
            f"or run: python eval/bootstrap_questions.py"
        )
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not questions:
        sys.exit("Question set is empty.")
    return questions


def corpus_paths():
    pdfs = sorted(str(p) for p in CORPUS.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {CORPUS}. Add the documents your questions refer to.")
    return pdfs


def evaluate(store, questions, k_values):
    """Retrieve once at the largest k, then score every smaller cutoff from it."""
    max_k = max(k_values)
    per_k = {k: [] for k in k_values}
    misses = []

    for question in questions:
        docs = store.similarity_search(question["question"], k=max_k)
        for k in k_values:
            per_k[k].append(score(docs, question, k))

        if not any(is_relevant(d, question) for d in docs):
            misses.append(
                {
                    "question": question["question"],
                    "expected": question.get("answer_span", "")[:90],
                    "got": [
                        {
                            "source": d.metadata.get("source"),
                            "page": d.metadata.get("page"),
                            "preview": " ".join(d.page_content.split())[:90],
                        }
                        for d in docs[:3]
                    ],
                }
            )

    return {k: aggregate(rows) for k, rows in per_k.items()}, misses


def judge(store, questions, k):
    """Optional answer-quality pass. Costs one LLM call per question."""
    from src.config import settings
    from src.rag import answer

    if not settings.llm_configured:
        print("  (skipping --judge: no LLM endpoint configured)")
        return None

    from src.models import chat

    scores = []
    for question in questions:
        reply, _ = answer(store, question["question"], k)
        verdict = chat(
            "You grade answers. Reply with exactly one word: CORRECT, PARTIAL, or WRONG.",
            f"Question: {question['question']}\n"
            f"Reference: {question.get('answer_span', '')}\n"
            f"Answer: {reply}",
        )
        word = (verdict or "").strip().upper()
        scores.append({"CORRECT": 1.0, "PARTIAL": 0.5}.get(word.split()[0] if word else "", 0.0))
    return sum(scores) / len(scores) if scores else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="+", default=["recursive", "semantic", "markdown"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[500, 1000, 2000])
    parser.add_argument("--overlap", type=int, default=200)
    parser.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.json")
    parser.add_argument("--judge", action="store_true", help="also grade generated answers")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    pdfs = corpus_paths()
    k_values = sorted(set(args.k))

    print(f"{len(questions)} questions · {len(pdfs)} PDF(s) · "
          f"{len(args.methods) * len(args.sizes)} configurations\n")

    runs = []
    for method in args.methods:
        for size in args.sizes:
            label = f"{method}/{size}"
            started = time.perf_counter()
            try:
                chunks = load_and_chunk(pdfs, method, size, args.overlap)
                if not chunks:
                    print(f"  {label:<20} no chunks produced — skipped")
                    continue
                store = build_vectorstore(chunks)
            except Exception as exc:
                print(f"  {label:<20} FAILED: {exc}")
                continue

            by_k, misses = evaluate(store, questions, k_values)
            elapsed = time.perf_counter() - started

            run = {
                "method": method,
                "chunk_size": size,
                "overlap": args.overlap,
                "chunks": len(chunks),
                "build_seconds": round(elapsed, 1),
                "metrics": {str(k): by_k[k] for k in k_values},
                "misses": misses,
            }
            if args.judge:
                run["answer_score"] = judge(store, questions, min(4, max(k_values)))
            runs.append(run)

            best = by_k[max(k_values)]
            print(f"  {label:<20} {len(chunks):>5} chunks  "
                  f"hit@{max(k_values)}={best['hit']:.2f}  mrr={best['mrr']:.2f}  "
                  f"({elapsed:.0f}s)")

    if not runs:
        sys.exit("\nNo configuration completed.")

    RESULTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (RESULTS / f"{stamp}.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")
    report = render(runs, k_values, len(questions))
    (RESULTS / f"{stamp}.md").write_text(report, encoding="utf-8")
    (RESULTS / "latest.md").write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"Written to eval/results/{stamp}.md")


def render(runs, k_values, n_questions):
    """Markdown report — paste the table straight into the README."""
    headline = max(k_values) if len(k_values) == 1 else (5 if 5 in k_values else max(k_values))

    lines = [
        "## Retrieval benchmark",
        "",
        f"{n_questions} questions · embeddings `BAAI/bge-small-en-v1.5` (local) · "
        f"ground truth is a verbatim span, so it survives re-chunking.",
        "",
        "| Config | Chunks | " + " | ".join(f"hit@{k}" for k in k_values)
        + f" | MRR | nDCG@{headline} | Build |",
        "|---|---:|" + "---:|" * len(k_values) + "---:|---:|---:|",
    ]

    ranked = sorted(runs, key=lambda r: -r["metrics"][str(headline)]["hit"])
    for run in ranked:
        cells = [f"{run['metrics'][str(k)]['hit']:.2f}" for k in k_values]
        head = run["metrics"][str(headline)]
        lines.append(
            f"| `{run['method']}` @ {run['chunk_size']} | {run['chunks']} | "
            + " | ".join(cells)
            + f" | {head['mrr']:.2f} | {head['ndcg']:.2f} | {run['build_seconds']}s |"
        )

    if any("answer_score" in r and r["answer_score"] is not None for r in ranked):
        lines += ["", "| Config | Answer score |", "|---|---:|"]
        for run in ranked:
            if run.get("answer_score") is not None:
                lines.append(f"| `{run['method']}` @ {run['chunk_size']} | "
                             f"{run['answer_score']:.2f} |")

    winner = ranked[0]
    lines += [
        "",
        f"**Best:** `{winner['method']}` at {winner['chunk_size']} — "
        f"hit@{headline} {winner['metrics'][str(headline)]['hit']:.2f}, "
        f"MRR {winner['metrics'][str(headline)]['mrr']:.2f}.",
    ]

    if winner["misses"]:
        lines += ["", f"**Still missed ({len(winner['misses'])}):**", ""]
        for miss in winner["misses"][:5]:
            lines.append(f"- {miss['question']}")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
