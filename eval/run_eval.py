#!/usr/bin/env python3
# eval/run_eval.py
#
# Master evaluation script.
# Runs retrieval eval and generation eval end-to-end, then:
#   • prints a formatted summary table to stdout
#   • saves eval/results/eval_results.json   (full per-query detail)
#   • saves eval/results/eval_summary.csv    (human-readable aggregate + per-query rows)
#
# Usage:
#   python3 -m eval.run_eval           # full run (retrieval + generation)
#   python3 -m eval.run_eval --retrieval-only
#   python3 -m eval.run_eval --generation-only

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from eval.query_set import QUERY_SET
from eval.retrieval_eval import RetrievalEvaluator, RetrievalResult
from eval.generation_eval import GenerationEvaluator, GenerationResult

RESULTS_DIR = Path(__file__).parent / "results"


# ── Serialisation helpers ────────────────────────────────────────────────────

def _result_to_dict(r: RetrievalResult | GenerationResult) -> dict:
    d = dataclasses.asdict(r)
    # Strip raw chunk texts from retrieval results to keep JSON tidy
    for key in ("rag_retrieved", "kw_retrieved", "retrieved_chunks"):
        if key in d:
            d[key] = [
                {
                    "chunk_id": c.get("chunk_id"),
                    "section":  c.get("metadata", {}).get("section"),
                    "hadm_id":  c.get("metadata", {}).get("hadm_id"),
                    "score":    round(c.get("score", 0), 4),
                }
                for c in d[key]
            ]
    return d


# ── Pretty-print helpers ─────────────────────────────────────────────────────

def _hline(widths: list[int], char: str = "─") -> str:
    return "┼".join(char * (w + 2) for w in widths)


def _row(cells: list[str], widths: list[int]) -> str:
    return "│".join(f" {c:<{w}} " for c, w in zip(cells, widths))


def print_retrieval_table(
    results: list[RetrievalResult],
    agg: dict,
) -> None:
    cols = ["ID", "Section", "Rel",
            "RAG P", "RAG R",
            "Scoped P", "Scoped R",
            "KW P", "KW R"]
    rows = [
        [
            r.query_id,
            r.expected_section[:22],
            str(r.total_relevant),
            f"{r.rag_precision:.2f}",
            f"{r.rag_recall:.2f}",
            f"{r.scoped_precision:.2f}",
            f"{r.scoped_recall:.2f}",
            f"{r.kw_precision:.2f}",
            f"{r.kw_recall:.2f}",
        ]
        for r in results
    ]
    widths = [max(len(cols[i]), max(len(row[i]) for row in rows)) for i in range(len(cols))]

    print("\n── RETRIEVAL EVALUATION ────────────────────────────────────────")
    print(_row(cols, widths))
    print(_hline(widths))
    for row in rows:
        print(_row(row, widths))
    print(_hline(widths))
    print(_row(
        ["MEAN", "", "",
         f"{agg['rag_mean_precision']:.2f}",
         f"{agg['rag_mean_recall']:.2f}",
         f"{agg['scoped_mean_precision']:.2f}",
         f"{agg['scoped_mean_recall']:.2f}",
         f"{agg['kw_mean_precision']:.2f}",
         f"{agg['kw_mean_recall']:.2f}"],
        widths,
    ))


def print_generation_table(
    results: list[GenerationResult],
    agg: dict,
) -> None:
    cols = ["ID", "RAG R1", "RAG R2", "RAG RL", "RAG BS", "RAG Faith",
            "NR R1", "NR R2", "NR RL", "NR BS", "Flag"]
    rows = [
        [
            r.query_id,
            f"{r.rag_rouge1:.2f}",
            f"{r.rag_rouge2:.2f}",
            f"{r.rag_rougeL:.2f}",
            f"{r.rag_bertscore:.2f}",
            f"{r.rag_faithfulness:.2f}",
            f"{r.norag_rouge1:.2f}",
            f"{r.norag_rouge2:.2f}",
            f"{r.norag_rougeL:.2f}",
            f"{r.norag_bertscore:.2f}",
            "⚠" if r.rag_faithfulness_flag else "✓",
        ]
        for r in results
    ]
    widths = [max(len(cols[i]), max(len(row[i]) for row in rows)) for i in range(len(cols))]

    print("\n── GENERATION EVALUATION ───────────────────────────────────────")
    print(_row(cols, widths))
    print(_hline(widths))
    for row in rows:
        print(_row(row, widths))
    print(_hline(widths))
    print(_row(
        ["MEAN",
         f"{agg['rag_mean_rouge1']:.2f}",
         f"{agg['rag_mean_rouge2']:.2f}",
         f"{agg['rag_mean_rougeL']:.2f}",
         f"{agg['rag_mean_bertscore']:.2f}",
         f"{agg['rag_mean_faithfulness']:.2f}",
         f"{agg['norag_mean_rouge1']:.2f}",
         f"{agg['norag_mean_rouge2']:.2f}",
         f"{agg['norag_mean_rougeL']:.2f}",
         f"{agg['norag_mean_bertscore']:.2f}",
         ""],
        widths,
    ))
    if agg.get("low_faithfulness_queries"):
        print(f"\n  ⚠ Low-faithfulness queries (<{0.5}): "
              f"{', '.join(agg['low_faithfulness_queries'])}")


# ── Persistence ──────────────────────────────────────────────────────────────

def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[run_eval] Saved JSON → {path}")


def save_csv(
    ret_results: list[RetrievalResult] | None,
    gen_results: list[GenerationResult] | None,
    agg: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    query_map = {q.query_id: q for q in QUERY_SET}

    # merge per-query rows
    ids = sorted(set(
        ([r.query_id for r in ret_results] if ret_results else []) +
        ([r.query_id for r in gen_results] if gen_results else [])
    ))

    ret_map = {r.query_id: r for r in (ret_results or [])}
    gen_map = {r.query_id: r for r in (gen_results or [])}

    for qid in ids:
        row: dict = {"query_id": qid, "query": query_map[qid].query}
        if qid in ret_map:
            r = ret_map[qid]
            row.update({
                "hadm_id":             r.hadm_id,
                "expected_section":    r.expected_section,
                "total_relevant":      r.total_relevant,
                "rag_precision":       round(r.rag_precision,    4),
                "rag_recall":          round(r.rag_recall,       4),
                "scoped_precision":    round(r.scoped_precision, 4),
                "scoped_recall":       round(r.scoped_recall,    4),
                "kw_precision":        round(r.kw_precision,     4),
                "kw_recall":           round(r.kw_recall,        4),
            })
        if qid in gen_map:
            g = gen_map[qid]
            row.update({
                "rag_rouge1":       round(g.rag_rouge1, 4),
                "rag_rouge2":       round(g.rag_rouge2, 4),
                "rag_rougeL":       round(g.rag_rougeL, 4),
                "rag_bertscore":    round(g.rag_bertscore, 4),
                "rag_faithfulness": round(g.rag_faithfulness, 4),
                "faith_flag":       g.rag_faithfulness_flag,
                "norag_rouge1":     round(g.norag_rouge1, 4),
                "norag_rouge2":     round(g.norag_rouge2, 4),
                "norag_rougeL":     round(g.norag_rougeL, 4),
                "norag_bertscore":  round(g.norag_bertscore, 4),
            })
        rows.append(row)

    # Append aggregate summary row
    agg_row = {"query_id": "AGGREGATE", "query": ""}
    for k, v in agg.items():
        if isinstance(v, float):
            agg_row[k] = round(v, 4)
        elif isinstance(v, list):
            agg_row[k] = "; ".join(v)

    rows.append(agg_row)

    fieldnames = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[run_eval] Saved CSV  → {path}")


# ── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CDSS RAG Evaluation")
    parser.add_argument("--retrieval-only",  action="store_true")
    parser.add_argument("--generation-only", action="store_true")
    args = parser.parse_args()

    run_retrieval  = not args.generation_only
    run_generation = not args.retrieval_only

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ret_results:  list[RetrievalResult]  = []
    gen_results:  list[GenerationResult] = []
    ret_agg = gen_agg = {}
    combined_agg: dict = {"timestamp": timestamp}

    # ── Retrieval eval ───────────────────────────────────────────────────────
    if run_retrieval:
        print("\n═══════════════════════════════════════")
        print(" STEP 1 / 2 — Retrieval Evaluation")
        print("═══════════════════════════════════════")
        evaluator = RetrievalEvaluator()
        ret_results = evaluator.run()
        ret_agg = RetrievalEvaluator.aggregate(ret_results)
        combined_agg.update({"retrieval": ret_agg})
        print_retrieval_table(ret_results, ret_agg)

    # ── Generation eval ──────────────────────────────────────────────────────
    if run_generation:
        print("\n═══════════════════════════════════════")
        print(" STEP 2 / 2 — Generation Evaluation")
        print("═══════════════════════════════════════")
        gen_evaluator = GenerationEvaluator()
        gen_results = gen_evaluator.run()
        gen_agg = GenerationEvaluator.aggregate(gen_results)
        combined_agg.update({"generation": gen_agg})
        print_generation_table(gen_results, gen_agg)
        n_chunk_refs = sum(1 for r in gen_results if r.reference_source == "chunk")
        print(f"\n  Reference source: {n_chunk_refs}/{len(gen_results)} queries "
              f"used real chunk text ({len(gen_results)-n_chunk_refs} keyword fallbacks)")

    # ── Save results ─────────────────────────────────────────────────────────
    full_payload = {
        "meta": combined_agg,
        "retrieval": [_result_to_dict(r) for r in ret_results],
        "generation": [_result_to_dict(r) for r in gen_results],
    }
    save_json(full_payload, RESULTS_DIR / "eval_results.json")
    save_csv(
        ret_results or None,
        gen_results or None,
        combined_agg,
        RESULTS_DIR / "eval_summary.csv",
    )

    print("\n✓ Evaluation complete.\n")


if __name__ == "__main__":
    main()
