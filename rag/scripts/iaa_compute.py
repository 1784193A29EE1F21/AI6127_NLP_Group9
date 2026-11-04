"""IAA (Q4): compute agreement / kappa from the filled sheets and keep valid records.

Usage: uv run python scripts/iaa_compute.py --annotators B C
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter
from pathlib import Path

from rag.config import PATHS
from rag.metrics import token_f1


def read_table(stem: Path) -> dict[str, dict]:
    if stem.with_suffix(".xlsx").exists():
        from openpyxl import load_workbook

        ws = load_workbook(stem.with_suffix(".xlsx"), read_only=True).active
        rows = list(ws.iter_rows(values_only=True))
        cols = [str(c) for c in rows[0]]
        data = [dict(zip(cols, r)) for r in rows[1:]]
    else:
        with open(stem.with_suffix(".csv"), encoding="utf-8-sig") as f:
            data = list(csv.DictReader(f))
    return {str(r["id"]): r for r in data if r.get("id")}


def to01(v) -> int | None:
    if v is None or str(v).strip() == "":
        return None
    label = str(v).strip().lower()
    if label in ("1", "1.0", "yes", "y", "true", "valid"):
        return 1
    if label in ("0", "0.0", "no", "n", "false", "invalid"):
        return 0
    raise ValueError(f"Invalid annotation label: {v!r}; use 0/1 or leave blank")


def cohen_kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def fleiss_kappa(labels: list[list[int]]) -> float:
    """labels[i] = list of 0/1 labels for item i (same number of raters)."""
    n_items, n_raters = len(labels), len(labels[0])
    p_j = [sum(l.count(c) for l in labels) / (n_items * n_raters) for c in (0, 1)]
    P_i = [(sum(l.count(c) ** 2 for c in (0, 1)) - n_raters) / (n_raters * (n_raters - 1)) for l in labels]
    P_bar, P_e = sum(P_i) / n_items, sum(p * p for p in p_j)
    return (P_bar - P_e) / (1 - P_e) if P_e < 1 else 1.0


def agreement_block(name: str, per_ann: dict[str, dict[str, int]], ids: list[str]) -> dict:
    anns = list(per_ann)
    rows = [[per_ann[a][i] for a in anns] for i in ids]
    out = {"n_items": len(ids), "annotators": anns,
           "percent_all_agree": round(sum(len(set(r)) == 1 for r in rows) / len(rows), 4),
           "positive_rate": {a: round(sum(per_ann[a][i] for i in ids) / len(ids), 4) for a in anns}}
    pair = {}
    for x, y in itertools.combinations(anns, 2):
        ax, ay = [per_ann[x][i] for i in ids], [per_ann[y][i] for i in ids]
        pair[f"{x}-{y}"] = {"percent_agreement": round(sum(p == q for p, q in zip(ax, ay)) / len(ids), 4),
                            "cohen_kappa": round(cohen_kappa(ax, ay), 4)}
    out["pairwise"] = pair
    if len(anns) >= 3:
        out["fleiss_kappa"] = round(fleiss_kappa(rows), 4)
    print(f"[{name}] {json.dumps(out)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotators", nargs="+", default=["B", "C"])
    ap.add_argument("--eval", type=Path, default=PATHS.eval_file)
    ap.add_argument("--dir", type=Path, default=PATHS.eval_dir / "iaa")
    args = ap.parse_args()
    if len(set(args.annotators)) < 2 or len(set(args.annotators)) != len(args.annotators):
        ap.error("Provide at least two distinct independent annotators")

    from rag.corpus import load_jsonl

    gold = {q["id"]: q for q in load_jsonl(args.eval)}
    blind = {a: read_table(args.dir / f"sheet_{a}_blind") for a in args.annotators}
    reveal = {a: read_table(args.dir / f"sheet_{a}_reveal") for a in args.annotators}

    valid = {a: {i: to01(r.get("valid")) for i, r in blind[a].items()} for a in args.annotators}
    agree = {a: {i: to01(r.get("agrees_with_A")) for i, r in reveal[a].items()} for a in args.annotators}
    ids_v = [i for i in gold if all(valid[a].get(i) is not None for a in args.annotators)]
    ids_a = [i for i in gold if all(agree[a].get(i) is not None for a in args.annotators)]

    report = {}
    if ids_v:
        report["validity"] = agreement_block("validity", {a: valid[a] for a in args.annotators}, ids_v)
    if ids_a:
        report["agrees_with_A"] = agreement_block("agrees_with_A", {a: agree[a] for a in args.annotators}, ids_a)
        # how often each annotator accepts A's answers
        report["acceptance_of_A"] = {a: round(sum(agree[a][i] for i in ids_a) / len(ids_a), 4)
                                     for a in args.annotators}

    f1s, f1_all = Counter(), []
    for a in args.annotators:
        vals = []
        for i, r in blind[a].items():
            own = str(r.get("own_answer") or "").strip()
            if own and i in gold:
                vals.append(token_f1(own, gold[i]["answer"]))
        if vals:
            f1s[a] = round(sum(vals) / len(vals), 4)
            f1_all += vals
    if f1_all:
        report["blind_answer_token_f1_vs_A"] = dict(f1s)
        report["blind_answer_pct_f1_ge_0.5"] = round(sum(v >= 0.5 for v in f1_all) / len(f1_all), 4)

    keep = []
    for i in gold:
        votes_v = [valid[a].get(i) for a in args.annotators if valid[a].get(i) is not None]
        votes_a = [agree[a].get(i) for a in args.annotators if agree[a].get(i) is not None]
        # annotator A counts as one "valid/agree" vote
        ok_v = (sum(votes_v) + 1) * 2 > len(votes_v) + 1 if votes_v else False
        ok_a = (sum(votes_a) + 1) * 2 > len(votes_a) + 1 if votes_a else False
        complete = len(votes_v) == len(args.annotators) and len(votes_a) == len(args.annotators)
        if complete and ok_v and ok_a:
            keep.append(i)
    report["n_kept"] = len(keep)
    report["n_total"] = len(gold)
    (args.dir / "valid_ids.txt").write_text("\n".join(keep), encoding="utf-8")
    with open(args.dir / "iaa_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    pa = report.get("agrees_with_A", report.get("validity", {})).get("percent_all_agree")
    if pa is not None and pa < 0.8:
        print("WARNING: agreement below the required 80% - discuss disagreements, refine the guideline, re-label.")


if __name__ == "__main__":
    main()
