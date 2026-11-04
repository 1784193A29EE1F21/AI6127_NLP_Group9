"""IAA (Q4): create blind labelling sheets for annotators B and C.

Usage: uv run python scripts/iaa_make_sheets.py --annotators B C
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from rag.config import PATHS
from rag.corpus import load_jsonl


def write_table(rows: list[dict], path: Path) -> Path:
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill

        wb = Workbook()
        ws = wb.active
        cols = list(rows[0].keys())
        ws.append(cols)
        for r in rows:
            ws.append([r[c] for c in cols])
        widths = {"question": 60, "evidence": 80, "answer_A": 50, "own_answer": 40, "source_url": 30}
        for i, c in enumerate(cols, 1):
            letter = ws.cell(1, i).column_letter
            ws.column_dimensions[letter].width = widths.get(c, 14)
            ws.cell(1, i).font = Font(bold=True)
            if c in ("valid", "own_answer", "agrees_with_A", "comment"):
                ws.cell(1, i).fill = PatternFill("solid", fgColor="FFF2CC")
        for row in ws.iter_rows(min_row=2):
            for cell in row:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "B2"
        path = path.with_suffix(".xlsx")
        wb.save(path)
    except ImportError:
        path = path.with_suffix(".csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, default=PATHS.eval_file)
    ap.add_argument("--annotators", nargs="+", default=["B", "C"])
    ap.add_argument("--out", type=Path, default=PATHS.eval_dir / "iaa")
    args = ap.parse_args()
    qs = load_jsonl(args.eval)
    args.out.mkdir(parents=True, exist_ok=True)
    for a in args.annotators:
        order = qs[:]
        random.Random(sum(map(ord, a))).shuffle(order)  # different order per annotator (reduces fatigue bias)
        blind = [{"id": q["id"], "question": q["question"], "evidence": q["evidence"],
                  "source_url": q["source_url"], "own_answer": "", "valid": "", "comment": ""} for q in order]
        reveal = [{"id": q["id"], "question": q["question"], "evidence": q["evidence"],
                   "answer_A": q["answer"], "agrees_with_A": "", "comment": ""} for q in order]
        p1 = write_table(blind, args.out / f"sheet_{a}_blind")
        p2 = write_table(reveal, args.out / f"sheet_{a}_reveal")
        print("wrote", p1, "and", p2)


if __name__ == "__main__":
    main()
