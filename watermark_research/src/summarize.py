#!/usr/bin/env python
"""
Turn one or more benchmark.py result folders into paper-style tables.

  python summarize.py ../results                       # Table 1 / Table 3 layout: method x dataset
  python summarize.py ../results_C1 ../results_C2 ../results_F1 --by run   # Table 2 layout: run x (dataset, attack)

Each result folder holds summary_all.csv (one row per dataset x method with det_acc, tpr, fpr and one sur__<attack>
column per attack). Output goes to stdout as a Markdown table and to --out as CSV.
"""
import argparse
import os

import pandas as pd


def load(paths):
    frames = []
    for p in paths:
        f = os.path.join(p, "summary_all.csv") if os.path.isdir(p) else p
        if not os.path.exists(f):
            print(f"[skip] {f} not found")
            continue
        df = pd.read_csv(f)
        df.insert(0, "run", os.path.basename(os.path.normpath(os.path.dirname(f) if not os.path.isdir(p) else p)))
        frames.append(df)
    if not frames:
        raise SystemExit("no summary_all.csv found")
    return pd.concat(frames, ignore_index=True)


def fmt_det(row):
    return f"{100 * row['det_acc']:.1f} ({100 * row['tpr']:.1f}/{100 * row['fpr']:.1f})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", help="result folders (or summary_all.csv files) written by benchmark.py")
    ap.add_argument("--by", choices=["method", "run"], default="method",
                    help="rows = method (default) or rows = run x method (for comparing optimization sets)")
    ap.add_argument("--xfer", action="store_true", help="also show transferability (Delta-Score > 0 rate) per attack")
    ap.add_argument("--out", default="summary_table.csv")
    args = ap.parse_args()

    df = load(args.results)
    attacks = [c[len("sur__"):] for c in df.columns if c.startswith("sur__")]
    rows = ["run", "method"] if args.by == "run" else ["method"]

    long = []
    for _, r in df.iterrows():
        base = {k: r[k] for k in rows}
        long.append({**base, "dataset": r["dataset"], "metric": "Det. acc (TPR/FPR)", "value": fmt_det(r)})
        for a in attacks:
            if pd.notna(r.get(f"sur__{a}")):
                long.append({**base, "dataset": r["dataset"], "metric": f"Sur. {a}", "value": f"{100 * r[f'sur__{a}']:.1f}"})
            if args.xfer and pd.notna(r.get(f"xfer__{a}")):
                long.append({**base, "dataset": r["dataset"], "metric": f"Xfer. {a}", "value": f"{100 * r[f'xfer__{a}']:.1f}"})
    table = pd.DataFrame(long).pivot_table(index=rows, columns=["dataset", "metric"], values="value", aggfunc="first")
    table = table.sort_index(axis=1)

    table.to_csv(args.out)
    with pd.option_context("display.max_columns", None, "display.width", 250):
        print(table.to_markdown() if hasattr(table, "to_markdown") else table.to_string())
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
