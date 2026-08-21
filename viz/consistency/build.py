"""Entry point: CSV in, four figures and one table out."""
import argparse

import pandas as pd

from . import figures, tables


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/consistency_runs.csv")
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    print(f"[build] {len(df):,} rows from {a.csv}")
    for name, (pdf, png) in figures.build_all(df, outdir=a.outdir).items():
        print(f"[build] {name}: {pdf}, {png}")
    print(f"[build] table: {tables.write_main_results(df, f'{a.outdir}/table_main.tex')}")


if __name__ == "__main__":
    main()
