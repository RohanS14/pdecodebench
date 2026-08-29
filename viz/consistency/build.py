"""Entry point: CSV in, four figures and one table out.

DEFAULTS ARE THE SYNTHETIC DEMO, not the experiment. --csv defaults to
results/consistency_runs.csv, which viz/consistency/synth.py generates with models
named model-a/b/c over solvers S00..; it exists so the figure code can be exercised
without a GPU run. The output therefore defaults to figures/synthetic_demo/ and NOT
to figures/, because it did default to figures/ and figures/table_main.tex sat in the
paper's figure directory for five days holding fabricated numbers under a caption
that read as a real result -- including a judge-confirmed column for an LLM-judge pass
that has never been run.

For the real thing use viz/export_consistency_figures.py, which loads the published
repos and writes figures/consistency_frozen/ and figures/consistency_roster/.
"""
import argparse

import pandas as pd

from . import figures, tables


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default="results/consistency_runs.csv")
    ap.add_argument("--outdir", default="figures/synthetic_demo")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    print(f"[build] {len(df):,} rows from {a.csv}")
    # Loud, because the numbers themselves look entirely plausible.
    if set(df.get("model", [])) <= {"model-a", "model-b", "model-c"}:
        print(f"[build] *** SYNTHETIC INPUT *** {a.csv} holds placeholder models. "
              f"Nothing written to {a.outdir} is a result. Use "
              f"viz/export_consistency_figures.py for the real rosters.")
    for name, (pdf, png) in figures.build_all(df, outdir=a.outdir).items():
        print(f"[build] {name}: {pdf}, {png}")
    print(f"[build] table: {tables.write_main_results(df, f'{a.outdir}/table_main.tex')}")


if __name__ == "__main__":
    main()
