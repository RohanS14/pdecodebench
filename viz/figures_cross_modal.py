"""Export the cross-modal consistency figures as standalone PDF/PNG.

Why this file exists: figures/fig1_blame_matrix.png and its five siblings were
produced by hand on 2026-08-20, straight out of a REPL, from the three-model frozen
run. Nothing in the repo could reproduce them and nothing on the figure said which
roster it was drawn on, so after the roster grew to eight models the PNGs on disk
showed three of them with no way to tell from the file. A figure whose provenance
lives only in someone's shell history is not a figure you can defend in review.

The two rosters are written to SEPARATE directories rather than to distinct filenames
in one directory, because the failure this guards against is overwriting: the frozen
figures back published claims (viz/consistency_claims.html is pinned to the same
rows), and one careless run with the wrong flag would silently replace them with
eight-model versions that answer a different question.

    figures/consistency_frozen/   3 models, greedy   -- the published run
    figures/consistency_roster/   8 models, k=3      -- the generational run

Usage:
    python viz/figures_cross_modal.py                  # both rosters
    python viz/figures_cross_modal.py --roster full    # just the eight
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz.consistency import figures as F                            # noqa: E402
from viz.consistency import style                                   # noqa: E402
from viz.consistency import tables                                  # noqa: E402
from viz.consistency.adapter import FROZEN_REPO, GENERATIONAL_REPOS, load_real  # noqa: E402

ROSTERS = {
    "frozen": (FROZEN_REPO, "figures/consistency_frozen"),
    "full": (GENERATIONAL_REPOS, "figures/consistency_roster"),
}


def export(roster, outdir):
    import matplotlib.pyplot as plt

    d = load_real(repo=roster)
    n_models = d["model"].nunique() if "model" in d.columns else 0
    print(f"[figures] {outdir}: {len(d):,} rows / {n_models} model(s)")

    written = []
    # Figure by figure rather than through build_all(), so one figure that cannot be
    # drawn on this roster does not take the other five with it. fig6 is the case
    # that matters: it plots against release date and is meaningless on three
    # checkpoints, so on the frozen roster it is expected to fail and the rest must
    # still be written.
    todo = [(n, fn, False) for n, fn in F.FIGURES.items()]
    todo += [(n, getattr(F, n), True) for n in F.CAPTIONED_FIGURES]
    for name, fn, captioned in todo:
        try:
            out = fn(d)
            fig = out[0] if captioned else out
            pdf, png = style.save(fig, name, outdir=outdir)
            plt.close(fig)
            written.append(name)
            print(f"[figures]   {png}")
        except Exception as exc:                                    # noqa: BLE001
            print(f"[figures]   SKIP {name}: {type(exc).__name__}: {exc}")

    # The main-results table travels with the figures it belongs to. It used to be
    # written only by viz/consistency/build.py, whose default input is the SYNTHETIC
    # demo frame -- which is how figures/table_main.tex came to hold model-a/b/c
    # numbers under a caption that reads as a real result.
    tex = tables.write_main_results(d, os.path.join(outdir, "table_main.tex"))
    print(f"[figures]   {tex}")
    written.append("table_main")

    # The obfuscation contrast as a table as well as a figure. fig5 is thirteen rows
    # stacked down a portrait axis -- the right form on screen, close to a full page
    # in a text column -- and every row of it already prints its before, its after
    # and its delta, so the only thing the dot plot adds is the interval, which is a
    # column. Both are written; the paper picks.
    try:
        tex = tables.write_obfuscation(d, os.path.join(outdir, "table_obfuscation.tex"))
        print(f"[figures]   {tex}")
        written.append("table_obfuscation")
    except Exception as exc:                                        # noqa: BLE001
        print(f"[figures]   SKIP table_obfuscation: {type(exc).__name__}: {exc}")
    # The dumbbell's quantity -- did it name the right view -- which is NOT the same
    # as where blame goes. fig5_obfuscation_dumbbell.png has no generator in this
    # repo (it was made by hand in 2026-08), so until it has one this table is the
    # only reproducible form those numbers exist in.
    try:
        tex = tables.write_obfuscation_accuracy(
            d, os.path.join(outdir, "table_obfuscation_accuracy.tex"))
        print(f"[figures]   {tex}")
        written.append("table_obfuscation_accuracy")
    except Exception as exc:                                        # noqa: BLE001
        print(f"[figures]   SKIP table_obfuscation_accuracy: "
              f"{type(exc).__name__}: {exc}")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", choices=["frozen", "full", "both"], default="both")
    a = ap.parse_args()
    names = ["frozen", "full"] if a.roster == "both" else [a.roster]
    for n in names:
        roster, outdir = ROSTERS[n]
        export(roster, outdir)


if __name__ == "__main__":
    main()
