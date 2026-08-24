"""
render_sims.py — render PDE simulation solutions to PNG for visual invalidity check.

Self-contained: reads the xlsx, executes each requested title's code, finds the main
solution array (time x space), and renders a 2-panel figure (line slices + space-time
heatmap), matching the style the author used.

Usage:
    python render_sims.py --xlsx data/pdedata_clean_v4.xlsx --out figs \
        --titles Heat_NoComm_InValid_3 Wave_NoComm_InValid_3 Burgers_Invalid2 Burgers_Invalid4
    # or --gt_samples Heat_3 Wave_3 Burgers_2 Burgers_4  (renders NoComm valid+invalid pair)
"""
import argparse, os, runpy, tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _norm(code):
    return str(code).replace("\\r\\n", "\n").replace("\\n\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def _pick_solution(ns):
    """Pick the most likely (time x space) solution array from an exec namespace."""
    best, best_score = None, -1
    for k, v in ns.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and np.issubdtype(v.dtype, np.floating) and v.size > 1:
            # prefer arrays whose axis-0 looks like time (varies across rows)
            score = v.shape[0] * v.shape[1]
            if k.lower() in ("u", "sol", "solution", "u_all", "phi_psi"):
                score *= 10
            if score > best_score:
                best, best_score = v, score
    return best


def run_code(code):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_norm(code)); path = f.name
    try:
        ns = runpy.run_path(path)
        if _pick_solution(ns) is None:
            ns = runpy.run_path(path, run_name="__main__")
        return _pick_solution(ns), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    finally:
        os.unlink(path)


def render(title, code, out_dir):
    sol, err = run_code(code)
    if err:
        print(f"  {title}: ERROR {err}")
        return False
    if sol is None:
        print(f"  {title}: no 2D solution array found")
        return False
    # orient so axis 0 = time (fewer distinct? use the longer axis as space if ambiguous)
    T, X = sol.shape
    x = np.arange(X)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    idxs = sorted(set(int(round(f * (T - 1))) for f in (0, 0.02, 0.05, 0.1, 0.3, 1.0)))
    for i in idxs:
        ax[0].plot(x, sol[i], lw=0.8, label=f"t_idx={i}")
    ax[0].set_xlabel("x"); ax[0].set_ylabel("value"); ax[0].legend(fontsize=7)
    ax[0].set_title(f"{title} — slices")
    im = ax[1].imshow(sol, aspect="auto", origin="lower", cmap="inferno",
                      extent=[0, X, 0, T])
    ax[1].set_xlabel("x"); ax[1].set_ylabel("time index")
    ax[1].set_title("space-time")
    fig.colorbar(im, ax=ax[1])
    stats = f"max|u|={np.max(np.abs(sol)):.3g}  min={np.min(sol):.3g}  nan={bool(np.isnan(sol).any())}"
    fig.suptitle(stats, fontsize=9)
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{title}.png")
    fig.tight_layout(); fig.savefig(p, dpi=90); plt.close(fig)
    print(f"  {title}: rendered -> {p}  ({stats})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="data/pdedata_clean_v4.xlsx")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--titles", nargs="*", default=[])
    ap.add_argument("--gt_samples", nargs="*", default=[])
    args = ap.parse_args()
    df = pd.read_excel(args.xlsx)

    titles = list(args.titles)
    for gt in args.gt_samples:
        for mt in ("NoComm_Valid", "NoComm_InValid"):
            cls, idx = gt.split("_"); titles.append(f"{cls}_{mt}_{idx}")

    print(f"Rendering {len(titles)} sims from {args.xlsx}")
    ok = 0
    for t in titles:
        rows = df[df["title"] == t]
        if rows.empty:
            print(f"  {t}: NOT FOUND"); continue
        ok += render(t, rows["code"].iloc[0], args.out)
    print(f"Done: {ok}/{len(titles)} rendered.")


if __name__ == "__main__":
    main()
