"""
Reads results_pca/{model}.jsonl files, runs PCA + linear probes per model.
Outputs pca_repr.html with:
  - Section A: probe accuracy heatmap (models × tasks)
  - Section B: 2D scatter grid (PC1 vs PC2, colored by pde_class)
  - Section C: 2D scatter grid colored by mod_type
  - Section D: 2D scatter grid colored by validity
"""
import json, glob, argparse, re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="results_pca")
parser.add_argument("--output", default="pca_repr.html")
args = parser.parse_args()

# ── Load ──────────────────────────────────────────────────────────────────────
by_model = {}
for f in sorted(glob.glob(f"{args.input_dir}/*.jsonl")):
    with open(f) as fh:
        for line in fh:
            r = json.loads(line)
            m = r["model"]
            if m not in by_model:
                by_model[m] = []
            by_model[m].append(r)

print(f"Loaded {sum(len(v) for v in by_model.values())} records from {len(by_model)} models")

# ── PCA + probes per model ────────────────────────────────────────────────────
results = []
for model, recs in sorted(by_model.items()):
    X = np.array([r["hidden"] for r in recs])
    nan_frac = np.isnan(X).mean()
    short = model.split("/")[-1]
    if nan_frac > 0.1:
        print(f"SKIP {short}: {nan_frac:.0%} NaN (float16 overflow)")
        continue

    meta = [{k: v for k, v in r.items() if k != "hidden"} for r in recs]
    dfm = pd.DataFrame(meta).reset_index(drop=True)

    pca = PCA(n_components=2)
    Xp = pca.fit_transform(StandardScaler().fit_transform(X))
    var = pca.explained_variance_ratio_

    probes = {}
    for target in ["pde_class", "mod_type", "gt_valid"]:
        le = LabelEncoder()
        y = le.fit_transform(dfm[target].astype(str))
        if len(np.unique(y)) < 2:
            probes[target] = None
            continue
        pipe = Pipeline([("sc", StandardScaler()), ("clf", LinearSVC(C=1.0, max_iter=2000))])
        scores = cross_val_score(pipe, X, y, cv=min(5, len(y) // 2), scoring="accuracy")
        probes[target] = round(float(scores.mean()), 3)

    results.append({
        "model": model, "short": short,
        "Xp": Xp, "dfm": dfm, "var": var, "probes": probes,
        "hidden_dim": X.shape[1],
    })
    print(f"{short} (dim={X.shape[1]}): probes={probes}, var={var.round(3)}")

n_models = len(results)
model_order = [r["short"] for r in results]

# ── Color palettes ────────────────────────────────────────────────────────────
PDE_COLORS = {
    "wave":          "#636EFA",
    "heat":          "#EF553B",
    "burgers":       "#00CC96",
    "navier-stokes": "#AB63FA",
}
MOD_COLORS = {
    "Comm_Valid":      "#1F77B4",
    "NoComm_Valid":    "#AEC7E8",
    "NoComm_InValid":  "#FF7F0E",
    "Comm_InValid":    "#FFBB78",
    "CorrComm":        "#2CA02C",
    "NoComm_CorrVar":  "#98DF8A",
}
VALID_COLORS = {"True": "#2CA02C", "False": "#D62728"}


def make_scatter_grid(results, color_col, palette, title_suffix, n_cols=4):
    """Return a plotly Figure with a 2D scatter grid (one subplot per model)."""
    n_rows = (len(results) + n_cols - 1) // n_cols
    subtitles = []
    for r in results:
        p = r["probes"].get(color_col)
        probe_str = f"probe={p:.2f}" if p is not None else "probe=—"
        subtitles.append(f"{r['short']}<br><sub>{probe_str} | var {r['var'][0]:.0%}+{r['var'][1]:.0%}</sub>")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=subtitles,
        horizontal_spacing=0.06,
        vertical_spacing=0.12,
    )

    legend_added = set()
    for idx, r in enumerate(results):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        Xp, dfm = r["Xp"], r["dfm"]
        labels = dfm[color_col].astype(str)

        for val in sorted(labels.unique()):
            mask = labels == val
            color = palette.get(val, "#888888")
            show_legend = val not in legend_added
            legend_added.add(val)
            fig.add_trace(
                go.Scatter(
                    x=Xp[mask, 0], y=Xp[mask, 1],
                    mode="markers",
                    marker=dict(size=6, color=color, opacity=0.85,
                                line=dict(width=0.5, color="white")),
                    name=val,
                    text=[f"{dfm.loc[j,'title']}<br>{dfm.loc[j,'mod_type']}" for j in dfm[mask].index],
                    hoverinfo="text+name",
                    legendgroup=val,
                    showlegend=show_legend,
                ),
                row=row, col=col,
            )

    fig.update_xaxes(title_text="PC1", showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(title_text="PC2", showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(
        title=dict(text=f"<b>PCA of last-token hidden states</b> — colored by {title_suffix}", x=0.5),
        height=260 * n_rows + 80,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(itemsizing="constant", tracegroupgap=4),
        margin=dict(t=80, b=20, l=20, r=20),
    )
    return fig


# ── Section A: probe heatmap ──────────────────────────────────────────────────
probe_tasks  = ["pde_class", "mod_type", "gt_valid"]
probe_labels = ["PDE class", "Mod type", "Phys valid"]
# Chance levels: PDE class=0.25 (4-way), mod_type=0.167 (6-way), valid=0.5
chance = [0.25, 0.167, 0.5]

z = []
for r in results:
    z.append([r["probes"].get(t) for t in probe_tasks])
z_arr = np.array([[v if v is not None else float("nan") for v in row] for row in z])

# Normalize by chance (lift over random)
z_lift = np.where(
    np.isnan(z_arr), np.nan,
    (z_arr - np.array(chance)) / (1.0 - np.array(chance))
)

text_ann = [[
    f"{z_arr[i, j]:.2f}" if not np.isnan(z_arr[i, j]) else "—"
    for j in range(len(probe_tasks))
] for i in range(len(results))]

fig_probe = go.Figure(go.Heatmap(
    z=z_lift,
    x=probe_labels,
    y=model_order,
    text=text_ann,
    texttemplate="%{text}",
    colorscale="RdYlGn",
    zmid=0,
    zmin=-0.3,
    zmax=0.7,
    colorbar=dict(title="Lift over<br>chance", tickformat=".0%"),
))
fig_probe.update_layout(
    title=dict(text="<b>Linear probe accuracy</b> (color = lift over random chance)", x=0.5),
    height=60 * len(results) + 100,
    xaxis=dict(side="top"),
    margin=dict(t=100, b=20, l=180, r=60),
    plot_bgcolor="white",
    paper_bgcolor="white",
)

# ── Section B/C/D: scatter grids ──────────────────────────────────────────────
fig_pde  = make_scatter_grid(results, "pde_class", PDE_COLORS, "PDE class (wave/heat/burgers/navier-stokes)")
fig_mod  = make_scatter_grid(results, "mod_type",  MOD_COLORS, "modification type")
fig_val  = make_scatter_grid(results, "gt_valid",  VALID_COLORS, "physical validity")

# ── Write combined HTML ───────────────────────────────────────────────────────
html_parts = [
    "<html><head><meta charset='utf-8'><title>PCA Representation Analysis</title>",
    "<style>body{font-family:sans-serif;background:#f8f9fa;padding:20px}",
    "h2{color:#333;margin-top:40px}.section{background:white;border-radius:8px;",
    "padding:20px;margin-bottom:30px;box-shadow:0 1px 4px rgba(0,0,0,.1)}</style></head><body>",
    "<h1 style='color:#222'>PCA of LLM Representation Space</h1>",
    "<p>Last-token hidden state (final layer) for 96 PDE code snippets per model. ",
    "PCA run independently per model (hidden dimensions differ across architectures). ",
    "Probe = 5-fold CV linear SVC accuracy.</p>",
    "<div class='section'><h2>Probe Accuracy</h2>",
    fig_probe.to_html(full_html=False, include_plotlyjs="cdn"),
    "</div><div class='section'><h2>Scatter by PDE class</h2>",
    fig_pde.to_html(full_html=False, include_plotlyjs=False),
    "</div><div class='section'><h2>Scatter by modification type</h2>",
    fig_mod.to_html(full_html=False, include_plotlyjs=False),
    "</div><div class='section'><h2>Scatter by physical validity</h2>",
    fig_val.to_html(full_html=False, include_plotlyjs=False),
    "</div></body></html>",
]

with open(args.output, "w") as f:
    f.write("\n".join(html_parts))
print(f"Written: {args.output}")

# ── Console summary ───────────────────────────────────────────────────────────
print(f"\n{'Model':<35} {'dim':>5} {'PDE':>6} {'Mod':>6} {'Valid':>6}")
for r in results:
    p = r["probes"]
    print(f"{r['short']:<35} {r['hidden_dim']:>5} "
          f"{p.get('pde_class',0) or 0:>6.3f} {p.get('mod_type',0) or 0:>6.3f} "
          f"{p.get('gt_valid',0) or 0:>6.3f}")
