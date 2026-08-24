"""
Reads pca_repr_results/{model}.jsonl files, runs PCA + linear probes,
outputs pca_repr.html with interactive 3D scatter.
"""
import json, glob, argparse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="results_pca")
parser.add_argument("--output", default="pca_repr.html")
args = parser.parse_args()

# Load all embeddings
records = []
for f in sorted(glob.glob(f"{args.input_dir}/*.jsonl")):
    with open(f) as fh:
        for line in fh:
            r = json.loads(line)
            records.append(r)

print(f"Loaded {len(records)} records from {len(set(r['model'] for r in records))} models")

df = pd.DataFrame([{k: v for k, v in r.items() if k != "hidden"} for r in records])
X = np.array([r["hidden"] for r in records])

models = df["model"].unique()
figs_data = []

for model in models:
    mask = df["model"] == model
    Xm = X[mask]
    dfm = df[mask].reset_index(drop=True)

    pca = PCA(n_components=3)
    Xp = pca.fit_transform(Xm)
    var = pca.explained_variance_ratio_

    # Probe accuracies
    probes = {}
    for target in ["pde_class", "mod_type", "gt_valid"]:
        le = LabelEncoder()
        y = le.fit_transform(dfm[target].astype(str))
        if len(np.unique(y)) < 2:
            probes[target] = 0.0
            continue
        clf = LinearSVC(C=1.0, max_iter=2000)
        scores = cross_val_score(clf, Xp, y, cv=min(5, len(y)//2), scoring="accuracy")
        probes[target] = round(float(scores.mean()), 3)

    figs_data.append({
        "model": model.split("/")[-1],
        "Xp": Xp,
        "dfm": dfm,
        "var": var,
        "probes": probes,
    })
    print(f"{model.split('/')[-1]}: probes={probes}, var={var.round(3)}")

# Build plotly: one 3D scatter per model, colored by pde_class
COLORS = {"wave": "#636EFA", "heat": "#EF553B", "burgers": "#00CC96", "navier-stokes": "#AB63FA"}

fig = go.Figure()
buttons = []
traces_per_model = {}

for i, fd in enumerate(figs_data):
    Xp, dfm = fd["Xp"], fd["dfm"]
    model_name = fd["model"]
    traces_per_model[model_name] = []

    for color_by in ["pde_class", "mod_type", "gt_valid"]:
        labels = dfm[color_by].astype(str)
        for val in labels.unique():
            m = labels == val
            t = go.Scatter3d(
                x=Xp[m, 0], y=Xp[m, 1], z=Xp[m, 2],
                mode="markers",
                marker=dict(size=5, opacity=0.8),
                name=f"{val}",
                text=[f"{dfm['title'][j]}<br>{dfm['mod_type'][j]}" for j in dfm[m].index],
                hoverinfo="text+name",
                visible=(i == 0 and color_by == "pde_class"),
                legendgroup=f"{model_name}_{color_by}",
            )
            fig.add_trace(t)
            traces_per_model[model_name].append((color_by, len(fig.data) - 1))

fig.update_layout(
    title="PCA of Model Representations (last token, final layer)",
    height=700,
    scene=dict(
        xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"
    ),
    margin=dict(l=0, r=200, t=60, b=0),
)

fig.write_html(args.output)
print(f"Written: {args.output}")

# Print probe summary table
print("\nProbe accuracy summary:")
print(f"{'Model':<30} {'PDE class':>10} {'Mod type':>10} {'Valid':>8}")
for fd in figs_data:
    p = fd["probes"]
    print(f"{fd['model']:<30} {p.get('pde_class',0):>10.3f} {p.get('mod_type',0):>10.3f} {p.get('gt_valid',0):>8.3f}")
