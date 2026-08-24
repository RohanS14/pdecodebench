"""
Analysis D: Layer-wise linear probing.

For each model, runs a forward pass on all 96 inputs and extracts the
last-token hidden state from EVERY layer. Then trains a linear probe
(5-fold CV LinearSVC) at each layer depth for three tasks:
  - pde_class (4-way: wave/heat/burgers/navier-stokes)
  - mod_type   (6-way: condition labels)
  - gt_valid   (binary: True/False)

Saves per-model results to results_layer_probes/{model_slug}_probes.json
(compact: probe accuracy per layer per task, plus metadata).
Also saves full hidden states as {model_slug}_hidden.npz for future use.

Outputs: layer_probes.html — line chart of probe accuracy vs normalised
         layer depth, one subplot per model, one line per task.

Usage:
    python run_layer_probes.py \
        --dataset data/pdedata_clean_v2.xlsx \
        --output_dir results_layer_probes

    # To also regenerate the HTML from existing probe JSONs (no GPU needed):
    python run_layer_probes.py --analyze_only \
        --output_dir results_layer_probes
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",      default="data/pdedata_clean_v2.xlsx")
parser.add_argument("--output_dir",   default="results_layer_probes")
parser.add_argument("--max_length",   type=int, default=2048)
parser.add_argument("--analyze_only", action="store_true",
                    help="Skip extraction; regenerate HTML from existing probe JSONs")
parser.add_argument("--models", nargs="+", default=[
    "microsoft/phi-4",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen3-32B",
    "Qwen/QwQ-32B",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-27b-it",
])
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
out_dir = Path(args.output_dir)

PROMPT_TEMPLATE = """You are analyzing a numerical simulation written in Python.

<code>
{code}
</code>

Answer the following about this simulation. Be concise.

Output only:
pde: ____
method: ____
behavior: ____
valid: ____

- pde: the type of PDE being solved
- method: numerical method(s) used — list all that apply
- behavior: dominant physical process(es) — list all that apply
- valid: is this a physically valid simulation?"""

TASKS   = ["pde_class", "mod_type", "gt_valid"]
# Chance level for each task (random baseline)
CHANCE  = {"pde_class": 0.25, "mod_type": 1/6, "gt_valid": 0.5}


# ── Probe runner ──────────────────────────────────────────────────────────────
def probe_layers(hidden_np: np.ndarray, df: pd.DataFrame):
    """
    hidden_np: (n_samples, n_layers, hidden_dim) float32
    Returns dict: task -> list of probe_acc per layer (length n_layers)
    """
    from sklearn.svm import LinearSVC
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline

    n_samples, n_layers, _ = hidden_np.shape
    results = {t: [] for t in TASKS}

    for layer_idx in range(n_layers):
        X = hidden_np[:, layer_idx, :]
        for task in TASKS:
            le = LabelEncoder()
            y = le.fit_transform(df[task].astype(str))
            if len(np.unique(y)) < 2:
                results[task].append(None)
                continue
            pipe = Pipeline([
                ("sc", StandardScaler()),
                ("clf", LinearSVC(C=1.0, max_iter=2000)),
            ])
            scores = cross_val_score(pipe, X, y,
                                     cv=min(5, n_samples // 2),
                                     scoring="accuracy")
            results[task].append(round(float(scores.mean()), 4))

        if (layer_idx + 1) % 10 == 0:
            print(f"  Probed {layer_idx + 1}/{n_layers} layers", flush=True)

    return results


# ── Extraction + probing loop ─────────────────────────────────────────────────
if not args.analyze_only:
    df = pd.read_excel(args.dataset)
    print(f"Dataset: {len(df)} rows", flush=True)

    for model_id in args.models:
        slug = model_id.replace("/", "__")
        probe_path = out_dir / f"{slug}_probes.json"
        npz_path   = out_dir / f"{slug}_hidden.npz"

        if probe_path.exists():
            print(f"[SKIP] {model_id} — probes already done", flush=True)
            continue

        print(f"\n{'='*60}", flush=True)
        print(f"Loading {model_id} ...", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model = model.to("cuda")
        model.eval()

        n_layers = model.config.num_hidden_layers
        hidden_dim = model.config.hidden_size
        print(f"  {n_layers} layers, hidden_dim={hidden_dim}", flush=True)

        # Collect last-token hidden states from all layers
        # Shape accumulator: list of (n_layers+1, hidden_dim) per sample
        all_hidden = []  # will become (n_samples, n_layers+1, hidden_dim)

        for i, (_, row) in enumerate(df.iterrows()):
            code_col = "code" if "code" in df.columns else df.columns[
                df.columns.str.contains("code", case=False)][0]
            prompt = PROMPT_TEMPLATE.format(code=row[code_col])
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
            ).to(model.device)

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

            # hidden_states: tuple of (1, seq_len, hidden_dim), one per layer (+embedding)
            # We want the last token from each layer
            layer_vecs = np.array([
                hs[0, -1, :].float().cpu().numpy()
                for hs in out.hidden_states[1:]  # skip embedding layer (index 0)
            ])  # shape: (n_layers, hidden_dim)

            all_hidden.append(layer_vecs)

            if (i + 1) % 16 == 0:
                print(f"  Extracted {i + 1}/{len(df)} rows", flush=True)

        # Stack to (n_samples, n_layers, hidden_dim)
        hidden_np = np.stack(all_hidden, axis=0).astype(np.float32)
        print(f"  Hidden array: {hidden_np.shape}", flush=True)

        # Check for NaN
        nan_frac = np.isnan(hidden_np).mean()
        if nan_frac > 0.01:
            print(f"  WARNING: {nan_frac:.1%} NaN — skipping probes for this model", flush=True)
            del model
            torch.cuda.empty_cache()
            continue

        # Save full hidden states
        np.savez_compressed(str(npz_path), hidden=hidden_np)
        print(f"  Saved hidden states → {npz_path}", flush=True)

        # Build metadata df aligned with hidden_np
        meta_df = df[["pde_class", "mod_type", "phys_valid"]].copy()
        meta_df = meta_df.rename(columns={"phys_valid": "gt_valid"})
        meta_df["gt_valid"] = meta_df["gt_valid"].astype(str)

        # Run probes
        print(f"  Running layer probes ...", flush=True)
        probe_results = probe_layers(hidden_np, meta_df)

        probe_data = {
            "model": model_id,
            "n_layers": n_layers,
            "hidden_dim": hidden_dim,
            "n_samples": len(df),
            "probes": probe_results,
        }
        with open(probe_path, "w") as f:
            json.dump(probe_data, f)
        print(f"  Saved probes → {probe_path}", flush=True)

        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache()

    print("\nExtraction complete.", flush=True)


# ── Visualization ─────────────────────────────────────────────────────────────
import plotly.graph_objects as go
from plotly.subplots import make_subplots

probe_files = sorted(out_dir.glob("*_probes.json"))
if not probe_files:
    print("No probe files found — nothing to visualize.", flush=True)
    sys.exit(0)

all_data = []
for pf in probe_files:
    with open(pf) as f:
        all_data.append(json.load(f))

print(f"\nGenerating visualization for {len(all_data)} models ...", flush=True)

TASK_COLORS = {
    "pde_class": "#636EFA",
    "mod_type":  "#EF553B",
    "gt_valid":  "#00CC96",
}
TASK_LABELS = {
    "pde_class": "PDE class (chance=25%)",
    "mod_type":  "Mod type (chance=17%)",
    "gt_valid":  "Validity (chance=50%)",
}

n_models = len(all_data)
n_cols = min(4, n_models)
n_rows = (n_models + n_cols - 1) // n_cols

subtitles = []
for d in all_data:
    best_pde = max((v for v in d["probes"]["pde_class"] if v is not None), default=0)
    subtitles.append(
        f"{d['model'].split('/')[-1]}<br>"
        f"<sub>{d['n_layers']} layers | best PDE={best_pde:.2f}</sub>"
    )

fig = make_subplots(
    rows=n_rows, cols=n_cols,
    subplot_titles=subtitles,
    horizontal_spacing=0.06,
    vertical_spacing=0.14,
    shared_yaxes=True,
)

for model_idx, d in enumerate(all_data):
    row = model_idx // n_cols + 1
    col = model_idx % n_cols + 1
    n_layers = d["n_layers"]
    xs = [i / (n_layers - 1) for i in range(n_layers)]  # normalized 0→1

    for task in TASKS:
        ys = d["probes"][task]
        ys_clean = [v if v is not None else float("nan") for v in ys]
        chance = CHANCE[task]

        fig.add_trace(
            go.Scatter(
                x=xs, y=ys_clean,
                mode="lines",
                name=TASK_LABELS[task],
                line=dict(color=TASK_COLORS[task], width=2),
                legendgroup=task,
                showlegend=(model_idx == 0),
                hovertemplate=f"{task}<br>layer: %{{x:.2f}}<br>acc: %{{y:.3f}}<extra></extra>",
            ),
            row=row, col=col,
        )
        # Chance baseline
        fig.add_hline(
            y=chance, line_dash="dot",
            line_color=TASK_COLORS[task],
            line_width=1, opacity=0.4,
            row=row, col=col,
        )

fig.update_xaxes(title_text="Layer depth (0=first, 1=last)", range=[0, 1])
fig.update_yaxes(title_text="Probe accuracy", range=[0, 1])
fig.update_layout(
    title=dict(
        text="<b>Layer-wise linear probing</b> — when do models encode PDE knowledge?",
        x=0.5,
    ),
    height=300 * n_rows + 100,
    plot_bgcolor="white",
    paper_bgcolor="white",
    legend=dict(
        orientation="h",
        y=-0.05,
        x=0.5,
        xanchor="center",
        itemsizing="constant",
    ),
    margin=dict(t=100, b=80, l=60, r=20),
)

# Add grid lines
fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)
fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", zeroline=False)

out_html = out_dir / "layer_probes.html"
fig.write_html(str(out_html))
print(f"Written: {out_html}", flush=True)

# Console summary
print(f"\n{'Model':<35} {'Layers':>6} {'Best PDE':>9} {'Best Mod':>9} {'Best Valid':>11}")
for d in all_data:
    p = d["probes"]
    best = lambda t: max((v for v in p[t] if v is not None), default=0)
    print(f"{d['model'].split('/')[-1]:<35} {d['n_layers']:>6} "
          f"{best('pde_class'):>9.3f} {best('mod_type'):>9.3f} {best('gt_valid'):>11.3f}")
