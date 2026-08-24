"""
Analysis C: PCA of representation space.
Extracts last-token hidden state (final layer) for all 96 inputs per model,
runs PCA, trains linear probes for pde_class / mod_type / phys_valid.
Outputs: pca_repr_results.jsonl (per-row embeddings + metadata)
         pca_repr_probes.json  (probe accuracies)
         pca_repr.html         (interactive 3D plotly scatter)
"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--dataset", default="data/pdedata_clean_v2.xlsx")
parser.add_argument("--output_dir", default="results_pca")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--max_length", type=int, default=2048)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
model_slug = args.model.replace("/", "__")
out_file = Path(args.output_dir) / f"{model_slug}.jsonl"

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

print(f"Loading model {args.model}...")
tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)
model = model.to("cuda")
model.eval()

df = pd.read_excel(args.dataset)
print(f"Dataset: {len(df)} rows")

# Skip already processed
done = set()
if out_file.exists():
    with open(out_file) as f:
        for line in f:
            r = json.loads(line)
            done.add((r["title"], r["mod_type"]))
    print(f"Resuming — {len(done)} rows already done")

rows_written = 0
with open(out_file, "a") as fout:
    for _, row in df.iterrows():
        key = (row["title"], row["mod_type"])
        if key in done:
            continue

        code_col = "code" if "code" in df.columns else df.columns[df.columns.str.contains("code", case=False)][0]
        prompt = PROMPT_TEMPLATE.format(code=row[code_col])

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
        ).to(model.device)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        # Last token, last layer hidden state
        h_last = out.hidden_states[-1][0, -1, :].float().cpu().numpy()

        result = {
            "model": args.model,
            "title": str(row["title"]),
            "pde_class": str(row["pde_class"]),
            "mod_type": str(row["mod_type"]),
            "gt_valid": bool(row["phys_valid"]),
            "hidden": h_last.tolist(),
        }
        fout.write(json.dumps(result) + "\n")
        fout.flush()
        rows_written += 1

        if rows_written % 10 == 0:
            print(f"[{args.model}] {rows_written} rows written")

print(f"Done. {rows_written} new rows -> {out_file}")
