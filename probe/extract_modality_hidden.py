"""
Hidden states for the non-code modalities of Experiment 2 Part II (plan §15).

extract_hidden.py embeds solver CODE. This embeds the two modalities that contain
no code at all:

    equation    the PDE in symbolic form   (data/equations_jul28.csv)
    trajectory  the executed dynamics      (trajectories_jul28.json)

Both are wrapped in a neutral prompt that names neither the PDE class nor the
numerical method, so a retrieval hit cannot be explained by the prompt handing
over the label.

Equations are embedded in all three notations (unicode / latex / ascii) so
cross-modal retrieval can be scored against notation as a nuisance: if a hit
survives changing notation, it is not a notation match.

Usage (from repo root):
    python probe/extract_modality_hidden.py \
        --modality equation \
        --input data/equations_jul28.csv \
        --model Qwen/Qwen2.5-Coder-7B-Instruct \
        --output_dir probe/hidden_states/
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.dataset_io import load_dataset  # noqa: E402

# Deliberately generic. No "PDE", no class name, no method name — the prompt must
# not supply the answer the retrieval test is asking the representation for.
PROMPTS = {
    "equation": (
        "Consider the following mathematical description of a physical system.\n\n"
        "{content}\n\n"
        "Think about what physical system this describes and how it evolves."
    ),
    "trajectory": (
        "Consider the following description of how a simulated physical field "
        "evolved.\n\n"
        "{content}\n\n"
        "Think about what physical system would produce this behaviour."
    ),
}
NOTATIONS = ["unicode", "latex", "ascii"]


def load_items(modality: str, path: str) -> list:
    """Return [{key, gt_sample, variant, content}] for the chosen modality."""
    items = []
    if modality == "equation":
        df = load_dataset(path)
        need = {"gt_sample", "equation_unicode", "equation_latex", "equation_ascii"}
        missing = need - set(df.columns)
        if missing:
            raise SystemExit(f"{path} missing columns: {sorted(missing)}")
        unreviewed = int(df.get("needs_review", 0).sum()) if "needs_review" in df else 0
        if unreviewed:
            print(f"WARNING: {unreviewed}/{len(df)} equations are still flagged "
                  f"needs_review=1. They are the physics ground truth for every "
                  f"cross-modal number — get them signed off before reporting.",
                  flush=True)
        for _, r in df.iterrows():
            for nota in NOTATIONS:
                content = str(r[f"equation_{nota}"]).strip()
                if not content:
                    continue
                items.append({"key": f"{r['gt_sample']}|{nota}",
                              "gt_sample": r["gt_sample"], "variant": nota,
                              "pde_class": r.get("pde_class", ""), "content": content})
    elif modality == "trajectory":
        with open(path) as f:
            traj = json.load(f)
        for k, rec in sorted(traj.items()):
            if not rec.get("text"):
                continue
            items.append({"key": k, "gt_sample": rec["gt_sample"],
                          "variant": rec["mod_type"],
                          "pde_class": rec.get("pde_class", ""),
                          "content": rec["text"]})
    else:
        raise ValueError(modality)
    if not items:
        raise SystemExit(f"no items built from {path}")
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", required=True, choices=["equation", "trajectory"])
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    ap.add_argument("--output_dir", default="probe/hidden_states/")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This script requires a GPU.", flush=True)
        sys.exit(1)

    items = load_items(args.modality, args.input)
    print(f"Modality: {args.modality}   items: {len(items)}", flush=True)
    print(f"Model:    {args.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # see extract_hidden.py: torch_dtype -> dtype in transformers v5
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True)
    model.eval()

    L = model.config.num_hidden_layers + 1
    D = model.config.hidden_size
    N = len(items)
    print(f"Allocating 2 × ({N}, {L}, {D}) float32 = "
          f"{2 * N * L * D * 4 / 1e6:.1f} MB", flush=True)

    # float32 for the same reason as extract_hidden.py — these feed cosine
    # comparisons against code representations.
    mean_pool = np.zeros((N, L, D), dtype=np.float32)
    last_tok = np.zeros((N, L, D), dtype=np.float32)

    for i, it in enumerate(items):
        text = PROMPTS[args.modality].format(content=it["content"])
        formatted = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False, add_generation_prompt=True)

        start = formatted.find(it["content"])
        if start == -1:
            span = None
        else:
            end = start + len(it["content"])
            enc0 = tok(formatted, return_offsets_mapping=True,
                       add_special_tokens=False)
            offs = enc0["offset_mapping"]
            idxs = [j for j, (s, e) in enumerate(offs)
                    if not (s == 0 and e == 0) and e > start and s < end]
            span = (idxs[0], idxs[-1] + 1) if idxs else None

        enc = tok(formatted, return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to("cuda")
        if span is None:
            span = (0, ids.shape[1])
            print(f"  WARNING item {i} ({it['key']}): content span not found, "
                  f"pooling over the full prompt", flush=True)

        with torch.no_grad():
            out = model(ids, output_hidden_states=True)
        for l, hs in enumerate(out.hidden_states):
            h = hs[0]
            mean_pool[i, l] = h[span[0]:span[1]].mean(dim=0).float().cpu().numpy()
            last_tok[i, l] = h[span[1] - 1].float().cpu().numpy()

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1:3d}/{N}] {it['key']}  content_toks={span[1]-span[0]}",
                  flush=True)

    for name, arr in [("mean_pool", mean_pool), ("last_tok", last_tok)]:
        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"ERROR: {name} contains NaN/Inf", flush=True)
            sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    slug = args.model.replace("/", "_")
    out_path = os.path.join(args.output_dir, f"{slug}_{args.modality}.npz")
    np.savez(
        out_path,
        mean_pool=mean_pool, last_tok=last_tok,
        keys=np.array([it["key"] for it in items]),
        gt_samples=np.array([it["gt_sample"] for it in items]),
        variants=np.array([it["variant"] for it in items]),
        pde_classes=np.array([it["pde_class"] for it in items]),
        contents=np.array([it["content"] for it in items]),
        modality=np.array(args.modality),
        model_name=np.array(args.model),
    )
    print(f"\nSaved: {out_path}  {mean_pool.shape} float32", flush=True)


if __name__ == "__main__":
    sys.exit(main())
