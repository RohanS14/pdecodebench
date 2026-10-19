"""
Upload helper for Experiment 1 (free-generation, run_eval.py).

Runs in a separate subprocess so HF imports never touch the vLLM process — vLLM's
EngineCore dies if huggingface_hub creates threads/connections inside its process.

Recovered from the cluster copy at projects/pde-llm-eval/code/upload_helper.py and
adapted to this repo's conventions: `--workspace` locates the vendored packages, matching
upload_helper_var.py, rather than assuming they sit next to this file.

Why it exists: run_mc_v3_all_models.sbatch defines an inline upload_partial() and calls it
after every model, so Experiment 2 streams partial artifacts. The free-gen sbatch uploads
only after all 10 models finish, so an 8-hour job produces nothing inspectable until the
end. Call this after each model to close that gap.

Usage (per model, from the sbatch loop):
    python upload_helper.py \
        --jsonl           results/Qwen__Qwen2.5-Coder-7B-Instruct.jsonl \
        --hf_dataset      bermaneh/pde-llm-eval-results-jul28 \
        --workspace       /home/ehb7466/pde-llm-eval \
        --packages_dir    /scratch/ehb7466/shared/raca-packages \
        --job_id          torch:12345 \
        --artifact_status partial

    # or aggregate every model written so far into one repo:
    python upload_helper.py --results_dir results/ --hf_dataset ... --workspace ...
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path


COLUMN_DESCRIPTIONS = {
    # ── k>1 sampling and answer-reached flags (added 2026-08-24) ─────────────
    "sample_idx": ("Which of the k sampled draws this row is, 0-based. Rows sharing "
                   "(title, mod_type, model, thinking) are draws of ONE item and are "
                   "NOT independent observations — pool them before any interval."),
    "k_draws": "How many draws were sampled per item in this run (3 for the xmodal roster).",
    "temperature": "Sampling temperature. 0.6 for the xmodal roster, mirroring the cross-representation runner.",
    "top_p": "Nucleus sampling cutoff. 0.95 for the xmodal roster.",
    "top_k": "Top-k sampling cutoff. 20 for the xmodal roster.",
    "sampling_seed": "Seed passed to vLLM SamplingParams, so the draws are reproducible.",
    "no_verdict": ("True when the run never reached an answer — the generation hit "
                   "its token cap (finish_reason 'length') or opened a reasoning "
                   "block it never closed. These rows must be DROPPED, not scored: "
                   "scoring them counts a run that said nothing as one that said "
                   "something false, and files it in the 'Hedged' confidence bucket."),
    "thinking": "Which reasoning arm produced the row, 'on' or 'off'. Part of the row identity.",
    "prompt_version": "Identifier of the prompt template used, so two prompt revisions never pool.",
    "dataset": "Basename of the item CSV this row was generated from.",
    "valid_conf": ("Confidence class of the model's `valid` answer — Confident / "
                   "Hedged / Absent. Canonical rule: freegen_static_judgments/parse_score.py."),
    "title":              "Dataset row identifier, e.g. Wave_Comm_Valid_1",
    "gt_sample":          "Base problem ID, e.g. Wave_1",
    "pde_class":          "Ground truth PDE class (wave/heat/burgers/navier-stokes)",
    "mod_type":           "Modification condition (one of the 8 mod_types)",
    "source":             "Base problem provenance: human or synthetic",
    "gt_pde":             "Ground truth PDE label",
    "gt_method":          "Ground truth numerical method (/-separated if multiple)",
    "gt_behavior":        "Ground truth physical behavior (/-separated if multiple)",
    "gt_valid":           "Ground truth physical validity (True/False)",
    "model_response":     "Full raw model output — never truncated",
    "parsed_pde":         "Parsed pde field from model response",
    "parsed_method":      "Parsed method field from model response",
    "parsed_behavior":    "Parsed behavior field from model response",
    "parsed_valid":       "Parsed valid field from model response",
    "pde_match":          "Binary keyword match for PDE field (0/1)",
    "pde_embed_sim":      "Cosine embedding similarity for PDE field, range [0,1]",
    "method_any_match":   "1 if any ground-truth method token appears in the response",
    "method_recall":      "Fraction of ground-truth method tokens found, range [0,1]",
    "behavior_any_match": "1 if any ground-truth behavior token appears in the response",
    "behavior_recall":    "Fraction of ground-truth behavior tokens found, range [0,1]",
    "valid_match":        "1 if the validity prediction matches ground truth",
    "finish_reason":      "vLLM stop reason — 'length' means the output was truncated",
    "model":              "Model ID used for inference",
    # ── Experiment 2 Part III, cross-modal consistency ────────────────────────
    "item_id":            "Cross-modal item key: system|condition|names|order_seed",
    "condition":          "A0 (all four views agree) or X_C / X_D / X_M / X_T_rand / "
                          "X_T_shuf / X_T_swap / X_T_exec (which view is corrupted)",
    "corrupted_view":     "code / description / math / trajectory, or none for A0",
    "traj_level":         "Which trajectory the item shows: valid, T_rand, T_shuf, "
                          "T_swap (the dataset's own swapped trajectory), or T_exec "
                          "(the invalid solver's executed output)",
    "names":              "real or obfuscated identifier names in the code view",
    "order_seed":         "Which of the two counterbalanced slot orders was used",
    "slot_1":             "Which representation is shown as View 1 (code/math/trajectory/description)",
    "slot_2":             "Which representation is shown as View 2",
    "slot_3":             "Which representation is shown as View 3",
    "slot_4":             "Which representation is shown as View 4",
    "outlier_slot":       "1-4, the slot holding the corrupted view; blank for A0",
    "thinking":           "on or off — whether the model's reasoning mode was enabled",
    "agree":              "Model's answer: yes if it judged all four views consistent",
    "outlier":            "Model's answer: view_1..view_4, or none",
    "system_pde_class":   "Model's answer: the PDE class the majority of views describe",
    "system_num_method":  "Model's answer: the numerical method the majority of views use",
    "justification":      "Model's free-text account of what is inconsistent",
    "parse_route":        "How the JSON was recovered: json / fenced_json / "
                          "embedded_json / failed. 'failed' rows score null, not wrong",
    "detection_correct":  "1 if agree matched whether the item was actually corrupted; "
                          "null when the response could not be parsed",
    "localization_correct": "1 if outlier named the right slot; null when detection "
                          "failed or the response could not be parsed",
    "is_corrupted":       "1 for the seven corrupted conditions, 0 for A0",
    "pde_class_match":    "1 if system_pde_class matched ground truth; null on parse failure",
    "num_method_match":   "1 if system_num_method matched ground truth; null on parse failure",
}


def load_jsonl(path: str) -> list[dict]:
    """Read a results JSONL, tolerating a truncated final line.

    Rows here carry full model responses (100k+ chars), so a job killed mid-append
    leaves a half-written last line. Failing the whole upload on it would discard
    every complete row alongside it -- and the partial row is not lost data, the
    generation loop rewrites it on resume. Anything malformed EARLIER in the file is
    a real problem and still raises.
    """
    rows = []
    with open(path) as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                print(f"[upload] {os.path.basename(path)}: dropping truncated "
                      f"final line ({len(line)} chars); {len(rows)} rows kept")
                break
            raise
    return rows


def union_schema(rows):
    """Give every row the union of all keys, filling absences with None.

    Guards the first-row-wins schema inference in Dataset.from_list: without this,
    appending a new column to the eval output silently drops it for the whole
    dataset whenever an older file happens to sort first.
    """
    keys = {}
    for r in rows:
        for k in r:
            keys.setdefault(k, None)
    missing = {k for r in rows for k in keys if k not in r}
    if missing:
        print(f"[upload_helper] schema union added {sorted(missing)} to rows "
              f"that lacked them", flush=True)
    return [{k: r.get(k) for k in keys} for r in rows]


def _hparams_from_rows(rows):
    """Decoding parameters as the DATA reports them, one entry per distinct value.

    A field absent from every row is reported as "not recorded" rather than guessed:
    older arms predate the sampling instrumentation, and "unknown" is the honest
    answer for them.
    """
    out = {}
    for key in ("thinking", "k_draws", "temperature", "top_p", "top_k",
                "sampling_seed", "prompt_version", "dataset"):
        vals = sorted({str(r[key]) for r in rows if r.get(key) is not None})
        if vals:
            out[key] = vals[0] if len(vals) == 1 else vals
        else:
            out[key] = "not recorded"
    caps = sorted({r["model"] + "=" + str(r.get("max_tokens"))
                   for r in rows if r.get("max_tokens") is not None})
    out["max_tokens"] = caps or "per-model, see run_eval.py MODEL_CONFIGS"
    nv = [r for r in rows if r.get("no_verdict")]
    out["rows_with_no_verdict"] = len(nv)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl",            default=None,
                        help="Single model's JSONL. Mutually exclusive with --results_dir.")
    parser.add_argument("--results_dir",      default=None,
                        help="Aggregate every *.jsonl in this dir (all models so far).")
    parser.add_argument("--hf_dataset",       required=True, help="HF dataset repo id")
    parser.add_argument("--workspace",        required=True,
                        help="Dir containing packages/key_handler and packages/hf_utility")
    parser.add_argument("--packages_dir",     default=None,
                        help="Directory that directly contains key_handler/ and hf_utility/, "
                             "overriding <workspace>/packages. On torch they are vendored at "
                             "/scratch/ehb7466/shared/raca-packages, not inside the repo.")
    parser.add_argument("--experiment",       default="pde-llm-eval",
                        help="HF NAMING slug. Every dataset from this experiment is "
                             "required to start with it, and it is passed to "
                             "push_dataset_to_hub as experiment_slug.")
    # Two different things used to share --experiment, and the dashboard was the
    # one that lost. hf_utility writes the manifest's experiment_id straight from
    # metadata["experiment_id"], which nothing here set, so all eight freegen_static_judgments arms
    # landed in RACA-PROJECT-MANIFEST with experiment_id=None -- and
    # import_experiments.py keeps only rows that HAVE one. The artifacts uploaded
    # fine, verified fine, and were invisible on the Artifacts tab.
    #
    # They are genuinely distinct: the HF slug here is `pde-llm-eval` (it prefixes
    # the dataset names) while the notes folder is `notes/experiments/
    # pde-freegen-xmodal`. Collapsing them would break the naming check.
    parser.add_argument("--experiment_id",    default=None,
                        help="Experiment FOLDER name under notes/experiments/, which "
                             "is what the dashboard joins artifacts on. Defaults to "
                             "--experiment when they happen to match.")
    parser.add_argument("--artifact_status",  default="partial", choices=["partial", "final"])
    parser.add_argument("--job_id",           default="local:0")
    parser.add_argument("--cluster",          default="torch")
    parser.add_argument("--canary",           default="false")
    parser.add_argument("--dataset_file",     default="data/merged_mod_jul28.csv",
                        help="Recorded in metadata so the artifact names its input dataset")
    args = parser.parse_args()

    if not args.jsonl and not args.results_dir:
        parser.error("one of --jsonl or --results_dir is required")

    is_canary = args.canary.lower() == "true"

    # Locate the vendored packages. Each one is its own source root, so both go on
    # sys.path — pointing at the parent alone does not make them importable.
    pkg_root = Path(args.packages_dir) if args.packages_dir else Path(args.workspace) / "packages"
    for name in ("key_handler", "hf_utility"):
        sys.path.insert(0, str(pkg_root / name))

    # Inject API keys
    try:
        from key_handler import KeyHandler
        KeyHandler.set_env_key()
    except Exception as e:
        print(f"[upload_helper] key_handler unavailable: {e}", flush=True)

    try:
        from hf_utility import push_dataset_to_hub
    except ImportError as e:
        sys.exit(f"[upload_helper] FAIL: hf_utility not importable from {pkg_root} ({e}). "
                 f"Pass --packages_dir at the directory holding key_handler/ and hf_utility/. "
                 f"Results are safe on disk; re-run this helper once the path is right.")
    from datasets import Dataset

    if args.results_dir:
        # RECURSIVE, and excluding the backup suffixes this repo writes beside a
        # file it rewrites. The one-directory-per-model layout puts arms at
        # <results_dir>/<model-slug>/<arm>.jsonl, and a flat "*.jsonl" finds nothing
        # there -- the same non-recursive glob that already cost this project silent
        # empty runs in aggregate_freegen.py and rescore_jsonl.py. Picking up a
        # .prerescore or .pretruncfix would be worse than finding nothing: those are
        # superseded copies of rows that are also present in the live file, so they
        # would upload as duplicates that look like real extra draws.
        paths = sorted(p for p in glob.glob(
            os.path.join(args.results_dir, "**", "*.jsonl"), recursive=True)
            if not p.endswith((".prerescore", ".pretruncfix", ".prenormalize")))
        rows = [r for p in paths for r in load_jsonl(p)]
        models = sorted({r.get("model") for r in rows if r.get("model")})
        print(f"[upload_helper] {len(rows)} rows from {len(paths)} files, "
              f"{len(models)} model(s)", flush=True)
        for m in models:
            print(f"[upload_helper]   {m}: {sum(1 for r in rows if r.get('model')==m)} rows",
                  flush=True)
    else:
        rows = load_jsonl(args.jsonl)
        print(f"[upload_helper] {len(rows)} rows from {args.jsonl}", flush=True)

    if not rows:
        print("[upload_helper] Nothing to upload — exiting without error.", flush=True)
        sys.exit(0)

    # Surface truncation immediately: a 'length' finish_reason is a failed row, not a datum.
    truncated = [r.get("title") for r in rows if r.get("finish_reason") == "length"]
    if truncated:
        print(f"[upload_helper] WARNING: {len(truncated)} truncated response(s): "
              f"{truncated[:10]}{' …' if len(truncated) > 10 else ''}", flush=True)

    models = sorted({r.get("model", "") for r in rows if r.get("model")})

    # hf_utility refuses to upload when a string column is empty in every row, on the
    # grounds that it usually means a parsing bug. Sometimes it means the opposite:
    # `protocol_violation` is empty exactly when no response broke the output
    # contract, which is the outcome we want. Name the empty columns out loud so the
    # signal survives, then allow the upload rather than losing the artifact.
    empty_cols = sorted(
        c for c in rows[0]
        if all(isinstance(r.get(c), str) and not r.get(c).strip() for r in rows))
    if empty_cols:
        print(f"[upload_helper] NOTE: column(s) empty in all {len(rows)} rows: "
              f"{empty_cols}. If that is not expected for this run, it is a parsing "
              f"bug and the artifact should not be trusted.", flush=True)
        os.environ["HF_ALLOW_EMPTY_COLUMNS"] = "1"

    # The hf_utility vendored on this cluster predates experiment_slug. Passing it
    # unconditionally raises TypeError and the upload silently never happens -- the
    # job keeps running and the artifact simply never appears. Detect support rather
    # than assume it, and enforce the naming rule locally when it is absent.
    import inspect
    supports_slug = "experiment_slug" in inspect.signature(push_dataset_to_hub).parameters
    slug_kwargs = {}
    if supports_slug:
        slug_kwargs["experiment_slug"] = args.experiment
    else:
        name = args.hf_dataset.split("/")[-1]
        if not name.startswith(args.experiment):
            sys.exit(f"[upload_helper] naming rule: '{name}' must start with "
                     f"'{args.experiment}' (checked locally; this hf_utility has no "
                     f"experiment_slug parameter)")
        print("[upload_helper] note: hf_utility here has no experiment_slug "
              "parameter; the naming rule was checked locally instead.", flush=True)

    # Dataset.from_list takes its schema from the FIRST row only, so a key that is
    # absent there is dropped from every row without a warning. Mixed old/new JSONLs
    # in one results dir put a `thinking`-less row first (QwQ sorts ahead of the
    # rest), which silently deleted the arm label from the artifact -- the same
    # "arm is not what it says" failure the run_eval fix exists to prevent.
    rows = union_schema(rows)

    push_dataset_to_hub(
        dataset=Dataset.from_list(rows),
        dataset_name=args.hf_dataset.split("/")[-1],
        **slug_kwargs,
        metadata={
            "script_name":     "run_eval.py",
            "model":           ", ".join(models) if models else "unknown",
            "description":     ((f"Cross-modal consistency (Experiment 2 Part III) — "
                                 if any("condition" in r and "traj_level" in r for r in rows[:5])
                                 else f"Free-generation PDE eval — ")
                                + f"{args.artifact_status}. {len(rows)} rows from "
                                f"{len(models)} model(s) on {args.dataset_file}."),
            "experiment_name": args.experiment,
            "experiment_id":   args.experiment_id or args.experiment,
            "job_id":          args.job_id,
            "cluster":         args.cluster,
            "artifact_status": args.artifact_status,
            "canary":          is_canary,
            "input_datasets":  [args.dataset_file],
            # READ OFF THE ROWS, never asserted. This dict used to be the literal
            # {"max_tokens": "per-model", "thinking": "suppressed"} regardless of
            # what ran -- so a --thinking on run published a README stating the
            # opposite. That is the F9 failure (an arm mislabelled with nothing
            # recording the truth) relocated into the metadata layer, where it is
            # harder to catch because the rows themselves are right.
            "hyperparameters": _hparams_from_rows(rows),
        },
        tags=sorted({args.experiment, args.experiment_id or args.experiment,
                     "free-gen", "jul28", args.artifact_status}),
        column_descriptions=COLUMN_DESCRIPTIONS,
    )

    print(f"[upload_helper] Done — {len(rows)} rows at {args.hf_dataset} "
          f"({args.artifact_status})", flush=True)


if __name__ == "__main__":
    main()
