#!/usr/bin/env python3
"""
Evaluate local or HuggingFace-compatible LLMs on PDE code understanding.

HPC usage examples:

  module load cuda
  source /scratch/$USER/venvs/pdecodebench/bin/activate
  python free_gen_exp/evaluate_llms_pde.py \
    --data-path data/pdedata.xlsx \
    --output-dir /scratch/$USER/pdecodebench/free_gen_results \
    --batch-size 4 \
    --device cuda \
    --max-new-tokens 96

Slurm sketch:

  #!/bin/bash
  #SBATCH --job-name=pdecodebench-llm
  #SBATCH --gres=gpu:1
  #SBATCH --cpus-per-task=8
  #SBATCH --mem=64G
  #SBATCH --time=08:00:00
  module load cuda
  source /scratch/$USER/venvs/pdecodebench/bin/activate
  cd /home/$USER/pdecodebench
  python free_gen_exp/evaluate_llms_pde.py --data-path data/pdedata.xlsx --device cuda

Optional embedding metric:
  pip install sentence-transformers
  python free_gen_exp/evaluate_llms_pde.py ... --embedding-similarity
"""

from __future__ import annotations

import argparse
import ast
import builtins
import gc
import json
import keyword
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


TARGET_FIELDS = ("equation", "method", "behavior")
GT_COLUMNS = {
    "equation": "pde_class",
    "method": "num_method",
    "behavior": "phys_process",
}
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-2-9b-it",
    "mistralai/Mistral-Nemo-Instruct-2407",
]

SYSTEM_PROMPT = """You are a precise scientific code analysis assistant.
Analyze Python source code for PDE simulations.
Return only the requested structured answer. Do not include explanations."""

USER_PROMPT_TEMPLATE = """Analyze the following PDE simulation code.

Infer:
- equation: the PDE or equation class represented by the code
- method: the numerical method used by the code
- behavior: the physical process represented by the code

Return ONLY this exact 3-line format:
equation: <...>
method: <...>
behavior: <...>

Code:
```python
{code}
```"""

BUILTIN_NAMES = set(dir(builtins))
KEYWORD_NAMES = set(keyword.kwlist)


@dataclass
class Example:
    row_id: int
    title: str
    code: str
    pde_class: str
    num_method: str
    phys_process: str
    phys_valid: bool
    mod_type: str
    condition: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM structured PDE-code understanding on pdedata.xlsx."
    )
    parser.add_argument("--data-path", default="pdedata.xlsx", help="Path to pdedata.xlsx.")
    parser.add_argument(
        "--sheet-name",
        default=0,
        help="Excel sheet name or integer index. Default: first sheet.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="One or more local paths or HuggingFace model IDs.",
    )
    parser.add_argument("--output-dir", default="free_gen_exp/results", help="Output directory.")
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device placement. Use auto for HF device_map='auto' when CUDA is available.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        help=(
            "Optional prompt token limit. Defaults to the model context window "
            "minus --max-new-tokens when that context length is known."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for smoke tests.")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["original", "obfuscated"],
        choices=["original", "obfuscated"],
        help="Code conditions to evaluate.",
    )
    parser.add_argument(
        "--embedding-similarity",
        action="store_true",
        help="Compute sentence embedding similarity between raw response and rubric reference.",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for optional embedding metric.",
    )
    parser.add_argument("--save-raw-json", action="store_true", help="Save raw generations JSON.")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def normalize_sheet_name(value: Any) -> Any:
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def resolve_data_path(data_path: str | Path) -> Path:
    path = Path(data_path)
    if path.exists():
        return path

    repo_data_path = Path(__file__).resolve().parents[1] / "data" / path.name
    if repo_data_path.exists():
        return repo_data_path

    raise FileNotFoundError(f"Could not find dataset at {path} or {repo_data_path}")


def normalize_code_text(code: Any) -> str:
    if pd.isna(code):
        return ""
    text = str(code)
    # Existing dataset cells often encode each logical newline as "\\n\n".
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")
    return text


def load_data(data_path: str | Path, sheet_name: Any = 0, limit: int | None = None) -> list[Example]:
    df = pd.read_excel(resolve_data_path(data_path), sheet_name=normalize_sheet_name(sheet_name))
    required = ["code", "pde_class", "num_method", "phys_process", "phys_valid"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    valid = df["phys_valid"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    df = df.loc[valid].copy()
    if limit is not None:
        df = df.head(limit).copy()

    examples: list[Example] = []
    for row_id, row in df.iterrows():
        examples.append(
            Example(
                row_id=int(row_id),
                title=str(row.get("title", row_id)),
                code=normalize_code_text(row["code"]),
                pde_class=str(row["pde_class"]),
                num_method=str(row["num_method"]),
                phys_process=str(row["phys_process"]),
                phys_valid=True,
                mod_type=str(row.get("mod_type", "")),
                condition="original",
            )
        )
    return examples


def expand_conditions(examples: list[Example], conditions: Iterable[str]) -> list[Example]:
    expanded: list[Example] = []
    for example in examples:
        if "original" in conditions:
            expanded.append(example)
        if "obfuscated" in conditions:
            expanded.append(
                Example(
                    row_id=example.row_id,
                    title=example.title,
                    code=obfuscate_variables(example.code),
                    pde_class=example.pde_class,
                    num_method=example.num_method,
                    phys_process=example.phys_process,
                    phys_valid=example.phys_valid,
                    mod_type=example.mod_type,
                    condition="obfuscated",
                )
            )
    return expanded


def build_user_prompt(code: str) -> str:
    return USER_PROMPT_TEMPLATE.format(code=code)


def build_prompt(tokenizer: Any, code: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(code)},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\n{messages[1]['content']}\n"


def load_generation_model(
    model_name: str,
    device: str,
    torch_dtype: str,
    trust_remote_code: bool,
    local_files_only: bool,
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[torch_dtype]

    kwargs = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
        "dtype": dtype,
    }
    if device == "auto" and torch.cuda.is_available():
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if kwargs["device_map"] is None:
        target_device = resolve_device(device)
        model = model.to(target_device)
    model.eval()
    return tokenizer, model, torch


def resolve_device(device: str) -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested --device mps, but MPS is not available.")
    return device


def generate_outputs(
    tokenizer: Any,
    model: Any,
    torch_mod: Any,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    max_input_tokens: int | None,
    temperature: float,
    top_p: float,
) -> list[str]:
    generations: list[str] = []
    do_sample = temperature > 0

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        input_token_limit = resolve_input_token_limit(tokenizer, model, max_new_tokens, max_input_tokens)
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=input_token_limit is not None,
            max_length=input_token_limit,
        )
        model_device = next(model.parameters()).device
        encoded = {key: value.to(model_device) for key, value in encoded.items()}

        with torch_mod.no_grad():
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            output_ids = model.generate(**encoded, **generation_kwargs)

        input_lengths = encoded["input_ids"].shape[1]
        new_tokens = output_ids[:, input_lengths:]
        generations.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
    return generations


def resolve_input_token_limit(
    tokenizer: Any,
    model: Any,
    max_new_tokens: int,
    max_input_tokens: int | None,
) -> int | None:
    if max_input_tokens is not None:
        return max_input_tokens

    candidates = []
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and tokenizer_limit < 1_000_000:
        candidates.append(tokenizer_limit)

    config = getattr(model, "config", None)
    for attr in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(config, attr, None)
        if isinstance(value, int) and value > 0:
            candidates.append(value)

    if not candidates:
        return None

    context_limit = min(candidates)
    return max(1, context_limit - max_new_tokens)


def parse_structured_output(text: str) -> dict[str, str]:
    parsed = {field: "" for field in TARGET_FIELDS}
    line_pattern = re.compile(r"^\s*(equation|method|behavior)\s*[:\-]\s*(.*?)\s*$", re.I)
    inline_pattern = re.compile(r"(equation|method|behavior)\s*[:\-]\s*(.*?)(?=\n\s*(?:equation|method|behavior)\s*[:\-]|\Z)", re.I | re.S)

    for line in text.strip().splitlines():
        match = line_pattern.match(line)
        if match:
            parsed[match.group(1).lower()] = clean_value(match.group(2))

    if any(not parsed[field] for field in TARGET_FIELDS):
        for match in inline_pattern.finditer(text.strip()):
            field = match.group(1).lower()
            if not parsed[field]:
                parsed[field] = clean_value(match.group(2))
    return parsed


def clean_value(value: str) -> str:
    value = value.strip().strip("`")
    value = re.sub(r"\s+", " ", value)
    return value.strip(" .;")


def normalize_label(value: Any) -> str:
    text = str(value).lower().strip()
    text = text.replace("_", "-")
    text = re.sub(r"[^a-z0-9/\-\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_keywords(value: Any) -> set[str]:
    normalized = normalize_label(value)
    pieces = re.split(r"[/,\s]+", normalized)
    return {piece for piece in pieces if piece and piece not in {"and", "or", "the", "a", "an"}}


def field_metrics(pred: str, truth: str) -> dict[str, float]:
    pred_norm = normalize_label(pred)
    truth_norm = normalize_label(truth)
    pred_kw = split_keywords(pred)
    truth_kw = split_keywords(truth)
    keyword_match = bool(truth_kw) and truth_kw.issubset(pred_kw)
    keyword_f1 = token_f1(pred_kw, truth_kw)
    return {
        "exact_match": float(pred == truth),
        "normalized_match": float(pred_norm == truth_norm),
        "keyword_match": float(keyword_match),
        "keyword_f1": keyword_f1,
    }


def token_f1(pred_tokens: set[str], truth_tokens: set[str]) -> float:
    if not pred_tokens and not truth_tokens:
        return 1.0
    if not pred_tokens or not truth_tokens:
        return 0.0
    overlap = len(pred_tokens & truth_tokens)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)


def build_reference_text(example: Example) -> str:
    return (
        f"equation: {example.pde_class}\n"
        f"method: {example.num_method}\n"
        f"behavior: {example.phys_process}"
    )


def cosine_similarity(vec_a: Any, vec_b: Any) -> float:
    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
    norm_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_embedding_similarities(
    references: list[str],
    responses: list[str],
    model_name: str,
    device: str,
) -> list[float]:
    from sentence_transformers import SentenceTransformer

    embed_device = resolve_device(device)
    model = SentenceTransformer(model_name, device=embed_device)
    ref_embeddings = model.encode(references, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    pred_embeddings = model.encode(responses, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
    return [cosine_similarity(ref, pred) for ref, pred in zip(ref_embeddings, pred_embeddings)]


def compute_prediction_rows(
    model_name: str,
    examples: list[Example],
    generations: list[str],
    embedding_scores: list[float] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, (example, generation) in enumerate(zip(examples, generations)):
        parsed = parse_structured_output(generation)
        row: dict[str, Any] = {
            "model": model_name,
            "row_id": example.row_id,
            "title": example.title,
            "condition": example.condition,
            "mod_type": example.mod_type,
            "pde_class": example.pde_class,
            "num_method": example.num_method,
            "phys_process": example.phys_process,
            "raw_generation": generation,
            "pred_equation": parsed["equation"],
            "pred_method": parsed["method"],
            "pred_behavior": parsed["behavior"],
        }

        for field in TARGET_FIELDS:
            truth = getattr(example, GT_COLUMNS[field])
            metrics = field_metrics(parsed[field], truth)
            for metric_name, metric_value in metrics.items():
                row[f"{field}_{metric_name}"] = metric_value

        row["all_exact_match"] = float(
            all(row[f"{field}_exact_match"] == 1.0 for field in TARGET_FIELDS)
        )
        row["all_normalized_match"] = float(
            all(row[f"{field}_normalized_match"] == 1.0 for field in TARGET_FIELDS)
        )
        row["all_keyword_match"] = float(
            all(row[f"{field}_keyword_match"] == 1.0 for field in TARGET_FIELDS)
        )
        if embedding_scores is not None:
            row["embedding_similarity"] = embedding_scores[idx]
        rows.append(row)
    return rows


def summarize_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col
        for col in predictions.columns
        if col.endswith(("_exact_match", "_normalized_match", "_keyword_match", "_keyword_f1"))
        or col in {"all_exact_match", "all_normalized_match", "all_keyword_match", "embedding_similarity"}
    ]
    grouped = predictions.groupby(["model", "condition"], dropna=False)
    summary = grouped[metric_cols].mean().reset_index()
    summary["n_examples"] = grouped.size().to_numpy()

    if set(["model", "row_id", "condition", "all_normalized_match"]).issubset(predictions.columns):
        degradation_rows = []
        for model_name, model_df in predictions.groupby("model"):
            if {"original", "obfuscated"}.issubset(set(model_df["condition"])):
                pivot = model_df.pivot_table(
                    index="row_id",
                    columns="condition",
                    values="all_normalized_match",
                    aggfunc="mean",
                ).dropna()
                if not pivot.empty:
                    degradation_rows.append(
                        {
                            "model": model_name,
                            "condition": "obfuscation_degradation",
                            "all_normalized_match": pivot["original"].mean() - pivot["obfuscated"].mean(),
                            "n_examples": len(pivot),
                        }
                    )
        if degradation_rows:
            summary = pd.concat([summary, pd.DataFrame(degradation_rows)], ignore_index=True, sort=False)
    return summary


class VariableCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imported_names: set[str] = set()
        self.function_names: set[str] = set()
        self.class_names: set[str] = set()
        self.candidate_names: set[str] = set()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.imported_names.add(alias.asname or alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            if alias.name != "*":
                self.imported_names.add(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_names.add(node.name)
        self._collect_args(node.args)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_names.add(node.name)
        self._collect_args(node.args)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_names.add(node.name)
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.candidate_names.add(node.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.candidate_names.add(node.name)
        self.generic_visit(node)

    def _collect_args(self, args: ast.arguments) -> None:
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            self.candidate_names.add(arg.arg)
        if args.vararg:
            self.candidate_names.add(args.vararg.arg)
        if args.kwarg:
            self.candidate_names.add(args.kwarg.arg)


class VariableRenamer(ast.NodeTransformer):
    def __init__(self, rename_map: dict[str, str]) -> None:
        self.rename_map = rename_map

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.rename_map:
            node.id = self.rename_map[node.id]
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        if node.arg in self.rename_map:
            node.arg = self.rename_map[node.arg]
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        self.generic_visit(node)
        if node.name in self.rename_map:
            node.name = self.rename_map[node.name]
        return node

    def visit_Global(self, node: ast.Global) -> ast.AST:
        node.names = [self.rename_map.get(name, name) for name in node.names]
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST:
        node.names = [self.rename_map.get(name, name) for name in node.names]
        return node


def obfuscate_variables(code: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code

    collector = VariableCollector()
    collector.visit(tree)
    excluded = (
        BUILTIN_NAMES
        | KEYWORD_NAMES
        | collector.imported_names
        | collector.function_names
        | collector.class_names
    )
    names = [name for name in sorted(collector.candidate_names) if name and name not in excluded]
    rename_map = {name: f"foobar_{idx}" for idx, name in enumerate(names, start=1)}
    if not rename_map:
        return code

    transformed = VariableRenamer(rename_map).visit(tree)
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed)


def safe_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name.strip("/"))


def save_results(
    output_dir: str | Path,
    predictions: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    save_raw_json: bool,
) -> tuple[Path, Path, Path | None]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame(predictions)
    summary_df = summarize_metrics(pred_df)

    predictions_path = out_dir / "per_example_predictions.csv"
    summary_path = out_dir / "summary_metrics.csv"
    pred_df.to_csv(predictions_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    raw_path = None
    if save_raw_json:
        raw_path = out_dir / "raw_generations.json"
        raw_path.write_text(json.dumps(raw_records, indent=2), encoding="utf-8")
    return predictions_path, summary_path, raw_path


def evaluate_model(model_name: str, examples: list[Example], args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    start_time = time.time()
    tokenizer, model, torch_mod = load_generation_model(
        model_name=model_name,
        device=args.device,
        torch_dtype=args.torch_dtype,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    prompts = [build_prompt(tokenizer, example.code) for example in examples]
    generations = generate_outputs(
        tokenizer=tokenizer,
        model=model,
        torch_mod=torch_mod,
        prompts=prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_input_tokens=args.max_input_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    embedding_scores = None
    if args.embedding_similarity:
        references = [build_reference_text(example) for example in examples]
        embedding_scores = compute_embedding_similarities(
            references=references,
            responses=generations,
            model_name=args.embedding_model,
            device=args.device,
        )

    prediction_rows = compute_prediction_rows(model_name, examples, generations, embedding_scores)
    raw_records = [
        {
            "model": model_name,
            "row_id": example.row_id,
            "title": example.title,
            "condition": example.condition,
            "prompt": prompt,
            "generation": generation,
        }
        for example, prompt, generation in zip(examples, prompts, generations)
    ]

    elapsed = time.time() - start_time
    print(f"Finished {model_name} on {len(examples)} prompts in {elapsed:.1f}s", flush=True)

    del model
    del tokenizer
    gc.collect()
    if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()
    return prediction_rows, raw_records


def main() -> None:
    args = parse_args()
    examples = load_data(args.data_path, args.sheet_name, args.limit)
    eval_examples = expand_conditions(examples, args.conditions)
    if not eval_examples:
        raise RuntimeError("No examples to evaluate after filtering.")

    all_predictions: list[dict[str, Any]] = []
    all_raw: list[dict[str, Any]] = []
    for model_name in args.models:
        print(f"Evaluating {model_name} on {len(eval_examples)} prompts", flush=True)
        predictions, raw = evaluate_model(model_name, eval_examples, args)
        all_predictions.extend(predictions)
        all_raw.extend(raw)

    predictions_path, summary_path, raw_path = save_results(
        args.output_dir,
        all_predictions,
        all_raw,
        args.save_raw_json,
    )
    print(f"Saved per-example predictions: {predictions_path}", flush=True)
    print(f"Saved summary metrics: {summary_path}", flush=True)
    if raw_path:
        print(f"Saved raw generations: {raw_path}", flush=True)


if __name__ == "__main__":
    main()
