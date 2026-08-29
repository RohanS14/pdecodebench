"""
End-to-end test of the free-generation eval path on the jul28 dataset, with vLLM
stubbed out. Covers everything except the GPU: dataset load, the mod_type
integrity assertion, the --gt_samples canary filter, record construction
(including the jul28 gt_sample/source/num_char/valid_conf fields), scoring, and
checkpoint resume.

Run: python tests/test_freegen_jul28.py
"""
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "freegen_static_judgments"))

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}" + (f"  [{detail}]" if detail else ""))
        failures.append(name)


# ── Stub vLLM so run_eval imports and runs without a GPU ─────────────────────
CANNED = ("pde: heat equation\n"
          "method: implicit (Crank-Nicolson)\n"
          "behavior: diffusion\n"
          "valid: possibly valid, but the timestep looks aggressive")


class _FakeOutput:
    def __init__(self, text):
        self.text = text
        self.finish_reason = "stop"


class _FakeRequestOutput:
    def __init__(self, text):
        self.outputs = [_FakeOutput(text)]


class _FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def chat(self, messages_batch, sampling_params=None, chat_template_kwargs=None):
        _FakeLLM.last_chat_kwargs = chat_template_kwargs
        _FakeLLM.last_prompt = messages_batch[0][0]["content"]
        return [_FakeRequestOutput(CANNED) for _ in messages_batch]


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


_vllm = types.ModuleType("vllm")
_vllm.LLM = _FakeLLM
_vllm.SamplingParams = _FakeSamplingParams
sys.modules["vllm"] = _vllm

# Block the sentence-transformer download; embed_sim is optional by design.
sys.modules["sentence_transformers"] = None

import run_eval  # noqa: E402

DATASET = os.path.join(ROOT, "data", "merged_mod_jul28.csv")


def run(argv):
    old = sys.argv
    sys.argv = ["run_eval.py"] + argv
    try:
        run_eval.main()
    finally:
        sys.argv = old


# run_eval writes one results file per reasoning arm; the fake model has no
# thinking mode, so it resolves to the "off" arm.
ARM_JSONL = "fake__model__think-off.jsonl"


def read(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


# ── Canary subset ────────────────────────────────────────────────────────────
print("\n── canary subset (--gt_samples) ──")

with tempfile.TemporaryDirectory() as tmp:
    run(["--model", "fake/model", "--dataset", DATASET, "--output_dir", tmp,
         "--gt_samples", "Burgers_1,Heat_5,NavierStokes_2,Wave_8", "--batch_size", "8"])
    rows = read(os.path.join(tmp, ARM_JSONL))

    check("32 rows (4 gt_samples x 8 mod_types)", len(rows) == 32, str(len(rows)))
    check("all 8 mod_types present", len({r["mod_type"] for r in rows}) == 8)
    check("all 4 pde_classes present", len({r["pde_class"] for r in rows}) == 4)
    check("both sources present", {r["source"] for r in rows} == {"human", "synthetic"},
          str({r["source"] for r in rows}))
    check("human samples are 1-4", all(r["source"] == "human"
                                       for r in rows if r["gt_sample"] in ("Burgers_1", "NavierStokes_2")))
    check("synthetic samples are 5-8", all(r["source"] == "synthetic"
                                           for r in rows if r["gt_sample"] in ("Heat_5", "Wave_8")))

    r0 = rows[0]
    for col in ("title", "gt_sample", "source", "pde_class", "mod_type", "num_char",
                "gt_pde", "gt_method", "gt_behavior", "gt_valid", "model_response",
                "parsed_pde", "parsed_method", "parsed_behavior", "parsed_valid",
                "valid_conf", "finish_reason", "model", "dataset",
                "pde_match", "method_any_match", "method_recall",
                "behavior_any_match", "behavior_recall", "valid_match"):
        check(f"record has '{col}'", col in r0, str(sorted(r0)))

    check("dataset provenance recorded", r0["dataset"] == "merged_mod_jul28.csv", r0["dataset"])
    check("num_char is an int", isinstance(r0["num_char"], int), repr(r0["num_char"]))
    check("gt_valid is a real bool", all(isinstance(r["gt_valid"], bool) for r in rows))
    check("gt_valid splits 16/16", sum(r["gt_valid"] for r in rows) == 16,
          str(sum(r["gt_valid"] for r in rows)))
    check("invalidity_note set on invalid rows only",
          all((r["invalidity_note"] is None) == r["gt_valid"] for r in rows))

    # The canned response hedges, so it must land in Hedged and still carry a lean.
    check("valid_conf = Hedged for the canned hedge",
          all(r["valid_conf"] == "Hedged" for r in rows), r0["valid_conf"])
    check("a hedge scores 0 on valid_match",
          all(r["valid_match"] == 0 for r in rows))

    # Scoring sanity: canned answer is heat/implicit/diffusion.
    heat = [r for r in rows if r["gt_pde"] == "heat"]
    check("heat rows score pde_match=1", all(r["pde_match"] == 1 for r in heat))
    wave = [r for r in rows if r["gt_pde"] == "wave"]
    check("wave rows score pde_match=0", all(r["pde_match"] == 0 for r in wave))

    # Prompt must be byte-identical to the published template.
    check("prompt matches PROMPT_TEMPLATE",
          _FakeLLM.last_prompt.startswith(
              "You are analyzing a numerical simulation written in Python."))
    # v2-valid-disambiguated: the compound v1 wording ("does this code RUN AND
    # produce...") let a model score the question on execution alone, which is the
    # confound the rewording exists to remove.
    check("prompt ends with the v2 validity question",
          _FakeLLM.last_prompt.rstrip().endswith(
              "Running without error is not sufficient."))
    check("prompt carries no v1 compound wording",
          "does this code run and produce" not in _FakeLLM.last_prompt)


# ── Resume ───────────────────────────────────────────────────────────────────
print("\n── checkpoint resume ──")

with tempfile.TemporaryDirectory() as tmp:
    run(["--model", "fake/model", "--dataset", DATASET, "--output_dir", tmp,
         "--gt_samples", "Wave_1", "--batch_size", "4"])
    first = read(os.path.join(tmp, ARM_JSONL))
    run(["--model", "fake/model", "--dataset", DATASET, "--output_dir", tmp,
         "--gt_samples", "Wave_1", "--batch_size", "4"])
    second = read(os.path.join(tmp, ARM_JSONL))
    check("re-run writes no duplicate rows", len(first) == len(second) == 8,
          f"{len(first)} then {len(second)}")

    # Widening the subset must add only the new rows.
    run(["--model", "fake/model", "--dataset", DATASET, "--output_dir", tmp,
         "--gt_samples", "Wave_1,Wave_5", "--batch_size", "4"])
    third = read(os.path.join(tmp, ARM_JSONL))
    check("widening the subset appends only new rows", len(third) == 16, str(len(third)))
    check("no duplicate (title, mod_type)",
          len({(r["title"], r["mod_type"]) for r in third}) == 16)


# ── Guards ───────────────────────────────────────────────────────────────────
print("\n── guards ──")

with tempfile.TemporaryDirectory() as tmp:
    try:
        run(["--model", "fake/model", "--dataset", DATASET, "--output_dir", tmp,
             "--gt_samples", "Wave_99"])
        check("unknown gt_sample is rejected", False, "no assertion raised")
    except AssertionError as e:
        check("unknown gt_sample is rejected", "Wave_99" in str(e), str(e))


# ── Full dataset shape ───────────────────────────────────────────────────────
print("\n── full dataset ──")

with tempfile.TemporaryDirectory() as tmp:
    run(["--model", "fake/model", "--dataset", DATASET, "--output_dir", tmp,
         "--batch_size", "64"])
    rows = read(os.path.join(tmp, ARM_JSONL))
    check("256 rows", len(rows) == 256, str(len(rows)))
    check("32 gt_samples", len({r["gt_sample"] for r in rows}) == 32)
    check("128 human / 128 synthetic",
          sum(r["source"] == "human" for r in rows) == 128
          and sum(r["source"] == "synthetic" for r in rows) == 128)
    check("128 valid / 128 invalid", sum(r["gt_valid"] for r in rows) == 128)
    check("64 rows per pde_class",
          all(sum(r["pde_class"] == c for r in rows) == 64
              for c in ("wave", "heat", "burgers", "navier-stokes")))
    check("32 rows per mod_type",
          all(sum(r["mod_type"] == m for r in rows) == 32
              for m in {r["mod_type"] for r in rows}))


print()
if failures:
    print(f"FAILED: {len(failures)} test(s): {failures}")
    sys.exit(1)
print("All tests passed.")
