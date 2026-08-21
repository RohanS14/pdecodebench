"""
consistency_prompts.py — prompt text, legend, and output schema for the
cross-modal consistency experiment (plan Part III).

Materializes the four views for an item from the manifest built by
crossmodal/datagen/build_multimodal_items.py and assembles the prompt. Kept separate from the
generation loop so the prompt copy and the schema can be reviewed without reading
the vLLM plumbing, matching how eval/frontier/agentic_prompts.py is split from its
orchestration.

Three things the prompt must not do, each of which would hand the answer away:

  * Name the PDE class or numerical method anywhere in the instructions or legend.
    The model is being asked to recover them.
  * Describe the views in language that favours one of them. The legend states what
    each view IS, in the same register, and nothing about reliability.
  * Let the slot order carry information. Order is permuted per item and
    counterbalanced so the corrupted view sits in each slot equally often.

Note on what the description view leaks by its nature: all 32 valid descriptions
name their PDE family in physical vocabulary ("heat conduction along a rod",
"wave", "fluid"), while none of them and none of the equations name a numerical
method. So system_pde_class is near ceiling whenever a clean description is present
and the informative cell is X_D, while system_num_method is carried only by the
code. That is a property of the data, not something the prompt can fix; it shapes
how the two identification measures are read.
"""
import csv
import os
import sys

# repo root: this file sits at crossmodal/<area>/, so three levels up
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
from crossmodal.datagen.build_multimodal_items import (                       # noqa: E402
    MOD_FOR_NAMES, canonical, load_code_variants, load_multimodal,
)
from crossmodal.datagen.corrupt_trajectory import build_ladder                # noqa: E402
from crossmodal.datagen.render_trajectory_table import (                      # noqa: E402
    choose_grid, parse_trajectory, render,
)

csv.field_size_limit(10 ** 9)

# Neutral one-line statements of what each view is. Same register, no hint about
# which is more trustworthy, no PDE or method vocabulary.
VIEW_LEGEND = {
    "code": "Python source code for a numerical solver",
    "math": "the governing equation, in LaTeX",
    "trajectory": "the numerical solution field produced by running a solver",
    "description": "a natural-language description of the physical process",
}

PROMPT_TEMPLATE = """\
Below are four representations of a physical system, labelled View 1 to View 4.

Legend:
{legend}

{views}

Do these representations all describe the same physical system? Either all four \
representations are in agreement, or exactly one of them is inconsistent with the \
other three.

Answer with a JSON object containing exactly these fields:
- "agree": "yes" if all four agree, "no" if one is inconsistent.
- "outlier": which view is inconsistent, as "view_1", "view_2", "view_3" or \
"view_4". Use "none" if and only if "agree" is "yes".
- "system_pde_class": the class of partial differential equation that the majority \
of the views describe.
- "system_num_method": the numerical method that the majority of the views use.
- "justification": what specifically is inconsistent, or why you judge the views \
consistent.

Output only the JSON object.\
"""

# Enforced by vLLM guided decoding where available. The "required iff agree == no"
# rule cannot be expressed in JSON Schema, so it is checked when parsing rather
# than constrained here -- see crossmodal/eval/parse_consistency.py.
CONSISTENCY_SCHEMA = {
    "type": "object",
    "properties": {
        "agree": {"type": "string", "enum": ["yes", "no"]},
        "outlier": {
            "type": "string",
            "enum": ["view_1", "view_2", "view_3", "view_4", "none"],
        },
        "system_pde_class": {"type": "string"},
        "system_num_method": {"type": "string"},
        "justification": {"type": "string"},
    },
    "required": [
        "agree", "outlier", "system_pde_class", "system_num_method", "justification",
    ],
    "additionalProperties": False,
}


def load_exec_trajectories(path="data/exec_trajectories.npz"):
    """T_exec arrays keyed by gt_sample, or {} if the job has not run.

    Absent entries are not an error: T_exec is unavailable for a handful of systems
    by construction (Heat_8 has no time loop; three more evolve a different grid
    than the stored trajectory, so their executed field is a different object than
    the valid view). Items for those systems are dropped rather than shown a
    substitute, and the coverage is reported.
    """
    import numpy as np

    if not os.path.exists(path):
        return {}
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


class ViewSources:
    """Holds everything needed to materialize a view, with per-system caching.

    Parsing the trajectory column is the expensive step -- the stored arrays run to
    3.5 MB of text -- so each system's valid trajectory, render grid and corruption
    ladder are built once and reused across that system's items.
    """

    def __init__(self, multimodal_csv, mod_dataset, exec_trajectories=None):
        self.mm = load_multimodal(multimodal_csv)
        self.codes = load_code_variants(mod_dataset)
        # {gt_sample: ndarray} from the cpu_short re-execution job. Absent until
        # that job lands, which is why T_exec is the one rung a first canary skips.
        self.exec_traj = exec_trajectories or {}
        self._cache = {}

    def _system(self, system):
        if system not in self._cache:
            valid = parse_trajectory(self.mm[system]["Trajectory"])
            self._cache[system] = {
                "valid": valid,
                "grid": choose_grid(valid.shape),
                "ladder": build_ladder(
                    valid, self.mm[system + "_wrong"]["Trajectory"], system,
                    include_time_shuffle=True),
            }
        return self._cache[system]

    def code(self, system, names, valid):
        return self.codes[system][MOD_FOR_NAMES[(names, valid)]]

    def trajectory(self, system, level):
        c = self._system(system)
        if level == "valid":
            arr = c["valid"]
        elif level == "T_exec":
            if system not in self.exec_traj:
                raise KeyError(
                    f"T_exec trajectory missing for {system}; run "
                    f"sbatch/run_build_modalities.sbatch first")
            arr = self.exec_traj[system]
        else:
            arr = c["ladder"][level]
        # Every rung goes through the same grid, so the four candidates render at
        # identical size and precision and the outlier cannot be spotted by shape.
        return render(arr, c["grid"])


def materialize_views(item, sources):
    """The four view bodies for one item, keyed by modality."""
    system = item["gt_sample"]
    corrupted = item["corrupted_view"]
    mm_valid, mm_wrong = sources.mm[system], sources.mm[system + "_wrong"]

    return {
        "code": sources.code(system, item["names"], corrupted != "code"),
        "math": (mm_wrong if corrupted == "math" else mm_valid)["Math Equation"].strip(),
        "description": (mm_wrong if corrupted == "description"
                        else mm_valid)["Written Description"].strip(),
        "trajectory": sources.trajectory(system, item["traj_level"]),
    }


def build_prompt(item, sources):
    """Assemble the full prompt for one item, honouring its slot permutation."""
    bodies = materialize_views(item, sources)
    slots = [item[f"slot_{i}"] for i in range(1, 5)]

    legend = "\n".join(
        f"  View {i}: {VIEW_LEGEND[m]}" for i, m in enumerate(slots, start=1))
    views = "\n\n".join(
        f"<view_{i}>\n{bodies[m]}\n</view_{i}>" for i, m in enumerate(slots, start=1))
    return PROMPT_TEMPLATE.format(legend=legend, views=views)


def build_messages(item, sources):
    return [{"role": "user", "content": build_prompt(item, sources)}]


def load_items(path):
    return list(csv.DictReader(open(path, newline="")))
