"""
instrument_history.py — records a time history from the solvers that do not keep
one, so the T_exec rung of the corruption ladder exists for them.

datagen/extract_trajectories.py reconstructs a trajectory "from the final namespace
plus any array the author kept as a history/list", which works for 27 of the 32
solvers. Five do not accumulate: Heat_5 and Heat_6 keep only four snapshots, and
Heat_8, NavierStokes_1 and NavierStokes_2 keep nothing but the final state. Without
a history their T_exec would be a 1- or 4-frame table sitting next to 10-frame
tables, which is precisely the kind of formatting difference that identifies the
outlier without any physics.

Three constraints, following the discipline the repo already used when it repaired
the sampling-cadence leak:

  1. THE INSTRUMENTED SOURCE IS NEVER THE CODE VIEW. It exists only to produce
     numbers. The code shown to a model stays the audited NoComm_* text, so the
     inserted line cannot become a fingerprint of the invalid condition -- which is
     exactly how `if n < 30 or n % 10 == 0` became a label leak the first time.
  2. WRITE-ONLY, PROVEN BY AST. The recorder is appended to and never read, so it
     cannot influence any computed value. assert_write_only() checks this rather
     than trusting it.
  3. VERIFIED DOWNSTREAM. full_audit_exec.py already asserts that a problem's
     surface conditions produce identical numbers; re-run it after instrumenting.

Only the real-identifier variants are executed. NoComm_CorrVar is AST-identical to
NoComm_Valid and produces the same numbers by construction, so its trajectory is the
same array -- executing it again would burn CPU to reproduce a known result.

Heat_8 is deliberately absent. It has no time loop at all: it is spectral and
evaluates a closed-form solution once at t_final. Producing ten frames would mean
evaluating its solution operator at ten times, which is writing a different program
rather than instrumenting this one, so it is excluded from T_exec and recorded as
such. See NOT_INSTRUMENTABLE.
"""
import ast
import copy

RECORDER = "_HISTORY"

# Which loop to record and which arrays are the solution, per solver. A small
# declarative table rather than a heuristic: with only four systems, a generic
# "find the interesting loop" pass would be more code and could silently pick the
# wrong array -- the pressure-Poisson inner loop in both NavierStokes samples is a
# convincing decoy, since it iterates and mutates an array that is not the solution.
#
# `function` names the enclosing def when the time loop is not at module level.
# Fields are matched by name, and the target loop is found as the outermost loop
# that mutates them -- by rebinding or by slice assignment -- so the spec survives
# the line-number shifts between a solver's valid and invalid variants.
INSTRUMENT_SPEC = {
    "Heat_5": {"fields": ["u"], "function": None},
    "Heat_6": {"fields": ["u"], "function": None},
    "NavierStokes_1": {"fields": ["u", "v"], "function": "cavity_flow"},
    "NavierStokes_2": {"fields": ["u", "v"], "function": None},
}

NOT_INSTRUMENTABLE = {
    "Heat_8": ("spectral solver with no time loop; evaluates a closed-form solution "
               "once at t_final, so a history would have to be a new program"),
}


def _mutated_names(node):
    """Names a statement subtree assigns to, by rebinding or by slice assignment.

    Both matter: Heat_5 rebinds `u = u_new`, while NavierStokes_2 mutates `u` in
    place with `u[1:-1, 1:-1] = ...` and never rebinds it. Looking only at Name
    targets would miss the solution array in three of the four solvers.
    """
    out = set()
    for x in ast.walk(node):
        if not isinstance(x, ast.Assign):
            continue
        targets = list(x.targets)
        for tgt in list(targets):
            if isinstance(tgt, ast.Tuple):
                targets.extend(tgt.elts)
        for tgt in targets:
            if isinstance(tgt, ast.Name):
                out.add(tgt.id)
            elif isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
                out.add(tgt.value.id)
    return out


def find_time_loop(tree, fields, function=None):
    """The outermost loop that mutates any of `fields`, within `function` if given."""
    scope = tree
    if function is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function:
                scope = node
                break
        else:
            raise ValueError(f"function {function!r} not found")

    wanted, best = set(fields), None
    for node in ast.walk(scope):
        if not isinstance(node, (ast.For, ast.While)):
            continue
        if not (_mutated_names(node) & wanted):
            continue
        # Outermost wins: the pressure-Poisson inner loop also touches an array,
        # but it runs many times per timestep and is not the time axis.
        if best is None or node.col_offset < best.col_offset:
            best = node
        elif node.col_offset == best.col_offset and node.lineno < best.lineno:
            best = node
    if best is None:
        raise ValueError(f"no loop mutating {sorted(wanted)}")
    return best


def _append_stmt(fields):
    """`_HISTORY.append([u.copy(), v.copy()])`

    .copy() is a method on the array itself, so this needs no numpy alias -- the
    solvers variously `import numpy` and `import numpy as np`, and assuming either
    would break the other.
    """
    return ast.Expr(value=ast.Call(
        func=ast.Attribute(value=ast.Name(id=RECORDER, ctx=ast.Load()),
                           attr="append", ctx=ast.Load()),
        args=[ast.List(
            elts=[ast.Call(func=ast.Attribute(value=ast.Name(id=f, ctx=ast.Load()),
                                              attr="copy", ctx=ast.Load()),
                           args=[], keywords=[])
                  for f in fields],
            ctx=ast.Load())],
        keywords=[]))


def instrument(code, system):
    """Return `code` with a write-only history recorder around its time loop."""
    if system in NOT_INSTRUMENTABLE:
        raise ValueError(f"{system} is not instrumentable: {NOT_INSTRUMENTABLE[system]}")
    spec = INSTRUMENT_SPEC[system]

    tree = ast.parse(code)
    loop = find_time_loop(tree, spec["fields"], spec["function"])
    loop = copy.deepcopy(loop) and loop            # operate in place on the parsed tree
    loop.body = list(loop.body) + [_append_stmt(spec["fields"])]

    tree.body.insert(0, ast.Assign(
        targets=[ast.Name(id=RECORDER, ctx=ast.Store())],
        value=ast.List(elts=[], ctx=ast.Load())))

    ast.fix_missing_locations(tree)
    out = ast.unparse(tree)
    assert_write_only(out)
    return out


def assert_write_only(code):
    """Prove the recorder cannot influence any computed value.

    Every load of the recorder must be the receiver of `.append(...)`. Anything else
    -- reading it, subscripting it, iterating it, passing it somewhere -- would mean
    the instrumentation had entered the computation, and the trajectory would no
    longer be the one the untouched solver produces.
    """
    tree = ast.parse(code)
    stores = appends = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == RECORDER:
            if isinstance(node.ctx, ast.Store):
                stores += 1
                continue
            parent_ok = False
            for other in ast.walk(tree):
                if (isinstance(other, ast.Attribute) and other.attr == "append"
                        and other.value is node):
                    parent_ok = True
                    break
            if not parent_ok:
                raise AssertionError(f"{RECORDER} is read somewhere other than .append")
            appends += 1
    assert stores == 1, f"{RECORDER} assigned {stores} times, expected exactly 1"
    assert appends >= 1, f"{RECORDER} is never appended to"
    return True


def stack_history(history):
    """Normalize a recorded history into the (T, X, Y, C) layout the renderer uses.

    Fields become channels: a 1-D scalar field gives (T, nx, 1, 1) and a pair of 2-D
    velocity components gives (T, ny, nx, 2), matching the shape convention the
    dataset's own trajectories already use.
    """
    import numpy as np

    if not history:
        raise ValueError("no frames recorded -- the time loop never ran")
    frames = []
    for step in history:
        arrays = [np.asarray(f, dtype=float) for f in step]
        arrays = [a[:, None] if a.ndim == 1 else a for a in arrays]
        frames.append(np.stack(arrays, axis=-1))
    out = np.stack(frames, axis=0)
    if out.ndim == 3:                       # (T, X, C) -> (T, X, 1, C)
        out = out[:, :, None, :]
    return out
