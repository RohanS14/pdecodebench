"""
augment_foobar_vars.py — Generate NoComm_CorrVar and NoComm_CorrVar_InValid rows.

NoComm_CorrVar:
  Run the AST renamer on NoComm_Valid code. Variable mapping is derived
  fresh from the code (sorted candidate names -> foobar_1, foobar_2, ...).

NoComm_CorrVar_InValid:
  Re-run the AST renamer on the corresponding NoComm_Valid to recover the
  mapping, then apply that exact mapping to NoComm_InValid code. Any variables
  in the invalid code not present in the mapping are assigned new foobar_N names
  continuing the existing numbering. This ensures shared variables have identical
  obfuscated names across validity conditions, preventing the probe from using
  name identity as a validity shortcut.
"""

import ast
import builtins
import keyword

import pandas as pd


BUILTIN_NAMES = set(dir(builtins))
KEYWORD_NAMES = set(keyword.kwlist)


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

class VariableCollector(ast.NodeVisitor):
    def __init__(self):
        self.imported_names = set()
        self.function_names = set()
        self.function_names_in_order = []
        self.class_names = set()
        self.candidate_names = set()
        self._class_depth = 0

    def _add_function_name(self, name):
        if name in self.function_names:
            return
        self.function_names.add(name)
        if self._class_depth == 0 and not (name.startswith("__") and name.endswith("__")):
            self.function_names_in_order.append(name)

    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name.split(".")[0]
            self.imported_names.add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            if alias.name == "*":
                continue
            name = alias.asname if alias.asname else alias.name
            self.imported_names.add(name)
        self.generic_visit(node)

    def _collect_args(self, args):
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            self.candidate_names.add(arg.arg)
        if args.vararg:
            self.candidate_names.add(args.vararg.arg)
        if args.kwarg:
            self.candidate_names.add(args.kwarg.arg)

    def visit_FunctionDef(self, node):
        self._add_function_name(node.name)
        self._collect_args(node.args)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        self._add_function_name(node.name)
        self._collect_args(node.args)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self.class_names.add(node.name)
        self._class_depth += 1
        self.generic_visit(node)
        self._class_depth -= 1

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.candidate_names.add(node.id)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.candidate_names.add(node.name)
        self.generic_visit(node)


class VariableRenamer(ast.NodeTransformer):
    def __init__(self, variable_rename_map, function_rename_map):
        self.variable_rename_map = variable_rename_map
        self.function_rename_map = function_rename_map
        self._class_depth = 0

    @property
    def rename_map(self):
        return {**self.variable_rename_map, **self.function_rename_map}

    def visit_FunctionDef(self, node):
        if self._class_depth == 0 and node.name in self.function_rename_map:
            node.name = self.function_rename_map[node.name]
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node):
        if self._class_depth == 0 and node.name in self.function_rename_map:
            node.name = self.function_rename_map[node.name]
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node):
        self._class_depth += 1
        self.generic_visit(node)
        self._class_depth -= 1
        return node

    def visit_Name(self, node):
        if node.id in self.rename_map:
            node.id = self.rename_map[node.id]
        return node

    def visit_arg(self, node):
        if node.arg in self.rename_map:
            node.arg = self.rename_map[node.arg]
        return node

    def visit_ExceptHandler(self, node):
        self.generic_visit(node)
        if node.name in self.rename_map:
            node.name = self.rename_map[node.name]
        return node

    def visit_Global(self, node):
        node.names = [self.rename_map.get(name, name) for name in node.names]
        return node

    def visit_Nonlocal(self, node):
        node.names = [self.rename_map.get(name, name) for name in node.names]
        return node


def _normalize(code: str) -> str:
    return code.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def _build_maps(code: str) -> tuple[dict, dict]:
    """Return (variable_rename_map, function_rename_map) for a piece of code."""
    tree = ast.parse(_normalize(code))
    collector = VariableCollector()
    collector.visit(tree)

    excluded = (
        BUILTIN_NAMES | KEYWORD_NAMES
        | collector.imported_names | collector.function_names | collector.class_names
    )
    names_to_rename = sorted(n for n in collector.candidate_names if n and n not in excluded)
    var_map = {name: f"foobar_{i}" for i, name in enumerate(names_to_rename, start=1)}

    fn_excluded = BUILTIN_NAMES | KEYWORD_NAMES | collector.imported_names | collector.class_names
    fn_map = {
        name: f"fn{i}"
        for i, name in enumerate(collector.function_names_in_order, start=1)
        if name not in fn_excluded
    }
    return var_map, fn_map


def _apply_maps(code: str, var_map: dict, fn_map: dict) -> str:
    source = _normalize(code)
    tree = ast.parse(source)
    new_tree = VariableRenamer(var_map, fn_map).visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


def _extend_map(base_var_map: dict, invalid_code: str) -> tuple[dict, dict]:
    """
    Return (extended_var_map, fn_map) for invalid_code.
    Shared variables keep their foobar_N names from base_var_map.
    New variables continue numbering from the highest N already used.
    """
    max_n = max((int(v.split("_")[1]) for v in base_var_map.values() if v.startswith("foobar_")), default=0)

    tree = ast.parse(_normalize(invalid_code))
    collector = VariableCollector()
    collector.visit(tree)
    excluded = (
        BUILTIN_NAMES | KEYWORD_NAMES
        | collector.imported_names | collector.function_names | collector.class_names
    )
    all_names = sorted(n for n in collector.candidate_names if n and n not in excluded)

    extended = dict(base_var_map)
    counter = max_n + 1
    for name in all_names:
        if name not in extended:
            extended[name] = f"foobar_{counter}"
            counter += 1

    fn_excluded = BUILTIN_NAMES | KEYWORD_NAMES | collector.imported_names | collector.class_names
    fn_map = {
        name: f"fn{i}"
        for i, name in enumerate(collector.function_names_in_order, start=1)
        if name not in fn_excluded
    }
    return extended, fn_map


# ---------------------------------------------------------------------------
# Public generation functions
# ---------------------------------------------------------------------------

def generate_foobar_rows(df: pd.DataFrame) -> list[dict]:
    """
    Returns new NoComm_CorrVar + NoComm_CorrVar_InValid rows to append to df.
    """
    no_comm_valid = df[df["mod_type"] == "NoComm_Valid"].reset_index(drop=True)
    no_comm_invalid = df[df["mod_type"] == "NoComm_InValid"].reset_index(drop=True)
    invalid_by_gt = no_comm_invalid.set_index("gt_sample")

    new_rows = []
    parse_failures = 0

    for _, row in no_comm_valid.iterrows():
        gt = row["gt_sample"]

        # --- NoComm_CorrVar ---
        try:
            var_map, fn_map = _build_maps(row["code"])
            transformed = _apply_maps(row["code"], var_map, fn_map)
        except Exception as e:
            print(f"WARNING: parse failure for {gt} (NoComm_CorrVar): {e}")
            parse_failures += 1
            continue

        new_row = row.copy()
        new_row["code"] = transformed
        new_row["mod_type"] = "NoComm_CorrVar"
        new_row["num_lines"] = len(transformed.splitlines())
        new_row["num_char"] = len(transformed)
        new_row["title"] = row["title"].replace("NoComm_Valid", "NoComm_CorrVar")
        new_rows.append(new_row.to_dict())

        # --- NoComm_CorrVar_InValid ---
        if gt not in invalid_by_gt.index:
            print(f"WARNING: no NoComm_InValid found for {gt}, skipping CorrVar_InValid")
            continue

        inv_row = no_comm_invalid[no_comm_invalid["gt_sample"] == gt].iloc[0]
        try:
            extended_var_map, inv_fn_map = _extend_map(var_map, inv_row["code"])
            inv_transformed = _apply_maps(inv_row["code"], extended_var_map, inv_fn_map)
        except Exception as e:
            print(f"WARNING: parse failure for {gt} (NoComm_CorrVar_InValid): {e}")
            parse_failures += 1
            continue

        inv_new_row = inv_row.copy()
        inv_new_row["code"] = inv_transformed
        inv_new_row["mod_type"] = "NoComm_CorrVar_InValid"
        inv_new_row["num_lines"] = len(inv_transformed.splitlines())
        inv_new_row["num_char"] = len(inv_transformed)
        inv_new_row["title"] = inv_row["title"].replace("NoComm_InValid", "NoComm_CorrVar_InValid")
        new_rows.append(inv_new_row.to_dict())

    print(f"Parse failures: {parse_failures}")
    corrvar = [r for r in new_rows if r["mod_type"] == "NoComm_CorrVar"]
    corrvar_inv = [r for r in new_rows if r["mod_type"] == "NoComm_CorrVar_InValid"]
    print(f"Generated {len(corrvar)} NoComm_CorrVar rows")
    print(f"Generated {len(corrvar_inv)} NoComm_CorrVar_InValid rows")
    return new_rows
