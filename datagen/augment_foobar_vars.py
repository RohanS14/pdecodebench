import ast
import builtins
import keyword
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment


INPUT_PATH = Path("/home/ehb7466/pdecodebench/data/pdedata.xlsx")
SOURCE_SHEET = 0
OUTPUT_SHEET = "FoobarVars"
BUILTIN_NAMES = set(dir(builtins))
KEYWORD_NAMES = set(keyword.kwlist)


def _is_truthy(value):
    return pd.notna(value) and str(value).strip().lower() in {"yes", "true", "1"}


def _find_column(df, candidates, required=True):
    column_lookup = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        column = column_lookup.get(candidate.lower())
        if column is not None:
            return column

    if required:
        raise KeyError(
            f"None of {candidates!r} were found in the DataFrame columns: "
            f"{list(df.columns)!r}"
        )
    return None


def _normalize_code_text(code):
    if pd.isna(code):
        raise ValueError("Code is null")

    source = str(code)
    return source.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def _serialize_code_text(code):
    return code.replace("\r\n", "\n").replace("\n", "\\n\n")


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


def obfuscate_code(code):
    source = _normalize_code_text(code)
    tree = ast.parse(source)

    collector = VariableCollector()
    collector.visit(tree)

    variable_excluded = (
        BUILTIN_NAMES
        | KEYWORD_NAMES
        | collector.imported_names
        | collector.function_names
        | collector.class_names
    )

    names_to_rename = [
        name for name in sorted(collector.candidate_names) if name and name not in variable_excluded
    ]

    rename_map = {
        original: f"foobar_{i}"
        for i, original in enumerate(names_to_rename, start=1)
    }
    function_excluded = (
        BUILTIN_NAMES
        | KEYWORD_NAMES
        | collector.imported_names
        | collector.class_names
    )
    function_rename_map = {
        original: f"fn{i}"
        for i, original in enumerate(collector.function_names_in_order, start=1)
        if original not in function_excluded
    }

    if not rename_map and not function_rename_map:
        return source

    new_tree = VariableRenamer(rename_map, function_rename_map).visit(tree)
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree)


def augment_dataframe_with_foobar_vars(df):
    df_work = df.dropna(how="all").copy()
    code_col = _find_column(df_work, ["code", "Code"])
    validity_col = _find_column(df_work, ["phys_valid", "Phys Valid"], required=False)

    if "foobar_vars" not in df_work.columns:
        df_work["foobar_vars"] = pd.NA

    df_work["foobar_vars"] = "No"

    augmented_rows = []
    eligible_rows = 0
    parse_failures = 0

    for _, row in df_work.iterrows():
        if validity_col is not None and not _is_truthy(row.get(validity_col)):
            continue

        eligible_rows += 1
        code_value = row.get(code_col)
        try:
            transformed_code = obfuscate_code(code_value)
        except Exception:
            parse_failures += 1
            continue

        new_row = row.copy()
        new_row[code_col] = _serialize_code_text(transformed_code)
        new_row["foobar_vars"] = "Yes"
        if "num_lines" in new_row.index:
            new_row["num_lines"] = len(transformed_code.splitlines())
        if "num_char" in new_row.index:
            new_row["num_char"] = len(new_row[code_col])
        augmented_rows.append(new_row)

    if augmented_rows:
        df_aug = pd.DataFrame(augmented_rows, columns=df_work.columns)
        df_out = pd.concat([df_work, df_aug], ignore_index=True)
    else:
        df_out = df_work.copy()

    print(f"Rows after removing empty Excel rows: {len(df_work)}")
    print(f"Eligible rows: {eligible_rows}")
    print(f"Parse failures skipped: {parse_failures}")
    print(f"Augmented rows created: {len(augmented_rows)}")
    print(f"Output rows: {len(df_out)}")
    return df_out


def _format_output_sheet(writer, df_out):
    worksheet = writer.sheets[OUTPUT_SHEET]
    code_col_idx = list(df_out.columns).index(_find_column(df_out, ["code", "Code"])) + 1

    worksheet.freeze_panes = "A2"
    worksheet.column_dimensions[worksheet.cell(row=1, column=code_col_idx).column_letter].width = 120

    wrapped_top = Alignment(wrap_text=True, vertical="top")
    for row in range(2, len(df_out) + 2):
        worksheet.cell(row=row, column=code_col_idx).alignment = wrapped_top


def main():
    df = pd.read_excel(INPUT_PATH, sheet_name=SOURCE_SHEET)
    df_out = augment_dataframe_with_foobar_vars(df)

    with pd.ExcelWriter(
        INPUT_PATH,
        engine="openpyxl",
        mode="a",
        if_sheet_exists="replace",
    ) as writer:
        df_out.to_excel(writer, sheet_name=OUTPUT_SHEET, index=False)
        _format_output_sheet(writer, df_out)

    print(f"Saved augmented data to {INPUT_PATH} sheet {OUTPUT_SHEET!r}")


if __name__ == "__main__":
    main()
