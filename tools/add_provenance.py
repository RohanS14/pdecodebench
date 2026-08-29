"""Join per-sample upstream provenance onto the two released datasets.

Adds origin / upstream_project / upstream_repo / upstream_path / upstream_commit /
upstream_license / upstream_citation to every row, keyed on gt_sample. Samples not
listed in PROVENANCE.csv are treated as synthetic (generated for this project, MIT).

    python3 tools/add_provenance.py            # rewrite CSVs in dataset_release/
    python3 tools/add_provenance.py --check    # report only, write nothing
"""
import argparse, pathlib, sys
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
REL = ROOT / "dataset_release"
COLS = ["origin", "upstream_project", "upstream_repo", "upstream_path",
        "upstream_commit", "upstream_license", "upstream_citation"]
SYNTHETIC = {"origin": "synthetic", "upstream_project": "", "upstream_repo": "",
             "upstream_path": "", "upstream_commit": "", "upstream_license": "MIT",
             "upstream_citation": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    prov = pd.read_csv(REL / "PROVENANCE.csv").set_index("gt_sample")
    unresolved = prov.index[prov.upstream_license.str.startswith("UNRESOLVED")].tolist()

    for name in ("dataset_cross_representation.csv", "dataset_code_perturbation.csv"):
        path = REL / name
        df = pd.read_csv(path)
        for c in COLS:
            df[c] = df.gt_sample.map(
                lambda s: prov.at[s, c] if s in prov.index else SYNTHETIC[c])
        df[COLS] = df[COLS].fillna("")
        counts = df.upstream_license.value_counts().to_dict()
        print(f"{name}: {len(df)} rows -> {counts}")
        if not args.check:
            df.to_csv(path, index=False, quoting=1)  # QUOTE_ALL, as originally written

    if unresolved:
        print(f"\nUNRESOLVED, must be settled before release: {', '.join(unresolved)}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
