"""Prove a HuggingFace cache is fully contained in another before deleting it.

Matching directory names are not evidence. When /scratch/ehb7466/.huggingface was
compared to /scratch/ehb7466/hf_cache on 2026-08-24, nineteen models appeared in
both -- and three of them still held blobs the destination did not have. They were
small (config and tokenizer files, 1.5-44KB), but "mostly duplicated" is not
duplicated, and a delete on the strength of the directory listing would have lost
them silently.

So this walks every regular file under SRC and requires a same-sized file at the
same relative path under DST. Symlinks are compared as symlinks: the HF layout
points snapshots/<rev>/<file> at ../../blobs/<sha>, and a link whose target differs
means the destination is pinned to a different revision.

Exits 0 only when SRC is a strict subset of DST. Anything else exits 1 and prints
what is missing, because the caller's next step is `rm -rf`.

Usage:
    python tools/verify_hf_merge.py /scratch/ehb7466/.huggingface /scratch/ehb7466/hf_cache
"""
import os
import sys


def walk(root):
    """Relative path -> ('link', target) or ('file', size). Directories are skipped:
    an empty directory carries nothing and its absence loses nothing."""
    out = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in filenames + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            full = os.path.join(dirpath, name)
            rel = os.path.relpath(full, root)
            if os.path.islink(full):
                out[rel] = ("link", os.readlink(full))
            else:
                try:
                    out[rel] = ("file", os.path.getsize(full))
                except OSError as e:
                    out[rel] = ("error", str(e))
    return out


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]

    print(f"[verify] scanning {src} ...", flush=True)
    a = walk(src)
    print(f"[verify]   {len(a)} entries", flush=True)
    print(f"[verify] scanning {dst} ...", flush=True)
    b = walk(dst)
    print(f"[verify]   {len(b)} entries", flush=True)

    missing, mismatched, unreadable = [], [], []
    for rel, val in a.items():
        if val[0] == "error":
            unreadable.append((rel, val[1])); continue
        if rel not in b:
            missing.append(rel); continue
        if b[rel] != val:
            mismatched.append((rel, val, b[rel]))

    def show(label, items, fmt=lambda x: x, limit=25):
        print(f"\n[verify] {label}: {len(items)}")
        for it in items[:limit]:
            print(f"    {fmt(it)}")
        if len(items) > limit:
            print(f"    ... and {len(items) - limit} more")

    if missing:
        show("MISSING from destination", missing)
    if mismatched:
        show("SIZE/TARGET MISMATCH", mismatched,
             lambda t: f"{t[0]}\n        src={t[1]}  dst={t[2]}")
    if unreadable:
        show("UNREADABLE in source", unreadable, lambda t: f"{t[0]}: {t[1]}")

    if missing or mismatched or unreadable:
        print(f"\n[verify] NOT SAFE TO DELETE {src}")
        sys.exit(1)
    print(f"\n[verify] OK — every one of the {len(a)} entries under {src} exists in "
          f"{dst} at the same size. Safe to delete.")


if __name__ == "__main__":
    main()
