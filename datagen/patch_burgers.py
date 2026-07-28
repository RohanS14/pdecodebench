import pandas as pd


def get_target_def(gt):
    if gt == 'Burgers_2':
        return 'def godunov'
    return 'def dudt'


def extract_initialization_block(code, target_def):
    """Extract everything before the first occurrence of target_def."""
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if line.startswith(target_def):
            return '\n'.join(lines[:i])
    return ''


def extract_comments_from_block(block):
    """Return all comment lines (stripped) from a block of code."""
    return [l for l in block.split('\n') if l.strip().startswith('#')]


def inject_initialization_block(old_code, new_init_block, target_def):
    """
    Replace the header of old_code (everything before target_def) with
    new_init_block, but preserve any comment lines that were in the old header.
    Comments are prepended to the new block so they aren't lost.
    """
    old_lines = old_code.split('\n')
    def_start = len(old_lines)
    for i, line in enumerate(old_lines):
        if line.startswith(target_def):
            def_start = i
            break

    old_header = '\n'.join(old_lines[:def_start])
    old_comments = extract_comments_from_block(old_header)
    rest = '\n'.join(old_lines[def_start:])

    parts = []
    if old_comments:
        parts.append('\n'.join(old_comments))
    parts.append(new_init_block.strip('\n'))
    parts.append(rest)
    return '\n\n'.join(parts)


def main():
    print("Loading pdedata_clean_v2.xlsx...")
    df = pd.read_excel('../data/pdedata_clean_v2.xlsx')

    for gt in ['Burgers_2', 'Burgers_3', 'Burgers_4']:
        nocomm_valid_row = df[(df['gt_sample'] == gt) & (df['mod_type'] == 'NoComm_Valid')]
        if nocomm_valid_row.empty:
            print(f"WARNING: NoComm_Valid not found for {gt}, skipping")
            continue
        nocomm_valid_code = nocomm_valid_row.iloc[0]['code']

        target_def = get_target_def(gt)
        init_block = extract_initialization_block(nocomm_valid_code, target_def)

        for mod in ['Comm_Valid', 'Comm_InValid', 'NoComm_InValid']:
            idx = df[(df['gt_sample'] == gt) & (df['mod_type'] == mod)].index
            if len(idx) > 0:
                old_code = df.loc[idx[0], 'code']
                new_code = inject_initialization_block(old_code, init_block, target_def)
                df.loc[idx[0], 'code'] = new_code

                # Verify comments are preserved
                old_n = sum(1 for l in old_code.split('\n') if l.strip().startswith('#'))
                new_n = sum(1 for l in new_code.split('\n') if l.strip().startswith('#'))
                print(f"Patched {gt} / {mod}  (comments: {old_n} -> {new_n})")
            else:
                print(f"WARNING: {gt} / {mod} not found")

    # Clean up any literal \\n escaping that might remain in the raw dataframe
    for i in range(len(df)):
        if isinstance(df.loc[i, 'code'], str):
            df.loc[i, 'code'] = df.loc[i, 'code'].replace('\\n\n', '\n').replace('\\n', '\n')

    out_path = '../data/pdedata_clean_v4_base.xlsx'
    df.to_excel(out_path, index=False)
    print(f"\nSaved patched base dataset to {out_path}")


if __name__ == '__main__':
    main()
