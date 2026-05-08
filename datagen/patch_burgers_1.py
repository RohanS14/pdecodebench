import pandas as pd
import difflib

INIT_WITH_COMMENTS = """# number of cells in 1D space
n = 100

# space
x0 = 0
xn = 1
dx = (xn - x0)/n
x_interface = np.linspace(x0, xn, n+1)
x = x_interface[0:n] + dx/2

# time
t0 = 0
tn = 0.5
t_steps = 200
t = np.linspace(t0, tn, t_steps)


# Initial condition
u_init = np.sin(2*np.pi*x)"""

def strip_comments(code):
    lines = code.split('\n')
    # Keep blank lines and non-comment lines
    lines = [l for l in lines if not l.strip().startswith('#')]
    return '\n'.join(lines)

INIT_NO_COMMENTS = strip_comments(INIT_WITH_COMMENTS)

def patch_code(code, init_block):
    lines = code.split('\n')
    
    # Find last import
    last_import_idx = -1
    for i, l in enumerate(lines):
        if l.startswith('import ') or l.startswith('from '):
            last_import_idx = i
            
    # Find first def
    first_def_idx = -1
    for i, l in enumerate(lines):
        if l.startswith('def '):
            first_def_idx = i
            break
            
    assert last_import_idx != -1, "Could not find import statement"
    assert first_def_idx != -1, "Could not find def statement"
    assert last_import_idx < first_def_idx, "Imports appear after def"
    
    # Check if already patched
    between = '\n'.join(lines[last_import_idx+1:first_def_idx])
    assert 'u_init' not in between, "u_init already defined in the block"
    
    # Insert with blank lines
    new_lines = lines[:last_import_idx+1] + ['\n' + init_block + '\n'] + lines[first_def_idx:]
    return '\n'.join(new_lines)

def main():
    print("Loading pdedata_clean_v4_base.xlsx...")
    df = pd.read_excel('data/pdedata_clean_v4_base.xlsx')
    
    mask = (df['gt_sample'] == 'Burgers_1') & (df['mod_type'].isin(['Comm_Valid', 'NoComm_Valid', 'Comm_InValid', 'NoComm_InValid']))
    rows = df[mask]
    
    assert len(rows) == 4, f"Expected 4 rows, found {len(rows)}"
    
    for idx, row in rows.iterrows():
        title = row['title']
        mod_type = row['mod_type']
        old_code = row['code']
        
        init_block = INIT_WITH_COMMENTS if 'NoComm' not in mod_type else INIT_NO_COMMENTS
        
        new_code = patch_code(old_code, init_block)
        
        # Verify with diff
        diff = list(difflib.unified_diff(
            old_code.splitlines(),
            new_code.splitlines(),
            lineterm="",
            n=2
        ))
        
        print(f"\n=== Diff for {title} ({mod_type}) ===")
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                print(f"\033[92m{line}\033[0m")
            elif line.startswith('-') and not line.startswith('---'):
                print(f"\033[91m{line}\033[0m")
            else:
                print(line)
                
        deleted = [l for l in diff if l.startswith("-") and not l.startswith("---")]
        assert len(deleted) == 0, f"Unexpected deletions in {title}: {deleted}"
        
        # Update metadata
        df.at[idx, 'code'] = new_code
        df.at[idx, 'num_lines'] = len(new_code.splitlines())
        df.at[idx, 'num_char'] = len(new_code)
        df.at[idx, 'num_comments'] = sum(1 for l in new_code.splitlines() if l.strip().startswith('#'))
        
        print(f"Patched {title}: Lines {row['num_lines']} -> {df.at[idx, 'num_lines']}, Comments {row['num_comments']} -> {df.at[idx, 'num_comments']}")
        
    df.to_excel('data/pdedata_clean_v4_base.xlsx', index=False)
    print("\nSaved patched data to pdedata_clean_v4_base.xlsx")

if __name__ == '__main__':
    main()
