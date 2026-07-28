import pandas as pd


def remove_asserts(code):
    lines = code.split('\n')
    lines = [l for l in lines if not l.strip().startswith('assert ')]
    return '\n'.join(lines)


def main():
    print("Loading pdedata_clean_v4_base.xlsx...")
    df = pd.read_excel('data/pdedata_clean_v4_base.xlsx')

    # Strip assertions from ALL InValid scripts across all PDE classes
    # Valid scripts are left untouched (by user decision)
    invalid_mask = (
        df['mod_type'].str.contains('InValid', na=False) |
        df['mod_type'].str.contains('Invalid', na=False)
    )

    before = df.loc[invalid_mask, 'code'].str.count('assert ').sum()
    df.loc[invalid_mask, 'code'] = df.loc[invalid_mask, 'code'].apply(remove_asserts)
    after = df.loc[invalid_mask, 'code'].str.count('assert ').sum()

    n_scripts = invalid_mask.sum()
    print(f"Processed {n_scripts} InValid scripts.")
    print(f"Removed {int(before - after)} assertion lines.")

    df.to_excel('data/pdedata_clean_v4_base.xlsx', index=False)
    print("Saved pdedata_clean_v4_base.xlsx.")


if __name__ == "__main__":
    main()
