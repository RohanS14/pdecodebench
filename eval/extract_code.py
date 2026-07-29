import pandas as pd
import os

def extract_code(excel_path, out_dir):
    print(f"Reading from {excel_path}...")
    df = pd.read_csv(excel_path) if str(excel_path).endswith(".csv") else pd.read_excel(excel_path)
    
    os.makedirs(out_dir, exist_ok=True)
    
    seen_titles = set()
    num_duplicates = 0
    num_extracted = 0
    
    for idx, row in df.iterrows():
        title = str(row['title']).strip()
        code = row['code']
        
        if pd.isna(code):
            print(f"Warning: Code missing for row {idx} with title {title}. Skipping.")
            continue
            
        base_title = title
        suffix = 2
        while title in seen_titles:
            title = f"{base_title}_{suffix}"
            suffix += 1
            
        if title != base_title:
            print(f"Warning: Duplicate title '{base_title}' found. Saving as '{title}.py'")
            num_duplicates += 1
            
        seen_titles.add(title)
        
        file_path = os.path.join(out_dir, f"{title}.py")
        with open(file_path, "w", encoding="utf-8") as f:
            # We preserve exact newlines and tabs by writing the string directly
            # but we remove the literal '\\n' that appears before actual newlines
            clean_code = str(code).replace('\\n\n', '\n').replace('\\n', '\n')
            f.write(clean_code)
            
        num_extracted += 1
        
    print(f"\nExtraction complete!")
    print(f"Extracted {num_extracted} files.")
    if num_duplicates > 0:
        print(f"Handled {num_duplicates} duplicate titles.")

if __name__ == "__main__":
    excel_path = "data/pdedata_clean_v4.xlsx"
    out_dir = "data/extracted_codes"
    extract_code(excel_path, out_dir)
