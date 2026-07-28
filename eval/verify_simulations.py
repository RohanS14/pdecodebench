import os
import glob
import runpy
import numpy as np
import traceback

def is_phys_valid(title):
    # based on dataset_overview.md, mod_type contains "Valid" or "InValid"
    # also the title string is something like Wave_Comm_Valid_1
    return "InValid" not in title and "Invalid" not in title

def has_float_arrays(namespace):
    return any(
        isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating)
        for v in namespace.values()
    )


def check_for_anomalies(namespace):
    has_nans = False
    has_spikes = False

    for k, v in namespace.items():
        if isinstance(v, np.ndarray):
            # Check for NaNs or Infs
            try:
                if np.isnan(v).any() or np.isinf(v).any():
                    has_nans = True
            except TypeError:
                # ignore non-numeric arrays
                pass
            
            # Check for spikes (values that are unreasonably large, e.g. > 1e6 for normalized physics sims)
            # Some PDE variables might naturally be larger, but usually > 1e6 indicates instability in these simple 1D solvers
            try:
                if np.max(np.abs(v)) > 1e6:
                    has_spikes = True
            except Exception:
                pass # ignore arrays that can't compute max (e.g. non-numeric)
                
    return has_nans, has_spikes

def main():
    extracted_dir = "data/extracted_codes"
    py_files = glob.glob(os.path.join(extracted_dir, "*.py"))
    
    print(f"Found {len(py_files)} extracted files to verify.")
    
    results = []
    
    for idx, py_file in enumerate(sorted(py_files)):
        title = os.path.splitext(os.path.basename(py_file))[0]
        expected_valid = is_phys_valid(title)
        
        execution_success = False
        nans_detected = False
        spikes_detected = False
        error_msg = None
        
        print(f"[{idx+1}/{len(py_files)}] Executing {title}...")
        
        no_arrays = False
        try:
            namespace = runpy.run_path(py_file)
            execution_success = True
            # Fallback: if no float arrays found, retry with __main__ so that
            # scripts whose computation lives inside if __name__=='__main__'
            # blocks are actually executed and checked.
            if not has_float_arrays(namespace):
                try:
                    namespace = runpy.run_path(py_file, run_name="__main__")
                except Exception:
                    pass
            if not has_float_arrays(namespace):
                no_arrays = True
            nans_detected, spikes_detected = check_for_anomalies(namespace)
        except Exception as e:
            error_msg = str(e)

        results.append({
            "title": title,
            "expected_valid": expected_valid,
            "execution_success": execution_success,
            "no_arrays": no_arrays,
            "nans": nans_detected,
            "spikes": spikes_detected,
            "error": error_msg
        })

    # Summary
    success_count = sum(1 for r in results if r["execution_success"])
    error_count = len(results) - success_count
    
    # Analysis of Invalid detection
    invalid_codes = [r for r in results if not r["expected_valid"] and r["execution_success"]]
    correctly_detected_invalid = [r for r in invalid_codes if r["nans"] or r["spikes"]]
    undetected_invalid = [r for r in invalid_codes if not (r["nans"] or r["spikes"])]
    
    valid_codes = [r for r in results if r["expected_valid"] and r["execution_success"]]
    false_positives = [r for r in valid_codes if r["nans"] or r["spikes"]]

    print("\n" + "="*50)
    print("VERIFICATION SUMMARY")
    print("="*50)
    no_arrays_count = sum(1 for r in results if r.get("no_arrays"))
    print(f"Total Scripts: {len(results)}")
    print(f"Execution Success: {success_count}")
    print(f"Execution Errors: {error_count}")
    print(f"No Arrays Found (inconclusive): {no_arrays_count}")
    if no_arrays_count > 0:
        print("  Scripts with no float arrays (JAX or empty execution):")
        for r in results:
            if r.get("no_arrays"):
                print(f"  - {r['title']}")
    
    if error_count > 0:
        print("\nErrors encountered in the following scripts:")
        for r in results:
            if not r["execution_success"]:
                print(f"  - {r['title']}: {r['error']}")
    
    print("\n" + "="*50)
    print("INVALIDITY DETECTION ANALYSIS")
    print("="*50)
    print(f"Ground Truth Invalid (Executed Successfully): {len(invalid_codes)}")
    
    # Breakdown of detected invalid
    nans_only = len([r for r in correctly_detected_invalid if r["nans"] and not r["spikes"]])
    spikes_only = len([r for r in correctly_detected_invalid if not r["nans"] and r["spikes"]])
    both_anomalies = len([r for r in correctly_detected_invalid if r["nans"] and r["spikes"]])
    
    print(f"Total Detected via NaN/Spike: {len(correctly_detected_invalid)}")
    print(f"  -> Detected via NaNs/Infs only: {nans_only}")
    print(f"  -> Detected via Spikes (>1e6) only: {spikes_only}")
    print(f"  -> Detected via Both: {both_anomalies}")
    
    print(f"\nUndetected Invalid Simulations (False Negatives): {len(undetected_invalid)}")
    if len(undetected_invalid) > 0:
        for r in undetected_invalid:
            print(f"  - {r['title']}")
            
    # Breakdown by PDE class
    print("\n--- Breakdown by PDE Class ---")
    from collections import defaultdict
    class_stats = defaultdict(lambda: {"total": 0, "detected": 0})
    for r in invalid_codes:
        pde = r["title"].split("_")[0]
        class_stats[pde]["total"] += 1
        if r["nans"] or r["spikes"]:
            class_stats[pde]["detected"] += 1
            
    for pde, stats in sorted(class_stats.items()):
        pct = (stats['detected'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {pde}: {stats['detected']} / {stats['total']} detected ({pct:.1f}%)")
    
    print("\nValid Simulation False Positives:")
    print(f"Ground Truth Valid (Executed Successfully): {len(valid_codes)}")
    print(f"Anomalies incorrectly detected in Valid code: {len(false_positives)}")
    if len(false_positives) > 0:
        for r in false_positives:
            print(f"  - {r['title']} (NaNs: {r['nans']}, Spikes: {r['spikes']})")

if __name__ == "__main__":
    main()
