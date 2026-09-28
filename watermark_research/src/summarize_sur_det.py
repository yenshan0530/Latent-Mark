import os
import argparse
import pandas as pd

def generate_watermark_report(base_dir):
    all_detectability = []
    all_survivability = []

    # 1. Walk through the directory structure
    # Expected: BASE_DIR / Dataset / Method / file.csv
    for dataset in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
            
        # Check subdirectories (Watermarking Methods)
        for method_dir in os.listdir(dataset_path):
            method_path = os.path.join(dataset_path, method_dir)
            if not os.path.isdir(method_path):
                continue
                
            # Path to the specific CSVs
            det_file = os.path.join(method_path, "combined_detectability_results.csv")
            surv_file = os.path.join(method_path, "qwen_benchmark_results.csv")
            
            # 2. Process Detectability (Accuracy)
            if os.path.exists(det_file):
                df_det = pd.read_csv(det_file)
                # We take the mean accuracy for the method in this dataset
                for _, row in df_det.iterrows():
                    all_detectability.append({
                        "Dataset": dataset,
                        "Method": row["Method"],
                        "Value": row["Accuracy"]
                    })
            
            # 3. Process Survivability (Pass Rate)
            if os.path.exists(surv_file):
                df_surv = pd.read_csv(surv_file)
                # Calculate the percentage of "PASS" results
                pass_rate_stats = df_surv.groupby("Method")["Survivability"].apply(
                    lambda x: (x == "PASS").mean()
                ).reset_index()
                
                for _, row in pass_rate_stats.iterrows():
                    all_survivability.append({
                        "Dataset": dataset,
                        "Method": row["Method"],
                        "Value": row["Survivability"]
                    })

    # 4. Create DataFrames and Merge
    df_det_final = pd.DataFrame(all_detectability)
    df_surv_final = pd.DataFrame(all_survivability)
    
    # Pivot Detectability
    pivot_det = df_det_final.pivot(index="Method", columns="Dataset", values="Value")
    pivot_det.columns = pd.MultiIndex.from_product([pivot_det.columns, ["Detectability"]])
    
    # Pivot Survivability
    pivot_surv = df_surv_final.pivot(index="Method", columns="Dataset", values="Value")
    pivot_surv.columns = pd.MultiIndex.from_product([pivot_surv.columns, ["Survivability"]])
    
    # Combine both pivots
    final_table = pd.concat([pivot_det, pivot_surv], axis=1).sort_index(axis=1)
    
    return final_table

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pivot per-method detectability and survivability into one table (expects results_dir/Dataset/Method/*.csv).")
    ap.add_argument("--results_dir", default="../results_snac", help="Results root")
    ap.add_argument("--out", default="watermark_summary_table", help="Output basename (writes .xlsx and .csv)")
    args = ap.parse_args()
    report = generate_watermark_report(args.results_dir)
    
    # Print to console
    print("\n--- Watermark Benchmark Table ---")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(report)
    
    # Export to Excel (best for viewing sub-columns/MultiIndex)
    output_excel = args.out + ".xlsx"
    report.to_excel(output_excel)
    print(f"\nTable exported to {output_excel}")
    
    # Optional: Export to CSV (Note: CSV doesn't support visual sub-columns well)
    report.to_csv(args.out + ".csv")
