import os
import pandas as pd
import glob
import argparse

def main():
    ap = argparse.ArgumentParser(description="Build the detectability / survivability summary table (Table 1 layout) from watermark_testing.py outputs.")
    ap.add_argument("--results_dir", default="../results_snac", help="Output root of watermark_testing.py (one subfolder per dataset)")
    ap.add_argument("--out", default="watermark_summary_table.csv", help="Output CSV path")
    args = ap.parse_args()
    results_dir = args.results_dir
    datasets = sorted([d.name for d in os.scandir(results_dir) if d.is_dir()])
    
    # Store aggregated data: method -> { dataset -> { "detect_acc": val, "survive_acc": val } }
    data = {}
    
    for dataset in datasets:
        dataset_dir = os.path.join(results_dir, dataset)
        
        # 1. Detectability Info
        detect_csv = os.path.join(dataset_dir, "combined_detectability_results.csv")
        if os.path.exists(detect_csv):
            df_det = pd.read_csv(detect_csv)
            for _, row in df_det.iterrows():
                method = row['Method']
                acc = row['Accuracy']
                if method not in data:
                    data[method] = {}
                if dataset not in data[method]:
                    data[method][dataset] = {"detect_acc": None, "survive_acc": None}
                data[method][dataset]["detect_acc"] = acc
        
        # 2. Survivability Info
        qwen_csv = os.path.join(dataset_dir, "qwen_benchmark_results.csv")
        if os.path.exists(qwen_csv):
            df_qwen = pd.read_csv(qwen_csv)
            # grouped by method
            groups = df_qwen.groupby("Method")
            for method, group in groups:
                if method not in data:
                    data[method] = {}
                if dataset not in data[method]:
                    data[method][dataset] = {"detect_acc": None, "survive_acc": None}
                # calculate accuracy: PASS / (PASS + FAIL)
                passes = (group['Survivability'] == 'PASS').sum()
                total = len(group)
                survive_acc = passes / total if total > 0 else 0
                data[method][dataset]["survive_acc"] = survive_acc

    # Format into DataFrame with MultiIndex columns
    # Rows: Method
    # Columns: Dataset -> (Detect_Acc, Survive_Acc)
    methods = list(data.keys())
    
    multi_columns = pd.MultiIndex.from_tuples(
        [(ds, "Detect") for ds in datasets] + [(ds, "Survive") for ds in datasets],
        names=["Dataset", "Metric"]
    )
    
    table = pd.DataFrame(index=methods, columns=multi_columns)
    
    for method in methods:
        for ds in datasets:
            if ds in data[method]:
                metrics = data[method][ds]
                if metrics["detect_acc"] is not None:
                    table.loc[method, (ds, "Detect")] = f'{metrics["detect_acc"]:.2f}'
            if metrics["survive_acc"] is not None:
                table.loc[method, (ds, "Survive")] = f'{metrics["survive_acc"]:.2f}'

    # Sort columns alphabetically by Dataset, then by Metric
    table = table.sort_index(axis=1, level=[0, 1])

    # --- Save to CSV ---
    output_csv = args.out
    table.to_csv(output_csv)

    print(f"Table successfully saved to {output_csv}")
    print("\n--- Summary Table ---")
    print(table.to_string())

if __name__ == "__main__":
    main()
