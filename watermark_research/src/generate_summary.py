import os
import csv
import glob
import argparse
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser(description="Print a per-dataset table of pass rates by optimization set x attack codec from transferbility_testing.py benchmark_summary_*.csv files.")
    ap.add_argument("--results_dir", default="../results_transferability", help="Output root of transferbility_testing.py")
    args = ap.parse_args()
    base_dir = args.results_dir
    
    # We want a table for each dataset
    
    for dataset in sorted(os.listdir(base_dir)):
        dataset_path = os.path.join(base_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue
            
        csv_files = glob.glob(os.path.join(dataset_path, 'benchmark_summary_*.csv'))

        if not csv_files:
            continue
            
        print(f"\n=== Dataset: {dataset} ===")
        
        # summary_data[opt_comb][attack_model] = "PASS/Total (rate%)"
        summary_data = defaultdict(dict)
        opt_combs = set()
        attack_models = set()
        
        for csv_file in csv_files:
            filename = os.path.basename(csv_file)
            # Remove prefix and suffix
            model_name = filename.replace('benchmark_summary_', '').replace('.csv', '')
            parts = model_name.split('_')
            
            # parts expected to be [codec1, codec2, codec3, attack]
            if len(parts) >= 4:
                attack_model = parts[-1]
                opt_comb = '_'.join(parts[:-1])
            else:
                attack_model = "unknown"
                opt_comb = model_name
                
            opt_combs.add(opt_comb)
            attack_models.add(attack_model)
            
            try:
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    pass_count = 0
                    total_count = 0
                    
                    # Columns might be Method, FAIL, Total, PASS_rate, PASS, ERROR
                    for row in reader:
                        if 'PASS' in row and 'Total' in row:
                            pass_count += int(row['PASS'])
                            total_count += int(row['Total'])
                    
                    if total_count > 0:
                        pass_rate = (pass_count / total_count) * 100
                        summary_data[opt_comb][attack_model] = f"{pass_count}/{total_count} ({pass_rate:.1f}%)"
                    else:
                        summary_data[opt_comb][attack_model] = "0/0 (0.0%)"
            except Exception as e:
                summary_data[opt_comb][attack_model] = "Error"
                
        opt_combs = sorted(list(opt_combs))
        attack_models = sorted(list(attack_models))
        
        if not opt_combs:
            continue
            
        # Determine column widths
        opt_col_width = max(len(c) for c in opt_combs) if opt_combs else 20
        # Include header in calculation
        opt_col_width = max(opt_col_width, len("Optimized Combination"))
        
        col_widths = [opt_col_width]
        for am in attack_models:
            max_w = len(am)
            for oc in opt_combs:
                cell_val = summary_data[oc].get(am, "-")
                max_w = max(max_w, len(cell_val))
            col_widths.append(max_w)
            
        def format_row(row_data):
            return " | ".join(f"{str(item):<{width}}" for item, width in zip(row_data, col_widths))
            
        # Print header
        print(format_row(["Optimized Combination"] + attack_models))
        print("-" * (sum(col_widths) + 3 * len(col_widths) - 3))
        
        # Print rows
        for oc in opt_combs:
            row = [oc]
            for am in attack_models:
                row.append(summary_data[oc].get(am, "-"))
            print(format_row(row))

if __name__ == "__main__":
    main()
