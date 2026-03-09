import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import datetime


def extract_info(path):
    """
    Extract Dataset and Model names from the file path.
    Assuming path structure is: .../Dataset_Name/Model_Name/Audio_Dir/1_original.wav
    """
    try:
        parts = path.split("/")
        dataset = parts[-4]  # The 4th level from the bottom is Dataset

        # Clean up dataset labels
        dataset = dataset.replace("_full_benchmark", "")

        model = parts[-3]  # The 3rd level from the bottom is Model
        model = model.replace("Manifold", "Latent")
        model = model.replace("Semantic", "Latent-")
        return pd.Series([dataset, model])
    except Exception:
        return pd.Series(["Unknown", "Unknown"])


def main():

    parser = argparse.ArgumentParser(
        description="Plot Boxplots and Distributions from Quality Results CSV"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Input CSV file path (e.g., quality_results.csv)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="plots_quality_results",
        help="Output image folder name",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Add this parameter to NOT show chart titles and generate pdf",
    )
    args = parser.parse_args()

    # --------------------------------
    # Read data
    # --------------------------------
    csv_path = args.csv
    if not os.path.exists(csv_path):
        print(f"Cannot find {csv_path}, please check the file location.")
        return

    df = pd.read_csv(csv_path)

    # --------------------------------
    # Data preprocessing
    # --------------------------------
    df[["dataset", "model"]] = df["clean"].apply(extract_info)

    # Handle STOI outliers
    if "stoi" in df.columns:
        df.loc[df["stoi"] <= 1.01e-5, "stoi"] = np.nan

    # --------------------------------
    # Plot settings
    # --------------------------------
    output_dir = args.out
    dist_dir = os.path.join(
        output_dir, "distributions"
    )  # Distribution plot subdirectory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(dist_dir, exist_ok=True)

    log_path = os.path.join(output_dir, "run_info.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(
            f"Execution time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        f.write(f"Source CSV file: {args.csv}\n")
        f.write(f"Output folder: {args.out}\n")
        f.write(f"For paper: {not args.paper}\n")

    metrics = [
        "si_snr_watermarked",
        "delta_si_snr",
        "snr",
        "lsd",
        "pesq",
        "stoi",
        "utmos_watermarked",
        "utmos_clean",
    ]
    sns.set_theme(style="whitegrid")

    # --------------------------------
    # Plot Boxplots
    # --------------------------------
    print("Plotting Boxplots...")
    for metric in metrics:
        if metric not in df.columns:
            continue

        plt.figure(figsize=(12, 6))

        ax = sns.boxplot(
            data=df,
            x="dataset",
            y=metric,
            hue="model",
            palette="Set2",
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
            },
            showfliers=False,
        )

        if "utmos" in metric:
            plt.ylim(0, 5)

        # ===============================================
        # Draw Clean baselines of each Dataset for UTMOS
        # ===============================================
        if metric == "utmos_watermarked" and "utmos_clean" in df.columns:
            # Calculate the clean mean for each dataset
            clean_means = df.groupby("dataset")["utmos_clean"].mean()

            # Get all dataset names on the X-axis and their coordinate positions (0, 1, 2...)
            xtick_labels = [t.get_text() for t in ax.get_xticklabels()]

            # Draw a horizontal red dashed line within the boxplot range of each dataset
            for i, ds in enumerate(xtick_labels):
                if ds in clean_means:
                    plt.hlines(
                        y=clean_means[ds],
                        xmin=i - 0.4,
                        xmax=i + 0.4,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        zorder=5,
                    )

            # Draw a hidden line purely to make it show up in the Legend
            plt.plot(
                [],
                [],
                color="red",
                linestyle="--",
                linewidth=2,
                label="Clean Baseline (Mean)",
            )
        # ==========================================

        if not args.paper:
            plt.title(
                f"{metric.upper()} Boxplot Comparison Across Datasets",
                fontsize=14,
                fontweight="bold",
                pad=15,
            )

        plt.xlabel("Dataset", fontsize=12, fontweight="bold")
        plt.ylabel(metric, fontsize=12, fontweight="bold")

        # Move the legend outside to the side of the chart to avoid blocking data
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(
            handles=handles,
            labels=labels,
            title="Watermark Method",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/{metric}_boxplot.png",
            dpi=300,
            bbox_inches="tight",
        )

        if args.paper:
            plt.savefig(
                f"{output_dir}/{metric}_boxplot.pdf",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close()

    # --------------------------------
    # Plot Distribution Plots
    # --------------------------------
    print("Plotting distribution plots...")
    unique_datasets = df["dataset"].unique()

    for metric in metrics:
        if metric not in df.columns or df[metric].isnull().all():
            continue

        for ds in unique_datasets:
            # Filter data for a specific Dataset
            sub_df = df[df["dataset"] == ds]

            if sub_df.empty:
                continue

            plt.figure(figsize=(10, 6))

            # Use histplot to plot distribution and add KDE curve
            # hue="model" will show different models in different colors
            try:
                sns.histplot(
                    data=sub_df,
                    x=metric,
                    hue="model",
                    kde=False,
                    element="step",  # Make overlapping parts clearer
                    palette="husl",
                    alpha=0.4,
                )

                if not args.paper:
                    plt.title(
                        f"{metric.upper()} Distribution - Dataset: {ds}",
                        fontsize=14,
                        fontweight="bold",
                    )
                plt.xlabel(metric, fontsize=12)
                plt.ylabel("Frequency", fontsize=12)

                # Save the image, filename includes dataset and metric
                file_name = f"{ds}_{metric}_dist.png".replace(" ", "_")
                plt.savefig(f"{dist_dir}/{file_name}", dpi=300, bbox_inches="tight")
            except Exception as e:
                print(
                    f"Error occurred while plotting distribution for {ds} - {metric}: {e}"
                )
            finally:
                plt.close()

    print(f"\n🎉 Plotting complete!")
    print(f"Boxplots saved to: ./{output_dir}/")
    print(f"Distribution Plots saved to: ./{dist_dir}/")


if __name__ == "__main__":
    main()
