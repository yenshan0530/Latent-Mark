import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def extract_info(path):
    try:
        parts = path.split("/")
        dataset = parts[-4].replace("_full_benchmark", "")
        model = parts[-3]
        return pd.Series([dataset, model])
    except Exception:
        return pd.Series(["Unknown", "Unknown"])


def load_and_merge_csvs(csv_paths):
    """Read and concatenate several CSVs; return an empty DataFrame if none exist."""
    dfs = []
    if not csv_paths:
        return pd.DataFrame()

    for csv_path in csv_paths:
        if not os.path.exists(csv_path):
            print(f"Not found, skipping: {csv_path}")
            continue
        print(f"Reading: {csv_path}")
        dfs.append(pd.read_csv(csv_path))

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    df[["dataset", "model"]] = df["clean"].apply(extract_info)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Generate Paper-Ready Plots (Separate Flags for Metrics)"
    )
    parser.add_argument(
        "--delta_si_snr_csv",
        type=str,
        nargs="+",
        help="CSV file(s) with delta_si_snr values (from evaluate_quality.py)",
    )
    parser.add_argument(
        "--utmos_csv",
        type=str,
        nargs="+",
        help="CSV file(s) with utmos_watermarked values (from evaluate_quality.py)",
    )
    parser.add_argument(
        "--out", type=str, default="plots_paper_figures", help="Output folder"
    )
    args = parser.parse_args()

    if not args.delta_si_snr_csv and not args.utmos_csv:
        print("Provide at least one of --delta_si_snr_csv or --utmos_csv.")
        return

    output_dir = args.out
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # --------------------------------
    # 1. Load and prepare each metric's data
    # --------------------------------
    df_delta_raw = load_and_merge_csvs(args.delta_si_snr_csv)
    df_utmos_raw = load_and_merge_csvs(args.utmos_csv)

    exclude_models = ["Latent-Joint"]

    # Delta SI-SNR
    if not df_delta_raw.empty and "delta_si_snr" in df_delta_raw.columns:
        df_delta = df_delta_raw.dropna(subset=["delta_si_snr"]).copy()

        # Drop datasets that are not plotted
        target_datasets_delta = ["Clotho", "LibriSpeech", "DAPS", "PCD", "jaCappella"]
        df_delta = df_delta[df_delta["dataset"].isin(target_datasets_delta)]

        # Drop models that are not plotted
        df_delta = df_delta[~df_delta["model"].isin(exclude_models)]
    else:
        df_delta = pd.DataFrame(columns=["dataset", "model", "delta_si_snr"])

    # UTMOS
    if not df_utmos_raw.empty and "utmos_watermarked" in df_utmos_raw.columns:
        df_utmos = df_utmos_raw.dropna(subset=["utmos_watermarked"]).copy()

        # Drop datasets that are not plotted
        target_datasets_utmos = ["DAPS", "LibriSpeech"]
        df_utmos = df_utmos[df_utmos["dataset"].isin(target_datasets_utmos)]

        # Drop models that are not plotted
        df_utmos = df_utmos[~df_utmos["model"].isin(exclude_models)]
    else:
        df_utmos = pd.DataFrame(columns=["dataset", "model", "utmos_watermarked"])

    # ==========================================
    # Global model list, sorted, used as hue_order
    # ==========================================
    all_models = set()
    if not df_delta.empty:
        all_models.update(df_delta["model"].unique())
    if not df_utmos.empty:
        all_models.update(df_utmos["model"].unique())

    preferred_order = [
        "AudioSeal",
        "SilentCipher",
        "WavMark",
        "Latent-Random",
        "Latent-PCA",
        "Latent-Cluster",
    ]

    # Filter and sort by preferred_order
    model_order = [m for m in preferred_order if m in all_models]

    # Models present in the data but absent from preferred_order are appended at the end
    for m in sorted(list(all_models)):
        if m not in model_order:
            model_order.append(m)

    print(f"Model order ({len(model_order)}): {model_order}")

    # Figure width scales with the number of datasets
    n_delta = len(df_delta["dataset"].unique())
    n_utmos = len(df_utmos["dataset"].unique())

    # Height, and base width per dataset
    plot_height = 5
    width_per_dataset = 1.5
    base_width = 1.0

    width_delta = n_delta * width_per_dataset + base_width
    width_utmos = n_utmos * width_per_dataset + base_width

    handles, labels = None, None

    # ==========================================
    # 2. Plot 1: Delta SI-SNR
    # ==========================================
    if n_delta > 0:
        fig_delta, ax_delta = plt.subplots(figsize=(width_delta, plot_height))

        sns.boxplot(
            data=df_delta,
            x="dataset",
            y="delta_si_snr",
            hue="model",
            hue_order=model_order,  # fixed order and colors
            palette="Set2",
            showmeans=True,
            showfliers=False,
            ax=ax_delta,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
            },
        )
        ax_delta.set_xlabel("Dataset", fontsize=12, fontweight="bold")
        ax_delta.set_ylabel(r"$\Delta$ SI-SNR (dB)", fontsize=12, fontweight="bold")

        handles, labels = ax_delta.get_legend_handles_labels()
        if ax_delta.get_legend() is not None:
            ax_delta.get_legend().remove()

        fig_delta.tight_layout()

        # PNG with title
        ax_delta.set_title(
            r"$\Delta$ SI-SNR (Higher is better)",
            fontsize=14,
            fontweight="bold",
            pad=10,
        )
        fig_delta.savefig(
            os.path.join(output_dir, "paper_figure_delta_si_snr.png"),
            dpi=300,
            bbox_inches="tight",
        )

        # PDF without title
        ax_delta.set_title("")
        fig_delta.savefig(
            os.path.join(output_dir, "paper_figure_delta_si_snr.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        plt.close(fig_delta)
        print(f"Saved Delta SI-SNR figure (width {width_delta} in)")

    # ==========================================
    # 3. Plot 2: UTMOS
    # ==========================================
    if n_utmos > 0:
        fig_utmos, ax_utmos = plt.subplots(figsize=(width_utmos, plot_height))

        sns.boxplot(
            data=df_utmos,
            x="dataset",
            y="utmos_watermarked",
            hue="model",
            hue_order=model_order,  # fixed order and colors
            palette="Set2",
            showmeans=True,
            showfliers=False,
            ax=ax_utmos,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
            },
        )
        ax_utmos.set_xlabel("Dataset", fontsize=12, fontweight="bold")
        ax_utmos.set_ylabel("MOS Score", fontsize=12, fontweight="bold")
        ax_utmos.set_ylim(0, 5)

        if not handles and not labels:
            handles, labels = ax_utmos.get_legend_handles_labels()

        if ax_utmos.get_legend() is not None:
            ax_utmos.get_legend().remove()

        fig_utmos.tight_layout()

        # PNG with title
        ax_utmos.set_title(
            "UTMOS (Higher is better)", fontsize=14, fontweight="bold", pad=10
        )
        fig_utmos.savefig(
            os.path.join(output_dir, "paper_figure_utmos.png"),
            dpi=300,
            bbox_inches="tight",
        )

        # PDF without title
        ax_utmos.set_title("")
        fig_utmos.savefig(
            os.path.join(output_dir, "paper_figure_utmos.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        plt.close(fig_utmos)
        print(f"Saved UTMOS figure (width {width_utmos} in)")

    # ==========================================
    # 4. Standalone legend
    # ==========================================
    if handles and labels:
        fig_leg = plt.figure(figsize=(8, 1.2))
        ax_leg = fig_leg.add_subplot(111)
        ax_leg.axis("off")

        leg = ax_leg.legend(
            handles, labels, loc="center", ncol=len(labels), frameon=False, fontsize=12
        )

        # Bold title with padding
        leg.set_title("Watermark Method\n", prop={"size": 12})
        leg.get_title().set_multialignment("center")

        # PNG with title
        fig_leg.savefig(
            os.path.join(output_dir, "paper_figure_legend_only.png"),
            dpi=300,
            bbox_inches="tight",
        )

        # PDF drops the title; comment this out to keep it
        # leg.set_title("")

        fig_leg.savefig(
            os.path.join(output_dir, "paper_figure_legend_only.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        plt.close(fig_leg)
        print("Saved standalone legend")


if __name__ == "__main__":
    main()
