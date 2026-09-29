import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------
# CONFIG
# --------------------------------
ap = argparse.ArgumentParser(description="Per-metric histograms with the clean-audio mean marked, one CSV per method.")
ap.add_argument("--dir", default=".", help="folder holding <Method>_quality_results.csv files from evaluate_quality.py")
ap.add_argument("--out", default="plots_distributions")
args = ap.parse_args()

METHODS = ["AudioSeal", "Latent-Cluster", "Latent-PCA", "Latent-Random", "SilentCipher", "WavMark"]
csv_files = {m: os.path.join(args.dir, f"{m}_quality_results.csv") for m in METHODS}

metrics = ["si_snr_watermarked", "delta_si_snr", "snr", "lsd", "pesq", "stoi"]

# Histogram colors per model
model_colors = {
    "AudioSeal": "#1f77b4",
    "Latent-Cluster": "#ff7f0e",
    "Latent-PCA": "#2ca02c",
    "Latent-Random": "#d62728",
    "SilentCipher": "#9467bd",
    "WavMark": "#8c564b",
}

# --------------------------------
# LOAD DATA
# --------------------------------
dfs = []
for model, path in csv_files.items():
    if not os.path.exists(path):
        print(f"[WARN] Missing {path}")
        continue
    df = pd.read_csv(path)
    df["model"] = model
    dfs.append(df)

all_data = pd.concat(dfs, ignore_index=True)

# --------------------------------
# PLOTS
# --------------------------------
output_dir = args.out
os.makedirs(output_dir, exist_ok=True)

for metric in metrics:
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot histograms + model-colored mean/std lines (BUT NO LEGEND ENTRIES)
    for model in all_data["model"].unique():
        sub = all_data[all_data["model"] == model]
        color = model_colors[model]

        # Histogram
        ax.hist(
            sub[metric],
            bins=30,
            density=False,
            alpha=0.25,
            color=color,
            label=model,   # <-- We will override legend later
        )

        # Mean & Std
        mean = sub[metric].mean()
        std = sub[metric].std()

        # Mean line — colored, NO legend entry
        ax.axvline(
            mean,
            color=color,
            linestyle="-",
            linewidth=2,
        )

        # Std lines — colored, NO legend entry
        ax.axvline(
            mean - std,
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
        )
        ax.axvline(
            mean + std,
            color=color,
            linestyle="--",
            linewidth=1.5,
            alpha=0.5,
        )

    # ax.set_title(f"Distribution of {metric}", fontsize=16)
    ax.set_xlabel(metric)
    ax.set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/{metric}_distribution.pdf", dpi=300)
    plt.savefig(f"{output_dir}/{metric}_distribution.png", dpi=300)

    plt.close()

    # ------------------------------------
    # Create SINGLE legend for line styles
    # ------------------------------------
    import matplotlib.lines as mlines

    mean_line = mlines.Line2D([], [], color="black", linestyle="-", linewidth=2, label="Mean")
    std_line = mlines.Line2D([], [], color="black", linestyle="--", linewidth=1.5, label="Std (±1)")

    # Model legend (histograms only)
    handles_hist, labels_hist = ax.get_legend_handles_labels()
    # Remove duplicates
    model_legend = dict(zip(labels_hist, handles_hist))

    # Collect every legend entry
    all_handles = list(model_legend.values()) + [mean_line, std_line]
    all_labels = list(model_legend.keys()) + ["Mean", "Std (±1)"]

    # 1. Create a small figure
    # Adjust figsize (width, height) to the number of labels
    fig_leg = plt.figure(figsize=(4, 2))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis('off')  # hide axes

    # 2. Draw the legend on the new figure
    # ncol sets the number of columns; use ncol=2 for many labels
    leg = ax_leg.legend(
        # all_handles,
        # all_labels,
        handles_hist,
        labels_hist,
        loc='center',
        fontsize=14,
        frameon=True
    )

    # 3. Save
    # bbox_inches='tight' trims whitespace; pad_inches sets the margin
    fig_leg.savefig('sound_quality_legend.pdf', bbox_inches='tight', pad_inches=0.1, dpi=300)

    # # Combine into one legend box
    # ax.legend(
    #     list(model_legend.values()) + [mean_line, std_line],
    #     list(model_legend.keys()) + ["Mean", "Std (±1)"],
    #     loc="upper left",
    #     fontsize=14
    # )


print(f"Saved histogram plots to ./{output_dir}/")
