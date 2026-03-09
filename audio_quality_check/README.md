# Audio Quality Check Pipeline

This directory contains automated tools for evaluating the quality of the watermarked audio and visualizing the results as statistical boxplots and distribution plots.

The pipeline consists of two main stages:
1. **Metrics Calculation (`evaluate_quality.py`)**: Scans audio files and calculates objective quality metrics (SNR, LSD, PESQ, STOI, UTMOS, and SI-SNR/$\Delta$SI-SNR).
2. **Visualization (`plot_quality_results.py`)**: Reads the computed CSV results and generates grouped boxplots and data distribution plots (histograms with KDE) for each dataset and model.

---

## Expected Directory Structure

To ensure the scripts correctly identify the original and watermarked audio pairs, as well as extract the dataset and model names for plotting, please organize your audio files according to the following structure (maintaining the relative hierarchy of the bottom 4 levels):

    Target_Directory/ (e.g., my_data/transferbility_audio/)
     └── Dataset_Name/             <-- 4th level from the bottom (X-axis in plots)
          └── Model_Name/          <-- 3rd level from the bottom (Legend in plots)
               └── Audio_Clip/     <-- Bottom level directory
                    ├── 1_original.wav      # Clean original audio
                    └── 2_watermarked.wav   # Watermarked audio

*Note: The plotting script will automatically remove the `_full_benchmark` suffix from dataset names for cleaner visualizations.*

---

## Step 1: Quality Evaluation (`evaluate_quality.py`)

This script recursively scans the specified directory for paired `1_original.wav` and `2_watermarked.wav` files and computes quality metrics including SNR, LSD, PESQ, STOI, UTMOS, and SI-SNR. The UTMOS metric utilizes a neural network model, so running it on a GPU is highly recommended to significantly reduce computation time.

### Basic Usage

    python evaluate_quality.py --dir <target_directory_path> --out <output_csv_filename>

---

## Step 2: Visualization (`plot_quality_results.py`)

This script reads the CSV file `output_csv_filename` generated in Step 1 and uses Seaborn to create high-quality boxplots (with outliers removed). It generates a `.png` image for every metric present in the CSV, including `utmos_watermarked`, `utmos_clean`, and `delta_si_snr`. Additionally, it automatically generates distribution plots (Histplots with KDE curves) for each metric per dataset.

### Usage

Pass the CSV file path via command-line arguments. You can also optionally specify an output directory name or use the `--paper` flag for publication-ready figures.

    # Basic usage (default output directory is plots_quality_results)
    python plot_quality_results.py --csv <output_csv_filename>

    # Specify a custom output directory
    python plot_quality_results.py --csv <output_csv_filename> --out plots_baseline_results

    # Generate plots without titles (ideal for LaTeX/Word paper insertion)
    python plot_quality_results.py --csv <output_csv_filename> --paper

---

## Dependencies & Environment Setup

Ensure the required Python packages are installed before running the scripts.

To install dependencies:
    
    pip install -r requirements.txt

*(The UTMOS model will be automatically downloaded via `torch.hub` during the first run.)*